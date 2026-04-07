# Code Architecture

## nanochat/gpt.py (The Transformer Model)

Implements the GPT language model architecture.

**Key components:**

- **GPTConfig** dataclass holding: `sequence_len`, `vocab_size`, `n_layer`, `n_head`, `n_kv_head`, `n_embd`, `window_pattern`
- **CausalSelfAttention** (the attention mechanism):
  - Projects input to Q, K, V via separate linear layers (`c_q`, `c_k`, `c_v`)
  - Applies Rotary Position Embeddings (RoPE) to Q and K, encoding position without a lookup table
  - Runs QK Norm (RMSNorm on Q and K before attention)
  - Calls `flash_attn` for the actual attention computation
  - Optional Value Embeddings for alternating layers (ResFormer-style)
- **MLP** feed-forward network: `c_fc` → ReLU² → `c_proj`
- **GPT.forward()** full forward pass:
  - Token embeddings via `wte`
  - Smear: mixes previous token's embedding into current (cheap bigram signal)
  - Loop through transformer blocks: `resid_lambdas[i] * x + x0_lambdas[i] * x0` → attention → MLP
  - Backout: subtracts mid-layer residual to remove low-level features
  - Final RMSNorm → LM head → softcap logits → cross-entropy loss

This is the core model. Every training run and inference call goes through it.

---

## nanochat/optim.py (MuonAdamW Optimizer)

Two fused optimizers: `MuonAdamW` (single GPU) and `DistMuonAdamW` (multi-GPU).

**Muon** (for matrix params like Q/K/V/MLP weights):
- Runs standard SGD momentum first
- Applies Polar Express, a Newton-Schulz-style iteration that orthogonalizes the update matrix (forces columns to be orthonormal)
- Adds NorMuon variance reduction: per-neuron adaptive LR that normalizes update scales after orthogonalization
- Result: more stable training with better-conditioned updates

**AdamW** (for embeddings, `lm_head`, scalars):
- Standard AdamW with decoupled weight decay
- Fused into a single compiled kernel

**DistMuonAdamW** (distributed):
- ZeRO-2 style: gradients are reduce-scattered so each rank holds 1/N of the gradient
- Optimizer state is sharded per rank (not replicated)
- 3-phase async pipeline: (1) launch all reduce ops, (2) wait + compute updates + launch gathers, (3) wait for gathers
- Communication overlapped with computation

Muon consistently outperforms AdamW on transformers. The fused kernels eliminate Python overhead. The distributed version enables multi-GPU training without PyTorch DDP overhead.

---

## nanochat/flash_attention.py (Unified Attention Interface)

Auto-switches between Flash Attention 3 (Hopper GPUs) and PyTorch SDPA fallback.

- On import, detects GPU architecture via `torch.cuda.get_device_capability()`
- If sm90 (Hopper/H100): loads FA3 from `varunneal/flash-attention-3` kernels
- Otherwise (Ada, Blackwell, MPS, CPU): uses PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`)
- Exposes two functions matching FA3's API: `flash_attn_func()` for training, `flash_attn_with_kvcache()` for inference

FA3 is 2-3x faster than SDPA on H100s. The transparent fallback means the rest of the codebase never needs to know which backend is active.

---

## nanochat/common.py (Shared Infrastructure)

Central utilities reused across all scripts.

- **COMPUTE_DTYPE**: auto-detected compute precision, `bfloat16` on Ampere+ (SM80+), `float32` otherwise. Set via `NANOCHAT_DTYPE` env var
- **compute_init()**: initializes CUDA/MPS/CPU device, sets NCCL env vars for DDP, calls `torch.set_float32_matmul_precision("high")` for TF32 matmuls on CUDA
- **get_dist_info()**: detects torchrun env vars (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) for DDP
- **TrainConfig** dataclass: all training hyperparameters in one place
- **DistributedConfig** dataclass: DDP settings auto-detected from torchrun env
- **wrap_model_for_ddp()**: wraps model with `DistributedDataParallel` (NCCL backend), no-op on single GPU
- **reduce_tensor()**, **ddp_barrier()**: cross-rank synchronization helpers
- **get_peak_flops()**: GPU peak FLOPS table for MFU calculation (H100, A100, L4, etc.)
- **ColoredFormatter**: pretty logging with ANSI colors

Keeps device/dtype/DDP logic consistent across all scripts.

---

## nanochat/engine.py (Inference Engine)

Efficient batch autoregressive generation with KV-cache and tool use.

- **KVCache**: pre-allocated `(B, T, H, D)` tensors for keys and values across all layers. FA3-compatible layout. Tracks `cache_seqlens` for variable-length sequences
- **Engine.generate()** full streaming inference:
  - Prefill: runs the full prompt through the model once, caches KV
  - Decode: autoregressive single-token steps, reusing cached KV
  - Tool use: detects `<|python_start|>`/`<|python_end|>` tokens, evaluates Python expressions via `eval()` with sandboxing
- **sample_next_token()**: temperature/top-k sampling from logits
- **RowState**: per-row state for batched generation (tracks forced tokens, python blocks, completion)

The KV-cache avoids recomputing attention for all previous tokens at every step. Batch generation with tool use enables interactive chat.

---

## nanochat/coco_dataset.py (COCO Captions Dataset)

PyTorch Dataset for COCO 2017 caption text.

- Loads `yerevann/coco-karpathy` from HuggingFace (~414K train, ~25K val captions)
- Tokenizes all captions with GPT-2 tiktoken tokenizer
- `collate_fn` pads sequences to max length within each batch, returns `input_ids` + `attention_mask`
- `create_dataloader()` adds `DistributedSampler` when torchrun is detected (auto from env)
- `print_dataset_stats()` reports caption count, avg length, vocab size, percentiles

Phase 1 trains the language model backbone on caption text. COCO is the same dataset used for vision-language training in Phase 2.

---

## nanochat/dataloader.py (Original DataLoader)

Loads nanochat's preprocessed parquet dataset with BOS-aligned best-fit packing.

- Reads parquet files of pre-tokenized documents (from `nanochat/dataset.py`)
- BOS-aligned best-fit packing: packs documents into sequences starting with BOS token, uses best-fit algorithm to minimize cropping
- Handles DDP sharding and training resumption
- `tokenizing_distributed_data_loader_bos_bestfit()` returns infinite iterator of `(x, y, state_dict)` tuples

The original nanochat pretraining uses this with 100% token utilization via BOS alignment.

---

## nanochat/fp8.py (FP8 Training)

Minimal FP8 training for H100+ GPUs (~150 lines, vs torchao's ~2000).

- Wraps `nn.Linear` with `Float8Linear` subclass
- On forward/backward: scales activations/gradients to FP8 range, calls `torch._scaled_mm` (cuBLAS FP8 kernel)
- Uses `float8_e4m3fn` for weights/inputs, `float8_e5m2` for gradients
- Tensorwise scaling recipe: one scale factor per tensor (not per-row)

FP8 matmuls are ~2x faster than BF16 on H100. Nanochat's implementation is 1% of torchao's complexity.

---

## nanochat/checkpoint_manager.py (Checkpoint Save/Load)

Saves and loads model + optimizer + metadata checkpoints.

- `save_checkpoint()` rank-0 writes, others barrier. Saves `.pt` file with model state, optimizer state, and JSON metadata
- `load_checkpoint()` loads model, optimizer state, and training metadata
- Patches missing config keys for backward compatibility (handles checkpoints from older versions)
- `load_model()` convenience function for loading a named model for inference

Enables training resumption, model sharing, and evaluation from saved checkpoints.

---

## nanochat/tokenizer.py (Tokenizer Wrapper)

Wraps tiktoken GPT-2 tokenizer with nanochat-specific special tokens.

- Encodes/decodes text ↔ token IDs
- Special token support: `<|bos|>`, `<|eos|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, etc.
- `encode_special()` respects special token strings
- Used everywhere for preprocessing prompts and decoding outputs

---

## nanochat/report.py (Report and W&B Logging)

Two systems: markdown report generation and W&B logging.

**Report class:**
- Writes structured markdown sections to `~/.cache/nanochat/report/`
- `generate()` assembles all sections into a final `report.md`
- Logs git info, GPU info, system info, cost estimates

**WandbLogger class:**
- `build_wandb_logger()` factory for DDP-aware W&B init (only rank-0)
- `log()` logs all metrics with consistent key prefixes
- `log_summary()` writes final metrics to W&B summary
- `DummyWandb` no-op for non-master ranks
- `get_gpu_memory_mb()` GPU memory helper

Clean separation of local reports and cloud W&B metrics.

---

## Training Pipeline Flow

```
1. python/torchrun script
       ↓
2. common.compute_init()      → device, DDP rank, world_size
       ↓
3. coco_dataset / dataloader  → tokenized batches
       ↓
4. gpt.GPT(config)            → model on meta device
       ↓
5. model.init_weights()       → random init with config
       ↓
6. optim.MuonAdamW()          → optimizer with per-group LRs
       ↓
7. Training loop:
       a. next(dataloader)     → batch of token IDs
       b. model.forward()      → loss (Flash Attention → LM head)
       c. loss.backward()      → gradients
       d. optimizer.step()     → Muon/AdamW update
       e. wandb_logger.log()   → metrics to W&B
       f. eval / checkpoint    → every N steps
       ↓
8. wandb.finish(), cleanup
```
