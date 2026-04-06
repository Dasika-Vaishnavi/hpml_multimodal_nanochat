# multimodal-nanochat

# Google Cloud VM Configuration
- Boot image: Google, Deep Learning VM with CUDA 12.4, M129, Debian 11, Python 3.10. With CUDA 12.4 preinstalled. (newest)
- GPU: NVIDIA L4
- Machine type: g2-standard-4 (4 vCPUs, 16 GB Memory) (default)
- Disk: 50 GB (should probably be larger)

# Git Configuration in VM
- SSH into VM (I used VSCode's Remote Explorer extension)
- when prompted, select yes to install NVIDIA drivers
- create SSH key using ssh-keygen
  - `ssh-keygen -t ed25519 -C "your-email@example.com"`
- Copy public key
  - `cat ~/.ssh/id_ed25519.pub`
- paste key in GitHub to authenticate 
- clone this repo

# Pytorch Configuration in VM
- I am using VSCode with Jupyter extensions
- We want to use pre-installed packages as much as possible. I read not to use Anaconda with a VM like this.
- python3 -m venv --system-site-packages nanochat_baseline
- source nanochat_baseline/bin/activate
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Setting up nanochat for inferencing and fine-tuning
## Installing dependencies
Per the nanochat README, it prefers using uv package manager:
- pip install uv
- uv sync --extra gpu
- source .venv/bin/activate

## Setting up pre-trained weights
Nanochat is meant for training, so using pretrained weights is a little complicated. I chose weights from https://huggingface.co/sdobson/nanochat because Karpathy's weights from https://huggingface.co/karpathy/nanochat-d32/tree/main were too large.

In your GCP VM:
```
pip install huggingface_hub

huggingface-cli download sdobson/nanochat model_000650.pt --local-dir ~/.cache/nanochat/chatsft_checkpoints/d20

huggingface-cli download sdobson/nanochat meta_000650.json --local-dir ~/.cache/nanochat/chatsft_checkpoints/d20

huggingface-cli download sdobson/nanochat tokenizer.pkl --local-dir ~/.cache/nanochat/tokenizer

huggingface-cli download sdobson/nanochat token_bytes.pt --local-dir ~/.cache/nanochat/tokenizer
```

## Running for inferencing
`NANOCHAT_DTYPE=bfloat16 python -m scripts.chat_cli --model-tag d20`

## Fine-tuning
As an intermediate step before introducing image embeddings, we can fine-tune nanochat on text. The following setup achieves this using nanochat's base_train.py script. This is slightly confusing because nanochat uses "pretraining" to mean next token prediction, which is trained with scripts/base_train.py. "Fine-tuning" refers to using conversational data like JSONL conversations, which is trained with chat_sft.py. So even though we "fine-tune" a model in a traditional sense of using pre-trained weights and slightly improving them, we use base_train.py for this in nanochat
- Download 1 partition of training data: from multimodal-nanochat/nanochat: `python -m nanochat.dataset -n 1`. This downloads training data into ~/.cache/nanochat/base_data_climbmix
- The following command trains for 10 iterations and then evaluates on many different test sets:
```
TORCH_COMPILE_DISABLE=1 NANOCHAT_DTYPE=bfloat16 python -m scripts.base_train \
    --model-tag d20 \
    --num-iterations 10 \
    --device-batch-size 1 \
    --total-batch-size 512 \
    --max-seq-len 512 \
    --eval-every 0
    ```
