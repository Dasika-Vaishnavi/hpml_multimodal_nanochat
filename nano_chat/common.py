"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
import math
import torch
import torch.distributed as dist
from filelock import FileLock
from dataclasses import dataclass, field

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        # Set NCCL env vars for reliable multi-node training
        if "NCCL_DEBUG" not in os.environ:
            os.environ["NCCL_DEBUG"] = "WARN"
        if "NCCL_SOCKET_IFNAME" not in os.environ:
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')


# -----------------------------------------------------------------------------
# Training Hyperparameters for Multimodal SLM Training
# These defaults are used in base_train_multimodal.py

@dataclass
class TrainConfig:
    """Training hyperparameters for multimodal SLM pretraining on COCO captions."""

    # Model (GPT config is set separately; see nanochat.gpt.GPTConfig)
    model_config_name: str = "small"  # "small", "tiny", or "mini"

    # Data
    dataset_name: str = "yerevann/coco-karpathy"
    dataset_split: str = "train"
    val_split: str = "validation"
    max_seq_len: int = 512
    max_length: int = 512  # caption truncation length

    # Batch & Accumulation
    device_batch_size: int = 8
    grad_accum_steps: int = 1
    total_batch_size: int = -1  # -1 = auto from device_batch_size * grad_accum_steps

    # Optimizer (MuonAdamW)
    matrix_lr: float = 0.02
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    scalar_lr: float = 0.5
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    beta1: float = 0.8
    beta2_mixed: float = 0.96
    beta2_scalar: float = 0.95
    eps: float = 1e-10

    # Training horizon
    num_iterations: int = -1  # -1 = calculate from tokens:param ratio
    target_param_data_ratio: float = 20.0  # Chinchilla-optimal ratio
    max_train_iters: int = 10000  # cap when auto-calculating

    # Evaluation & Logging
    eval_every: int = 500
    sample_every: int = 2000
    save_every: int = -1  # -1 = only at end
    wandb_run: str = "dummy"  # "dummy" disables wandb

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from_step: int = -1

    # Device
    device_type: str = ""  # "" = auto-detect

    def resolve_batch_size(self, world_size: int = 1) -> int:
        """Compute total_batch_size = device_batch_size * grad_accum_steps * world_size."""
        if self.total_batch_size > 0:
            return self.total_batch_size
        return self.device_batch_size * self.grad_accum_steps * world_size

    def resolve_num_iterations(self, num_params: int, num_scaling_params: int) -> int:
        """Resolve number of training iterations from ratio or explicit value."""
        if self.num_iterations > 0:
            return self.num_iterations
        target_tokens = int(self.target_param_data_ratio * num_scaling_params)
        total_batch = self.resolve_batch_size()
        iters = max(1, target_tokens // total_batch)
        return min(iters, self.max_train_iters)


@dataclass
class DistributedConfig:
    """Distributed training configuration, auto-detected from torchrun env."""

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @classmethod
    def auto(cls) -> "DistributedConfig":
        """Detect DDP settings from torchrun environment variables."""
        if is_ddp_requested():
            return cls(
                enabled=True,
                rank=int(os.environ["RANK"]),
                local_rank=int(os.environ["LOCAL_RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
                master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
                master_port=int(os.environ.get("MASTER_PORT", "29500")),
            )
        return cls()

    def __post_init__(self):
        self.is_master = self.rank == 0


def wrap_model_for_ddp(model, local_rank: int = None):
    """
    Wrap model with PyTorch DDP using NCCL backend.
    Returns the (possibly wrapped) model. Does nothing in single-GPU mode.

    Uses find_unused_parameters=False for efficiency; all nanochat parameters
    are used in the forward pass.
    """
    if not is_ddp_requested():
        return model

    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    model = model.to(device)

    # NCCL is fastest for multi-GPU on CUDA
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    return model


def reduce_tensor(tensor, world_size: int):
    """All-reduce a scalar tensor across all ranks (sum, then average)."""
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt.div_(world_size)
    return rt


def gather_tensor(tensor, dst_rank: int = 0):
    """Gather tensors from all ranks to destination rank. Returns list on dst, None elsewhere."""
    if not is_ddp_requested():
        return tensor

    world_size = get_dist_info()[3]
    if world_size == 1:
        return tensor

    # Gather to dst_rank
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


def ddp_barrier():
    """Synchronize all ranks. No-op if not in DDP mode."""
    if is_ddp_initialized():
        dist.barrier()
