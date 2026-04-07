"""
COCO Captions Dataset for text-only pretraining of the language model.

This module loads the COCO 2017 captions dataset from HuggingFace for
Phase 1 (text-only pretraining). In Phase 2, multimodal capabilities will be added.
"""

import os
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

from nanochat.common import get_base_dir, is_ddp_requested

# -----------------------------------------------------------------------------
# Dataset configuration

DATASET_NAME = "yerevann/coco-karpathy"
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "coco_captions_data")

# -----------------------------------------------------------------------------
# COCO Caption Dataset

class COCOCaptionsDataset(Dataset):
    """
    PyTorch Dataset for COCO 2017 captions.
    Tokenizes all captions using the GPT-2 tokenizer.

    Uses yerevann/coco-karpathy dataset which has the Karpathy splits:
    - train: ~82,783 images with 5 captions each (~413K captions)
    - validation: ~5,000 images (~25K captions)
    - test: ~5,000 images (~25K captions)
    - restval: ~30,504 images (~153K captions)
    """

    def __init__(self, split="train", tokenizer=None, max_length=512):
        """
        Args:
            split: "train", "validation", "test", or "restval"
            tokenizer: tiktoken encoding (defaults to GPT-2)
            max_length: maximum sequence length (truncates longer captions)
        """
        self.split = split
        self.max_length = max_length

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        else:
            self.tokenizer = tokenizer

        # Download and load COCO captions
        print(f"Loading COCO Karpathy dataset (split: {split})...")
        self.dataset = load_dataset(DATASET_NAME, split=split)
        print(f"Loaded {len(self.dataset)} {split} images")

        # Extract captions (each image has multiple captions in "sentences" field)
        self._extract_captions()

        # Pre-tokenize all captions for faster training
        self._preprocess()

    def _extract_captions(self):
        """Extract individual captions from the dataset structure."""
        self.captions = []
        for item in self.dataset:
            # Each item has a "sentences" list with multiple caption strings
            for caption in item["sentences"]:
                self.captions.append(caption)

        print(f"Extracted {len(self.captions):,} captions")

    def _preprocess(self):
        """Pre-tokenize all captions to speed up training."""
        print(f"Tokenizing {len(self.captions):,} captions...")
        self.tokenized = []

        for i, caption in enumerate(self.captions):
            if i % 100000 == 0 and i > 0:
                print(f"  Tokenized {i:,}/{len(self.captions):,} captions...")

            # Tokenize with truncation
            token_ids = self.tokenizer.encode(caption)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]

            self.tokenized.append(token_ids)

        print(f"Tokenization complete.")

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        """Return tokenized caption as tensor."""
        return torch.tensor(self.tokenized[idx], dtype=torch.long)


def collate_fn(batch):
    """
    Collate function to pad sequences to same length within a batch.
    Returns:
        - input_ids: padded tensor of shape (batch_size, max_len)
        - attention_mask: tensor indicating real tokens (1) vs padding (0)
    """
    # Get lengths and find max
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)

    # Pad sequences
    padded_batch = []
    attention_masks = []

    for seq in batch:
        pad_length = max_len - len(seq)
        padded_seq = torch.cat([
            seq,
            torch.zeros(pad_length, dtype=torch.long)
        ])
        padded_batch.append(padded_seq)

        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.cat([
            torch.ones(len(seq), dtype=torch.long),
            torch.zeros(pad_length, dtype=torch.long)
        ])
        attention_masks.append(mask)

    return {
        "input_ids": torch.stack(padded_batch),
        "attention_mask": torch.stack(attention_masks)
    }


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    ddp_rank=0,
    ddp_world_size=1,
):
    """
    Create a DataLoader for the COCO captions dataset.

    Args:
        dataset: COCOCaptionsDataset instance
        batch_size: batch size
        shuffle: whether to shuffle data (ignored when DDP is active)
        num_workers: number of worker processes for data loading
        pin_memory: pin memory for faster GPU transfer
        drop_last: drop last incomplete batch
        ddp_rank: current DDP rank (default 0, no DDP)
        ddp_world_size: total number of DDP processes (default 1, no DDP)

    Returns:
        DataLoader instance
    """
    # Use DistributedSampler when running under torchrun
    sampler = None
    if is_ddp_requested():
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Shuffle is controlled by sampler in DDP mode
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
        sampler=sampler,
    )


def print_dataset_stats(dataset, tokenizer):
    """Print statistics about the dataset."""
    print("\n" + "=" * 60)
    print("COCO Captions Dataset Statistics")
    print("=" * 60)

    # Number of captions
    n_captions = len(dataset)
    print(f"Number of captions: {n_captions:,}")

    # Average caption length
    total_tokens = sum(len(seq) for seq in dataset.tokenized)
    avg_length = total_tokens / n_captions
    print(f"Average caption length (tokens): {avg_length:.2f}")

    # Token length distribution
    lengths = [len(seq) for seq in dataset.tokenized]
    print(f"Min caption length: {min(lengths)} tokens")
    print(f"Max caption length: {max(lengths)} tokens")
    print(f"Median caption length: {sorted(lengths)[len(lengths)//2]} tokens")

    # 95th percentile
    sorted_lengths = sorted(lengths)
    p95_idx = int(len(sorted_lengths) * 0.95)
    print(f"95th percentile length: {sorted_lengths[p95_idx]} tokens")

    # Vocab size
    vocab_size = tokenizer.n_vocab
    print(f"Vocab size: {vocab_size}")

    # Sample caption
    print(f"\nSample caption (raw): \"{dataset.captions[0]}\"")
    print(f"Sample tokenized (first 10): {dataset[0][:10].tolist()}")

    print("=" * 60 + "\n")


def get_coco_data_loader(
    split="train",
    batch_size=32,
    max_length=512,
    num_workers=4,
    shuffle=True,
    tokenizer=None
):
    """
    Convenience function to create a COCO captions DataLoader.

    Args:
        split: "train", "validation", "test", or "restval"
        batch_size: training batch size
        max_length: max sequence length
        num_workers: data loading workers
        shuffle: shuffle data
        tokenizer: optional tiktoken tokenizer

    Returns:
        DataLoader, Dataset
    """
    # Use GPT-2 tokenizer by default
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = COCOCaptionsDataset(
        split=split,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Print stats
    print_dataset_stats(dataset, tokenizer)

    # Auto-detect DDP settings
    ddp_rank = 0
    ddp_world_size = 1
    if is_ddp_requested():
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    return dataloader, dataset


# -----------------------------------------------------------------------------
# Main script for testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO Captions Dataset")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test", "restval"],
                        help="Dataset split to load (default: validation for quick testing)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    args = parser.parse_args()

    # Create dataset
    dataset = COCOCaptionsDataset(
        split=args.split,
        max_length=args.max_length
    )

    # Print statistics
    print_dataset_stats(dataset, dataset.tokenizer)

    # Create DataLoader
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Test iteration
    print(f"Testing DataLoader with {len(dataloader)} batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Test first 3 batches
            break
        print(f"  Batch {i}: input_ids shape={batch['input_ids'].shape}, "
              f"attention_mask shape={batch['attention_mask'].shape}")

    print("Dataset and DataLoader working correctly!")