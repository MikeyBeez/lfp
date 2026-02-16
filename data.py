"""Data loading for WikiText with GPT-2 tokenization."""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText dataset chunked into fixed-length sequences."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        # drop the tail that doesn't fill a complete sequence
        n_seqs = (len(tokens) - 1) // seq_len  # -1 for targets offset
        self.tokens = tokens[: n_seqs * seq_len + 1]

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def load_wikitext(name: str = "wikitext-2", seq_len: int = 256):
    """Load and tokenize WikiText, returning train/val/test datasets.

    Args:
        name: "wikitext-2" or "wikitext-103"
        seq_len: sequence length for chunking
    """
    from datasets import load_dataset

    ds_name = "wikitext"
    ds_config = "wikitext-2-raw-v1" if name == "wikitext-2" else "wikitext-103-raw-v1"

    raw = load_dataset(ds_name, ds_config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    splits = {}
    for split_name in ["train", "validation", "test"]:
        text = "\n".join(raw[split_name]["text"])
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        splits[split_name] = WikiTextDataset(tokens, seq_len)

    return splits, tokenizer


def build_dataloaders(
    splits: dict,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Wrap datasets in DataLoaders."""
    loaders = {}
    for name, ds in splits.items():
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(name == "train"),
        )
    return loaders
