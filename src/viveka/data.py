"""torch Dataset / collator over Example records."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .formatting import encode_batch
from .schema import Example


class SufficiencyDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class Collator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[Example]) -> dict[str, Any]:
        enc = encode_batch(
            self.tokenizer,
            [e.question for e in batch],
            [e.evidence for e in batch],
            max_length=self.max_length,
        )
        out: dict[str, Any] = dict(enc)
        out["labels"] = torch.tensor([e.label for e in batch], dtype=torch.float32)
        out["categories"] = [e.category for e in batch]
        return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
