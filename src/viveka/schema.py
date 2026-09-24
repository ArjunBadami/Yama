"""Canonical example schema and JSONL IO.

Every dataset source (public QA, synthetic, teacher-labelled) is normalised into
`Example` records so that the training, evaluation and inference code never has
to know where the data came from.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Difficulty / construction categories. Used for stratified evaluation so that a
# good aggregate score cannot hide a failure on one shortcut-prone slice.
CATEGORIES = (
    "obvious_sufficient",  # all required facts, little noise
    "obvious_insufficient",  # unrelated or empty evidence
    "relevant_insufficient",  # on-topic evidence that does not resolve the question
    "missing_one_fact",  # all but one required fact present
    "distractor_heavy",  # all required facts buried in lots of relevant noise
    "multi_hop",  # sufficiency depends on combining facts
    "contradiction",  # required facts present but one is contradicted
    "long_insufficient",  # long evidence, still insufficient (length shortcut)
    "short_sufficient",  # very short evidence that is nonetheless sufficient
    "teacher_labelled",  # label produced by a teacher model (Definition B)
    "unknown",
)


@dataclass
class Example:
    question: str
    evidence: list[str]
    sufficient: bool
    source: str = "unknown"  # e.g. hotpotqa, squad_v2, toy, synthetic
    category: str = "unknown"  # one of CATEGORIES
    id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}; expected one of {CATEGORIES}")
        self.evidence = [e.strip() for e in self.evidence if e and e.strip()]

    @property
    def label(self) -> float:
        return 1.0 if self.sufficient else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            question=d["question"],
            evidence=list(d.get("evidence", [])),
            sufficient=bool(d["sufficient"]),
            source=d.get("source", "unknown"),
            category=d.get("category", "unknown"),
            id=d.get("id"),
            meta=dict(d.get("meta", {})),
        )


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, limit: int | None = None) -> list[Example]:
    out: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Example.from_dict(json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    return out


def iter_jsonl(path: str | Path) -> Iterator[Example]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Example.from_dict(json.loads(line))


def split_examples(
    examples: list[Example],
    val_frac: float,
    seed: int = 0,
    group_key: str | None = "id",
) -> tuple[list[Example], list[Example]]:
    """Split into train/val.

    If `group_key` is set, all examples sharing that key (e.g. the several
    degraded variants of the same source question) land in the same split.
    Without this, the model would see the "sufficient" version of a question in
    training and its "missing one fact" sibling in validation, which leaks.
    """
    rng = random.Random(seed)
    if group_key is None:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        val_idx = set(idx[:n_val])
        train = [ex for i, ex in enumerate(examples) if i not in val_idx]
        val = [ex for i, ex in enumerate(examples) if i in val_idx]
        return train, val

    groups: dict[str, list[Example]] = {}
    for i, ex in enumerate(examples):
        key = _group_of(ex, group_key) or f"__solo_{i}"
        groups.setdefault(key, []).append(ex)
    keys = list(groups)
    rng.shuffle(keys)
    n_val = int(len(keys) * val_frac)
    val_keys = set(keys[:n_val])
    train = [ex for k in keys if k not in val_keys for ex in groups[k]]
    val = [ex for k in keys if k in val_keys for ex in groups[k]]
    return train, val


def _group_of(ex: Example, group_key: str) -> str | None:
    if group_key == "id":
        return ex.meta.get("group_id") or (ex.id.rsplit("::", 1)[0] if ex.id else None)
    return ex.meta.get(group_key)
