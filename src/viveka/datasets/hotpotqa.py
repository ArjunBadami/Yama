"""HotpotQA -> sufficiency examples.

HotpotQA (distractor setting) gives us, per question:
  - the answer,
  - 10 paragraphs: 2 gold + 8 retrieved distractors (TF-IDF over the question),
  - the exact supporting sentences within the gold paragraphs.

That is almost exactly the annotation we need:
  gold                  = the supporting sentences (typically 2-4, from 2 paragraphs)
  relevant distractors  = other sentences from the gold paragraphs + sentences from
                          the 8 retrieved distractor paragraphs (on-topic by construction)
  unrelated             = sentences from a different random question's paragraphs

Because most questions are "bridge" questions needing facts from two paragraphs,
dropping one supporting sentence yields a natural multi-hop hard negative.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable

from ..schema import Example
from .degrade import VariantSpec, build_variants

log = logging.getLogger(__name__)

HF_NAME = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"


def _load(split: str, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, HF_CONFIG, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _row_sentences(row) -> tuple[list[str], list[str], list[str]]:
    """Return (gold, relevant_distractors, all_sentences_flat)."""
    titles: list[str] = row["context"]["title"]
    sents: list[list[str]] = row["context"]["sentences"]
    sf_titles: list[str] = row["supporting_facts"]["title"]
    sf_ids: list[int] = row["supporting_facts"]["sent_id"]

    gold_keys = set(zip(sf_titles, sf_ids, strict=False))
    gold, distract, flat = [], [], []
    for t, para in zip(titles, sents, strict=False):
        for j, s in enumerate(para):
            s = s.strip()
            if not s:
                continue
            flat.append(s)
            if (t, j) in gold_keys:
                gold.append(s)
            else:
                distract.append(s)
    return gold, distract, flat


def build_hotpotqa(
    split: str = "train",
    limit: int | None = None,
    seed: int = 0,
    spec: VariantSpec | None = None,
    min_gold: int = 1,
) -> list[Example]:
    rng = random.Random(seed)
    spec = spec or VariantSpec(
        n_clean_sufficient=1,
        n_distractor_heavy=1,
        n_missing_one=1,
        n_relevant_insufficient=1,
        n_long_insufficient=1,
        n_obvious_insufficient=0,
        light_noise_max=3,
        heavy_noise_min=6,
        heavy_noise_max=14,
    )
    ds = _load(split, limit)
    n = len(ds)
    log.info("hotpotqa[%s]: %d rows", split, n)

    # Pre-extract so that "unrelated" pools can be drawn from other rows cheaply.
    rows = [_row_sentences(r) for r in ds]
    out: list[Example] = []
    for i, r in enumerate(ds):
        gold, distract, _ = rows[i]
        if len(gold) < min_gold:
            continue
        j = rng.randrange(n)
        if j == i:
            j = (j + 1) % n
        unrelated = rows[j][2]
        out.extend(
            build_variants(
                r["question"].strip(),
                gold,
                distract,
                unrelated,
                source="hotpotqa",
                group_id=f"hotpotqa-{r['id']}",
                rng=rng,
                spec=VariantSpec(
                    **{**spec.__dict__, "extra_meta": {"hp_type": r.get("type"), "hp_level": r.get("level")}}
                ),
                answer=r.get("answer"),
                multi_hop=(r.get("type") == "bridge") or len(set(r["supporting_facts"]["title"])) > 1,
            )
        )
    return out


def iter_hotpotqa_rows(split: str = "validation", limit: int | None = None) -> Iterable[dict]:
    """Raw rows (question, answer, gold, distractors) for teacher labelling."""
    ds = _load(split, limit)
    for r in ds:
        gold, distract, _ = _row_sentences(r)
        yield {
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "gold": gold,
            "relevant_distractors": distract,
            "type": r.get("type"),
        }
