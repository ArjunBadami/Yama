"""SQuAD 2.0 -> sufficiency examples.

SQuAD 2.0 contributes two things HotpotQA does not:

  1. Single-hop, single-sentence sufficiency: the answer span lives in one
     sentence of the paragraph. That sentence is gold; the rest of the paragraph
     is on-topic noise.
  2. ~50k *unanswerable* questions that were adversarially written to look
     answerable from the paragraph. Paragraph + unanswerable question is a
     high-quality `relevant_insufficient` example with strong lexical overlap,
     i.e. exactly the shortcut we want the model to un-learn.

Label caveat: "the sentence containing the answer span is sufficient" is an
approximation. Some questions need a neighbouring sentence for coreference.
This is acceptable noise for v0 and is what the teacher labeller is for.
"""

from __future__ import annotations

import logging
import random

from ..schema import Example
from .degrade import VariantSpec, build_variants
from .text import sentence_spans

log = logging.getLogger(__name__)

HF_NAME = "rajpurkar/squad_v2"


def _load(split: str, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset(HF_NAME, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def build_squad_v2(
    split: str = "train",
    limit: int | None = None,
    seed: int = 0,
    spec: VariantSpec | None = None,
    unanswerable_noise_max: int = 3,
) -> list[Example]:
    rng = random.Random(seed)
    spec = spec or VariantSpec(
        n_clean_sufficient=1,
        n_distractor_heavy=1,
        n_missing_one=1,  # falls back to relevant_insufficient (1 gold)
        n_relevant_insufficient=0,
        n_long_insufficient=0,
        n_obvious_insufficient=0,
        light_noise_max=2,
        heavy_noise_min=3,
        heavy_noise_max=8,
    )
    ds = _load(split, limit)
    log.info("squad_v2[%s]: %d rows", split, len(ds))

    contexts = [r["context"] for r in ds]
    out: list[Example] = []
    for i, r in enumerate(ds):
        spans = sentence_spans(r["context"])
        if not spans:
            continue
        sentences = [s for _, _, s in spans]
        j = rng.randrange(len(ds))
        if j == i:
            j = (j + 1) % len(ds)
        unrelated = [s for _, _, s in sentence_spans(contexts[j])]

        answers = r["answers"]
        if len(answers["text"]) == 0:
            # Unanswerable: the whole on-topic paragraph is insufficient.
            k = (
                len(sentences)
                if len(sentences) <= unanswerable_noise_max + 1
                else rng.randint(2, min(len(sentences), unanswerable_noise_max + 3))
            )
            ev = rng.sample(sentences, k)
            out.append(
                Example(
                    question=r["question"].strip(),
                    evidence=ev,
                    sufficient=False,
                    source="squad_v2",
                    category="relevant_insufficient",
                    id=f"squad_v2-{r['id']}::0",
                    meta={"group_id": f"squad_v2-{r['id']}", "unanswerable": True},
                )
            )
            continue

        start = int(answers["answer_start"][0])
        gold_idx = next((n for n, (a, b, _) in enumerate(spans) if a <= start < b), None)
        if gold_idx is None:
            continue
        gold = [sentences[gold_idx]]
        distract = [s for n, s in enumerate(sentences) if n != gold_idx]
        out.extend(
            build_variants(
                r["question"].strip(),
                gold,
                distract,
                unrelated,
                source="squad_v2",
                group_id=f"squad_v2-{r['id']}",
                rng=rng,
                spec=spec,
                answer=answers["text"][0],
                multi_hop=False,
            )
        )
    return out
