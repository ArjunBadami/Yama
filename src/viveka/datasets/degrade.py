"""The controlled curriculum: turn one annotated question into many labelled variants.

Given a question, the set of facts that are *required* to answer it (gold), a
pool of on-topic but non-required sentences (relevant distractors) and a pool
of off-topic sentences (unrelated), we construct:

    all gold, little noise                     -> sufficient    (obvious_sufficient / short_sufficient / multi_hop)
    all gold + lots of relevant noise          -> sufficient    (distractor_heavy)
    gold minus one fact (+ noise)              -> insufficient  (missing_one_fact)
        skipped when the answer text is still in what remains
    no gold, only relevant noise               -> insufficient  (relevant_insufficient)
    no gold, lots of relevant noise            -> insufficient  (long_insufficient)
    unrelated noise only / empty               -> insufficient  (obvious_insufficient)

The point of the mixture is to make every cheap shortcut fail:
  - relevance != sufficiency   (relevant_insufficient, long_insufficient)
  - length    != sufficiency   (long_insufficient vs short_sufficient)
  - overlap   != sufficiency   (missing_one_fact keeps most of the question's vocabulary)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..schema import Example


@dataclass
class VariantSpec:
    """How many of each variant to emit per source question, and noise sizes."""

    # Positive variants
    n_clean_sufficient: int = 1  # gold + 0..light_noise_max distractors
    n_distractor_heavy: int = 1  # gold + heavy_noise_min..max distractors
    # Negative variants
    n_missing_one: int = 1  # gold minus one fact + light noise (needs >=2 gold)
    n_relevant_insufficient: int = 1  # relevant noise only
    n_long_insufficient: int = 0  # lots of relevant noise only
    n_obvious_insufficient: int = 0  # unrelated noise only (or empty)
    light_noise_max: int = 2
    heavy_noise_min: int = 4
    heavy_noise_max: int = 10
    # Fraction of missing_one variants where, when only 1 gold fact exists, we
    # still emit a "no gold" example labelled relevant_insufficient.
    single_gold_fallback: bool = True
    shuffle_evidence: bool = True
    extra_meta: dict = field(default_factory=dict)


def _norm(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text]
    return " ".join("".join(chars).split())


def answer_still_in_evidence(answer: str | None, evidence: list[str]) -> bool:
    """True when the gold answer string is still readable in `evidence`.

    Yes/no answers and very short strings are ignored: "no" and "Bury" show up
    inside unrelated words often enough that a substring check would throw away
    real negatives. If this returns True, the example must not be labeled insufficient.
    """
    if not answer:
        return False
    normalized = _norm(answer)
    # "no"/"yes" and tiny strings ("2002", "Bury") occur inside unrelated sentences.
    if normalized in {"yes", "no"} or len(normalized) < 8:
        return False
    return normalized in _norm(" ".join(evidence))


def _sample(pool: list[str], k: int, rng: random.Random) -> list[str]:
    if k <= 0 or not pool:
        return []
    return rng.sample(pool, min(k, len(pool)))


def _mk(
    question: str,
    evidence: list[str],
    sufficient: bool,
    category: str,
    source: str,
    group_id: str,
    idx: int,
    rng: random.Random,
    spec: VariantSpec,
    meta: dict,
) -> Example:
    ev = list(evidence)
    if spec.shuffle_evidence:
        rng.shuffle(ev)
    return Example(
        question=question,
        evidence=ev,
        sufficient=sufficient,
        source=source,
        category=category,
        id=f"{group_id}::{idx}",
        meta={"group_id": group_id, **spec.extra_meta, **meta},
    )


def build_variants(
    question: str,
    gold: list[str],
    relevant_distractors: list[str],
    unrelated: list[str],
    *,
    source: str,
    group_id: str,
    rng: random.Random,
    spec: VariantSpec | None = None,
    answer: str | None = None,
    multi_hop: bool = False,
) -> list[Example]:
    spec = spec or VariantSpec()
    gold = [g.strip() for g in gold if g and g.strip()]
    relevant_distractors = [d for d in {s.strip() for s in relevant_distractors} if d and d not in gold]
    unrelated = [u for u in {s.strip() for s in unrelated} if u]
    if not gold:
        return []

    out: list[Example] = []
    base_meta = {"n_gold": len(gold)}
    if answer is not None:
        base_meta["answer"] = answer
    i = 0

    # --- positives -------------------------------------------------------
    for _ in range(spec.n_clean_sufficient):
        noise = _sample(relevant_distractors, rng.randint(0, spec.light_noise_max), rng)
        if multi_hop and len(gold) >= 2:
            cat = "multi_hop"
        elif not noise and sum(len(g) for g in gold) < 160:
            cat = "short_sufficient"
        else:
            cat = "obvious_sufficient"
        out.append(_mk(question, gold + noise, True, cat, source, group_id, i, rng, spec, base_meta))
        i += 1

    for _ in range(spec.n_distractor_heavy):
        k = rng.randint(spec.heavy_noise_min, spec.heavy_noise_max)
        noise = _sample(relevant_distractors, k, rng)
        if len(noise) < spec.heavy_noise_min:
            noise += _sample(unrelated, spec.heavy_noise_min - len(noise), rng)
        if not noise:
            continue
        out.append(_mk(question, gold + noise, True, "distractor_heavy", source, group_id, i, rng, spec, base_meta))
        i += 1

    # --- negatives -------------------------------------------------------
    for _ in range(spec.n_missing_one):
        if len(gold) >= 2:
            # Try every sentence to drop. Keep the first deletion that actually
            # removes the answer. If every deletion leaves the answer in view,
            # emit nothing: that would be a sufficient passage labeled insufficient.
            drop_order = list(range(len(gold)))
            rng.shuffle(drop_order)
            emitted = False
            for drop in drop_order:
                kept = [g for j, g in enumerate(gold) if j != drop]
                noise = _sample(relevant_distractors, rng.randint(0, spec.light_noise_max), rng)
                evidence = kept + noise
                if answer_still_in_evidence(answer, evidence):
                    continue
                out.append(
                    _mk(
                        question,
                        evidence,
                        False,
                        "missing_one_fact",
                        source,
                        group_id,
                        i,
                        rng,
                        spec,
                        {**base_meta, "dropped_fact": gold[drop]},
                    )
                )
                emitted = True
                break
            if emitted:
                i += 1
        elif spec.single_gold_fallback and relevant_distractors:
            noise = _sample(relevant_distractors, rng.randint(1, max(1, spec.light_noise_max)), rng)
            out.append(_mk(question, noise, False, "relevant_insufficient", source, group_id, i, rng, spec, base_meta))
            i += 1

    for _ in range(spec.n_relevant_insufficient):
        noise = _sample(relevant_distractors, rng.randint(1, max(1, spec.light_noise_max + 1)), rng)
        if not noise:
            continue
        out.append(_mk(question, noise, False, "relevant_insufficient", source, group_id, i, rng, spec, base_meta))
        i += 1

    for _ in range(spec.n_long_insufficient):
        k = rng.randint(spec.heavy_noise_min, spec.heavy_noise_max)
        noise = _sample(relevant_distractors, k, rng)
        if len(noise) < spec.heavy_noise_min:
            noise += _sample(unrelated, spec.heavy_noise_min - len(noise), rng)
        if len(noise) < 2:
            continue
        out.append(_mk(question, noise, False, "long_insufficient", source, group_id, i, rng, spec, base_meta))
        i += 1

    for _ in range(spec.n_obvious_insufficient):
        noise = _sample(unrelated, rng.randint(0, spec.light_noise_max + 1), rng)
        out.append(_mk(question, noise, False, "obvious_insufficient", source, group_id, i, rng, spec, base_meta))
        i += 1

    return out


def label_balance(examples: list[Example]) -> dict[str, int]:
    pos = sum(1 for e in examples if e.sufficient)
    return {"total": len(examples), "sufficient": pos, "insufficient": len(examples) - pos}


def category_counts(examples: list[Example]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in examples:
        out[e.category] = out.get(e.category, 0) + 1
    return dict(sorted(out.items()))
