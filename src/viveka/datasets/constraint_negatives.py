"""Build constraint-mismatch negatives from examples the model already calls sufficient.

The confident mistakes on the held-out set are mostly this shape: the evidence
contains a plausible answer span, and the question asks for something that span
does not satisfy (the wrong person, a total instead of a part, a list with no
superlative). These examples are built only from a training or validation file.
They must not be built from the eval file, or the score would be leaked.

The construction is mechanical. In a sufficient example whose answer string sits
in one evidence sentence, that string is replaced by a different name or year
taken from another sentence in the same example. The question stays the same.
The label is insufficient, because the sentence now states the wrong fact.

    python -m viveka.datasets.constraint_negatives \
        --src data/processed-v2/train.jsonl \
        --out data/processed-v3/train.jsonl \
        --repeat 3

`--src` rows are copied through, then the new negatives are appended `repeat` times.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from ..schema import Example, read_jsonl, write_jsonl
from .degrade import _norm, answer_still_in_evidence, category_counts, label_balance

# A proper name (optionally with of/the/de between the words), or a year.
# A single capitalized word is skipped: it is usually just the start of a sentence.
_SPAN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+(?:of|the|de|da|van|von|di)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)+|(?:1[0-9]{3}|20[0-9]{2}))\b"
)


def _replacements(evidence: list[str], answer: str) -> list[str]:
    answer_norm = _norm(answer)
    found: list[str] = []
    seen: set[str] = set()
    for sentence in evidence:
        if answer_norm in _norm(sentence):
            continue
        for match in _SPAN.finditer(sentence):
            span = match.group(0)
            key = _norm(span)
            if key in seen or key == answer_norm or len(key) < 4:
                continue
            seen.add(key)
            found.append(span)
    return found


def _replace_answer(sentence: str, answer: str, replacement: str) -> str | None:
    match = re.search(re.escape(answer), sentence, flags=re.IGNORECASE)
    if match is None:
        return None
    return sentence[: match.start()] + replacement + sentence[match.end() :]


def make_constraint_negative(ex: Example, rng: random.Random) -> Example | None:
    answer = str(ex.meta.get("answer") or "").strip()
    if not ex.sufficient or not answer_still_in_evidence(answer, ex.evidence):
        return None
    options = _replacements(ex.evidence, answer)
    if not options:
        return None
    replacement = rng.choice(options)
    new_evidence: list[str] = []
    swapped = False
    for sentence in ex.evidence:
        if not swapped:
            replaced = _replace_answer(sentence, answer, replacement)
            if replaced is not None:
                new_evidence.append(replaced)
                swapped = True
                continue
        new_evidence.append(sentence)
    if not swapped or answer_still_in_evidence(answer, new_evidence):
        return None
    return Example(
        question=ex.question,
        evidence=new_evidence,
        sufficient=False,
        source=ex.source,
        category="constraint_mismatch",
        id=f"{ex.id}::constraint" if ex.id else None,
        meta={
            "group_id": ex.meta.get("group_id"),
            "answer": answer,
            "swapped_in": replacement,
            "parent_id": ex.id,
        },
    )


def augment(examples: list[Example], n: int, repeat: int, seed: int) -> tuple[list[Example], int]:
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    made: list[Example] = []
    for i in order:
        if len(made) >= n:
            break
        neg = make_constraint_negative(examples[i], rng)
        if neg is not None:
            made.append(neg)
    extra = [e for _ in range(repeat) for e in made]
    return examples + extra, len(made)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="train or val jsonl. Never the eval file.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=20000, help="how many distinct swapped negatives to build")
    p.add_argument("--repeat", type=int, default=3, help="append that many copies, so training upweights them")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if "eval" in args.src.name:
        raise SystemExit("refusing to build training negatives from a file named like an eval set")
    examples = read_jsonl(args.src)
    mixed, n_made = augment(examples, args.n, args.repeat, args.seed)
    write_jsonl(args.out, mixed)
    print(
        {
            "src": len(examples),
            "distinct_negatives": n_made,
            "written": len(mixed),
            "balance": label_balance(mixed),
            "categories": category_counts(mixed),
        }
    )


if __name__ == "__main__":
    main()
