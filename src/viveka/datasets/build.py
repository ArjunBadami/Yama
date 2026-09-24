"""Build train / val / eval JSONL files from one or more sources.

    python -m viveka.datasets.build --sources toy --toy-questions 400 --out data/processed/toy
    python -m viveka.datasets.build --sources hotpotqa squad_v2 --limit 20000 --out data/processed

Splits are made by *group* (all variants of one source question stay together)
so degraded siblings never leak across train/val/eval.

`eval` is carved from the HotpotQA/SQuAD *validation* splits when available so
that it is disjoint from training data at the source level, not just at the
example level. The doc's Phase 0 hand-reviewed eval set should live at
data/eval/ and be curated separately; this file is the automatic one.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from ..schema import Example, split_examples, write_jsonl
from .degrade import category_counts, label_balance

log = logging.getLogger(__name__)


def _build_source(name: str, args: argparse.Namespace, split: str, seed: int) -> list[Example]:
    if name == "toy":
        from .toy import generate_toy

        n = args.toy_questions if split == "train" else max(20, args.toy_questions // 10)
        return generate_toy(n, seed=seed)
    if name == "hotpotqa":
        from .hotpotqa import build_hotpotqa

        return build_hotpotqa(split=split, limit=args.limit, seed=seed)
    if name == "squad_v2":
        from .squad_v2 import build_squad_v2

        return build_squad_v2(split=split, limit=args.limit, seed=seed)
    raise ValueError(f"unknown source {name!r}")


def _balance(examples: list[Example], rng: random.Random, target_pos_rate: float | None) -> list[Example]:
    """Downsample the majority label to hit `target_pos_rate` (approximately)."""
    if target_pos_rate is None:
        return examples
    pos = [e for e in examples if e.sufficient]
    neg = [e for e in examples if not e.sufficient]
    # Want pos / (pos + neg) = r  =>  neg = pos * (1 - r) / r
    want_neg = int(len(pos) * (1 - target_pos_rate) / target_pos_rate)
    want_pos = int(len(neg) * target_pos_rate / (1 - target_pos_rate))
    if len(neg) > want_neg:
        neg = rng.sample(neg, want_neg)
    elif len(pos) > want_pos:
        pos = rng.sample(pos, want_pos)
    out = pos + neg
    rng.shuffle(out)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True, choices=["toy", "hotpotqa", "squad_v2"])
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--limit", type=int, default=None, help="max source rows per public dataset split")
    p.add_argument("--toy-questions", type=int, default=400)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument(
        "--eval-from-validation",
        action="store_true",
        default=True,
        help="build eval.jsonl from the public datasets' validation splits (default on)",
    )
    p.add_argument("--no-eval", action="store_true", help="skip eval.jsonl")
    p.add_argument("--eval-limit", type=int, default=2000, help="max source rows for eval per dataset")
    p.add_argument(
        "--target-pos-rate", type=float, default=None, help="downsample train to this positive rate, e.g. 0.5"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_all: list[Example] = []
    for s in args.sources:
        exs = _build_source(s, args, "train", args.seed)
        log.info("%s/train: %s %s", s, label_balance(exs), category_counts(exs))
        train_all.extend(exs)

    train, val = split_examples(train_all, args.val_frac, seed=args.seed)
    train = _balance(train, rng, args.target_pos_rate)
    rng.shuffle(train)
    n_tr = write_jsonl(args.out / "train.jsonl", train)
    n_va = write_jsonl(args.out / "val.jsonl", val)
    log.info("wrote %d train, %d val -> %s", n_tr, n_va, args.out)

    stats = {
        "train": {"balance": label_balance(train), "categories": category_counts(train)},
        "val": {"balance": label_balance(val), "categories": category_counts(val)},
    }

    if not args.no_eval:
        eval_all: list[Example] = []
        for s in args.sources:
            eval_args = argparse.Namespace(**{**vars(args), "limit": args.eval_limit})
            split = "validation" if s != "toy" else "eval"
            exs = _build_source(s, eval_args, split, args.seed + 1)
            eval_all.extend(exs)
        rng.shuffle(eval_all)
        n_ev = write_jsonl(args.out / "eval.jsonl", eval_all)
        stats["eval"] = {"balance": label_balance(eval_all), "categories": category_counts(eval_all)}
        log.info("wrote %d eval", n_ev)

    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
