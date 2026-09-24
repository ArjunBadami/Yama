"""Definition-B labelling: sufficiency as an operational, measurable quantity.

    Evidence is sufficient if a competent target reasoner, restricted to ONLY
    that evidence, reliably answers the question correctly.

For each example that carries a gold answer in `meta["answer"]` we ask the
teacher K times (sampled) to answer from the evidence alone, or say it cannot.
The example is relabelled sufficient iff at least `min_correct` of K answers
match the gold answer. The original heuristic label is preserved in meta so
we can measure agreement between the two definitions.

    python -m viveka.teacher.label --in data/processed/val.jsonl \
        --out data/processed/val_teacher.jsonl --k 3 --limit 500

Examples without a gold answer are passed through unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..schema import Example, read_jsonl, write_jsonl
from .client import (
    DEFAULT_MAX_CALLS,
    CachedTeacher,
    TeacherBudgetExceeded,
    estimate_calls,
    estimate_cost_usd,
    make_teacher,
    parse_json,
)

log = logging.getLogger(__name__)

PROMPT = """You are answering a question using ONLY the evidence provided. Do not use any outside knowledge.

If the evidence contains enough information to determine the answer, give the answer.
If the evidence does NOT contain enough information, say so instead of guessing.

Question: {question}

Evidence:
{evidence}

Respond with JSON only, in this exact shape:
{{"answerable": true or false, "answer": "<short answer, or empty string if not answerable>"}}"""


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def answers_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # yes/no questions
    if g in ("yes", "no"):
        return p.split()[0] == g
    # containment either way (SQuAD-style leniency)
    return g in p or p in g


def judge(teacher, ex: Example, k: int, temperature: float) -> dict:
    gold = str(ex.meta.get("answer", "")).strip()
    prompt = PROMPT.format(
        question=ex.question,
        evidence="\n".join(f"- {e}" for e in ex.evidence) if ex.evidence else "(no evidence)",
    )
    votes = []
    for s in range(k):
        kwargs = {"temperature": temperature if k > 1 else 0.0, "json_mode": True}
        if isinstance(teacher, CachedTeacher):
            kwargs["sample_idx"] = s
        raw = teacher.complete(prompt, **kwargs)
        obj = parse_json(raw) or {}
        answerable = bool(obj.get("answerable", False))
        answer = str(obj.get("answer", ""))
        votes.append(
            {"answerable": answerable, "answer": answer, "correct": answerable and answers_match(answer, gold)}
        )
    n_correct = sum(v["correct"] for v in votes)
    return {"votes": votes, "n_correct": n_correct, "k": k, "teacher": teacher.name}


def relabel(
    examples: list[Example],
    teacher,
    k: int = 3,
    temperature: float = 0.7,
    min_correct: int | None = None,
    workers: int = 8,
) -> tuple[list[Example], dict]:
    min_correct = k if min_correct is None else min_correct
    todo = [ex for ex in examples if ex.meta.get("answer")]
    passthrough = [ex for ex in examples if not ex.meta.get("answer")]
    out: list[Example] = list(passthrough)
    agree = disagree = 0
    budget_hit = False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(judge, teacher, ex, k, temperature): ex for ex in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="teacher-label"):
            ex = futs[fut]
            try:
                res = fut.result()
            except TeacherBudgetExceeded:
                # Stop spending: cancel everything not yet started, pass the rest through unlabelled.
                for f in futs:
                    f.cancel()
                budget_hit = True
                out.append(ex)
                continue
            except Exception as e:
                log.warning("teacher failed on %s: %s", ex.id, e)
                out.append(ex)
                continue
            new_label = res["n_correct"] >= min_correct
            if new_label == ex.sufficient:
                agree += 1
            else:
                disagree += 1
            out.append(
                Example(
                    question=ex.question,
                    evidence=ex.evidence,
                    sufficient=new_label,
                    source=ex.source,
                    category="teacher_labelled",
                    id=ex.id,
                    meta={
                        **ex.meta,
                        "heuristic_label": ex.sufficient,
                        "heuristic_category": ex.category,
                        "teacher": res,
                    },
                )
            )
    stats = {
        "labelled": agree + disagree,
        "passthrough": len(passthrough),
        "agreement": agree / max(1, agree + disagree),
        "flipped_to_sufficient": sum(
            1 for e in out if e.category == "teacher_labelled" and e.sufficient and not e.meta["heuristic_label"]
        ),
        "flipped_to_insufficient": sum(
            1 for e in out if e.category == "teacher_labelled" and not e.sufficient and e.meta["heuristic_label"]
        ),
        "budget_hit": budget_hit,
    }
    if budget_hit:
        log.error("teacher call cap reached; %d examples passed through unlabelled", len(todo) - (agree + disagree))
    return out, stats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=3, help="samples per example")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min-correct", type=int, default=None, help="default: all k must be correct")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--cache", type=Path, default=Path("data/teacher_cache/label.jsonl"))
    p.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help=f"hard cap on billed teacher calls (default {DEFAULT_MAX_CALLS}); the job stops when reached",
    )
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    examples = read_jsonl(args.inp, limit=args.limit)
    n_label = sum(1 for e in examples if e.meta.get("answer"))
    model = args.model or os.environ.get("VIVEKA_TEACHER_MODEL", "gemini-2.5-flash")
    n_calls = estimate_calls(n_label, args.k)
    cap = (
        args.max_calls
        if args.max_calls is not None
        else int(os.environ.get("VIVEKA_TEACHER_MAX_CALLS", DEFAULT_MAX_CALLS))
    )
    print(
        f"labelling {n_label} examples x k={args.k} = up to {n_calls} calls to {model} "
        f"(~${estimate_cost_usd(n_calls, model):.2f} at list price; cache hits are free). Hard cap: {cap} calls."
    )
    if n_calls > cap:
        print(f"NOTE: {n_calls} > cap {cap}; the job will stop at the cap. Pass --max-calls {n_calls} to allow all.")
    if not args.yes and sys.stdin.isatty():
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    teacher = make_teacher(cache_path=args.cache, model=args.model, max_calls=cap)
    out, stats = relabel(
        examples, teacher, k=args.k, temperature=args.temperature, min_correct=args.min_correct, workers=args.workers
    )
    write_jsonl(args.out, out)
    stats["teacher_usage"] = teacher.usage()
    print(json.dumps(stats, indent=2))
    if stats["budget_hit"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
