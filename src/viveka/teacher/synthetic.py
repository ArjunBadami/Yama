"""Synthetic (question, required facts, distractors) generation with a teacher model.

The teacher writes the *ingredients*; the curriculum in datasets/degrade.py does
the labelling by construction (all facts -> sufficient, minus one -> insufficient,
etc.). This keeps labels mechanical and auditable rather than asking the teacher
"is this sufficient?" directly, which would import its biases.

    python -m viveka.teacher.synthetic --n 300 --out data/processed/synthetic.jsonl \
        --domains medical_devices customer_support finance_compliance

To reduce generator artefacts (section 38.4 of the plan): vary domains, vary
the number of required facts, and ask for distractors that share vocabulary
with the question. Mixing two teacher models is a further mitigation; pass
--model to run the script twice with different generators.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ..datasets.degrade import VariantSpec, build_variants, category_counts, label_balance
from ..schema import Example, write_jsonl
from .client import (
    DEFAULT_MAX_CALLS,
    CachedTeacher,
    TeacherBudgetExceeded,
    estimate_cost_usd,
    make_teacher,
    parse_json,
)

log = logging.getLogger(__name__)

DOMAINS = [
    "medical_devices_regulatory",
    "customer_support_saas",
    "finance_compliance",
    "insurance_claims",
    "employment_law",
    "supply_chain_logistics",
    "software_incident_response",
    "real_estate_contracts",
    "clinical_trial_eligibility",
    "travel_and_visa_rules",
    "academic_admissions",
    "consumer_electronics_warranty",
]

PROMPT = """You are generating training data for a model that judges whether a set of evidence is SUFFICIENT to answer a question.

Domain: {domain}
Number of required facts: {n_facts}

Write ONE realistic question from this domain whose answer can be determined ONLY by combining exactly {n_facts} separate facts. No single fact should be enough on its own.

Then write:
- "required_facts": the {n_facts} facts, each a standalone declarative sentence. Together they fully determine the answer.
- "answer": the short correct answer.
- "distractors": 6 to 8 sentences that are clearly about the same entities/topic and reuse words from the question, but do NOT help answer it. Make some of them look like they might be relevant. None may contradict the required facts or reveal the answer.

Vary sentence style; do not make the required facts stylistically distinguishable from the distractors (similar length, register, and specificity).

Respond with JSON only:
{{"question": "...", "answer": "...", "required_facts": ["...", ...], "distractors": ["...", ...]}}"""


def generate_one(teacher, domain: str, n_facts: int, sample_idx: int, temperature: float) -> dict | None:
    prompt = PROMPT.format(domain=domain, n_facts=n_facts)
    kwargs = {"temperature": temperature, "json_mode": True}
    if isinstance(teacher, CachedTeacher):
        kwargs["sample_idx"] = sample_idx
    raw = teacher.complete(prompt, **kwargs)
    obj = parse_json(raw)
    if not isinstance(obj, dict):
        return None
    facts = [str(f).strip() for f in obj.get("required_facts", []) if str(f).strip()]
    distract = [str(d).strip() for d in obj.get("distractors", []) if str(d).strip()]
    q = str(obj.get("question", "")).strip()
    if not q or len(facts) < 1 or len(distract) < 3:
        return None
    return {
        "question": q,
        "answer": str(obj.get("answer", "")).strip(),
        "gold": facts,
        "distractors": distract,
        "domain": domain,
        "n_facts_requested": n_facts,
    }


def generate(
    teacher,
    n: int,
    domains: list[str],
    seed: int = 0,
    temperature: float = 1.0,
    workers: int = 8,
    spec: VariantSpec | None = None,
) -> tuple[list[Example], list[dict]]:
    rng = random.Random(seed)
    spec = spec or VariantSpec(
        n_clean_sufficient=1,
        n_distractor_heavy=1,
        n_missing_one=1,
        n_relevant_insufficient=1,
        n_long_insufficient=0,
        n_obvious_insufficient=0,
        light_noise_max=2,
        heavy_noise_min=4,
        heavy_noise_max=8,
    )
    jobs = [(rng.choice(domains), rng.choice([2, 2, 3, 3, 4]), i) for i in range(n)]
    raws: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(generate_one, teacher, d, k, i, temperature) for d, k, i in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="synthetic"):
            try:
                r = fut.result()
            except TeacherBudgetExceeded:
                for f in futs:
                    f.cancel()
                log.error("teacher call cap reached; stopping generation with %d questions", len(raws))
                break
            except Exception as e:
                log.warning("generation failed: %s", e)
                continue
            if r:
                raws.append(r)

    # Unrelated pool: distractors from a different generated item.
    out: list[Example] = []
    for i, r in enumerate(raws):
        other = raws[(i + 1) % len(raws)] if len(raws) > 1 else r
        out.extend(
            build_variants(
                r["question"],
                r["gold"],
                r["distractors"],
                other["distractors"],
                source="synthetic",
                group_id=f"synthetic-{seed}-{i}",
                rng=rng,
                spec=VariantSpec(**{**spec.__dict__, "extra_meta": {"domain": r["domain"], "teacher": teacher.name}}),
                answer=r["answer"],
                multi_hop=len(r["gold"]) >= 2,
            )
        )
    return out, raws


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="number of source questions to generate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw-out", type=Path, default=None, help="also dump raw teacher outputs")
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None)
    p.add_argument("--cache", type=Path, default=Path("data/teacher_cache/synthetic.jsonl"))
    p.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help=f"hard cap on billed teacher calls (default {DEFAULT_MAX_CALLS}); generation stops when reached",
    )
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model = args.model or os.environ.get("VIVEKA_TEACHER_MODEL", "gemini-2.5-flash")
    cap = (
        args.max_calls
        if args.max_calls is not None
        else int(os.environ.get("VIVEKA_TEACHER_MAX_CALLS", DEFAULT_MAX_CALLS))
    )
    # generation prompts are ~350 tokens in, ~450 out
    print(
        f"generating {args.n} questions = {args.n} calls to {model} "
        f"(~${estimate_cost_usd(args.n, model, in_tokens=350, out_tokens=450):.2f} at list price). Hard cap: {cap} calls."
    )
    if args.n > cap:
        print(f"NOTE: {args.n} > cap {cap}; generation will stop at the cap. Pass --max-calls {args.n} to allow all.")
    if not args.yes and sys.stdin.isatty():
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return

    teacher = make_teacher(cache_path=args.cache, model=args.model, max_calls=cap)
    examples, raws = generate(
        teacher, args.n, args.domains, seed=args.seed, temperature=args.temperature, workers=args.workers
    )
    write_jsonl(args.out, examples)
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raws), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_questions": len(raws),
                "examples": label_balance(examples),
                "categories": category_counts(examples),
                "teacher_usage": teacher.usage(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
