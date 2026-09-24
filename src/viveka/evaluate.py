"""Evaluate a checkpoint on a JSONL file: discrimination, calibration, per-category, selective.

    viveka-evaluate --ckpt runs/lora-modernbert-base/best --data data/processed/eval.jsonl \
        --out runs/lora-modernbert-base/eval_report.json

    # Re-fit temperature on a held-out calibration file and write it into the checkpoint
    viveka-evaluate --ckpt ... --data data/processed/val.jsonl --fit-temperature

Reports are computed with the checkpoint's stored temperature applied
(`--raw` to disable). Use `--dump-predictions preds.jsonl` for error analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .inference import pick_device, predict_logits
from .metrics import format_report, full_report
from .model import SufficiencyModel, fit_temperature, load_tokenizer
from .schema import read_jsonl

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="write full JSON report here")
    p.add_argument("--dump-predictions", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--raw", action="store_true", help="ignore stored temperature")
    p.add_argument("--fit-temperature", action="store_true", help="fit T on --data and save into --ckpt")
    p.add_argument("--hi", type=float, default=0.98)
    p.add_argument("--lo", type=float, default=0.20)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    device = pick_device(args.device)
    model = SufficiencyModel.load(args.ckpt, device=device)
    tok = load_tokenizer(model.model_cfg["name"])
    max_len = int(args.max_length or model.model_cfg.get("max_length", 2048))
    examples = read_jsonl(args.data, limit=args.limit)
    labels = np.array([e.label for e in examples])
    cats = [e.category for e in examples]

    logits = predict_logits(
        model,
        tok,
        examples,
        batch_size=args.batch_size,
        max_length=max_len,
        device=device,
        bf16=device.type == "cuda",
        desc="eval",
    )

    if args.fit_temperature:
        T = fit_temperature(logits, labels)
        model.temperature.fill_(T)
        model.save(args.ckpt)
        log.info("fitted temperature=%.4f and saved to %s", T, args.ckpt)

    T = 1.0 if args.raw else float(model.temperature.item())
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    rep = full_report(probs, labels, categories=cats, selective_hi=args.hi, selective_lo=args.lo)
    rep["temperature"] = T
    rep["checkpoint"] = str(args.ckpt)
    rep["data"] = str(args.data)
    print(format_report(rep))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log.info("wrote report -> %s", args.out)
    if args.dump_predictions:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_predictions.open("w", encoding="utf-8") as f:
            for e, pr, z in zip(examples, probs, logits, strict=False):
                f.write(
                    json.dumps(
                        {
                            "id": e.id,
                            "source": e.source,
                            "category": e.category,
                            "label": e.sufficient,
                            "p_sufficient": float(pr),
                            "logit": float(z),
                            "question": e.question,
                            "evidence": e.evidence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        log.info("wrote predictions -> %s", args.dump_predictions)


if __name__ == "__main__":
    main()
