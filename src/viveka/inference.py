"""Inference: P(sufficient) for (question, evidence).

Library use:

    from viveka.inference import SufficiencyPredictor
    pred = SufficiencyPredictor("runs/lora-modernbert-base/best")
    p = pred.predict("Does the device require biocompatibility testing?",
                     ["Device contacts intact skin.", "Contact duration exceeds 30 days."])

CLI:

    viveka-predict --ckpt runs/lora-modernbert-base/best \
        --question "Does the user have API access?" \
        --evidence "The user is on the Enterprise plan." "Enterprise plans permit API access."

    viveka-predict --ckpt ... --jsonl data/eval/eval.jsonl --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Collator, SufficiencyDataset, model_inputs, to_device
from .formatting import encode_batch
from .model import SufficiencyModel, load_tokenizer
from .schema import Example, read_jsonl

log = logging.getLogger(__name__)


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict_logits(
    model: SufficiencyModel,
    tokenizer: Any,
    examples: list[Example],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    bf16: bool = False,
    num_workers: int = 0,
    desc: str | None = None,
) -> np.ndarray:
    """Raw (uncalibrated) logits for a list of examples."""
    model.eval()
    dl = DataLoader(
        SufficiencyDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer, max_length),
        num_workers=num_workers,
    )
    out: list[np.ndarray] = []
    it: Iterable = dl
    if desc:
        from tqdm import tqdm

        it = tqdm(dl, desc=desc, leave=False)
    for batch in it:
        batch = to_device(batch, device)
        with autocast_ctx(device, bf16):
            logits = model(**model_inputs(batch))
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)


class SufficiencyPredictor:
    def __init__(self, ckpt_dir: str | Path, device: str | None = None, max_length: int | None = None) -> None:
        self.device = pick_device(device)
        self.model = SufficiencyModel.load(ckpt_dir, device=self.device)
        base_name = self.model.model_cfg["name"]
        self.tokenizer = load_tokenizer(base_name)
        self.max_length = int(max_length or self.model.model_cfg.get("max_length", 2048))
        self.temperature = float(self.model.temperature.item())

    @torch.no_grad()
    def predict_batch(self, questions: list[str], evidences: list[list[str]]) -> np.ndarray:
        enc = encode_batch(self.tokenizer, questions, evidences, max_length=self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with autocast_ctx(self.device, self.device.type == "cuda"):
            probs = self.model.predict_proba(**model_inputs(enc))
        return probs.float().cpu().numpy()

    def predict(self, question: str, evidence: list[str]) -> float:
        return float(self.predict_batch([question], [evidence])[0])

    def decide(self, question: str, evidence: list[str], hi: float = 0.98, lo: float = 0.20) -> dict[str, Any]:
        """Three-way control signal for an agent loop."""
        p = self.predict(question, evidence)
        if p >= hi:
            action = "answer"
        elif p < lo:
            action = "retrieve_more"
        else:
            action = "uncertain"
        return {"p_sufficient": p, "action": action, "hi": hi, "lo": lo}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--question")
    p.add_argument("--evidence", nargs="*", default=[])
    p.add_argument("--jsonl", type=Path, help="score every example in this file")
    p.add_argument("--out", type=Path, help="write predictions JSONL (with --jsonl)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--hi", type=float, default=0.98)
    p.add_argument("--lo", type=float, default=0.20)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pred = SufficiencyPredictor(args.ckpt, device=args.device)
    if args.jsonl:
        exs = read_jsonl(args.jsonl)
        probs: list[float] = []
        for i in range(0, len(exs), args.batch_size):
            chunk = exs[i : i + args.batch_size]
            probs.extend(pred.predict_batch([e.question for e in chunk], [e.evidence for e in chunk]).tolist())
        rows = [
            {"id": e.id, "p_sufficient": pr, "label": e.sufficient, "category": e.category}
            for e, pr in zip(exs, probs, strict=False)
        ]
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            print(f"wrote {len(rows)} predictions -> {args.out}")
        else:
            for r in rows:
                print(json.dumps(r))
        return
    if not args.question:
        p.error("provide --question (with --evidence) or --jsonl")
    print(json.dumps(pred.decide(args.question, args.evidence, hi=args.hi, lo=args.lo), indent=2))


if __name__ == "__main__":
    main()
