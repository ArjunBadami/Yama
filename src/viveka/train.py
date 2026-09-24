"""Train the sufficiency model.

    viveka-train --config configs/lora.yaml
    viveka-train --config configs/lora.yaml train.lr=1e-4 output.gcs_dir=gs://bucket/runs/x

Features
  - frozen / LoRA / full modes from config
  - bf16 autocast on CUDA
  - linear warmup + linear decay
  - periodic validation with the full metrics report (log loss is the model-selection metric)
  - resumable: `<out>/last/` holds model + optimizer + scheduler + RNG + step;
    re-running the same command continues where it left off (spot-VM friendly)
  - optional GCS mirroring of the output dir after every save
  - after training: temperature-calibrate the best checkpoint on the val set

Outputs under output.dir:
  config.yaml, metrics.jsonl, last/, best/ (best/val_report.json, best/viveka_model.json[temperature])
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import gcs
from .config import Config, load_config, save_config
from .data import Collator, SufficiencyDataset, model_inputs, to_device
from .inference import autocast_ctx, pick_device, predict_logits
from .metrics import format_report, full_report
from .model import SufficiencyModel, fit_temperature, load_tokenizer
from .schema import read_jsonl

log = logging.getLogger("viveka.train")


# ----------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_bf16(flag: Any, device: torch.device) -> bool:
    if isinstance(flag, str) and flag.lower() == "auto":
        return device.type == "cuda" and torch.cuda.is_bf16_supported()
    return bool(flag) and device.type == "cuda"


def linear_schedule(optimizer: torch.optim.Optimizer, warmup: int, total: int) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only parameters that train (+ buffers we care about). Small for LoRA/frozen."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items() if k in names or k == "temperature"}


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.out = Path(cfg.output.dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.gcs_dir: str | None = cfg.output.get("gcs_dir") or None
        self.device = pick_device(os.environ.get("VIVEKA_DEVICE"))
        self.bf16 = resolve_bf16(cfg.train.get("bf16", "auto"), self.device)
        set_seed(int(cfg.get("seed", 0)))

        save_config(cfg, self.out / "config.yaml")
        if self.gcs_dir:
            gcs.pull_if_missing(self.gcs_dir, self.out)

        # data
        max_train = cfg.data.get("max_train_examples")
        self.train_examples = read_jsonl(cfg.data.train, limit=max_train)
        self.val_examples = read_jsonl(cfg.data.val)
        log.info(
            "train=%d val=%d device=%s bf16=%s",
            len(self.train_examples),
            len(self.val_examples),
            self.device,
            self.bf16,
        )

        # model
        self.tokenizer = load_tokenizer(cfg.model.name)
        self.model = SufficiencyModel.build(dict(cfg.model)).to(self.device)
        self.max_length = int(cfg.model.get("max_length", 2048))

        # optimisation
        t = cfg.train
        self.batch_size = int(t.batch_size)
        self.grad_accum = int(t.get("grad_accum", 1))
        self.epochs = int(t.epochs)
        self.steps_per_epoch = math.ceil(len(self.train_examples) / self.batch_size / self.grad_accum)
        self.total_steps = int(t.get("max_steps") or self.steps_per_epoch * self.epochs)
        warmup = int(self.total_steps * float(t.get("warmup_ratio", 0.06)))
        groups = self.model.param_groups(
            float(t.lr), float(t.get("encoder_lr", 0.0)), float(t.get("weight_decay", 0.0))
        )
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)
        self.scheduler = linear_schedule(self.optimizer, warmup, self.total_steps)
        pos_weight = torch.tensor(float(t.get("pos_weight", 1.0)), device=self.device)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.max_grad_norm = float(t.get("max_grad_norm", 1.0))

        self.step = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.metrics_path = self.out / "metrics.jsonl"
        self._maybe_resume()

    # ------------------------------------------------------------- checkpoints
    def _save(self, tag: str, extra: dict[str, Any] | None = None) -> Path:
        d = self.out / tag
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d)
        if tag == "last":
            torch.save(
                {
                    "step": self.step,
                    "epoch": self.epoch,
                    "best_val": self.best_val,
                    "model_state": trainable_state(self.model),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                },
                d / "trainer_state.pt",
            )
        for k, v in (extra or {}).items():
            (d / k).write_text(json.dumps(v, indent=2), encoding="utf-8")
        if self.gcs_dir:
            gcs.rsync(self.out, self.gcs_dir)
        return d

    def _maybe_resume(self) -> None:
        state_path = self.out / "last" / "trainer_state.pt"
        if not state_path.exists():
            return
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        _, unexpected = self.model.load_state_dict(st["model_state"], strict=False)
        if unexpected:
            log.warning("unexpected keys on resume: %s", unexpected[:5])
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        self.step = int(st["step"])
        self.epoch = int(st["epoch"])
        self.best_val = float(st["best_val"])
        rng = st.get("rng") or {}
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:  # pragma: no cover
            log.warning("could not restore RNG state: %s", e)
        log.info(
            "resumed from %s at step %d (epoch %d, best_val=%.4f)", state_path, self.step, self.epoch, self.best_val
        )

    # ----------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, examples, desc: str = "val") -> tuple[dict[str, Any], np.ndarray]:
        logits = predict_logits(
            self.model,
            self.tokenizer,
            examples,
            batch_size=self.batch_size * 2,
            max_length=self.max_length,
            device=self.device,
            bf16=self.bf16,
            desc=desc,
        )
        probs = 1.0 / (1.0 + np.exp(-logits))
        labels = np.array([e.label for e in examples])
        cats = [e.category for e in examples]
        rep = full_report(probs, labels, categories=cats)
        self.model.train()
        return rep, logits

    def _run_validation(self) -> None:
        rep, _ = self.evaluate(self.val_examples)
        rec = {
            "step": self.step,
            "epoch": self.epoch,
            "split": "val",
            "log_loss": rep["log_loss"],
            "brier": rep["brier"],
            "auroc": rep["auroc"],
            "ece": rep["ece"],
            "acc@0.5": rep["at_threshold"]["0.5"]["accuracy"],
            "false_suff@0.98": rep["selective"]["false_sufficient_among_calls"],
            "coverage@sel": rep["selective"]["coverage"],
        }
        append_jsonl(self.metrics_path, rec)
        log.info("[val step %d] %s", self.step, format_report(rep).replace("\n", " | "))
        if rep["log_loss"] < self.best_val:
            self.best_val = rep["log_loss"]
            self._save("best", {"val_report.json": rep})
            log.info("new best val log_loss=%.4f -> %s", self.best_val, self.out / "best")

    # ----------------------------------------------------------------- train
    def _loader(self, epoch: int) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(int(self.cfg.get("seed", 0)) + epoch)
        return DataLoader(
            SufficiencyDataset(self.train_examples),
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            collate_fn=Collator(self.tokenizer, self.max_length),
            num_workers=int(self.cfg.train.get("num_workers", 0)),
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

    def train(self) -> Path:
        t = self.cfg.train
        log_every = int(t.get("log_every_steps", 20))
        eval_every = int(t.get("eval_every_steps", 500))
        save_every = int(t.get("save_every_steps", 500))
        self.model.train()
        log.info("total_steps=%d steps/epoch=%d", self.total_steps, self.steps_per_epoch)

        start_epoch = self.epoch
        t0 = time.time()
        running, running_n = 0.0, 0
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            loader = self._loader(epoch)
            # skip already-consumed optimizer steps when resuming mid-epoch
            skip_batches = (
                (self.step - epoch * self.steps_per_epoch) * self.grad_accum
                if self.step > epoch * self.steps_per_epoch
                else 0
            )
            micro = 0
            self.optimizer.zero_grad(set_to_none=True)
            for batch in loader:
                if skip_batches > 0:
                    skip_batches -= 1
                    continue
                batch = to_device(batch, self.device)
                with autocast_ctx(self.device, self.bf16):
                    logits = self.model(**model_inputs(batch))
                    loss = self.loss_fn(logits.float(), batch["labels"]) / self.grad_accum
                loss.backward()
                running += loss.item() * self.grad_accum
                running_n += 1
                micro += 1
                if micro % self.grad_accum != 0:
                    continue

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad), self.max_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1

                if self.step % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    rec = {
                        "step": self.step,
                        "epoch": epoch,
                        "split": "train",
                        "loss": running / max(1, running_n),
                        "lr": lr,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    append_jsonl(self.metrics_path, rec)
                    log.info("step %d/%d loss=%.4f lr=%.2e", self.step, self.total_steps, rec["loss"], lr)
                    running, running_n = 0.0, 0
                if self.step % eval_every == 0:
                    self._run_validation()
                if self.step % save_every == 0:
                    self._save("last")
                if self.step >= self.total_steps:
                    break
            if self.step >= self.total_steps:
                break
            self._save("last")

        # final validation + save
        self.epoch = self.epochs
        self._run_validation()
        self._save("last")
        return self._finalize()

    # -------------------------------------------------------------- finalize
    def _finalize(self) -> Path:
        best_dir = self.out / "best"
        if not best_dir.exists():
            best_dir = self.out / "last"
        # Temperature-scale the best checkpoint on validation logits.
        best = SufficiencyModel.load(best_dir, device=self.device)
        logits = predict_logits(
            best,
            self.tokenizer,
            self.val_examples,
            batch_size=self.batch_size * 2,
            max_length=self.max_length,
            device=self.device,
            bf16=self.bf16,
            desc="calib",
        )
        labels = np.array([e.label for e in self.val_examples])
        T = fit_temperature(logits, labels)
        best.temperature.fill_(T)
        best.save(best_dir)
        probs_raw = 1 / (1 + np.exp(-logits))
        probs_cal = 1 / (1 + np.exp(-logits / T))
        rep_raw = full_report(probs_raw, labels)
        rep_cal = full_report(probs_cal, labels)
        summary = {
            "temperature": T,
            "val_raw": {k: rep_raw[k] for k in ("log_loss", "brier", "ece", "auroc")},
            "val_calibrated": {k: rep_cal[k] for k in ("log_loss", "brier", "ece", "auroc")},
        }
        (best_dir / "calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(
            "temperature=%.3f  val ece raw=%.4f -> cal=%.4f  logloss raw=%.4f -> cal=%.4f",
            T,
            rep_raw["ece"],
            rep_cal["ece"],
            rep_raw["log_loss"],
            rep_cal["log_loss"],
        )
        if self.gcs_dir:
            gcs.rsync(self.out, self.gcs_dir)
        return best_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-4")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_config(args.config, args.overrides)
    best = Trainer(cfg).train()
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
