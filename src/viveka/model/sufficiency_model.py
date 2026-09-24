"""The Viveka model: pretrained encoder -> pooled vector -> one logit -> sigmoid.

    h = Encoder(question, evidence)          (pooled: CLS or mean)
    z = w^T h + b
    P(sufficient) = sigmoid(z / T)           (T = temperature, 1.0 until calibrated)

Three training modes, selected in config:
    frozen : encoder weights frozen, only the head trains
    lora   : encoder frozen, LoRA adapters + head train
    full   : everything trains
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

_MODEL_CFG_FILE = "viveka_model.json"
_HEAD_FILE = "head.pt"
_ADAPTER_DIR = "adapter"
_ENCODER_DIR = "encoder"


def load_tokenizer(name_or_path: str) -> Any:
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token or tok.eos_token
    return tok


def _resolve_attn_implementation(requested: str) -> str | None:
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401

                return "flash_attention_2"
            except Exception:
                return "sdpa"
        return None  # let transformers pick its default on CPU
    return requested


def _load_encoder(name: str, attn_implementation: str) -> nn.Module:
    impl = _resolve_attn_implementation(attn_implementation)
    kwargs: dict[str, Any] = {}
    if impl:
        kwargs["attn_implementation"] = impl
    try:
        return AutoModel.from_pretrained(name, **kwargs)
    except (ValueError, ImportError) as e:
        if impl:
            log.warning("attn_implementation=%s unavailable (%s); falling back to default", impl, e)
            return AutoModel.from_pretrained(name)
        raise


class SufficiencyModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        pooling: str = "cls",
        dropout: float = 0.1,
        model_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be cls|mean, got {pooling!r}")
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        # Calibration temperature. Not trained by the main loss; fit post hoc.
        self.register_buffer("temperature", torch.ones(()))
        self.model_cfg: dict[str, Any] = dict(model_cfg or {})
        self._accepts_token_type_ids = "token_type_ids" in inspect.signature(_unwrap(encoder).forward).parameters
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, model_cfg: dict[str, Any]) -> SufficiencyModel:
        name = model_cfg["name"]
        mode = model_cfg.get("mode", "lora")
        encoder = _load_encoder(name, model_cfg.get("attn_implementation", "auto"))
        hidden = encoder.config.hidden_size

        if mode == "frozen":
            for p in encoder.parameters():
                p.requires_grad_(False)
        elif mode == "lora":
            encoder = _apply_lora(encoder, model_cfg.get("lora", {}))
        elif mode == "full":
            pass
        else:
            raise ValueError(f"unknown model.mode {mode!r}; expected frozen|lora|full")

        if mode in ("lora", "full") and model_cfg.get("gradient_checkpointing", True):
            # Backward through 2048-token sequences does not fit a 24GB L4 at batch 16.
            # Recompute activations instead of storing them. LoRA needs input grads or the
            # checkpointed layers receive no gradient.
            if hasattr(encoder, "enable_input_require_grads"):
                encoder.enable_input_require_grads()
            if hasattr(encoder, "gradient_checkpointing_enable"):
                try:
                    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    encoder.gradient_checkpointing_enable()
                log.info("gradient checkpointing enabled")

        model = cls(
            encoder,
            hidden,
            pooling=model_cfg.get("pooling", "cls"),
            dropout=float(model_cfg.get("dropout", 0.1)),
            model_cfg=model_cfg,
        )
        log.info(
            "built SufficiencyModel: %s mode=%s pooling=%s trainable=%s/%s",
            name,
            mode,
            model.pooling,
            f"{model.num_trainable():,}",
            f"{model.num_params():,}",
        )
        return model

    # ---------------------------------------------------------------- forward
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: Any) -> torch.Tensor:
        enc_kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._accepts_token_type_ids and "token_type_ids" in kw and kw["token_type_ids"] is not None:
            enc_kwargs["token_type_ids"] = kw["token_type_ids"]
        out = self.encoder(**enc_kwargs)
        hidden = out.last_hidden_state  # [B, L, d]
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: Any) -> torch.Tensor:
        """Returns raw (uncalibrated) logits of shape [B]."""
        pooled = self.encode(input_ids, attention_mask, **kw)
        return self.head(self.dropout(pooled)).squeeze(-1)

    def calibrated_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kw: Any) -> torch.Tensor:
        return torch.sigmoid(self.calibrated_logits(self.forward(input_ids, attention_mask, **kw)))

    # ------------------------------------------------------------------ utils
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_groups(self, head_lr: float, encoder_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Head (and LoRA adapters) at `head_lr`; base encoder weights at `encoder_lr`."""
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        lora_params, base_params = [], []
        for n, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if "lora_" in n else base_params).append(p)
        groups: list[dict[str, Any]] = []
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": 0.0})
        if lora_params:
            groups.append({"params": lora_params, "lr": head_lr, "weight_decay": weight_decay})
        if base_params:
            groups.append({"params": base_params, "lr": encoder_lr, "weight_decay": weight_decay})
        return groups

    # --------------------------------------------------------------- save/load
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        mode = self.model_cfg.get("mode", "lora")
        meta = {
            "model_cfg": self.model_cfg,
            "pooling": self.pooling,
            "temperature": float(self.temperature.item()),
            "hidden_size": self.head.in_features,
        }
        (out / _MODEL_CFG_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        torch.save(self.head.state_dict(), out / _HEAD_FILE)
        if mode == "lora":
            self.encoder.save_pretrained(out / _ADAPTER_DIR)
        elif mode == "full":
            _unwrap(self.encoder).save_pretrained(out / _ENCODER_DIR)
        # frozen: base weights are re-loaded from the hub by name.

    @classmethod
    def load(cls, ckpt_dir: str | Path, device: str | torch.device | None = None) -> SufficiencyModel:
        d = Path(ckpt_dir)
        meta = json.loads((d / _MODEL_CFG_FILE).read_text(encoding="utf-8"))
        model_cfg = meta["model_cfg"]
        mode = model_cfg.get("mode", "lora")
        attn = model_cfg.get("attn_implementation", "auto")

        if mode == "full" and (d / _ENCODER_DIR).exists():
            encoder = _load_encoder(str(d / _ENCODER_DIR), attn)
        else:
            encoder = _load_encoder(model_cfg["name"], attn)
            if mode == "lora":
                from peft import PeftModel

                encoder = PeftModel.from_pretrained(encoder, str(d / _ADAPTER_DIR))
        model = cls(
            encoder,
            meta["hidden_size"],
            pooling=meta["pooling"],
            dropout=float(model_cfg.get("dropout", 0.1)),
            model_cfg=model_cfg,
        )
        model.head.load_state_dict(torch.load(d / _HEAD_FILE, map_location="cpu"))
        model.temperature.fill_(float(meta.get("temperature", 1.0)))
        model.eval()
        if device is not None:
            model.to(device)
        return model


def _unwrap(m: nn.Module) -> nn.Module:
    """Return the underlying transformers model beneath any PEFT wrapper."""
    if hasattr(m, "get_base_model"):
        return m.get_base_model()
    return m


def _apply_lora(encoder: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    targets = list(lora_cfg.get("target_modules") or _default_lora_targets(encoder))
    cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=targets,
        bias="none",
        # FEATURE_EXTRACTION keeps peft from expecting a task head of its own.
        task_type="FEATURE_EXTRACTION",
    )
    peft_model = get_peft_model(encoder, cfg)
    return peft_model


def _default_lora_targets(encoder: nn.Module) -> list[str]:
    model_type = getattr(encoder.config, "model_type", "")
    if model_type == "modernbert":
        return ["Wqkv", "Wo", "Wi"]
    # bert / roberta / deberta-style names
    return ["query", "key", "value", "dense"]


def hidden_size_of(name: str) -> int:
    return AutoConfig.from_pretrained(name).hidden_size
