"""Discrimination, calibration and selective-prediction metrics.

All functions take numpy arrays: `probs` in [0, 1] and binary `labels`.
No sklearn dependency; everything here is a few lines of numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-7


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    p, y = _clip(probs), np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p, y = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled by average rank."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    pred = np.asarray(probs) >= threshold
    y = np.asarray(labels).astype(bool)
    return {
        "tp": int((pred & y).sum()),
        "fp": int((pred & ~y).sum()),
        "tn": int((~pred & ~y).sum()),
        "fn": int((~pred & y).sum()),
    }


def classification_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    c = confusion(probs, labels, threshold)
    n = sum(c.values())
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # The expensive error for an agent: we said "enough" and it was not.
        "false_sufficient_rate": fp / (fp + tn) if fp + tn else 0.0,
        # The cheap error: we said "not enough" and it was.
        "false_insufficient_rate": fn / (fn + tp) if fn + tp else 0.0,
        **c,
    }


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width confidence bins. Each bin: mean predicted prob vs empirical rate."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": cnt,
                "mean_confidence": float(p[m].mean()) if cnt else float("nan"),
                "empirical_rate": float(y[m].mean()) if cnt else float("nan"),
            }
        )
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE for a binary probabilistic forecast: sum_b (n_b/N) |conf_b - rate_b|."""
    bins = reliability_bins(probs, labels, n_bins)
    n = sum(b["count"] for b in bins)
    if n == 0:
        return float("nan")
    return float(sum(b["count"] / n * abs(b["mean_confidence"] - b["empirical_rate"]) for b in bins if b["count"] > 0))


def selective_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    hi: float = 0.98,
    lo: float = 0.20,
) -> dict[str, float]:
    """Three-way decision: P>=hi -> sufficient, P<lo -> insufficient, else abstain.

    Reports coverage (fraction of decisions the model takes) and the error rate
    on the decisions it does take, split by direction.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n = len(p)
    say_suff = p >= hi
    say_insuff = p < lo
    abstain = ~(say_suff | say_insuff)
    n_suff, n_insuff = int(say_suff.sum()), int(say_insuff.sum())
    return {
        "hi": hi,
        "lo": lo,
        "coverage": float((n_suff + n_insuff) / n) if n else float("nan"),
        "abstain_rate": float(abstain.mean()) if n else float("nan"),
        "sufficient_calls": n_suff,
        "insufficient_calls": n_insuff,
        # Of the times we confidently said "sufficient", how often were we wrong?
        "false_sufficient_among_calls": float((say_suff & ~y).sum() / n_suff) if n_suff else 0.0,
        "false_insufficient_among_calls": float((say_insuff & y).sum() / n_insuff) if n_insuff else 0.0,
        # Fraction of all truly-insufficient cases we would wrongly wave through.
        "false_sufficient_rate": float((say_suff & ~y).sum() / max(1, (~y).sum())),
    }


def full_report(
    probs: np.ndarray,
    labels: np.ndarray,
    categories: list[str] | None = None,
    thresholds: tuple[float, ...] = (0.5, 0.9, 0.98),
    selective_hi: float = 0.98,
    selective_lo: float = 0.20,
    n_bins: int = 10,
) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    rep: dict[str, Any] = {
        "n": len(probs),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "log_loss": log_loss(probs, labels),
        "brier": brier(probs, labels),
        "auroc": auroc(probs, labels),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "reliability": reliability_bins(probs, labels, n_bins),
        "at_threshold": {str(t): classification_metrics(probs, labels, t) for t in thresholds},
        "selective": selective_metrics(probs, labels, selective_hi, selective_lo),
    }
    if categories is not None:
        cats = np.asarray(categories)
        per_cat: dict[str, Any] = {}
        for c in sorted(set(cats.tolist())):
            m = cats == c
            sub_p, sub_y = probs[m], labels[m]
            per_cat[c] = {
                "n": int(m.sum()),
                "positive_rate": float(sub_y.mean()),
                "accuracy@0.5": classification_metrics(sub_p, sub_y, 0.5)["accuracy"],
                "log_loss": log_loss(sub_p, sub_y),
                "brier": brier(sub_p, sub_y),
                "mean_prob": float(sub_p.mean()),
            }
        rep["per_category"] = per_cat
    return rep


def format_report(rep: dict[str, Any]) -> str:
    lines = [
        f"n={rep['n']}  pos_rate={rep['positive_rate']:.3f}",
        f"log_loss={rep['log_loss']:.4f}  brier={rep['brier']:.4f}  auroc={rep['auroc']:.4f}  ece={rep['ece']:.4f}",
    ]
    for t, m in rep["at_threshold"].items():
        lines.append(
            f"@{t}: acc={m['accuracy']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} false_suff={m['false_sufficient_rate']:.3f}"
        )
    s = rep["selective"]
    lines.append(
        f"selective[hi={s['hi']},lo={s['lo']}]: coverage={s['coverage']:.3f} "
        f"false_suff_among_calls={s['false_sufficient_among_calls']:.3f} "
        f"false_insuff_among_calls={s['false_insufficient_among_calls']:.3f}"
    )
    if "per_category" in rep:
        lines.append("per category:")
        for c, m in rep["per_category"].items():
            lines.append(
                f"  {c:<24} n={m['n']:<6} acc={m['accuracy@0.5']:.3f} "
                f"logloss={m['log_loss']:.3f} mean_p={m['mean_prob']:.3f} pos={m['positive_rate']:.2f}"
            )
    return "\n".join(lines)
