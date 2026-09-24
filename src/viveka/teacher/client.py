"""Teacher client abstraction with a Gemini-on-Vertex implementation and a disk cache.

Auth on GCP: Application Default Credentials.
  local : gcloud auth application-default login
  VM    : the attached service account (needs roles/aiplatform.user)

Environment:
  GOOGLE_CLOUD_PROJECT   (default: gcloud's active project)
  GOOGLE_CLOUD_LOCATION  (default: us-central1)
  VIVEKA_TEACHER_MODEL   (default: gemini-2.5-flash)

Cost control: every GeminiTeacher has a hard cap on the number of *billed* calls
(`max_calls`, default 1000; cache hits don't count). Exceeding it raises
TeacherBudgetExceeded so a runaway script stops instead of spending. `usage()` reports
calls, tokens and an estimated cost from the PRICES table (approximate list prices; verify).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Approximate Vertex list prices; update when they change.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_MAX_CALLS = 1000


class TeacherBudgetExceeded(RuntimeError):
    pass


class Teacher(Protocol):
    name: str

    def complete(self, prompt: str, *, temperature: float = 0.0, json_mode: bool = False) -> str: ...

    def usage(self) -> dict: ...


class GeminiTeacher:
    def __init__(
        self,
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
        max_retries: int = 5,
        max_calls: int | None = None,
    ) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'viveka[teacher]'  (google-genai) to use GeminiTeacher") from e

        self.model = model or os.environ.get("VIVEKA_TEACHER_MODEL", "gemini-2.5-flash")
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.max_retries = max_retries
        self.max_calls = int(
            max_calls if max_calls is not None else os.environ.get("VIVEKA_TEACHER_MAX_CALLS", DEFAULT_MAX_CALLS)
        )
        self.name = f"vertex:{self.model}"
        self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
        self._types = __import__("google.genai.types", fromlist=["types"])
        self._lock = threading.Lock()
        self._calls = 0
        self._in_tokens = 0
        self._out_tokens = 0

    def complete(self, prompt: str, *, temperature: float = 0.0, json_mode: bool = False) -> str:
        with self._lock:
            if self._calls >= self.max_calls:
                raise TeacherBudgetExceeded(
                    f"teacher call cap reached ({self.max_calls}); est. spend so far ${self.usage()['est_cost_usd']:.2f}. "
                    "Raise --max-calls / VIVEKA_TEACHER_MAX_CALLS to continue."
                )
            self._calls += 1
        cfg = self._types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                um = getattr(resp, "usage_metadata", None)
                if um is not None:
                    with self._lock:
                        self._in_tokens += int(getattr(um, "prompt_token_count", 0) or 0)
                        self._out_tokens += int(getattr(um, "candidates_token_count", 0) or 0)
                return resp.text or ""
            except Exception as e:  # rate limits / transient 5xx
                if attempt == self.max_retries - 1:
                    raise
                log.warning("teacher call failed (%s); retrying in %.1fs", e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return ""  # unreachable

    def usage(self) -> dict:
        price_in, price_out = PRICES.get(self.model, PRICES["gemini-2.5-flash"])
        cost = self._in_tokens / 1e6 * price_in + self._out_tokens / 1e6 * price_out
        return {
            "model": self.model,
            "billed_calls": self._calls,
            "max_calls": self.max_calls,
            "input_tokens": self._in_tokens,
            "output_tokens": self._out_tokens,
            "est_cost_usd": round(cost, 4),
            "price_known": self.model in PRICES,
        }


class CachedTeacher:
    """Wraps any Teacher with an append-only JSONL cache keyed by (model, prompt, temperature, json_mode, sample_idx).

    Sampling with temperature > 0 needs several *distinct* completions per prompt, so
    callers pass `sample_idx` to keep them apart.
    """

    def __init__(self, inner: Teacher, cache_path: str | Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["k"]] = rec["v"]
                    except Exception:
                        continue
            log.info("teacher cache: %d entries from %s", len(self._cache), self.path)

    def _key(self, prompt: str, temperature: float, json_mode: bool, sample_idx: int) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.name, prompt, temperature, json_mode, sample_idx]).encode("utf-8"))
        return h.hexdigest()

    def complete(self, prompt: str, *, temperature: float = 0.0, json_mode: bool = False, sample_idx: int = 0) -> str:
        k = self._key(prompt, temperature, json_mode, sample_idx)
        with self._lock:
            if k in self._cache:
                return self._cache[k]
        v = self.inner.complete(prompt, temperature=temperature, json_mode=json_mode)
        with self._lock:
            self._cache[k] = v
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        return v

    def usage(self) -> dict:
        return self.inner.usage()


def make_teacher(cache_path: str | Path | None = None, **kw) -> Teacher:
    t: Teacher = GeminiTeacher(**kw)
    if cache_path:
        return CachedTeacher(t, cache_path)
    return t


def estimate_calls(n_examples: int, k: int = 1) -> int:
    return n_examples * k


def estimate_cost_usd(
    n_calls: int, model: str = "gemini-2.5-flash", in_tokens: int = 450, out_tokens: int = 40
) -> float:
    """Rough pre-run estimate for a labelling job: default token counts match label.py's prompt."""
    price_in, price_out = PRICES.get(model, PRICES["gemini-2.5-flash"])
    return n_calls * (in_tokens / 1e6 * price_in + out_tokens / 1e6 * price_out)


def parse_json(text: str) -> dict | list | None:
    """Tolerant JSON extraction from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the outermost {...} or [...]
    for open_, close in (("{", "}"), ("[", "]")):
        i, j = text.find(open_), text.rfind(close)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None
