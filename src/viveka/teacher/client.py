"""Teacher client abstraction with a Gemini-on-Vertex implementation and a disk cache.

Auth on GCP: Application Default Credentials.
  local : gcloud auth application-default login
  VM    : the attached service account (needs roles/aiplatform.user)

Environment:
  GOOGLE_CLOUD_PROJECT   (default: gcloud's active project)
  GOOGLE_CLOUD_LOCATION  (default: us-central1)
  VIVEKA_TEACHER_MODEL   (default: gemini-2.5-flash)
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


class Teacher(Protocol):
    name: str

    def complete(self, prompt: str, *, temperature: float = 0.0, json_mode: bool = False) -> str: ...


class GeminiTeacher:
    def __init__(
        self,
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
        max_retries: int = 5,
    ) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'viveka[teacher]'  (google-genai) to use GeminiTeacher") from e

        self.model = model or os.environ.get("VIVEKA_TEACHER_MODEL", "gemini-2.5-flash")
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.max_retries = max_retries
        self.name = f"vertex:{self.model}"
        self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
        self._types = __import__("google.genai.types", fromlist=["types"])

    def complete(self, prompt: str, *, temperature: float = 0.0, json_mode: bool = False) -> str:
        cfg = self._types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                return resp.text or ""
            except Exception as e:  # rate limits / transient 5xx
                if attempt == self.max_retries - 1:
                    raise
                log.warning("teacher call failed (%s); retrying in %.1fs", e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return ""  # unreachable


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


def make_teacher(cache_path: str | Path | None = None, **kw) -> Teacher:
    t: Teacher = GeminiTeacher(**kw)
    if cache_path:
        return CachedTeacher(t, cache_path)
    return t


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
