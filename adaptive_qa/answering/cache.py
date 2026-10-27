"""Disk-backed cache for LLM answer generation in evaluations.

Provides resumable batch evaluation: completed answers are stored immediately
with their generated text, token usage, latency, model, and metadata.
Re-running evaluation reuses previously cached answers, avoiding redundant
expensive model calls and enabling graceful resumption after interruptions or timeouts.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import PROJECT_ROOT


DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "eval_answer_cache.jsonl"


def _cache_key(question: str, grounded: bool, model: str, chunk_ids: Optional[Sequence[str]] = None) -> str:
    norm_q = " ".join(question.strip().lower().split())
    chunks_str = ";".join(sorted(chunk_ids)) if chunk_ids else ""
    raw = f"{norm_q}||{grounded}||{model.strip().lower()}||{chunks_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    """Thread-safe append-only JSONL answer cache."""

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self.path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        k = record.get("key")
                        if k:
                            self._entries[k] = record
                    except Exception:
                        continue
        except Exception:
            pass

    def get(
        self,
        question: str,
        grounded: bool,
        model: str,
        chunk_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        k = _cache_key(question, grounded, model, chunk_ids)
        return self._entries.get(k)

    def put(
        self,
        question: str,
        grounded: bool,
        model: str,
        chunk_ids: Optional[Sequence[str]],
        text: Optional[str],
        tokens: Optional[Dict[str, Any]],
        gen_ms: Optional[float],
        backend: str = "llm",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        k = _cache_key(question, grounded, model, chunk_ids)
        record = {
            "key": k,
            "question": question,
            "grounded": grounded,
            "model": model,
            "chunk_ids": list(chunk_ids) if chunk_ids else [],
            "text": text,
            "tokens": tokens,
            "gen_ms": gen_ms,
            "backend": backend,
            "error": error,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._entries[k] = record
        # Append immediately for crash/timeout tolerance
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return record

    def size(self) -> int:
        return len(self._entries)
