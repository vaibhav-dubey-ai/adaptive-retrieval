"""Shared building blocks: data types, config loading, timing, JSON/JSONL IO and cost accounting.

Everything else in the package imports from here; this module imports nothing from the package.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ======================================================================
# Shared dataclass definitions used across the pipeline.
# ======================================================================

class Decision(str, Enum):
    RETRIEVE = "RETRIEVE"
    DO_NOT_RETRIEVE = "DO_NOT_RETRIEVE"


@dataclass(frozen=True)
class QuestionRecord:
    """One question with its reference answer (evaluation-only fields)."""

    question_id: str
    question: str
    answer: str
    gold_titles: Tuple[str, ...]
    row_type: str = ""
    level: str = ""
    split: str = ""


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    sentences: Tuple[str, ...]
    source_questions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved passage.

    ``method`` records which retriever produced the hit (``bm25`` / ``dense`` /
    ``hybrid``). ``component_ranks``/``component_scores`` are filled by the
    hybrid retriever so per-component behaviour stays inspectable; they are
    diagnostic only and never reach the decision engine or the answer prompt.
    """

    doc_id: str
    title: str
    text: str
    score: float
    method: str = ""
    component_ranks: Dict[str, int] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class GeneratedAnswer:
    text: str
    abstained: bool = False
    evidence_doc_ids: Tuple[str, ...] = ()
    backend: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTimings:
    """Latency of one pipeline stage, in milliseconds."""

    decision_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_method_a_ms: Optional[float] = None
    generation_method_b_ms: Optional[float] = None
    adaptive_e2e_ms: Optional[float] = None
    cross_verify_ms: Optional[float] = None
    evaluation_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "decision_ms": self.decision_ms,
            "retrieval_ms": self.retrieval_ms,
            "generation_method_a_ms": self.generation_method_a_ms,
            "generation_method_b_ms": self.generation_method_b_ms,
            "adaptive_e2e_ms": self.adaptive_e2e_ms,
            "cross_verify_ms": self.cross_verify_ms,
            "evaluation_ms": self.evaluation_ms,
        }


PRIMITIVE_STR = Any
ListLike = List


# ======================================================================
# Configuration loading: one JSON tree, sections per component.
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a config file and deep-merge it over ``configs/default.json``.

    Any section the user config omits stays the documented default, so runs
    are reproducible with minimal config files.
    """
    defaults = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if path is None:
        return defaults
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    user_cfg = json.loads(p.read_text(encoding="utf-8"))
    merged = _deep_merge(defaults, user_cfg)
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path_value: str) -> Path:
    """Resolve a possibly-relative path in the config against the project root."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def data_root() -> Path:
    return PROJECT_ROOT / "data"


# ======================================================================
# Monotonic, high-resolution timing utilities.
# ======================================================================

PLATFORM_RESOLUTION_NS = time.get_clock_info("perf_counter").resolution


def now_ns() -> int:
    """Monotonic, high-resolution clock reading in nanoseconds.

    Uses ``time.perf_counter_ns`` which is guaranteed monotonic
    (non-decreasing) and is distinct from wall-clock time.
    """
    return time.perf_counter_ns()


class Timer:
    """Context manager measuring elapsed time with the monotonic clock."""

    def __init__(self) -> None:
        self._start: Optional[int] = None
        self._elapsed_ns: int = 0

    def __enter__(self) -> "Timer":
        self._start = now_ns()
        return self

    def __exit__(self, *exc) -> None:
        if self._start is not None:
            self._elapsed_ns += now_ns() - self._start
            self._start = None

    def elapsed_ns(self) -> int:
        if self._start is not None:
            return now_ns() - self._start
        return self._elapsed_ns

    def elapsed_ms(self) -> float:
        return elapsed_ms_from_ns(self.elapsed_ns())


def elapsed_ms_from_ns(ns: int) -> float:
    """Convert nanoseconds to milliseconds with sub-ms precision."""
    return ns / 1_000_000.0


# ======================================================================
# Safe run-artifact serialization (JSONL for per-question rows, JSON for
# ======================================================================

def write_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# Cost model for measured/estimated expense.
# ======================================================================

def _policy_uses_retrieval(policy: str, rec: Dict[str, Any]) -> bool:
    """Does THIS policy retrieve on THIS question? Computed per policy —
    retrieval frequency must NEVER be copied across the four policies
    (bug fix: the previous implementation inferred is_b from ``decision``
    for every policy, making never/always/oracle identical)."""
    if policy == "always_retrieve":
        return True
    if policy == "never_retrieve":
        return False
    if policy == "adaptive":
        return rec.get("decision") == "RETRIEVE"
    if policy == "oracle":
        outcome = rec.get("outcome") or {}
        if outcome.get("retrieval_better"):
            return True
        if outcome.get("no_retrieval_better"):
            return False
        # tied / ambiguous: follow the decision engine when available
        return rec.get("decision") == "RETRIEVE"
    # fallback: explicit per-question policy when present
    return rec.get("primary_method") == "B"


def cost_summary(
    records: List[Dict[str, Any]],
    cost_per_retrieval_usd: Optional[float] = None,
    cost_per_llm_call_usd: Optional[float] = None,
) -> Dict[str, Any]:
    n = max(1, len(records))
    # every question costs exactly one model call (closed-book generation still calls the model)
    llm_rate = 1.0 if records else 0.0

    def for_policy(policy: str) -> Dict[str, Any]:
        rate = sum(_policy_uses_retrieval(policy, rec) for rec in records) / n
        return {
            "policy": policy,
            "retrieval_frequency_per_question": round(rate, 4),
            "llm_calls_per_question": llm_rate,
            "estimated_cost_per_question_usd": round(
                rate * (cost_per_retrieval_usd or 0.0)
                + llm_rate * (cost_per_llm_call_usd or 0.0),
                8,
            ),
            "note": (
                "0.0 unless hosted per-call prices are configured; local "
                "deterministic backends have zero API spend."
            ),
        }

    return {p: for_policy(p) for p in ("never_retrieve", "always_retrieve", "adaptive", "oracle")}
