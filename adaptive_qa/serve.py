"""Interactive answering flow for the research dashboard.

Hard sequencing contract (requirement §2):

  1. The adaptive decision engine runs FIRST on the question alone. Neither
     path has executed yet and nothing retrieved exists.
  2. AFTER the decision (and its reasons/features/duration) is recorded, both
     answering paths run INDEPENDENTLY for research comparison. Path A never
     sees the document collection; Path B searches ONLY the selected
     collection.
  3. A SEPARATE method, ``run_adaptive_only``, executes the deployed
     workflow: decision then ONLY the selected path. Its wall-clock is real
     measured operational latency; it must never be conflated with the
     comparison run's total time.

Timing uses ``timeutil.now_ns`` (monotonic high-resolution). Token usage is
provider-reported when the model supplies it, otherwise estimated and
labeled; decision latency of the heuristic/learned engine is real measured
time but carries NO invented LLM tokens (it does not call the model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .timeutil import now_ns
from .types import Decision


def _ms(ns: int) -> float:
    return round(ns / 1_000_000.0, 3)


@dataclass
class BranchResult:
    label: str                     # "A" or "B"
    ok: bool = True
    error: Optional[str] = None
    answer: Any = None             # GeneratedAnswer
    generation_ms: Optional[float] = None
    path_total_ms: Optional[float] = None      # generation (+ retrieval for B)
    retrieval_ms: Optional[float] = None       # Option B only; None for A
    retrieval_hits: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_method: Optional[str] = None     # bm25 | dense | hybrid (B only)
    retrieval_diagnostics: Optional[Dict[str, Any]] = None
    tokens: Optional[Dict[str, Any]] = None    # {in,out,total,source}

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "label": self.label,
            "ok": self.ok,
            "error": self.error,
            "generation_ms": self.generation_ms,
            "path_total_ms": self.path_total_ms,
            "retrieval_ms": self.retrieval_ms,
            "retrieval_hits": self.retrieval_hits,
            "retrieval_method": self.retrieval_method,
            "retrieval_diagnostics": self.retrieval_diagnostics,
            "tokens": self.tokens,
            "performed_retrieval": self.label == "B",
        }
        if self.answer is not None:
            out["text"] = self.answer.text
            out["abstained"] = self.answer.abstained
            out["backend"] = self.answer.backend
            out["meta"] = self.answer.meta
        return out


def _hit_entry(h: Any, i: int) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "rank": i + 1,
        "doc_id": h.doc_id,
        "score": round(float(h.score), 4),
        "excerpt": h.text[:400],
    }
    src = getattr(h, "source_file", None)
    if src is not None:
        entry["source_file"] = src
        page = getattr(h, "page", None)
        entry["page"] = page if page is not None else None
    else:
        entry["title"] = h.title
    method = getattr(h, "method", "")
    if method:
        entry["retrieval_method"] = method
    ranks = getattr(h, "component_ranks", None)
    if ranks:
        entry["component_ranks"] = dict(ranks)
    return entry


def _err_str(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _decision_branch(decision: str) -> str:
    return "B" if decision == Decision.RETRIEVE.value else "A"


class ServeFlow:
    """Decision-first execution with per-branch isolation and error capture.

    A failing branch records its error and keeps the other branch's result
    visible; failing the whole request would violate that requirement.
    """

    def __init__(
        self,
        decision_engine,
        method_a,
        method_b,
        benchmark_retriever=None,
        collection_retriever=None,
        top_k: int = 8,
    ) -> None:
        self.decision_engine = decision_engine
        self.method_a = method_a
        self.method_b = method_b
        self.benchmark_retriever = benchmark_retriever
        self.collection_retriever = collection_retriever
        self.top_k = int(top_k)

    # ------------------------------------------------------------- decision
    def decide_first(self, question: str) -> Dict[str, Any]:
        t0 = now_ns()
        decision, score, meta = self.decision_engine.decide(question)
        decision_ms = _ms(now_ns() - t0)
        return {
            "decision": decision.value,
            "decision_score": score,
            "decision_features": meta,
            "decision_ms": decision_ms,
        }

    # ------------------------------------------------------------ branches
    def _branch_a(self, question: str) -> BranchResult:
        t0 = now_ns()
        try:
            answer = self.method_a.answer(question)  # closed-book: question only
        except Exception as exc:  # branch-failure isolation
            return BranchResult(label="A", ok=False, error=_err_str(exc))
        info = BranchResult(
            label="A",
            ok=True,
            answer=answer,
            generation_ms=_ms(now_ns() - t0),
            tokens=(answer.meta or {}).get("tokens"),
        )
        return info

    def _branch_b(self, question: str) -> BranchResult:
        retriever = self.collection_retriever or self.benchmark_retriever
        if retriever is None:
            return BranchResult(label="B", ok=False, error="No retriever available for Option B.")
        method = getattr(retriever, "method", None)
        t0 = now_ns()
        try:
            hits = retriever.retrieve(question, self.top_k)
        except Exception as exc:
            return BranchResult(label="B", ok=False, error=_err_str(exc),
                                retrieval_ms=_ms(now_ns() - t0))
        retrieval_ms = _ms(now_ns() - t0)
        # Collected AFTER the query so lazy model-load and cache statistics
        # describe the call that actually happened.
        diagnostics = None
        if hasattr(retriever, "diagnostics"):
            try:
                diagnostics = retriever.diagnostics()
            except Exception:  # diagnostics must never break answering
                diagnostics = None
        t1 = now_ns()
        try:
            answer = self.method_b.answer(question, hits)
        except Exception as exc:
            return BranchResult(
                label="B", ok=False, error=_err_str(exc),
                retrieval_ms=retrieval_ms,
                retrieval_hits=[_hit_entry(h, i) for i, h in enumerate(hits)],
                retrieval_method=method,
                retrieval_diagnostics=diagnostics,
            )
        return BranchResult(
            label="B",
            ok=True,
            answer=answer,
            generation_ms=_ms(now_ns() - t1),
            retrieval_ms=retrieval_ms,
            retrieval_hits=[_hit_entry(h, i) for i, h in enumerate(hits)],
            retrieval_method=method,
            retrieval_diagnostics=diagnostics,
            tokens=(answer.meta or {}).get("tokens"),
        )

    # --------------------------------------------------------------- flows
    def run_comparison(self, question: str, question_id: str = "") -> Dict[str, Any]:
        """Decision FIRST, then both paths. Per-branch end-to-end latencies
        (t_branch_start -> t_branch_end) are measured independently; the total
        comparison wall-clock is reported separately and is NOT the latency of
        a deployed adaptive request. When the collection retriever is absent,
        Option B searches the benchmark corpus."""
        d = self.decide_first(question)

        t_a0 = now_ns()
        a = self._branch_a(question)
        a_end = now_ns()
        a.path_total_ms = _ms(a_end - t_a0)

        t_b0 = now_ns()
        b = self._branch_b(question)
        b_end = now_ns()
        if b.ok or b.path_total_ms is None:
            b.path_total_ms = _ms(b_end - t_b0)

        return {
            "mode": "compare",
            "question_id": question_id,
            "question": question,
            "decision": d,
            "selected_branch": _decision_branch(d["decision"]),
            "branch_a": a.to_dict(),
            "branch_b": b.to_dict(),
            "comparison": _compare_metrics(a, b),
            "comparison_total_ms": _ms(b_end - t_a0),
            "note": (
                "Both branches executed for research comparison; the total is the "
                "time of the full comparison request, NOT deployed adaptive latency."
            ),
        }

    def run_adaptive_only(self, question: str, question_id: str = "") -> Dict[str, Any]:
        """Deployed workflow: decision, then ONLY the selected path executes.
        The wall-clock returned here (decision start -> branch end) is real
        operational latency: decision + retrieval (B only) + generation."""
        d = self.decide_first(question)
        t0 = now_ns()

        if d["decision"] == Decision.RETRIEVE.value:
            b = self._branch_b(question)
            total_ms = _ms(now_ns() - t0)
            return {
                "mode": "adaptive",
                "question_id": question_id,
                "question": question,
                "decision": d,
                "selected_branch": "B",
                "executed_branch": b.to_dict(),
                "operations": {
                    "decision_ms": d["decision_ms"],
                    "retrieval_ms": b.retrieval_ms if b.ok else None,
                    "generation_ms": b.generation_ms if b.ok else None,
                },
                "adaptive_total_ms": total_ms,
                "token_usage": {"decision_tokens": None, "selected_branch": b.tokens},
                "note": "Only the selected path executed — real deployed latency.",
            }
        a = self._branch_a(question)
        total_ms = _ms(now_ns() - t0)
        return {
            "mode": "adaptive",
            "question_id": question_id,
            "question": question,
            "decision": d,
            "selected_branch": "A",
            "executed_branch": a.to_dict(),
            "operations": {
                "decision_ms": d["decision_ms"],
                "retrieval_ms": None,  # retrieval never ran on this path
                "generation_ms": a.generation_ms if a.ok else None,
            },
            "adaptive_total_ms": total_ms,
            "token_usage": {"decision_tokens": None, "selected_branch": a.tokens},
            "note": "Only the selected path executed — real deployed latency (no retrieval performed).",
        }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 3)


def _pct_saving(base: float, other: float) -> Optional[float]:
    if not base:
        return None
    return round((base - other) / base * 100.0, 2)


def _compare_metrics(a: BranchResult, b: BranchResult) -> Dict[str, Any]:
    """Comparison arithmetic (requirement §8).

    time difference = Option B total latency − Option A total latency
      (positive => the no-retrieval path was FASTER; negative => increase).
    token difference = Option B total tokens − Option A total tokens
      (positive => no-retrieval used FEWER tokens). Percentages are only
      produced when the baseline is nonzero (otherwise display '—').
    """
    a_t = a.path_total_ms if a.ok else None
    b_t = b.path_total_ms if b.ok else None
    a_tok = (a.tokens or {}).get("total_tokens") if a.ok else None
    b_tok = (b.tokens or {}).get("total_tokens") if b.ok else None

    def pct(base: Optional[float], other: Optional[float]) -> Optional[float]:
        if not base:
            return None
        if other is None:
            return None
        return round((base - other) / base * 100.0, 2)

    return {
        "formula_time": "time difference = Option B total latency − Option A total latency",
        "time_difference_ms": _delta(a_t, b_t),
        "time_saved_pct": _pct_saving(b_t, a_t) if (a_t is not None and b_t is not None) else None,
        "token_formula": "token difference = Option B total tokens − Option A total tokens",
        "token_difference": (b_tok - a_tok) if (a_tok is not None and b_tok is not None) else None,
        "tokens_a": a_tok,
        "tokens_b": b_tok,
        "token_saved_pct_when_B_used_more": pct(b_tok, a_tok) if b_tok else None,
        "retrieval_ms_brings_only_B": b.retrieval_ms if b.ok else None,
        "units": {"time": "milliseconds", "tokens": "tokens"},
    }
