"""Live document QA: answer-vs-answer comparison helpers and ServeFlow (decision -> retrieval -> answer, with timings)."""

from __future__ import annotations

import re
from ..common import Decision
from ..common import now_ns
from ..retrieval.bm25 import tokenize
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
from typing import Any, Dict, List, Optional


# ======================================================================
# Evidence and answer-difference analysis for the comparison dashboard.
# ======================================================================

# Documented thresholds on lexical overlap (share of a sentence's content terms
# that appear in a passage). Status names describe OVERLAP, not entailment:
HIGH_OVERLAP_AT = 0.60
PARTIAL_OVERLAP_AT = 0.35
MIN_TERMS_TO_CHECK = 3

STATUS_HIGH = "high_overlap"
STATUS_PARTIAL = "partial_overlap"
STATUS_NONE = "no_overlap"
STATUS_UNCHECKED = "unchecked"

# Terms too generic to be interesting in a "what did B use that A did not"
# comparison.
_GENERIC = frozenset(
    "also such which would could should their there these those about using used "
    "each other into from with that this have has been were was are is".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    for piece in _SENTENCE_SPLIT.split((text or "").strip()):
        s = piece.strip().lstrip("-*•0123456789. )(").strip()
        if s:
            out.append(s)
    return out


def term_set(text: str) -> set:
    return set(tokenize(text or ""))


def _informative(terms: Iterable[str]) -> List[str]:
    return sorted(
        t for t in set(terms) if len(t) >= 4 and t not in _GENERIC
    )


def sentence_overlap_support(
    answer_text: str,
    passages: Sequence[Dict[str, Any]],
    *,
    high_at: float = HIGH_OVERLAP_AT,
    partial_at: float = PARTIAL_OVERLAP_AT,
) -> List[Dict[str, Any]]:
    """LEXICAL OVERLAP of each answer sentence against the retrieved passages.

    Status meanings (all of them describe shared vocabulary, NOT
    entailment, NOT factual verification):

    - ``high_overlap``   : ≥ ``high_at`` of the sentence's content terms occur
      in the passage (the sentence largely reuses passage vocabulary).
    - ``partial_overlap``: between ``partial_at`` and ``high_at``.
    - ``no_overlap``     : below ``partial_at``.
    - ``unchecked``      : sentence too short to test (fewer than
      MIN_TERMS_TO_CHECK content terms).

    A sentence marked ``high_overlap`` can still be semantically WRONG
    (reversing a passage's meaning reuses its vocabulary), and a well-formed
    paraphrase can score ``no_overlap``. Use this as a pointer back to the
    passage the wording likely came from — nothing more.
    """
    passage_terms = [term_set(p.get("text") or p.get("excerpt") or "") for p in passages]
    rows: List[Dict[str, Any]] = []
    for sentence in split_sentences(answer_text):
        terms = term_set(sentence)
        if len(terms) < MIN_TERMS_TO_CHECK:
            rows.append(
                {
                    "sentence": sentence,
                    "overlap_status": STATUS_UNCHECKED,
                    "overlap": None,
                    "citations": [],
                    "terms_in_evidence": [],
                    "note": "too short to check lexically",
                }
            )
            continue
        scores = []
        for i, pt in enumerate(passage_terms):
            overlap = len(terms & pt) / max(1, len(terms))
            scores.append((overlap, i + 1))
        scores.sort(key=lambda t: (-t[0], t[1]))
        best_overlap, best_rank = scores[0]
        citing = [rank for ov, rank in scores if ov >= high_at][:3]
        if best_overlap >= high_at:
            status = STATUS_HIGH
        elif best_overlap >= partial_at:
            status = STATUS_PARTIAL
        else:
            status = STATUS_NONE
        evidence_terms = sorted(terms & passage_terms[best_rank - 1]) if passage_terms else []
        rows.append(
            {
                "sentence": sentence,
                "overlap_status": status,
                "overlap": round(best_overlap, 3),
                "best_passage": best_rank,
                "citations": [f"[{r}]" for r in citing],
                "terms_in_evidence": _informative(evidence_terms)[:10],
            }
        )
    return rows


def sentence_support(answer_text, passages, **kw):
    """Backwards-compatible alias for :func:`sentence_overlap_support`."""
    return sentence_overlap_support(answer_text, passages, **kw)


def assign_citations(passages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy passages and attach 1-based citation labels + HTML anchors."""
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(passages, start=1):
        entry = dict(p)
        entry["rank"] = i
        entry["citation"] = f"[{i}]"
        entry["anchor"] = f"passage-{i}"
        out.append(entry)
    return out


def compare_answers(
    answer_a: str,
    answer_b: str,
    passages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Lexical-overlap map for B plus a passage-by-passage answer difference summary."""
    cited = assign_citations(passages)
    b_support = sentence_overlap_support(answer_b, cited)
    a_support = sentence_overlap_support(answer_a, cited)

    a_terms = term_set(answer_a)
    b_terms = term_set(answer_b)
    shared = a_terms & b_terms

    per_passage: List[Dict[str, Any]] = []
    for p in cited:
        p_terms = term_set(p.get("text") or "")
        in_b = b_terms & p_terms
        in_a = a_terms & p_terms
        per_passage.append(
            {
                "rank": p["rank"],
                "citation": p["citation"],
                "anchor": p["anchor"],
                "source_file": p.get("source_file"),
                "page": p.get("page"),
                "terms_b_not_a": _informative(in_b - in_a)[:8],
                "terms_a_not_b": _informative(in_a - in_b)[:8],
                "coverage_a": round(len(in_a) / max(1, len(a_terms)), 3),
                "coverage_b": round(len(in_b) / max(1, len(b_terms)), 3),
            }
        )

    def _count(rows, status):
        return sum(1 for r in rows if r["overlap_status"] == status)

    return {
        "passages": cited,
        "b_sentences": b_support,
        "a_sentences": a_support,
        "per_passage": per_passage,
        "answer_diff": {
            "only_in_a": _informative(a_terms - b_terms)[:15],
            "only_in_b": _informative(b_terms - a_terms)[:15],
            "shared_terms": len(shared),
            "jaccard": round(len(shared) / max(1, len(a_terms | b_terms)), 3),
        },
        "summary": {
            "n_passages": len(cited),
            "b_high_overlap": _count(b_support, STATUS_HIGH),
            "b_partial_overlap": _count(b_support, STATUS_PARTIAL),
            "b_no_overlap": _count(b_support, STATUS_NONE),
            "b_unchecked": _count(b_support, STATUS_UNCHECKED),
            "a_high_overlap": _count(a_support, STATUS_HIGH),
            "a_no_overlap": _count(a_support, STATUS_NONE),
        },
        "method": "lexical_overlap_only",
        "status_names": {
            STATUS_HIGH: "≥ HIGH_OVERLAP_AT of the sentence's content terms occur in the passage",
            STATUS_PARTIAL: "between PARTIAL_OVERLAP_AT and HIGH_OVERLAP_AT",
            STATUS_NONE: "below PARTIAL_OVERLAP_AT",
            STATUS_UNCHECKED: "sentence too short to test",
        },
        "method_note": (
            "This map measures LEXICAL OVERLAP ONLY: the share of a sentence's "
            "content terms that occur in a retrieved passage. It is NOT "
            "entailment and NOT a correctness check — a sentence can reuse a "
            "passage's vocabulary while contradicting it, and a factual "
            "paraphrase can score no_overlap. Treat these statuses as shared- "
            "vocabulary labels, never as 'the passage proves this sentence'."
        ),
        "thresholds": {
            "high_overlap_at": HIGH_OVERLAP_AT,
            "partial_overlap_at": PARTIAL_OVERLAP_AT,
        },
    }


def citation_map(passages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{"[1]": {anchor, source_file, page, ...}}`` for frontend validation."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in assign_citations(passages):
        out[p["citation"]] = {
            "anchor": p["anchor"],
            "source_file": p.get("source_file"),
            "page": p.get("page"),
            "score": p.get("score"),
        }
    return out


def invalid_citations(answer_text: str, passages: Sequence[Dict[str, Any]]) -> List[str]:
    """Citations in the answer that do not map to a supplied passage ([] if ok).

    Used by tests and the dashboard to catch a model citing ``[9]`` when only
    five passages were given.
    """
    supplied = {f"[{i}]" for i in range(1, len(passages) + 1)}
    found = set(re.findall(r"\[(\d+)\]", answer_text or ""))
    return sorted(f"[{f}]" for f in found if f"[{f}]" not in supplied)


# ======================================================================
# Interactive answering flow for the research dashboard.
# ======================================================================

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
    retrieved_context_chars: Optional[int] = None
    generation_cold: Optional[bool] = None  # first call of this backend instance

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
            "n_passages": len(self.retrieval_hits) if self.label == "B" else 0,
            "retrieved_context_chars": self.retrieved_context_chars,
            # Flat copy from answer.meta so the UI can badge cold/warm without
            # digging into the nested meta dict.
            "generation_cold": (self.answer.meta or {}).get("generation_cold") if self.answer else None,
        }
        if self.answer is not None:
            out["text"] = self.answer.text
            out["abstained"] = self.answer.abstained
            out["backend"] = self.answer.backend
            out["meta"] = self.answer.meta
        return out


def _hit_entry(h: Any, i: int, retriever: Any = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "rank": i + 1,
        "doc_id": h.doc_id,
        "score": round(float(h.score), 4),
        "excerpt": h.text[:400],
        # Full passage text: the dashboard can expand a passage and the
        # citation/support analysis must see exactly what the model received.
        "text": h.text,
        "title": h.title,
    }
    # Source provenance (upload filename + PDF page) lives in the document
    # store, not on the frozen RetrievalHit. Join it when the retriever exposes
    # source_meta(); otherwise fall back to the hit's title.
    source = getattr(h, "source_file", None)
    page = getattr(h, "page", None)
    if retriever is not None and hasattr(retriever, "source_meta"):
        try:
            sm = retriever.source_meta(h) or {}
            source = sm.get("filename", source)
            page = sm.get("page", page)
        except Exception:
            pass
    if source is not None:
        entry["source_file"] = source
        entry["page"] = page if page is not None else None
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
        collection_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.decision_engine = decision_engine
        self.method_a = method_a
        self.method_b = method_b
        self.benchmark_retriever = benchmark_retriever
        self.collection_retriever = collection_retriever
        self.top_k = int(top_k)
        self.collection_info = collection_info or {}

    # ------------------------------------------------------------- decision
    def decide_first(self, question: str) -> Dict[str, Any]:
        """Run the decision engine (question only) and report its FULL trace.

        ``decision_ms`` is the wall clock of the whole decision step, which for
        the heuristic engine includes the documented collection preflight;
        ``decision_engine_ms`` and ``preflight_ms`` split that cost so neither
        is hidden.
        """
        t0 = now_ns()
        decision, score, meta = self.decision_engine.decide(question)
        decision_ms = _ms(now_ns() - t0)
        preflight = meta.get("preflight") or {"available": False}
        preflight_ms = preflight.get("ms") if isinstance(preflight, dict) else None
        engine_ms = (
            round(max(0.0, decision_ms - float(preflight_ms)), 3)
            if isinstance(preflight_ms, (int, float))
            else decision_ms
        )
        return {
            "decision": decision.value,
            "engine_name": getattr(self.decision_engine, "name", None),
            "decision_score": score,
            "decision_features": meta,
            "decision_ms": decision_ms,
            "decision_engine_ms": engine_ms,
            "preflight_ms": preflight_ms,
            "threshold": meta.get("threshold"),
            "decision_rule": meta.get("decision_rule"),
            "score_definition": meta.get("score_definition"),
            "rationale": meta.get("rationale"),
            "signals": meta.get("components") or {},
            "preflight": preflight,
            "inputs": (
                "question text + optional model-memory signal + aggregate collection "
                "relevance preflight; NEVER reference answers, gold chunk ids, "
                "oracle labels or retrieved passages"
            ),
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
            generation_cold=(answer.meta or {}).get("generation_cold"),
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
                retrieval_hits=[_hit_entry(h, i, retriever) for i, h in enumerate(hits)],
                retrieval_method=method,
                retrieval_diagnostics=diagnostics,
            )
        return BranchResult(
            label="B",
            ok=True,
            answer=answer,
            generation_ms=_ms(now_ns() - t1),
            retrieval_ms=retrieval_ms,
            retrieval_hits=[_hit_entry(h, i, retriever) for i, h in enumerate(hits)],
            retrieval_method=method,
            retrieval_diagnostics=diagnostics,
            tokens=(answer.meta or {}).get("tokens"),
            retrieved_context_chars=sum(len(h.text) for h in hits),
            generation_cold=(answer.meta or {}).get("generation_cold"),
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

        total_ms = _ms(b_end - t_a0)
        selected = _decision_branch(d["decision"])
        analysis = None
        if a.ok and b.ok:
            try:
                analysis = compare_answers(
                    a.answer.text if a.answer else "",
                    b.answer.text if b.answer else "",
                    b.retrieval_hits,
                )
            except Exception as exc:  # analysis must never break answering
                analysis = {"error": _err_str(exc), "method": "lexical_overlap_only"}
        selected_path_ms = round(
            d["decision_ms"]
            + ((a.path_total_ms or 0.0) if selected == "A" else (b.path_total_ms or 0.0)),
            3,
        )
        return {
            "mode": "compare",
            "question_id": question_id,
            "question": question,
            "decision": d,
            "selected_branch": selected,
            "branch_a": a.to_dict(),
            "branch_b": b.to_dict(),
            "comparison": _compare_metrics(a, b),
            "comparison_total_ms": total_ms,
            "analysis": analysis,
            "collection": self.collection_info,
            "top_k": self.top_k,
            "timings": {
                "preflight_ms": d.get("preflight_ms"),
                "decision_engine_ms": d.get("decision_engine_ms"),
                "decision_ms": d["decision_ms"],
                "retrieval_ms": b.retrieval_ms if b.ok else None,
                "generation_a_ms": a.generation_ms if a.ok else None,
                "generation_b_ms": b.generation_ms if b.ok else None,
                "path_a_total_ms": a.path_total_ms if a.ok else None,
                "path_b_total_ms": b.path_total_ms if b.ok else None,
                "comparison_total_ms": total_ms,
                "selected_path_projected_ms": selected_path_ms,
                "units": "milliseconds",
                "note": (
                    "selected_path_projected_ms = measured decision + the measured "
                    "selected branch of THIS run; it is an estimate of deployed "
                    "latency, distinct from the measured adaptive-only mode."
                ),
            },
            "note": (
                "Both branches executed for research comparison; the total is the "
                "time of the full comparison request, NOT deployed adaptive latency."
            ),
        }

    def run_adaptive_only(self, question: str, question_id: str = "") -> Dict[str, Any]:
        """Deployed workflow: decision, then ONLY the selected path executes.

        ``adaptive_total_ms`` is measured from BEFORE the decision to the end of
        the selected branch, i.e. decision (incl. preflight) + retrieval (B
        only) + generation. That is the real operational latency of a deployed
        request; it is never merged with the comparison run's wall clock.
        """
        t_all = now_ns()
        d = self.decide_first(question)

        if d["decision"] == Decision.RETRIEVE.value:
            b = self._branch_b(question)
            total_ms = _ms(now_ns() - t_all)
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
                "timings": {
                    "preflight_ms": d.get("preflight_ms"),
                    "decision_engine_ms": d.get("decision_engine_ms"),
                    "decision_ms": d["decision_ms"],
                    "retrieval_ms": b.retrieval_ms,
                    "generation_ms": b.generation_ms,
                    "adaptive_total_ms": total_ms,
                    "units": "milliseconds",
                },
                "collection": self.collection_info,
                "top_k": self.top_k,
                "note": "Only the selected path executed — real deployed latency.",
            }
        a = self._branch_a(question)
        total_ms = _ms(now_ns() - t_all)
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
            "timings": {
                "preflight_ms": d.get("preflight_ms"),
                "decision_engine_ms": d.get("decision_engine_ms"),
                "decision_ms": d["decision_ms"],
                "retrieval_ms": None,
                "generation_ms": a.generation_ms,
                "adaptive_total_ms": total_ms,
                "units": "milliseconds",
            },
            "collection": self.collection_info,
            "top_k": self.top_k,
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
    cold_a = a.generation_cold if a.ok else None
    cold_b = b.generation_cold if b.ok else None

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
        "generation_cold_a": cold_a,
        "generation_cold_b": cold_b,
        # Fairness: A always runs first in the comparison, so its generation
        # may include the one-off model load/warm-up while B's already-warm
        # call does not pay it. A single sequential run therefore cannot be
        # read as "retrieval is faster/slower" — use the interleaved medians
        # from scripts/bench_latency.py for that claim.
        "fairness_note": (
            "Single sequential comparison: Option A ran first and may carry "
            "first-call model-load cost (marked below); differences from ONE "
            "run are not evidence that either branch is faster. See "
            "scripts/bench_latency.py for interleaved trials with medians."
        ),
        "units": {"time": "milliseconds", "tokens": "tokens"},
    }
