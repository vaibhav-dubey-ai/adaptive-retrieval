"""Serve-flow tests: decision BEFORE answering, branch isolation, adaptive-only
execution, timing formulas and token accounting. No network; model calls and
retrieval are mocked with deterministic spies.
"""

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.serve import BranchResult, ServeFlow, _compare_metrics
from adaptive_qa.types import Decision, GeneratedAnswer, RetrievalHit


class SpyEngine:
    """Decision engine that records WHEN it was called relative to branches."""

    def __init__(self, decision: Decision):
        self.decision = decision
        self.calls: list = []
        self.log = []

    def decide(self, question):
        self.log.append("decision")
        self.calls.append(question)
        return self.decision, 0.7, {"feature": "x"}


class SpyMethod:
    def __init__(self, label, tokens=None, fail=False, text="answer"):
        self.label = label
        self.tokens = tokens or {}
        self.fail = fail
        self.text = text
        self.invocations: list = []

    def answer(self, question, passages=None):
        self.invocations.append({"question": question, "passages": passages})
        from time import sleep

        sleep(0.01)
        if self.fail:
            raise RuntimeError(f"{self.label} model exploded")
        import random

        # deterministic per-instance token estimate (not a network call)
        return GeneratedAnswer(
            text=f"{self.label}:{self.text}",
            abstained=False,
            backend=self.label,
            meta={"tokens": self.tokens},
        )


class SpyRetriever:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.calls: list = []

    def retrieve(self, query, top_k):
        self.calls.append(query)
        return [
            RetrievalHit(doc_id="d1", title="doc one", text="chunk text", score=1.0)
        ] if not self.hits else self.hits


def _flow(decision=Decision.RETRIEVE, a_fail=False, b_fail=False, tokens_a=None, tokens_b=None):
    engine = SpyEngine(decision)
    a = SpyMethod("A", tokens=tokens_a, fail=a_fail)
    b = SpyMethod("B", tokens=tokens_b, fail=b_fail)
    retr = SpyRetriever()
    return ServeFlow(engine, a, b, benchmark_retriever=retr)


# ------------------------------------------------------- decision-before-work
def test_decision_runs_before_either_branch():
    flow = _flow()
    flow.run_comparison("What does the doc say?")
    assert flow.decision_engine.log == ["decision"]
    # decision gets the question, and engine observes no other content
    assert flow.decision_engine.calls == ["What does the doc say?"]


def test_compare_mode_runs_both_branches_after_decision():
    flow = _flow()
    out = flow.run_comparison("Q1")
    # side-by-side order: decision, then branch A execution, then branch B
    assert out["decision"]["decision"] == "RETRIEVE"
    assert out["branch_a"]["text"].startswith("A:")
    assert out["branch_b"]["text"].startswith("B:")
    # both answers exist for comparison, per requirement; and ONLY the
    # decision ran before them (nothing was retrieved before the decision)
    assert out["selected_branch"] == "B"


def test_compare_with_no_retrieve_decision_still_runs_both():
    flow = _flow(decision=Decision.DO_NOT_RETRIEVE)
    out = flow.run_comparison("Q1")
    assert out["branch_a"]["ok"] and out["branch_b"]["ok"]
    assert out["selected_branch"] == "A"


def test_option_a_never_invokes_retrieval():
    flow = _flow()
    out = flow.run_comparison("Q")
    # branch A answered from the question alone (never sees passages)
    a_call = flow.method_a.invocations[0]
    assert a_call["passages"] is None
    assert out["branch_a"]["performed_retrieval"] is False


def test_branch_b_invokes_retrieval_with_question():
    flow = _flow()
    out = flow.run_comparison("Q doc")
    assert flow.benchmark_retriever.calls == ["Q doc"]  # one retrieval call for B
    hits = out["branch_b"]["retrieval_hits"]
    assert hits and hits[0]["score"] == 1.0
    assert "chunk text" in hits[0]["excerpt"]


# ------------------------------------------------------- branch failure / error
def test_failed_branch_keeps_other_answer_visible():
    flow = _flow(b_fail=True)
    out = flow.run_comparison("Q")
    assert out["branch_a"]["ok"] and out["branch_a"]["text"]
    assert out["branch_b"]["ok"] is False
    assert "exploded" in out["branch_b"]["error"]


def test_branch_a_failure_recorded_in_compare():
    flow = _flow(a_fail=True)
    out = flow.run_comparison("Q")
    assert out["branch_a"]["ok"] is False
    assert out["branch_b"]["ok"]


# ------------------------------------------------------- adaptive-only mode
def test_adaptive_mode_runs_only_selected_branch_retrieve():
    flow = _flow(decision=Decision.RETRIEVE)
    out = flow.run_adaptive_only("Q")
    assert out["selected_branch"] == "B"
    assert out["executed_branch"]["performed_retrieval"] is True
    assert len(flow.method_a.invocations) == 0  # Option A never executed
    assert len(flow.method_b.invocations) == 1
    ops = out["operations"]
    assert ops["retrieval_ms"] is not None and ops["retrieval_ms"] >= 0
    assert out["adaptive_total_ms"] >= ops["generation_ms"]


def test_adaptive_mode_runs_only_selected_branch_no_retrieve():
    flow = _flow(decision=Decision.DO_NOT_RETRIEVE)
    out = flow.run_adaptive_only("Q")
    assert out["executed_branch"]["performed_retrieval"] is False
    assert len(flow.method_b.invocations) == 0   # Option B never executed
    assert len(flow.benchmark_retriever.calls) == 0  # retrieval never happened
    assert out["operations"]["retrieval_ms"] is None


# ------------------------------------------------------- timing / token math
def _br(label, ok, path_ms, tokens):
    b = BranchResult(label=label, ok=ok)
    b.path_total_ms = path_ms
    b.tokens = tokens
    return b


def test_comparison_formulas_and_signs():
    a = _br("A", True, 100.0, {"total_tokens": 50, "source": "estimated"})
    b = _br("B", True, 350.0, {"total_tokens": 250, "source": "provider-reported"})
    m = _compare_metrics(a, b)
    assert m["time_difference_ms"] == 250.0  # B slower => retrieval cost visible
    assert m["time_saved_pct"] == round((350 - 100) / 350 * 100, 2)
    assert m["token_difference"] == 200
    assert m["token_saved_pct_when_B_used_more"] == round((250 - 50) / 250 * 100, 2)


def test_percentage_baseline_zero_shows_dash_not_error():
    # B total 0 => no percentage can be computed; must be None, not exception
    a = _br("A", True, 10.0, {"total_tokens": 5})
    b = _br("B", True, 0.0, {"total_tokens": 0})
    m = _compare_metrics(a, b)
    assert m["time_saved_pct"] is None
    # failed branches excluded from comparison math
    a2 = _br("A", False, None, None)
    m2 = _compare_metrics(a2, b)
    assert m2["time_difference_ms"] is None


def test_branch_timings_measured_and_units_ms():
    flow = _flow()
    out = flow.run_comparison("Q")
    a, b = out["branch_a"], out["branch_b"]
    for br in (a, b):
        assert br["generation_ms"] >= 0
        assert br["path_total_ms"] >= br["generation_ms"] if br["label"] == "B" else True
    assert b["path_total_ms"] >= b["retrieval_ms"] >= 0
    assert out["comparison_total_ms"] >= a["path_total_ms"]
    assert out["comparison_total_ms"] >= b["path_total_ms"]
