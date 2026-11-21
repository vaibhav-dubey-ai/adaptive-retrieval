"""Serve flow: decision trace pass-through, citations, timings, failure isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.live.serve import ServeFlow  # noqa: E402
from adaptive_qa.common import Decision, GeneratedAnswer, RetrievalHit  # noqa: E402


class Engine:
    def __init__(self, decision=Decision.RETRIEVE, preflight_ms=2.0):
        self.decision = decision
        self.preflight_ms = preflight_ms

    def decide(self, question):
        import time

        time.sleep(0.006)  # a real decision takes measurable time
        meta = {
            "engine": "heuristic",
            "threshold": 0.5,
            "decision_rule": "RETRIEVE if score >= threshold, else DO_NOT_RETRIEVE",
            "score_definition": "weighted mean of available signals (retrieval-need score)",
            "rationale": "RETRIEVE: score 0.72 >= threshold 0.50",
            "memory_available": False,
            "components": {
                "collection_relevance": {
                    "value": 0.8, "weight": 0.7, "contribution": 0.62,
                    "available": True, "explanation": "document relevance 0.80",
                },
                "question_cue": {
                    "value": 0.1, "weight": 0.15, "contribution": 0.02,
                    "available": True, "explanation": "open WH question",
                },
                "memory_uncertainty": {
                    "value": None, "weight": 0.2, "contribution": 0.0,
                    "available": False, "explanation": "no familiarity probe",
                },
            },
            "preflight": {
                "available": True,
                "ms": self.preflight_ms,
                "cold": False,
                "signals": {"collection_relevance": 0.8, "corpus_coverage": 0.9},
                "collection": {"n_chunks": 80, "top_k_probed": 5},
            },
        }
        return self.decision, 0.72, meta


class MethodA:
    def answer(self, question):
        return GeneratedAnswer(
            text="Composite attributes are structured values.",
            backend="llm_closedbook",
            meta={"model": "qwen3:4b-instruct",
                  "tokens": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50,
                             "source": "provider-reported"}},
        )


class MethodB:
    def __init__(self, fail=False):
        self.fail = fail

    def answer(self, question, passages):
        assert passages, "grounded branch must receive passages"
        if self.fail:
            raise RuntimeError("Provider returned an empty response")
        return GeneratedAnswer(
            text="Composite attributes are divided into component attributes. "
                 "Diamonds represent relationship sets.",
            backend="llm_grounded",
            meta={"model": "qwen3:4b-instruct",
                  "tokens": {"input_tokens": 400, "output_tokens": 40, "total_tokens": 440,
                             "source": "provider-reported"}},
        )


class Retriever:
    method = "bm25"

    def retrieve(self, query, top_k):
        return [
            RetrievalHit("udoc_x:0017", "ch6.pdf (page 17)", 
                         "Composite attributes allow attributes to be divided into subparts "
                         "called component attributes.", 3.2, method="bm25"),
            RetrievalHit("udoc_x:0010", "ch6.pdf (page 19)",
                         "Diamonds represent relationship sets in an ER diagram.", 2.7,
                         method="bm25"),
        ][:top_k]

    def diagnostics(self):
        return {"retriever": "bm25", "n_chunks": 80}

    def source_meta(self, hit):
        page = 17 if hit.doc_id.endswith("0017") else 19
        return {"filename": "ch6.pdf", "page": page, "doc_id": hit.doc_id.split(":")[0]}


def _flow(fail_b=False, decision=Decision.RETRIEVE):
    return ServeFlow(
        Engine(decision),
        MethodA(),
        MethodB(fail=fail_b),
        collection_retriever=Retriever(),
        top_k=2,
        collection_info={"n_chunks": 80, "retrieval_method": "bm25", "top_k": 2},
    )


def test_decision_reports_engine_and_preflight_time_separately():
    out = _flow().run_comparison("What are composite attributes?")
    d = out["decision"]
    assert d["decision"] == "RETRIEVE"
    assert d["preflight_ms"] == 2.0
    assert d["decision_engine_ms"] == pytest.approx(d["decision_ms"] - 2.0, abs=0.02)
    assert d["decision_engine_ms"] > 0
    assert d["threshold"] == 0.5
    assert "retrieval-need score" in d["score_definition"]
    assert "score 0.72" in d["rationale"]
    assert d["signals"]["memory_uncertainty"]["available"] is False
    assert "reference answers" in d["inputs"]


def test_passages_carry_source_page_and_full_text():
    out = _flow().run_comparison("q")
    hits = out["branch_b"]["retrieval_hits"]
    assert [h["source_file"] for h in hits] == ["ch6.pdf", "ch6.pdf"]
    assert [h["page"] for h in hits] == [17, 19]
    assert "component attributes" in hits[0]["text"]
    assert out["branch_b"]["n_passages"] == 2
    assert out["branch_b"]["retrieved_context_chars"] > 0
    assert out["top_k"] == 2
    assert out["collection"]["n_chunks"] == 80


def test_analysis_maps_citations_to_the_right_passages():
    out = _flow().run_comparison("q")
    an = out["analysis"]
    assert an["method"] == "lexical_overlap_only"
    assert [p["citation"] for p in an["passages"]] == ["[1]", "[2]"]
    assert an["passages"][1]["page"] == 19
    supported = [s for s in an["b_sentences"] if s["overlap_status"] == "high_overlap"]
    assert supported, an["b_sentences"]
    citation = supported[0]["citations"][0]
    rank = int(citation.strip("[]"))
    assert an["passages"][rank - 1]["anchor"] == f"passage-{rank}"
    # the diamond sentence must cite the passage that actually contains "diamonds"
    diamond = next(s for s in an["b_sentences"] if "diamonds" in s["sentence"].lower())
    assert diamond["citations"] == ["[2]"]


def test_timings_block_separates_retrieval_generation_and_decision():
    out = _flow().run_comparison("q")
    t = out["timings"]
    assert t["units"] == "milliseconds"
    assert t["path_a_total_ms"] >= t["generation_a_ms"]
    assert t["path_b_total_ms"] >= (t["retrieval_ms"] or 0) + (t["generation_b_ms"] or 0)
    assert t["selected_path_projected_ms"] >= (t["decision_ms"] or 0)
    assert out["comparison"]["token_difference"] == 440 - 50


def test_failed_branch_keeps_answer_and_reports_incomplete_comparison():
    out = _flow(fail_b=True).run_comparison("q")
    assert out["branch_a"]["ok"] is True and out["branch_a"]["text"]
    assert out["branch_b"]["ok"] is False
    assert "empty response" in out["branch_b"]["error"]
    assert out["analysis"] is None  # no comparison analysis for a failed branch
    assert out["comparison"]["tokens_b"] is None
    assert out["comparison"]["time_difference_ms"] is None


def test_adaptive_mode_reports_measured_breakdown_without_other_branch():
    out = _flow().run_adaptive_only("q")
    assert out["selected_branch"] == "B"
    t = out["timings"]
    assert t["preflight_ms"] == 2.0
    assert t["retrieval_ms"] is not None
    assert t["generation_ms"] is not None
    assert out["adaptive_total_ms"] >= (t["decision_ms"] or 0)
    assert out["token_usage"]["selected_branch"]["total_tokens"] == 440


def test_adaptive_mode_do_not_retrieve_has_no_retrieval_timing():
    flow = _flow(decision=Decision.DO_NOT_RETRIEVE)
    out = flow.run_adaptive_only("q")
    assert out["selected_branch"] == "A"
    assert out["timings"]["retrieval_ms"] is None
    assert flow._branch_b.__self__ is flow  # sanity: same flow object
