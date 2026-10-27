"""Experiments view (saved results), demo examples and policy evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.eval.policy import evaluate_policies  # noqa: E402
from adaptive_qa.eval.retrieval_eval import EvalExample  # noqa: E402
from adaptive_qa.experiments_view import (  # noqa: E402
    build_retrieval_view,
    list_retrieval_runs,
    load_policy_view,
    load_retrieval_view,
)
from adaptive_qa.types import Decision, GeneratedAnswer, RetrievalHit  # noqa: E402


def _write_run(root: Path, run_id: str, table, methods):
    d = root / run_id
    d.mkdir(parents=True)
    payload = {
        "generated_at_utc": "2026-01-01T00:00:00Z",
        "dataset": {"path": "eval_data/x.jsonl", "sha256": "abc123"},
        "metrics": {"recall_k": 5, "rank_k": 10},
        "comparison_table": table,
        "methods": methods,
    }
    (d / "retrieval_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def _table():
    return [
        {"method": "bm25", "n_questions": 2, "n_scored_questions": 2, "recall_at_5": 0.5,
         "mrr_at_10": 0.5, "ndcg_at_10": 0.5, "hit_rate_at_1": 0.5, "no_evidence_rate": 0.0,
         "build_ms": 10.0, "median_query_ms": 0.12, "p95_query_ms": 0.2, "cold_first_query_ms": 0.3},
        {"method": "dense", "n_questions": 2, "n_scored_questions": 2, "recall_at_5": 0.75,
         "mrr_at_10": 0.6, "ndcg_at_10": 0.7, "hit_rate_at_1": 0.5, "no_evidence_rate": 0.0,
         "build_ms": 200.0, "model_load_ms": 150.0, "median_query_ms": 20.0,
         "p95_query_ms": 30.0, "cold_first_query_ms": 900.0},
        {"method": "hybrid", "n_questions": 2, "n_scored_questions": 2, "recall_at_5": 1.0,
         "mrr_at_10": 1.0, "ndcg_at_10": 1.0, "hit_rate_at_1": 1.0, "no_evidence_rate": 0.0,
         "build_ms": 50.0, "median_query_ms": 25.0, "p95_query_ms": 40.0, "cold_first_query_ms": 1200.0},
    ]


def _methods():
    def per_question(relevant, retrieved, recall, rr, ndcg):
        return [{
            "question_id": "q1", "question": "What is X?", "category": "definition",
            "relevant": relevant, "retrieved": retrieved, "recall_at_k": recall,
            "reciprocal_rank": rr, "ndcg": ndcg, "retrieval_ms": 0.1,
        }]
    return {
        "bm25": {"per_question": per_question(["c:1"], ["c:2", "c:1"], 1.0, 0.5, 0.4)},
        "dense": {"per_question": per_question(["c:1"], ["c:1", "c:2"], 1.0, 1.0, 0.9)},
        "hybrid": {"per_question": per_question(["c:1"], ["c:1"], 1.0, 1.0, 1.0)},
    }


def test_view_is_built_from_the_saved_file_with_real_deltas(tmp_path: Path):
    _write_run(tmp_path, "20260101_000000", _table(), _methods())
    view = build_retrieval_view(tmp_path / "20260101_000000")
    assert view["run_id"] == "20260101_000000"
    rows = {r["method"]: r for r in view["comparison_table"]}
    assert rows["bm25"]["vs_bm25"] == {}
    assert rows["hybrid"]["vs_bm25"]["recall_at_5"] == 0.5
    assert rows["dense"]["vs_bm25"]["recall_at_5"] == 0.25
    assert view["best"]["recall_at_5"] == ["hybrid"]
    # latency columns preserved honestly, including cold start
    assert rows["dense"]["model_load_ms"] == 150.0
    assert rows["dense"]["cold_first_query_ms"] == 900.0


def test_view_merges_per_question_rows_across_methods(tmp_path: Path):
    _write_run(tmp_path, "r1", _table(), _methods())
    view = build_retrieval_view(tmp_path / "r1")
    assert len(view["questions"]) == 1
    q = view["questions"][0]
    assert q["relevant_chunk_ids"] == ["c:1"]
    assert set(q["methods"]) == {"bm25", "dense", "hybrid"}
    assert q["methods"]["bm25"]["reciprocal_rank"] == 0.5
    assert view["metric_help"]["recall_at_5"].startswith("Recall@5")


def test_latest_run_is_selected_and_listed(tmp_path: Path):
    _write_run(tmp_path, "20260101_000000", _table(), _methods())
    _write_run(tmp_path, "20260102_000000", _table(), _methods())
    runs = list_retrieval_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ["20260102_000000", "20260101_000000"]
    assert load_retrieval_view(tmp_path)["run_id"] == "20260102_000000"
    assert load_retrieval_view(tmp_path, run_id="20260101_000000")["run_id"] == "20260101_000000"
    assert load_retrieval_view(tmp_path, run_id="nope") is None


def test_missing_view_returns_none(tmp_path: Path):
    assert load_retrieval_view(tmp_path / "absent") is None


def test_api_experiments_retrieval_and_question_inspector(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient

    import api.main as api_main

    _write_run(tmp_path, "r1", _table(), _methods())
    monkeypatch.setattr(api_main, "RESULTS_RETRIEVAL_DIR", tmp_path)

    class FakeStore:
        def get(self, doc_id):
            return {"filename": "ch6.pdf", "page": None}

        def load_chunks(self, doc_id):
            return [
                {"chunk_id": "c:1", "page": 19, "text": "Diamonds represent relationship sets."},
                {"chunk_id": "c:2", "page": 17, "text": "Composite attributes have subparts."},
            ]

    monkeypatch.setattr(api_main, "_docs", lambda: FakeStore())
    client = TestClient(api_main.app)

    resp = client.get("/experiments/retrieval")
    assert resp.status_code == 200
    assert resp.json()["view"]["comparison_table"][0]["method"] == "bm25"

    q = client.get("/experiments/retrieval/question/q1")
    assert q.status_code == 200
    body = q.json()
    assert body["evaluation_only"] is True
    assert body["relevant_chunk_ids"] == ["c:1"]
    bm25_hits = body["methods"]["bm25"]["retrieved"]
    assert bm25_hits[0]["chunk_id"] == "c:2" and bm25_hits[0]["is_relevant"] is False
    assert bm25_hits[1]["is_relevant"] is True
    assert bm25_hits[1]["source"]["page"] == 19

    assert client.get("/experiments/retrieval/question/q404").status_code == 404


def test_labels_are_only_in_the_experiments_payload(monkeypatch, tmp_path: Path):
    """The live comparison endpoint must not carry gold labels."""
    from fastapi.testclient import TestClient

    import api.main as api_main
    from adaptive_qa.types import RetrievalHit

    _write_run(tmp_path, "r1", _table(), _methods())
    monkeypatch.setattr(api_main, "RESULTS_RETRIEVAL_DIR", tmp_path)

    class FakeRetriever:
        method = "bm25"

        def retrieve(self, q, k):
            return [RetrievalHit("c:1", "ch6.pdf", "text", 1.0, method="bm25")]

        def diagnostics(self):
            return {"retriever": "bm25"}

    class FakeStore:
        def active_documents(self):
            return [{"filename": "ch6.pdf", "n_chunks": 1}]

    monkeypatch.setattr(api_main, "_docs", lambda: FakeStore())
    monkeypatch.setattr(api_main, "_collection_info", lambda m, k: {})
    monkeypatch.setattr(api_main, "_collection_retriever", lambda m: FakeRetriever())
    client = TestClient(api_main.app)
    body = client.post("/retrieval/compare", json={"question": "What is X?"}).json()
    assert "relevant_chunk_ids" not in json.dumps(body)
    assert body["labels_included"] is False


# ------------------------------------------------------------ demo examples
def test_demo_examples_cover_the_categories_without_reference_answers(
    monkeypatch, tmp_path: Path
):
    from fastapi.testclient import TestClient

    import api.main as api_main

    lines = [
        {"question_id": "q1", "category": "exact_terminology", "question": "What is a diamond?",
         "relevant_chunk_ids": ["c:1"], "reference_answer": "SECRET GOLD"},
        {"question_id": "q2", "category": "paraphrase", "question": "What does the diamond shape mean?",
         "relevant_chunk_ids": ["c:1"], "reference_answer": "SECRET GOLD 2"},
        {"question_id": "q3", "category": "unanswerable", "question": "Who won the 2022 World Cup?",
         "relevant_chunk_ids": [], "reference_answer": ""},
    ]
    (tmp_path / "tiny.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines), encoding="utf-8"
    )
    monkeypatch.setattr(api_main, "EVAL_SET_DIR", tmp_path)
    client = TestClient(api_main.app)
    body = client.get("/demo/examples").json()
    assert body["available"] is True
    assert body["n_questions"] == 3
    names = {c["name"]: c for c in body["categories"]}
    assert names["unanswerable"]["answerable"] is False
    assert names["exact_terminology"]["count"] == 1
    assert "SECRET GOLD" not in json.dumps(body)  # reference answers are not exposed
    assert "relevant_chunk_ids" not in json.dumps(body)


# ----------------------------------------------------------- policy metrics
class FakeRetriever:
    method = "bm25"

    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, question, top_k):
        return [RetrievalHit(cid, "d", "t", 1.0) for cid in self.mapping.get(question, [])][:top_k]


def _examples():
    return [
        EvalExample("q1", "answerable one", ("c:1",), "definition", "gold one"),
        EvalExample("q2", "answerable two", ("c:2",), "definition", "gold two"),
        EvalExample("q3", "unanswerable one", (), "unanswerable", ""),
    ]


def _decide(question):
    return (Decision.RETRIEVE if question != "answerable two" else Decision.DO_NOT_RETRIEVE), 0.6, {}


def test_policies_differ_and_are_computed_separately():
    retriever = FakeRetriever({"answerable one": ["c:1"], "answerable two": ["c:2"], "unanswerable one": ["c:1"]})
    out = evaluate_policies(_examples(), _decide, retriever, top_k=5)
    pol = out["policies"]
    assert pol["never_retrieve"]["evidence_recall_at_5"] == 0.0
    assert pol["always_retrieve"]["evidence_recall_at_5"] == 1.0
    # adaptive skipped q2 (answerable) => recall 0.5
    assert pol["adaptive"]["evidence_recall_at_5"] == 0.5
    assert pol["oracle"]["evidence_recall_at_5"] == 1.0
    # adaptive retrieved for q1 and for the unanswerable q3
    assert pol["adaptive"]["retrieval_calls"] == 2
    assert pol["always_retrieve"]["retrieval_calls"] == 3
    assert pol["never_retrieve"]["retrieval_calls"] == 0
    assert pol["adaptive"]["unanswerable_probe_rate"] == 1.0
    assert pol["always_retrieve"]["unanswerable_probe_rate"] == 1.0
    assert out["decision_quality"]["skipped_but_answerable"] == ["q2"]
    assert out["decision_quality"]["retrieved_but_unanswerable"] == ["q3"]
    assert out["decision_quality"]["accuracy"] == round(1 / 3, 4)
    # latency definitions differ per policy
    assert "decision" in pol["adaptive"]["latency_definition"]


def test_answer_layer_scores_and_oracle_preference():
    class MethodA:
        def answer(self, q):
            return GeneratedAnswer(text="wrong", backend="a",
                                   meta={"tokens": {"total_tokens": 10, "source": "estimated"}})

    class MethodB:
        def answer(self, q, passages):
            raise RuntimeError("provider returned an empty response")

    retriever = FakeRetriever({"answerable one": ["c:1"]})
    out = evaluate_policies(
        _examples()[:1], _decide, retriever, top_k=5, method_a=MethodA(), method_b=MethodB()
    )
    row = out["per_question"][0]
    # a failed generation is recorded, not counted as an answer
    assert row["answer_b"] is None
    assert "empty response" in row["error_b"]
    assert out["policies"]["always_retrieve"]["n_rows_with_answer"] == 0
    assert out["policies"]["never_retrieve"]["n_rows_with_answer"] == 1
    # quality means only cover successful answers
    assert out["policies"]["always_retrieve"]["mean_composite"] is None
    assert out["policies"]["never_retrieve"]["mean_total_tokens"] == 10
    # a failed generation contributes no latency sample
    assert out["policies"]["always_retrieve"]["n_latency_samples"] == 0
    assert out["policies"]["never_retrieve"]["n_latency_samples"] == 1
    assert out["policies"]["never_retrieve"]["mean_latency_ms"] is not None
    assert out["policies"]["never_retrieve"]["mean_retrieval_ms"] is None


def test_oracle_uses_reference_scores_only_in_evaluation():
    class MethodA:
        def answer(self, q):
            return GeneratedAnswer(text="gold one", backend="a", meta={"tokens": {"total_tokens": 3}})

    class MethodB:
        def answer(self, q, passages):
            return GeneratedAnswer(text="something else entirely", backend="b",
                                   meta={"tokens": {"total_tokens": 30}})

    retriever = FakeRetriever({"answerable one": ["c:1"]})
    out = evaluate_policies(
        _examples()[:1], _decide, retriever, top_k=5, method_a=MethodA(), method_b=MethodB()
    )
    row = out["per_question"][0]
    assert row["oracle_prefers"] == "A"
    assert row["em_a"] == 1.0
    assert row["composite_a"] > row["composite_b"]
    assert out["policies"]["oracle"]["mean_total_tokens"] == 3


def test_policy_view_loads_latest_file(tmp_path: Path):
    root = tmp_path / "policy_eval"
    for run_id, n in (("20260101_000000", 5), ("20260102_000000", 9)):
        d = root / run_id
        d.mkdir(parents=True)
        (d / "policy_results.json").write_text(
            json.dumps({"generated_at_utc": "x", "settings": {"n_questions": n}, "policies": {}}),
            encoding="utf-8",
        )
    view = load_policy_view(root)
    assert view is not None and view["run_id"] == "20260102_000000"
    assert view["settings"]["n_questions"] == 9
    assert "adaptive" in view["policy_help"]
