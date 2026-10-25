"""Tests for the labelled retrieval-evaluation set and the harness around it.

The shipped dataset is parsed (schema + categories + label form). Metrics are
exercised with stub retrievers so nothing here depends on an uploaded document
or on model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.eval.retrieval_eval import (  # noqa: E402
    EvalExample,
    compare_methods,
    evaluate_retriever,
    load_eval_examples,
    validate_labels,
    write_results,
)
from adaptive_qa.types import RetrievalHit  # noqa: E402

DATASET = PROJECT_ROOT / "eval_data" / "retrieval_eval" / "ch6_er_model_questions.jsonl"


def _example(qid, question, rel, category="definition"):
    return EvalExample(
        question_id=qid,
        question=question,
        relevant_chunk_ids=tuple(rel),
        category=category,
        reference_answer="reference",
    )


# ---------------------------------------------------------------- shipped set
def test_shipped_dataset_parses_and_is_balanced():
    examples = load_eval_examples(DATASET)
    assert len(examples) == 75
    categories = {e.category for e in examples}
    assert categories == {
        "exact_terminology", "paraphrase", "definition", "multi_passage", "unanswerable",
    }
    answerable = [e for e in examples if e.answerable]
    unanswerable = [e for e in examples if not e.answerable]
    assert len(answerable) == 67
    assert len(unanswerable) == 8
    # labels are stable chunk ids of the form "<doc_id>:<n>"
    for ex in answerable:
        assert ex.relevant_chunk_ids
        for cid in ex.relevant_chunk_ids:
            assert ":" in cid and cid.rsplit(":", 1)[1].isdigit()
        assert ex.reference_answer, ex.question_id
    # multi-passage questions really have more than one labelled chunk
    multi = [e for e in examples if e.category == "multi_passage"]
    assert multi and all(len(e.relevant_chunk_ids) >= 2 for e in multi)


def test_loader_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        json.dumps({"question_id": "x", "question": "one", "relevant_chunk_ids": []}) + "\n",
        encoding="utf-8",
    )
    assert len(load_eval_examples(p)) == 1
    p.write_text(
        json.dumps({"question_id": "x", "question": "one"}) + "\n"
        + json.dumps({"question_id": "x", "question": "two"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as ei:
        load_eval_examples(p)
    assert "duplicate question_id" in str(ei.value)


def test_loader_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"question": "no id"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        load_eval_examples(p)
    assert "question_id" in str(ei.value)


def test_loader_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_examples(p)


def test_validate_labels_flags_unknown_chunk_ids():
    examples = [_example("q1", "Question?", ["doc:0001", "doc:9999"])]
    report = validate_labels(examples, ["doc:0001", "doc:0002"])
    assert report["ok"] is False
    assert report["unknown_chunk_ids"] == [{"question_id": "q1", "chunk_id": "doc:9999"}]
    assert validate_labels([_example("q2", "Q?", ["doc:0002"])], ["doc:0002"])["ok"] is True


# ------------------------------------------------------------------- harness
class StubRetriever:
    """Returns ids per question, so metric outcomes are fully predictable."""

    def __init__(self, mapping, method="stub"):
        self.mapping = mapping
        self.method = method
        self.calls = 0

    def retrieve(self, question, top_k):
        self.calls += 1
        ids = self.mapping.get(question, [])[:top_k]
        return [RetrievalHit(i, "t", "text", 1.0) for i in ids]


def test_evaluate_retriever_records_per_question_metrics_and_latency():
    examples = [
        _example("q1", "A", ["c1"]),
        _example("q2", "B", ["c2"]),
        _example("q3", "C", []),
    ]
    retriever = StubRetriever({"A": ["c1", "x"], "B": ["x", "c2"], "C": ["x"]})
    out = evaluate_retriever(retriever, examples, top_k=5)
    rows = {r["question_id"]: r for r in out["per_question"]}
    assert rows["q1"]["recall_at_k"] == 1.0
    assert rows["q1"]["reciprocal_rank"] == 1.0
    assert rows["q2"]["reciprocal_rank"] == 0.5
    assert rows["q3"]["recall_at_k"] is None       # unanswerable
    agg = out["aggregate"]
    assert agg["n_scored_questions"] == 2
    assert agg["recall_at_5"] == 1.0
    assert agg["mrr_at_10"] == pytest.approx(0.75)
    assert out["latency"]["steady_state"]["median_ms"] is not None
    assert retriever.calls == 3


def test_evaluate_retriever_repeat_only_affects_latency_sampling():
    examples = [_example("q1", "A", ["c1"])]
    retriever = StubRetriever({"A": ["c1"]})
    out = evaluate_retriever(retriever, examples, top_k=3, repeat=3)
    assert len(out["per_question"]) == 1          # metrics from the first pass
    assert out["latency"]["cold_first_query_ms"] is not None
    # the first query is reported as cold start and excluded from steady state
    assert out["latency"]["steady_state"]["n"] == 2
    assert retriever.calls == 3


class _FakeStore:
    def __init__(self):
        self.built = []

    def build_retriever(self, doc_id=None, method="bm25", retrieval_cfg=None):
        self.built.append(method)
        mapping = {"A": ["c1"]} if method == "bm25" else {"A": ["x", "c1"]}
        return StubRetriever(mapping, method=method)


def test_compare_methods_runs_each_method_over_the_same_questions(tmp_path):
    store = _FakeStore()
    examples = [_example("q1", "A", ["c1"])]
    results = compare_methods({"retrieval": {}}, examples, ["bm25", "dense"], store, top_k=5)
    assert store.built == ["bm25", "dense"]
    assert results["bm25"]["aggregate"]["mrr_at_10"] == 1.0
    assert results["dense"]["aggregate"]["mrr_at_10"] == 0.5
    assert results["bm25"]["build_ms"] is not None


def test_write_results_emits_json_and_csv(tmp_path):
    store = _FakeStore()
    examples = [_example("q1", "A", ["c1"])]
    results = compare_methods({"retrieval": {}}, examples, ["bm25"], store)
    dataset = tmp_path / "set.jsonl"
    dataset.write_text(json.dumps({"question_id": "q1", "question": "A"}) + "\n", encoding="utf-8")
    paths = write_results(tmp_path / "out", results, dataset_path=dataset)
    for key in ("json", "summary_csv", "per_question_csv"):
        assert Path(paths[key]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["metrics"] == {"recall_k": 5, "rank_k": 10}
    assert payload["dataset"]["sha256"]
    assert payload["comparison_table"][0]["method"] == "bm25"
    csv_text = Path(paths["summary_csv"]).read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("method,")
    per_q = Path(paths["per_question_csv"]).read_text(encoding="utf-8")
    assert "relevant_chunk_ids" in per_q and "c1" in per_q
