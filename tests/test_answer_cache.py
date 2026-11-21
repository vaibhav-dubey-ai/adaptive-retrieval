"""Tests for the AnswerCache and evaluation caching/metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from adaptive_qa.answering.cache import AnswerCache, _cache_key
from adaptive_qa.eval.policy import _per_class_metrics, evaluate_policies
from adaptive_qa.eval.retrieval_eval import EvalExample
from adaptive_qa.common import GeneratedAnswer


def test_answer_cache_put_get_persistence(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    cache = AnswerCache(cache_file)
    assert cache.size() == 0

    q = "What is an entity set?"
    model = "test-model"
    chunks = ["c:1", "c:2"]

    record = cache.put(
        question=q,
        grounded=True,
        model=model,
        chunk_ids=chunks,
        text="An entity set is a collection of entities.",
        tokens={"total_tokens": 42},
        gen_ms=123.4,
        backend="mock",
    )
    assert record["key"]
    assert cache.size() == 1

    # Get from memory
    hit = cache.get(q, True, model, chunks)
    assert hit is not None
    assert hit["text"] == "An entity set is a collection of entities."
    assert hit["tokens"]["total_tokens"] == 42

    # Reload from disk in a fresh instance
    cache2 = AnswerCache(cache_file)
    assert cache2.size() == 1
    hit2 = cache2.get(q, True, model, chunks)
    assert hit2 is not None
    assert hit2["text"] == "An entity set is a collection of entities."


def test_per_class_metrics_handles_zero_denominators():
    # Case where TP=0, FP=0 (Never-retrieve scenario)
    cm = {
        "tp_retrieve_answerable": 0,
        "fn_skip_answerable": 10,
        "fp_retrieve_unanswerable": 0,
        "tn_skip_unanswerable": 5,
        "n_rows": 15,
    }
    m = _per_class_metrics(cm)
    assert m["precision"] is None  # 0/0 -> None
    assert m["recall"] == 0.0
    assert m["f1"] is None
    assert m["specificity"] == 1.0  # 5/5
    assert m["balanced_accuracy"] == 0.5  # (0.0 + 1.0) / 2
    assert m["accuracy"] == round(5 / 15, 4)


def test_evaluate_policies_with_cache(tmp_path: Path):
    cache_file = tmp_path / "cache.jsonl"
    cache = AnswerCache(cache_file)

    examples = [
        EvalExample(question_id="q1", question="Question 1", relevant_chunk_ids=("c:1",), reference_answer="Ref 1"),
        EvalExample(question_id="q2", question="Question 2", relevant_chunk_ids=(), reference_answer=""),
    ]

    class FakeRetriever:
        def retrieve(self, q, k):
            from adaptive_qa.common import RetrievalHit
            return [RetrievalHit(doc_id="c:1", score=1.0, rank=1, text="Text 1")]

    call_count = {"a": 0, "b": 0}

    class MockMethodA:
        def answer(self, q):
            call_count["a"] += 1
            return GeneratedAnswer(text=f"Answer A for {q}", backend="mock_a", meta={"tokens": {"total_tokens": 10}})

    class MockMethodB:
        def answer(self, q, hits):
            call_count["b"] += 1
            return GeneratedAnswer(text=f"Answer B for {q}", backend="mock_b", meta={"tokens": {"total_tokens": 20}})

    def mock_decide(q):
        return "RETRIEVE", 0.8, {}

    # First run without pre-existing cache
    res1 = evaluate_policies(
        examples,
        mock_decide,
        FakeRetriever(),
        top_k=1,
        method_a=MockMethodA(),
        method_b=MockMethodB(),
        max_answers=1,
        cache=cache,
    )
    assert call_count["a"] == 1
    assert call_count["b"] == 1
    assert res1["per_question"][0]["answer_a"] == "Answer A for Question 1"

    # Second run with cache present: model answer() should NOT be called again
    call_count["a"] = 0
    call_count["b"] = 0
    res2 = evaluate_policies(
        examples,
        mock_decide,
        FakeRetriever(),
        top_k=1,
        method_a=MockMethodA(),
        method_b=MockMethodB(),
        max_answers=1,
        cache=cache,
    )
    assert call_count["a"] == 0  # Reused from cache!
    assert call_count["b"] == 0  # Reused from cache!
    assert res2["per_question"][0]["answer_a"] == "Answer A for Question 1"
    assert res2["per_question"][0].get("cached_a") is True
