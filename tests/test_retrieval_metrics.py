"""Retrieval metric tests with hand-calculated expected values."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.eval.retrieval_metrics import (  # noqa: E402
    aggregate,
    latency_stats,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)


# --------------------------------------------------------------- Recall@k
def test_recall_at_k_counts_the_share_of_labelled_chunks_found():
    relevant = ["a", "b"]
    assert recall_at_k(["x", "a", "y", "b"], relevant, 5) == 1.0
    assert recall_at_k(["x", "a", "y", "z"], relevant, 5) == 0.5
    assert recall_at_k(["x", "a"], relevant, 1) == 0.0      # k truncates
    assert recall_at_k([], relevant, 5) == 0.0


def test_recall_is_undefined_without_labels():
    assert recall_at_k(["a"], [], 5) is None


# ------------------------------------------------------------------- MRR@k
def test_mrr_uses_rank_of_first_relevant_hit():
    assert reciprocal_rank_at_k(["x", "a", "b"], ["a"], 10) == 0.5
    assert reciprocal_rank_at_k(["a"], ["a"], 10) == 1.0
    assert reciprocal_rank_at_k(["x", "y"], ["a"], 10) == 0.0
    # beyond k the hit does not count
    assert reciprocal_rank_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_mrr_undefined_without_labels():
    assert reciprocal_rank_at_k(["a"], [], 10) is None


# ------------------------------------------------------------------ nDCG@k
def test_ndcg_matches_manual_dcg_over_idcg():
    relevant = ["a", "b"]
    retrieved = ["x", "a", "y", "b"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, 10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 10) == pytest.approx(1.0)


def test_ndcg_penalises_low_ranked_relevance():
    high = ndcg_at_k(["a", "x", "y"], ["a"], 10)
    low = ndcg_at_k(["x", "y", "a"], ["a"], 10)
    assert high > low


def test_ndcg_empty_retrieval_is_zero():
    assert ndcg_at_k([], ["a"], 10) == 0.0


def test_ndcg_undefined_without_labels():
    assert ndcg_at_k(["a"], [], 10) is None


# --------------------------------------------------------------- aggregation
def _row(retrieved, relevant, category="definition", latency=1.0):
    return {
        "retrieved": retrieved,
        "relevant": relevant,
        "category": category,
        "retrieval_ms": latency,
    }


def test_aggregate_excludes_unanswerable_questions_and_reports_them():
    rows = [
        _row(["a", "x"], ["a"]),           # recall 1.0, rr 1.0
        _row(["x", "b"], ["b"]),           # recall 1.0, rr 0.5
        _row([], []),                      # unanswerable -> excluded
        _row(["x"], []),                   # unanswerable, returned something
    ]
    agg = aggregate(rows, k_recall=5, k_rank=10)
    assert agg["n_questions"] == 4
    assert agg["n_scored_questions"] == 2
    assert agg["n_unanswerable_questions"] == 2
    assert agg["recall_at_5"] == 1.0
    assert agg["mrr_at_10"] == pytest.approx((1.0 + 0.5) / 2)
    # one of the two unanswerable questions returned no evidence at all
    assert agg["no_evidence_rate"] == 0.5
    assert agg["hit_rate_at_1"] == 0.5


def test_aggregate_breakdown_by_category():
    rows = [
        _row(["a"], ["a"], category="exact"),
        _row(["x"], ["a"], category="paraphrase"),
    ]
    agg = aggregate(rows)
    assert agg["by_category"]["exact"]["recall_at_5"] == 1.0
    assert agg["by_category"]["paraphrase"]["recall_at_5"] == 0.0


def test_aggregate_of_only_unanswerable_questions_has_no_quality_value():
    agg = aggregate([_row([], []), _row([], [])])
    assert agg["recall_at_5"] is None
    assert agg["mrr_at_10"] is None
    assert agg["no_evidence_rate"] == 1.0


# ------------------------------------------------------------------- latency
def test_latency_stats_report_median_and_p95():
    stats = latency_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats["n"] == 5
    assert stats["median_ms"] == 3.0
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0
    assert stats["p95_ms"] > 4.0


def test_latency_stats_empty():
    assert latency_stats([])["median_ms"] is None


def test_percentile_interpolates():
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) is None
