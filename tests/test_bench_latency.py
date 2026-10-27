"""Unit tests for the interleaved-benchmark summary math (no LLM, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.bench_latency import median, summarize  # noqa: E402


@pytest.fixture
def trials():
    """Two interleaved trials: cold first calls, one failed trial."""
    trials = [
        {
            "order": "A,B", "ok_a": True, "ok_b": True,
            "a_ms": 30_000.0, "b_ms": 9_000.0,          # A cold (30s), B warm
            "a_cold": True, "b_cold": False,
            "a_tokens": {"total_tokens": 100}, "b_tokens": {"total_tokens": 1400},
            "retrieval_ms": 900.0, "decision_ms": 3.0,
            "generation_a_ms": 30_000.0, "generation_b_ms": 8_000.0,
        },
        {
            "order": "B,A", "ok_a": True, "ok_b": True,
            "a_ms": 7_500.0, "b_ms": 7_800.0,           # A warm here (order swapped)
            "a_cold": False, "b_cold": True,            # B cold in swapped trial
            "a_tokens": {"total_tokens": 100}, "b_tokens": {"total_tokens": 1500},
            "retrieval_ms": 1_100.0, "decision_ms": 5.0,
            "generation_a_ms": 7_500.0, "generation_b_ms": 6_700.0,
        },
        {
            "order": "A,B", "ok_a": False, "ok_b": True,
            "a_ms": None, "b_ms": 8_000.0, "a_cold": False, "b_cold": False,
            "a_tokens": None, "b_tokens": {"total_tokens": 1600},
            "retrieval_ms": 1_000.0, "decision_ms": 4.0,
            "generation_a_ms": None, "generation_b_ms": 7_000.0,
            "error_a": "RuntimeError: simulated failure",
        },
    ]
    return trials


def test_median_basic():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([]) is None
    assert median([None, None]) is None


def test_complete_pairs_exclude_failed_trials(trials):
    s = summarize(trials)
    assert s["n_trials"] == 3
    assert s["n_complete_pairs"] == 2  # failed A excluded everywhere
    assert s["n_failed"] == 1


def test_cold_samples_are_reported_separately_not_averaged(trials):
    s = summarize(trials)
    # cold first call per branch: A's 30 000 ms (trial 1), B's 7 800 (trial 2)
    assert s["branch_A"]["cold_first_call_ms"] == 30_000.0
    assert s["branch_B"]["cold_first_call_ms"] == 7_800.0
    # warm medians must NOT include the cold A sample (hand math):
    assert s["branch_A"]["median_path_ms_warm"] == 7_500.0
    assert s["branch_B"]["median_path_ms_warm"] == 9_000.0
    assert s["branch_A"]["n_warm_samples"] == 1
    assert s["branch_B"]["n_warm_samples"] == 1


def test_delta_warm_and_retrieval_decision_medians(trials):
    s = summarize(trials)
    assert s["median_delta_warm_ms"] == round(9_000.0 - 7_500.0, 3)
    # retrieval median mixes values from complete pairs and failed trial? No:
    # only complete pairs contribute (documented behavior)
    assert s["median_retrieval_ms"] == median([900.0, 1_100.0])
    assert s["median_decision_ms"] == median([3.0, 5.0])


def test_token_averages_only_over_counted_answers(trials):
    s = summarize(trials)
    assert s["branch_A"]["mean_total_tokens"] == 100.0
    assert s["branch_B"]["mean_total_tokens"] == 1450.0


def test_empty_trials_degrade_to_none():
    s = summarize([])
    assert s["n_trials"] == 0 and s["n_complete_pairs"] == 0
    assert s["median_delta_warm_ms"] is None
    assert s["branch_A"]["median_path_ms_warm"] is None
