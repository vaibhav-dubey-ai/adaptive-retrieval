"""Routing evaluation: quality metric, labelling, split hygiene, API view."""

import json
import random

from adaptive_qa.eval.em_f1 import f1_score, quality_score, recall_score
from adaptive_qa.eval.routing import (
    _tune_threshold,
    evaluate_split,
    label_rows,
    repeated_splits,
)


def test_quality_does_not_punish_a_correct_wordy_answer():
    gold = "Paris."
    wordy = "The capital of France is Paris, a large city on the Seine."
    assert recall_score(gold, wordy) == 1.0
    assert f1_score(gold, wordy) < 0.3
    assert quality_score(gold, wordy) > f1_score(gold, wordy)


def test_quality_is_zero_for_a_wrong_answer_and_one_for_exact():
    assert quality_score("Jupiter.", "Saturn is large") == 0.0
    assert quality_score("Jupiter.", "jupiter") == 1.0


def _row(i, helps, score, qa=0.1, qb=0.9):
    qa, qb = (qa, qb) if helps else (0.9, 0.9)
    return {
        "question_id": f"q{i}", "reference_answer": "x", "answer_a": "a", "answer_b": "b",
        "heuristic_score": score,
        "preflight": {"collection_relevance": score, "corpus_coverage": score,
                      "topk_coverage": score, "top1_coverage": score, "n_question_terms": 4},
        "features": {"q_len_tokens": 0.3, "q_entity_density": 0.0}, "wh_open": 1.0,
        "tokens_a": 80, "tokens_b": 800, "gen_a_ms": 3000.0, "gen_b_ms": 20000.0,
        "retrieval_ms": 1.0, "quality_a": qa, "quality_b": qb,
        "retrieval_helps": helps, "retrieval_hurts": False,
    }


def test_label_rows_skips_unscorable_and_applies_gap():
    ok = {"reference_answer": "Paris", "answer_a": "Paris", "answer_b": "Paris"}
    helped = {"reference_answer": "Paris", "answer_a": "I don't know", "answer_b": "Paris"}
    no_ref = {"reference_answer": "", "answer_a": "x", "answer_b": "y"}
    no_answer = {"reference_answer": "Paris", "answer_a": None, "answer_b": "Paris"}
    out = label_rows([ok, helped, no_ref, no_answer], gap=0.10)
    assert len(out) == 2
    assert out[0]["retrieval_helps"] is False and out[1]["retrieval_helps"] is True


def test_threshold_tuning_separates_a_separable_signal():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    targets = [False, False, False, True, True, True]
    t = _tune_threshold(scores, targets)
    assert 0.3 < t <= 0.7


def test_oracle_dominates_and_never_is_cheapest():
    rows = [_row(i, helps=i % 2 == 0, score=0.9 if i % 2 == 0 else 0.1) for i in range(20)]
    res = evaluate_split(rows[:14], rows[14:])
    assert res["oracle"]["quality"] >= res["always"]["quality"] - 1e-9
    assert res["never"]["tokens"] < res["always"]["tokens"]
    assert res["learned"]["balanced_accuracy"] == 1.0  # perfectly separable synthetic signal


def test_repeated_splits_never_leak_train_into_test(monkeypatch):
    rows = [_row(i, helps=i % 3 == 0, score=random.Random(i).random()) for i in range(30)]
    seen = []
    import adaptive_qa.eval.routing as routing

    real = routing.evaluate_split

    def spy(train, test, **kw):
        assert not {r["question_id"] for r in train} & {r["question_id"] for r in test}
        seen.append(len(test))
        return real(train, test, **kw)

    monkeypatch.setattr(routing, "evaluate_split", spy)
    out = repeated_splits(rows, n_splits=5)
    assert len(seen) == 5 and "learned" in out


def test_routing_endpoint_serves_saved_results(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    run = tmp_path / "r1"
    run.mkdir()
    (run / "routing_results.json").write_text(json.dumps({"settings": {"n_scored": 1}, "policies": {}}))
    (run / "per_question.jsonl").write_text(json.dumps(_row(1, True, 0.9)) + "\n")
    monkeypatch.setattr(api_main, "RESULTS_ROUTING_DIR", tmp_path)
    body = TestClient(api_main.app).get("/experiments/routing").json()
    assert body["view"]["run_id"] == "r1"
    assert body["view"]["questions"][0]["question_id"] == "q1"
    assert body["view"]["branch_costs"]["tokens_b"] == 800
