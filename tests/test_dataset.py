import json

import pytest

from adaptive_qa.dataset import (
    assign_splits,
    choose_offset,
    normalize_hotpot_row,
    to_question_records,
)


def _raw_row(qid: str, question: str, answer: str) -> dict:
    """Mimic a HF datasets-server hotpot_qa distractor row."""
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": "bridge",
        "level": "medium",
        "supporting_facts": {"title": ["P1", "P2"], "sent_id": [0, 1]},
        "context": [
            {"title": "P1", "sentences": ["Alpha sentence one.", "Alpha two."]},
            {"title": "Distractor", "sentences": ["Noise."]},
            {"title": "P2", "sentences": ["Beta sentence."]},
        ],
    }


def test_normalize_row_extracts_gold_titles_and_paragraphs():
    rec = normalize_hotpot_row(_raw_row("q1", "Who did X?", "Jones"))
    assert rec["gold_titles"] == ["P1", "P2"]
    assert rec["paragraphs"][0]["title"] == "P1"
    assert len(rec["paragraphs"]) == 3
    assert rec["answer"] == "Jones"


def test_normalize_row_handles_dict_context_payload():
    raw = _raw_row("q1", "Q?", "A")
    raw["context"] = {"title": [["P1", "P2"]], "sentences": [[["s"]]]}
    rec = normalize_hotpot_row(raw)
    assert rec["paragraphs"], "context dict variant must still parse"
    assert rec["gold_titles"] == ["P1", "P2"]


def test_splits_are_question_level_and_disjoint():
    records = [json.loads(json.dumps(
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "answer": f"a{i}",
            "gold_titles": [],
            "paragraphs": [],
        }
    )) for i in range(20)]
    out = assign_splits(records, train_frac=0.4, validation_frac=0.3, seed=13)
    labels = [r["split"] for r in out]
    assert len(labels) == 20
    assert any(s == "train" for s in labels)
    assert any(s == "validation" for s in labels)
    assert any(s == "test" for s in labels)
    # each question assigned exactly once
    assert sum(labels.count(s) for s in {"train", "validation", "test"}) == 20
    # deterministic
    out2 = assign_splits(
        [json.loads(json.dumps(r)) for r in out_copy_for_reset(out)],
        train_frac=0.4, validation_frac=0.3, seed=13,
    )
    assert [r["split"] for r in out2] == labels


def out_copy_for_reset(records):
    return [
        {**r, "split": ""}
        for r in records
    ]


def test_to_question_records_fields():
    rec = normalize_hotpot_row(_raw_row("q7", "Who did X?", "Jones"))
    rec["split"] = "train"
    qr = to_question_records([rec])[0]
    assert qr.question_id == "q7"
    assert qr.answer == "Jones"
    assert qr.gold_titles == ("P1", "P2")
    assert qr.split == "train"


def test_choose_offset_deterministic_and_bounded():
    a = choose_offset(1000, n=200, seed=3)
    b = choose_offset(1000, n=200, seed=3)
    c = choose_offset(1000, n=200, seed=4)
    assert a == b
    assert 0 <= a <= 1000 - 200
    assert c != a
