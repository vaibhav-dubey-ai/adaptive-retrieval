import json

from adaptive_qa.corpus import build_corpus, load_corpus

import pathlib

from adaptive_qa.corpus import save_corpus, save_eval_manifest


def _records():
    return [
        {
            "question_id": "q1",
            "paragraphs": [
                {"title": "P1", "sentences": ["Alpha one.", "Alpha two."]},
                {"title": "D1", "sentences": ["Noise sentence."]},
            ],
            "gold_titles": ["P1"],
        },
        {
            "question_id": "q2",
            "paragraphs": [
                {"title": "P1", "sentences": ["Alpha one.", "Alpha two."]},  # duplicate doc
                {"title": "P2", "sentences": ["Beta sentence."]},
            ],
            "gold_titles": ["P2"],
        },
    ]


def test_corpus_dedups_and_builds_ids(tmp_path: pathlib.Path):
    docs, meta = build_corpus(_records())
    assert len(docs) == 3
    ids = [d.doc_id for d in docs]
    assert len(set(ids)) == 3
    assert all(i.startswith("doc_") for i in ids)


def test_gold_map_names_only_gold_paragraph_docs(tmp_path: pathlib.Path):
    docs, meta = build_corpus(_records())
    gold_by_question = meta["gold_doc_ids_by_question"]
    title_to_id = {d.title: d.doc_id for d in docs}
    assert set(gold_by_question["q1"]) == {title_to_id["P1"]}
    assert set(gold_by_question["q2"]) == {title_to_id["P2"]}
    # distractor doc is not gold for anyone
    distractor_id = title_to_id["D1"]
    assert distractor_id not in set(sum(gold_by_question.values(), []))


def test_save_load_roundtrip(tmp_path: pathlib.Path):
    docs, meta = build_corpus(_records())
    p = tmp_path / "corpus.jsonl"
    m = tmp_path / "manifest.json"
    save_corpus(docs, str(p))
    save_eval_manifest(meta, str(m))
    loaded = load_corpus(str(p))
    assert [d.doc_id for d in loaded] == [d.doc_id for d in docs]
    assert [d.text for d in loaded] == [d.text for d in docs]

    payload = json.loads(m.read_text(encoding="utf-8"))
    assert "gold_doc_ids_by_question" in payload
    assert "EVALUATION ONLY" in payload["_note"]
