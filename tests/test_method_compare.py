"""Live comparison of BM25 / dense / hybrid on one question."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.retrieval.compare import compare_methods  # noqa: E402
from adaptive_qa.common import RetrievalHit  # noqa: E402


def _hit(cid, score, method, text="passage text", ranks=None):
    return RetrievalHit(cid, f"ch6.pdf", text, score, method=method, component_ranks=ranks or {})


class FakeRetriever:
    def __init__(self, method, hits, fail=False):
        self.method = method
        self.hits = hits
        self.fail = fail
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        if self.fail:
            raise RuntimeError("retriever exploded")
        return self.hits[:top_k]

    def diagnostics(self):
        return {"retriever": self.method, "n_chunks": 40}

    def source_meta(self, hit):
        return {"filename": "ch6.pdf", "page": 19 if hit.doc_id.endswith("10") else 17}


def _provider(retrievers):
    def provider(method):
        obj = retrievers[method]
        if isinstance(obj, Exception):
            raise obj
        return obj
    return provider


def test_all_three_methods_run_on_the_same_question_and_top_k():
    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("c:0010", 4.5, "bm25"), _hit("c:0017", 3.1, "bm25")]),
        "dense": FakeRetriever("dense", [_hit("c:0017", 0.72, "dense"), _hit("c:0010", 0.55, "dense")]),
        "hybrid": FakeRetriever(
            "hybrid",
            [_hit("c:0010", 0.032, "hybrid", ranks={"bm25": 1, "dense": 2}),
             _hit("c:0017", 0.029, "hybrid", ranks={"bm25": 2, "dense": 1})],
        ),
    }
    out = compare_methods(
        "What does a diamond represent?", ["bm25", "dense", "hybrid"], 5, _provider(retrievers)
    )
    for r in retrievers.values():
        assert r.calls == [("What does a diamond represent?", 5)]
    assert out["labels_included"] is False
    assert out["top_k"] == 5
    for method in ("bm25", "dense", "hybrid"):
        assert out["methods"][method]["ok"] is True
        assert out["methods"][method]["hits"]


def test_method_specific_scores_and_ranks_are_preserved():
    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("c:0010", 4.5, "bm25")]),
        "dense": FakeRetriever("dense", [_hit("c:0017", 0.72, "dense")]),
        "hybrid": FakeRetriever(
            "hybrid", [_hit("c:0010", 0.032, "hybrid", ranks={"bm25": 1, "dense": 0})]
        ),
    }
    out = compare_methods("q", ["bm25", "dense", "hybrid"], 5, _provider(retrievers))
    assert out["methods"]["bm25"]["hits"][0]["score"] == 4.5
    assert out["methods"]["dense"]["hits"][0]["score"] == 0.72
    hybrid_top = out["methods"]["hybrid"]["hits"][0]
    assert hybrid_top["component_ranks"] == {"bm25": 1, "dense": 0}
    # score scales are labelled, never blended into one number
    assert "BM25" in out["methods"]["bm25"]["score_kind"]
    assert "cosine" in out["methods"]["dense"]["score_kind"]
    assert out["methods"]["bm25"]["score_kind"] != out["methods"]["dense"]["score_kind"]


def test_overlap_marks_shared_and_unique_passages_with_rank_notes():
    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("c:0010", 4.5, "bm25"), _hit("c:0007", 3.0, "bm25")]),
        "dense": FakeRetriever("dense", [_hit("c:0010", 0.9, "dense"), _hit("c:0018", 0.8, "dense")]),
        "hybrid": FakeRetriever(
            "hybrid", [_hit("c:0010", 0.05, "hybrid"), _hit("c:0018", 0.03, "hybrid")]
        ),
    }
    out = compare_methods("q", ["bm25", "dense", "hybrid"], 5, _provider(retrievers))
    ov = out["overlap"]
    assert ov["shared_by_all"] == ["c:0010"]
    assert ov["unique_to"]["bm25"] == ["c:0007"]
    assert "c:0018" in ov["shared_by_two_or_more"]
    assert ov["pairwise_jaccard"]["bm25|dense"] == round(1 / 3, 3)
    shared_hit = out["methods"]["dense"]["hits"][0]
    assert shared_hit["shared"] is True
    assert shared_hit["ranks_by_method"] == {"bm25": 1, "dense": 1, "hybrid": 1}
    assert "bm25 #1" in shared_hit["rank_note"]


def test_source_metadata_is_joined_for_display():
    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("udoc_x:0010", 2.0, "bm25")]),
    }
    out = compare_methods("q", ["bm25"], 5, _provider(retrievers))
    hit = out["methods"]["bm25"]["hits"][0]
    assert hit["source_file"] == "ch6.pdf"
    assert hit["page"] == 19
    assert hit["text"] == "passage text"


def test_one_method_failing_does_not_hide_the_others():
    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("c:1", 1.0, "bm25")]),
        "dense": RuntimeError("embeddings unavailable"),
        "hybrid": FakeRetriever("hybrid", [_hit("c:1", 0.01, "hybrid")]),
    }
    out = compare_methods("q", ["bm25", "dense", "hybrid"], 3, _provider(retrievers))
    assert out["methods"]["bm25"]["ok"] is True
    assert out["methods"]["hybrid"]["ok"] is True
    assert out["methods"]["dense"]["ok"] is False
    assert "embeddings unavailable" in out["methods"]["dense"]["error"]
    # overlap analysis only considers the methods that actually ran
    assert out["overlap"]["shared_by_all"] == ["c:1"]


def test_no_active_collection_is_reported_per_method():
    out = compare_methods("q", ["bm25"], 5, lambda m: None)
    assert out["methods"]["bm25"]["ok"] is False
    assert "collection" in out["methods"]["bm25"]["error"].lower()


# ----------------------------------------------------------------- HTTP API
def test_api_retrieval_compare_uses_all_methods(monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps

    retrievers = {
        "bm25": FakeRetriever("bm25", [_hit("c:0010", 4.0, "bm25")]),
        "dense": FakeRetriever("dense", [_hit("c:0017", 0.8, "dense")]),
        "hybrid": FakeRetriever("hybrid", [_hit("c:0010", 0.03, "hybrid")]),
    }

    class FakeStore:
        def active_documents(self):
            return [{"filename": "ch6.pdf", "n_chunks": 80}]

    monkeypatch.setattr(deps, "_docs", lambda: FakeStore())
    monkeypatch.setattr(deps, "_collection_info", lambda method, k: {"n_chunks": 80})
    monkeypatch.setattr(deps, "_collection_retriever", _provider(retrievers))

    client = TestClient(api_main.app)
    resp = client.post("/retrieval/compare", json={"question": "What is a diamond?", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["methods"]) == {"bm25", "dense", "hybrid"}
    assert body["methods"]["dense"]["hits"][0]["doc_id"] == "c:0017"
    assert body["labels_included"] is False
    assert "relevant" not in str(body).lower()  # no gold labels on the live path


def test_api_compare_requires_a_question(monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps

    client = TestClient(api_main.app)
    assert client.post("/retrieval/compare", json={"question": " "}).status_code == 400


def test_api_compare_validates_top_k(monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps

    class FakeStore:
        def active_documents(self):
            return [{"filename": "x.pdf", "n_chunks": 1}]

    monkeypatch.setattr(deps, "_docs", lambda: FakeStore())
    client = TestClient(api_main.app)
    assert client.post("/retrieval/compare", json={"question": "q", "top_k": 0}).status_code == 400
    assert client.post("/retrieval/compare", json={"question": "q", "top_k": 99}).status_code == 400
