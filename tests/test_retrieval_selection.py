"""Retrieval-method selection: config, store scoping, API plumbing, chunk ids.

Dense inference is replaced by an injected deterministic embedder so the suite
stays fast and never downloads model weights.
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.live.documents import DocumentStore, chunk_pages  # noqa: E402
from adaptive_qa.retrieval import (  # noqa: E402
    DEFAULT_METHOD,
    RETRIEVAL_METHODS,
    build_retriever_from_config,
    load_retriever,
    normalize_method,
)
from adaptive_qa.common import CorpusDoc, Decision, GeneratedAnswer, RetrievalHit  # noqa: E402


class FakeEmbedder:
    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
        return out


def _docs():
    return [
        CorpusDoc("d1", "er", "Composite attributes are divided into component attributes.", ()),
        CorpusDoc("d2", "keys", "A weak entity set is existence dependent on its identifying entity.", ()),
    ]


# ------------------------------------------------------------------ selection
def test_registered_methods_and_default_are_backward_compatible():
    assert RETRIEVAL_METHODS == ("bm25", "dense", "hybrid")
    assert DEFAULT_METHOD == "bm25"


def test_normalize_method_accepts_case_and_rejects_unknown():
    assert normalize_method("BM25") == "bm25"
    assert normalize_method(None) == "bm25"
    with pytest.raises(ValueError) as ei:
        normalize_method("faiss")
    assert "bm25, dense, hybrid" in str(ei.value)


def test_config_builder_defaults_to_bm25_and_honours_override():
    cfg = {
        "retrieval": {
            "retriever": "bm25",
            "bm25": {"k1": 1.2},
            "dense": {"embedder": FakeEmbedder(), "use_cache": False},
            "hybrid": {},
        }
    }
    assert type(build_retriever_from_config(cfg, _docs())).__name__ == "BM25Retriever"
    # an explicit method wins over the config default
    r2 = build_retriever_from_config(cfg, _docs(), method="dense")
    assert type(r2).__name__ == "DenseRetriever"


def test_bm25_kwargs_never_leak_into_dense_constructor():
    # k1/b are BM25-only; passing them must not raise for dense
    r = load_retriever(
        "dense",
        _docs(),
        bm25_kwargs={"k1": 9.0, "b": 0.5},
        dense_kwargs={"embedder": FakeEmbedder(), "use_cache": False},
    )
    assert len(r.retrieve("weak entity set", 2)) == 2


# ------------------------------------------------------- store scoping by method
def _store(tmp_path) -> DocumentStore:
    return DocumentStore(root=tmp_path / "documents")


def _dense_cfg():
    return {"dense": {"embedder": FakeEmbedder(), "use_cache": False}, "bm25": {}, "hybrid": {}}


def test_dense_retriever_searches_only_active_documents(tmp_path):
    store = _store(tmp_path)
    active = store.save_document("active.txt", ("Composite attributes are divided into subparts. " * 12).encode())
    inactive = store.save_document("inactive.txt", ("Quantum chromodynamics is a gauge theory. " * 12).encode())
    store.set_active(inactive["doc_id"], False)

    retr = store.build_retriever(method="dense", retrieval_cfg=_dense_cfg())
    assert retr.method == "dense"
    hits = retr.retrieve("composite attributes subparts", top_k=5)
    assert hits
    assert all(h.doc_id.startswith(active["doc_id"]) for h in hits)
    meta = retr.source_meta(hits[0])
    assert meta["filename"] == "active.txt"


def test_hybrid_retriever_searches_only_active_documents(tmp_path):
    store = _store(tmp_path)
    active = store.save_document("active.txt", ("A weak entity set uses a discriminator. " * 12).encode())
    inactive = store.save_document("other.txt", ("Boiling pasta makes it tender. " * 12).encode())
    store.set_active(inactive["doc_id"], False)

    retr = store.build_retriever(method="hybrid", retrieval_cfg=_dense_cfg())
    hits, _ = retr.retrieve_with_components("weak entity discriminator", top_k=4)
    assert hits
    assert all(h.doc_id.startswith(active["doc_id"]) for h in hits)
    assert retr.diagnostics()["retriever"] == "hybrid"


def test_default_store_retriever_stays_bm25(tmp_path):
    store = _store(tmp_path)
    store.save_document("a.txt", b"bm25 default content " * 20)
    retr = store.build_retriever()
    assert retr.method == "bm25"
    assert retr.diagnostics()["retriever"] == "bm25"


# ------------------------------------------------------------- chunk identity
def test_chunk_ids_are_unique_across_pages():
    pages = [
        {"page": 1, "text": "a" * 120},
        {"page": 2, "text": "b" * 120},
        {"page": 3, "text": "c" * 120},
    ]
    chunks = chunk_pages(pages, doc_id="D", target_chars=60, overlap_chars=10)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), ids
    # ids follow a document-wide counter, not a per-page one
    assert ids[0] == "D:0000"
    assert ids[-1] == f"D:{len(ids) - 1:04d}"


def test_reindex_keeps_alive_and_rewrites_unique_ids(tmp_path):
    store = _store(tmp_path)
    meta = store.save_document("doc.txt", ("Reindex me please. " * 40).encode())
    store.set_active(meta["doc_id"], False)
    updated = store.reindex(meta["doc_id"])
    assert updated is not None
    assert updated["active"] is False           # activation state preserved
    chunks = store.load_chunks(meta["doc_id"])
    assert chunks and len({c["chunk_id"] for c in chunks}) == len(chunks)


# -------------------------------------------------------------------- HTTP API
def test_branch_b_diagnostics_are_collected_after_the_query():
    """Diagnostics must describe the call that happened (model loaded, cache used)."""
    from adaptive_qa.live.serve import ServeFlow

    class DiagRetriever:
        method = "dense"

        def __init__(self):
            self.queried = False

        def retrieve(self, q, top_k):
            self.queried = True
            return [RetrievalHit("c1", "t", "passage", 0.5, method="dense")]

        def diagnostics(self):
            return {"retriever": "dense", "queried": self.queried, "model_load_ms": 12.5}

    class Engine:
        def decide(self, q):
            return Decision.RETRIEVE, 0.9, {}

    class Method:
        def answer(self, q, passages=None):
            return GeneratedAnswer(text="ok", backend="fake", meta={})

    retriever = DiagRetriever()
    flow = ServeFlow(Engine(), Method(), Method(), collection_retriever=retriever)
    out = flow.run_comparison("Q?")
    diag = out["branch_b"]["retrieval_diagnostics"]
    assert diag["queried"] is True
    assert diag["model_load_ms"] == 12.5
    assert out["branch_b"]["retrieval_method"] == "dense"


def test_api_retrieval_config_lists_methods():
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps

    client = TestClient(api_main.app)
    body = client.get("/retrieval/config").json()
    assert body["methods"] == ["bm25", "dense", "hybrid"]
    assert body["default"] in body["methods"]


def test_api_rejects_unknown_retrieval_method():
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps

    client = TestClient(api_main.app)
    resp = client.post(
        "/ask?mode=compare",
        json={"question": "anything?", "retrieval_method": "faiss"},
    )
    assert resp.status_code == 400
    assert "retrieval" in resp.json()["detail"].lower() or "bm25" in resp.json()["detail"]


def test_api_passes_requested_method_to_the_flow(monkeypatch):
    """The dashboard's method selector must reach the Option B retriever."""
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps
    from adaptive_qa.live.serve import ServeFlow

    captured = {}

    class Engine:
        def decide(self, q):
            return Decision.RETRIEVE, 0.8, {"reason": "test"}

    class MethodA:
        def answer(self, q):
            return GeneratedAnswer(text="closed-book answer", backend="llm_closedbook",
                                   meta={"tokens": {"input_tokens": 5, "output_tokens": 5,
                                                    "total_tokens": 10, "source": "estimated"}})

    class MethodB:
        def answer(self, q, passages):
            assert passages, "Option B must receive retrieved passages"
            return GeneratedAnswer(text="grounded answer", backend="llm_grounded",
                                   meta={"tokens": {"input_tokens": 30, "output_tokens": 5,
                                                    "total_tokens": 35, "source": "estimated"}})

    class ScopedRetriever:
        def __init__(self, method):
            self.method = method

        def retrieve(self, q, top_k):
            return [RetrievalHit("chunk:0001", "doc.txt (page 2)", "passage text", 0.9,
                                 method=self.method)]

        def diagnostics(self):
            return {"retriever": self.method, "n_chunks": 1}

    def fake_serve_flow(retrieval_method=None):
        captured["method"] = retrieval_method
        return ServeFlow(
            decision_engine=Engine(),
            method_a=MethodA(),
            method_b=MethodB(),
            collection_retriever=ScopedRetriever(retrieval_method or "bm25"),
        )

    monkeypatch.setattr(deps, "_serve_flow", fake_serve_flow)
    client = TestClient(api_main.app)

    resp = client.post("/ask?mode=compare", json={"question": "Q?", "retrieval_method": "hybrid"})
    assert resp.status_code == 200
    body = resp.json()
    assert captured["method"] == "hybrid"
    assert body["branch_b"]["retrieval_method"] == "hybrid"
    assert body["branch_b"]["retrieval_diagnostics"]["retriever"] == "hybrid"
    assert body["branch_b"]["retrieval_hits"][0]["retrieval_method"] == "hybrid"
    assert body["branch_a"]["performed_retrieval"] is False
    assert body["decision"]["decision"] == "RETRIEVE"
    assert body["selected_branch"] == "B"
    # both answers are visible in comparison mode
    assert body["branch_a"]["text"] == "closed-book answer"
    assert body["branch_b"]["text"] == "grounded answer"
