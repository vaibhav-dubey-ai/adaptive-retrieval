"""Dense retriever tests.

No network and no model download: a deterministic hashed bag-of-words embedder
is injected through the documented ``embedder`` parameter. One test additionally
proves the real-model code path fails loudly (rather than silently falling back
to BM25) when ``sentence-transformers`` is missing.
"""

from __future__ import annotations

import builtins
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.retrieval.dense import (
    DenseRetriever,
    EmbeddingCache,
    EmbeddingModelError,
    cache_key,
    cosine_similarity,
    l2_normalize,
)
from adaptive_qa.types import CorpusDoc


class FakeEmbedder:
    """Deterministic lexical embedder: identical API shape, no model weights."""

    dim = 64

    def __init__(self) -> None:
        self.calls = 0
        self.encoded_texts: list = []

    def encode(self, texts):
        self.calls += 1
        self.encoded_texts.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        return out


def _docs():
    return [
        CorpusDoc("d1", "tower", "The Eiffel Tower is located in Paris.", ()),
        CorpusDoc("d2", "physics", "Quantum chromodynamics describes the strong interaction.", ()),
        CorpusDoc("d3", "cooking", "Boil the pasta until it is tender.", ()),
    ]


# ------------------------------------------------------------------ vector math
def test_l2_normalize_rows_unit_norm():
    mat = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.isclose(np.linalg.norm(mat[0]), 1.0)
    assert np.allclose(mat[1], 0.0)  # zero rows survive without NaN


def test_cosine_similarity_matches_manual_dot_product():
    q = np.array([1.0, 0.0], dtype=np.float32)
    m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    sims = cosine_similarity(q, m)
    assert sims.shape == (3,)
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[1], 0.0)
    assert np.isclose(sims[2], 1 / np.sqrt(2))


def test_cosine_similarity_empty_matrix():
    assert cosine_similarity(np.ones(4, dtype=np.float32), np.zeros((0, 4), dtype=np.float32)).shape == (0,)


# --------------------------------------------------------------------- ranking
def test_dense_ranking_and_metadata(tmp_path):
    fake = FakeEmbedder()
    r = DenseRetriever(_docs(), embedder=fake, cache_dir=tmp_path, namespace="rank")
    hits = r.retrieve("Where is the Eiffel Tower located?", top_k=2)
    assert hits and hits[0].doc_id == "d1"
    assert hits[0].text.startswith("The Eiffel Tower")
    assert hits[0].title == "tower"
    assert hits[0].method == "dense"
    assert hits[0].score >= hits[-1].score
    assert r.dim == FakeEmbedder.dim
    assert r.n_docs == 3


def test_dense_returns_at_most_top_k_and_empty_corpus():
    fake = FakeEmbedder()
    r = DenseRetriever(_docs(), embedder=fake, use_cache=False)
    assert len(r.retrieve("tower", top_k=1)) == 1
    empty = DenseRetriever([], embedder=fake, use_cache=False)
    assert empty.retrieve("anything", top_k=5) == []
    # blank query never invents hits
    assert r.retrieve("   ", top_k=5) == []


def test_dense_ties_break_by_doc_id():
    class ConstantEmbedder:
        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    docs = [CorpusDoc("b", "t", "same text", ()), CorpusDoc("a", "t", "same text", ())]
    hits = DenseRetriever(docs, embedder=ConstantEmbedder(), use_cache=False).retrieve("q", 2)
    assert [h.doc_id for h in hits] == ["a", "b"]


# ------------------------------------------------------------------ the cache
def test_embedding_cache_written_and_reused(tmp_path):
    fake = FakeEmbedder()
    d = _docs()
    first = DenseRetriever(d, embedder=fake, cache_dir=tmp_path, namespace="c1")
    assert fake.calls == 1
    assert first.cache_reused == 0
    cache_file = tmp_path / "sentence-transformers__all-MiniLM-L6-v2" / "c1.npz"
    assert cache_file.exists()

    fake2 = FakeEmbedder()
    second = DenseRetriever(d, embedder=fake2, cache_dir=tmp_path, namespace="c1")
    assert fake2.calls == 0, "unchanged chunks must not be re-encoded"
    assert second.cache_reused == len(d)
    # identical vectors to the first pass
    assert np.allclose(second._matrix, first._matrix)


def test_cache_invalidated_when_chunk_text_changes(tmp_path):
    fake = FakeEmbedder()
    DenseRetriever(_docs(), embedder=fake, cache_dir=tmp_path, namespace="c2")
    changed = [
        CorpusDoc("d1", "tower", "The Eiffel Tower stands in Paris, France.", ()),
        CorpusDoc("d2", "physics", "Quantum chromodynamics describes the strong interaction.", ()),
    ]
    fake2 = FakeEmbedder()
    r2 = DenseRetriever(changed, embedder=fake2, cache_dir=tmp_path, namespace="c2")
    assert fake2.calls == 1
    # only the changed chunk was encoded; the untouched one came from cache
    assert fake2.encoded_texts == ["tower\nThe Eiffel Tower stands in Paris, France."]
    assert r2.cache_reused == 1


def test_cache_key_depends_on_model_and_title_flag():
    a = cache_key("m1", "text", True)
    assert a != cache_key("m2", "text", True)
    assert a != cache_key("m1", "text", False)
    assert a == cache_key("m1", "text", True)


def test_corrupt_cache_is_ignored_not_fatal(tmp_path):
    fake = FakeEmbedder()
    r = DenseRetriever(_docs(), embedder=fake, cache_dir=tmp_path, namespace="bad")
    r.cache.path.write_bytes(b"not an npz file")
    fake2 = FakeEmbedder()
    r2 = DenseRetriever(_docs(), embedder=fake2, cache_dir=tmp_path, namespace="bad")
    assert fake2.calls == 1  # re-encoded instead of crashing
    assert len(r2.retrieve("tower", 1)) == 1


def test_cache_stats_report_entries(tmp_path):
    fake = FakeEmbedder()
    r = DenseRetriever(_docs(), embedder=fake, cache_dir=tmp_path, namespace="stats")
    stats = EmbeddingCache(tmp_path, namespace="stats").stats()
    assert stats["entries"] == 3
    assert r.diagnostics()["retriever"] == "dense"


# ------------------------------------------------------- missing model handling
def test_missing_sentence_transformers_raises_instead_of_fallback(monkeypatch):
    """A missing model must NOT silently degrade to lexical retrieval."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("sentence-transformers is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbeddingModelError) as ei:
        DenseRetriever(_docs(), use_cache=False)
    assert "sentence-transformers" in str(ei.value)


def test_unusable_model_raises_clear_error(monkeypatch):
    class BrokenModel:
        def __init__(self, *a, **k):
            raise RuntimeError("cannot download weights")

    import types

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = BrokenModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(EmbeddingModelError) as ei:
        DenseRetriever(_docs(), model_name="some/model", use_cache=False)
    assert "some/model" in str(ei.value)
