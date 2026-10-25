"""Hybrid (BM25 + dense) retrieval tests: RRF maths, dedupe, components."""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.retrieval.bm25 import BM25Retriever
from adaptive_qa.retrieval.dense import DenseRetriever
from adaptive_qa.retrieval.hybrid import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from adaptive_qa.types import CorpusDoc


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
        CorpusDoc("c1", "er", "Composite attributes can be divided into subparts called component attributes.", ()),
        CorpusDoc("c2", "keys", "A weak entity set uses an identifying entity and a discriminator.", ()),
        CorpusDoc("c3", "uml", "UML class diagrams correspond to ER diagrams but differ in notation.", ()),
    ]


def _hybrid(tmp_path=None, **kwargs):
    return HybridRetriever(
        _docs(),
        bm25_kwargs={},
        dense_kwargs={"embedder": FakeEmbedder(), "use_cache": False},
        **kwargs,
    )


# ---------------------------------------------------------------- RRF scoring
def test_rrf_formula_and_ordering():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    # b appears in both rankings, so it must lead; a outranks c
    assert [doc for doc, _ in fused] == ["b", "a", "c"]


def test_rrf_dedupes_a_chunk_present_in_both_rankings():
    fused = reciprocal_rank_fusion([["x", "y"], ["x", "y"]], k=60)
    assert [doc for doc, _ in fused] == ["x", "y"]
    assert len(fused) == 2


def test_rrf_single_ranking_matches_its_order():
    fused = reciprocal_rank_fusion([["p", "q", "r"]], k=60)
    assert [doc for doc, _ in fused] == ["p", "q", "r"]


def test_rrf_ties_break_by_doc_id():
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def test_default_k_is_the_documented_constant():
    assert DEFAULT_RRF_K == 60


# ------------------------------------------------------------------ retriever
def test_hybrid_returns_top_k_with_fused_scores():
    r = _hybrid()
    hits = r.retrieve("composite attributes divided into subparts", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "c1"
    assert all(h.method == "hybrid" for h in hits)
    assert hits[0].score >= hits[1].score
    assert hits[0].text.startswith("Composite attributes")


def test_hybrid_hits_carry_component_ranks_and_scores():
    r = _hybrid()
    hits, components = r.retrieve_with_components("weak entity set discriminator", top_k=3)
    top = hits[0]
    assert top.doc_id == "c2"
    assert top.component_ranks["bm25"] >= 1
    assert top.component_ranks["dense"] >= 1
    assert "bm25" in top.component_scores and "dense" in top.component_scores
    assert components["rrf_k"] == 60
    # BM25 only returns chunks sharing a query term; dense ranks the whole corpus
    assert 1 <= len(components["bm25"]) <= 3
    assert len(components["dense"]) == 3


def test_hybrid_candidate_pool_is_wider_than_top_k():
    r = _hybrid(candidate_k=50)
    hits = r.retrieve("UML class diagrams", top_k=1)
    assert len(hits) == 1
    _, components = r.retrieve_with_components("UML", top_k=1)
    # fusion ranks a wider pool than the requested depth
    assert components["candidate_k"] == 50


def test_hybrid_empty_corpus_returns_nothing():
    r = HybridRetriever([], bm25_kwargs={}, dense_kwargs={"embedder": FakeEmbedder(), "use_cache": False})
    assert r.retrieve("anything", 3) == []


def test_hybrid_fused_score_equals_rrf_of_component_ranks():
    """Every fused score must be reproducible from its component ranks.

    A chunk ranked by only one component contributes a single RRF term (the
    missing component's rank is reported as 0), which is exactly how chunks
    found by just one ranking stay in the pool.
    """
    r = _hybrid()
    hits, _ = r.retrieve_with_components("weak entity discriminator", top_k=3)
    assert len({h.doc_id for h in hits}) == len(hits)  # deduplicated by chunk id
    for h in hits:
        expected = sum(1.0 / (60 + rank) for rank in h.component_ranks.values() if rank)
        assert h.score == pytest.approx(expected)


def test_hybrid_diagnostics_expose_both_components():
    diag = _hybrid().diagnostics()
    assert diag["retriever"] == "hybrid"
    assert diag["rrf_k"] == 60
    assert diag["components"]["dense"]["embedding_dim"] == FakeEmbedder.dim
    assert diag["components"]["bm25"]["retriever"] == "bm25"


def test_hybrid_is_deterministic():
    r = _hybrid()
    first = [(h.doc_id, round(h.score, 8)) for h in r.retrieve("ER diagram notation", 3)]
    second = [(h.doc_id, round(h.score, 8)) for h in r.retrieve("ER diagram notation", 3)]
    assert first == second
