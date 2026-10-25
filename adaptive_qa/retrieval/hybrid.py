"""Hybrid retrieval: BM25 (lexical) + dense (semantic) fused with RRF.

Reciprocal Rank Fusion (Cormack et al., 2009):

    RRF(d) = sum_i  1 / (k + rank_i(d))

with ``k = 60`` (the value used in the original paper and in most IR
pipelines) and ``rank_i(d)`` the 1-based rank of chunk ``d`` in ranking ``i``.

Only *ranks* are fused: raw BM25 scores and cosine similarities live on
incomparable scales, so adding or averaging the raw scores would be arbitrary.
A chunk that appears in only one ranking simply contributes one term from that
ranking, and chunks are deduplicated by their stable ``doc_id`` (the chunk id).
Component rankings and scores stay available for debugging and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import Retriever
from .bm25 import BM25Retriever
from .dense import DEFAULT_MODEL_NAME, DenseRetriever, Embedder
from ..types import CorpusDoc, RetrievalHit

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 50


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K
) -> List[Tuple[str, float]]:
    """Fuse ranked id lists; returns ``[(doc_id, rrf_score)]`` best first.

    Ties break on ``doc_id`` ascending so the fusion is deterministic.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever(Retriever):
    """BM25 + dense retrieval fused with RRF over the same chunk set."""

    def __init__(
        self,
        docs: Sequence[CorpusDoc],
        k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        bm25_kwargs: Optional[Dict[str, Any]] = None,
        dense_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.docs: List[CorpusDoc] = list(docs)
        self.k = int(k)
        self.candidate_k = max(1, int(candidate_k))
        self.bm25 = BM25Retriever(self.docs, **(bm25_kwargs or {}))
        self.dense = DenseRetriever(self.docs, **{**(dense_kwargs or {})})
        self._by_id = {d.doc_id: d for d in self.docs}

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def _candidates(self, query: str, top_k: int) -> int:
        # Fuse a wider candidate pool than the requested depth, otherwise RRF
        # can only reorder the (already narrow) lexical/dense top-k.
        return max(self.candidate_k, top_k)

    def retrieve_with_components(
        self, query: str, top_k: int
    ) -> Tuple[List[RetrievalHit], Dict[str, Any]]:
        depth = self._candidates(query, top_k)
        bm25_hits = self.bm25.retrieve(query, depth)
        dense_hits = self.dense.retrieve(query, depth)

        bm25_rank = {h.doc_id: i for i, h in enumerate(bm25_hits, start=1)}
        dense_rank = {h.doc_id: i for i, h in enumerate(dense_hits, start=1)}
        bm25_score = {h.doc_id: h.score for h in bm25_hits}
        dense_score = {h.doc_id: h.score for h in dense_hits}

        fused = reciprocal_rank_fusion(
            [list(bm25_rank.keys()), list(dense_rank.keys())], k=self.k
        )

        hits: List[RetrievalHit] = []
        for doc_id, rrf in fused[: max(0, int(top_k))]:
            doc = self._by_id[doc_id]
            hits.append(
                RetrievalHit(
                    doc_id=doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=float(rrf),
                    method="hybrid",
                    component_ranks={
                        "bm25": bm25_rank.get(doc_id, 0),
                        "dense": dense_rank.get(doc_id, 0),
                    },
                    component_scores={
                        "bm25": float(bm25_score.get(doc_id, 0.0)),
                        "dense": float(dense_score.get(doc_id, 0.0)),
                    },
                )
            )
        components = {
            "bm25": [
                {"rank": i, "doc_id": h.doc_id, "score": round(float(h.score), 6)}
                for i, h in enumerate(bm25_hits, start=1)
            ],
            "dense": [
                {"rank": i, "doc_id": h.doc_id, "score": round(float(h.score), 6)}
                for i, h in enumerate(dense_hits, start=1)
            ],
            "rrf_k": self.k,
            "candidate_k": depth,
        }
        return hits, components

    def retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        hits, _ = self.retrieve_with_components(query, top_k)
        return hits

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "retriever": "hybrid",
            "rrf_k": self.k,
            "candidate_k": self.candidate_k,
            "n_documents": self.n_docs,
            "components": {"bm25": {"retriever": "bm25"}, "dense": self.dense.diagnostics()},
        }
