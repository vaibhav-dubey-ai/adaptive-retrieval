"""Retriever interfaces and factory.

Three interchangeable retrievers share one ``retrieve(query, top_k)`` contract:

- ``bm25``   — lexical Okapi BM25 over the chunk corpus (baseline, in-repo).
- ``dense``  — semantic cosine similarity over pretrained sentence embeddings.
- ``hybrid`` — both rankings fused with Reciprocal Rank Fusion (k = 60).

Selection is configuration-driven (``retrieval.retriever``) and can be
overridden per request; ``bm25`` stays the default so existing requests and
saved experiment runs keep their original behaviour.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import CorpusDoc, RetrievalHit

RETRIEVAL_METHODS: Tuple[str, ...] = ("bm25", "dense", "hybrid")
DEFAULT_METHOD = "bm25"


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        """Return up to ``top_k`` hits ordered by decreasing score.

        Deterministic: ties are broken by doc_id, ascending.
        """


def normalize_method(name: Optional[str]) -> str:
    method = (name or DEFAULT_METHOD).strip().lower()
    if method not in RETRIEVAL_METHODS:
        raise ValueError(
            f"unknown retrieval method {name!r}; expected one of {', '.join(RETRIEVAL_METHODS)}"
        )
    return method


def load_retriever(
    retriever_name: str,
    docs: Sequence[CorpusDoc],
    bm25_kwargs: Optional[Dict[str, Any]] = None,
    dense_kwargs: Optional[Dict[str, Any]] = None,
    hybrid_kwargs: Optional[Dict[str, Any]] = None,
) -> Retriever:
    """Build a retriever over ``docs``.

    Keyword groups are kept separate so a BM25 parameter (``k1``/``b``) can
    never leak into the dense retriever's constructor and vice versa.
    """
    method = normalize_method(retriever_name)
    if method == "bm25":
        from .bm25 import BM25Retriever

        return BM25Retriever(docs, **(bm25_kwargs or {}))
    if method == "dense":
        from .dense import DenseRetriever

        return DenseRetriever(docs, **(dense_kwargs or {}))
    from .hybrid import HybridRetriever

    merged = dict(hybrid_kwargs or {})
    merged.setdefault("bm25_kwargs", bm25_kwargs or {})
    merged.setdefault("dense_kwargs", dense_kwargs or {})
    return HybridRetriever(docs, **merged)


def build_retriever_from_config(
    cfg: Dict[str, Any], docs: Sequence[CorpusDoc], method: Optional[str] = None
) -> Retriever:
    """Build the retriever described by a project config (optionally overridden)."""
    rcfg = cfg.get("retrieval", {}) or {}
    bm25_kwargs = dict(rcfg.get("bm25", {}) or {})
    dense_kwargs = dict(rcfg.get("dense", {}) or {})
    hybrid_kwargs = dict(rcfg.get("hybrid", {}) or {})
    return load_retriever(
        method or rcfg.get("retriever", DEFAULT_METHOD),
        docs,
        bm25_kwargs=bm25_kwargs,
        dense_kwargs=dense_kwargs,
        hybrid_kwargs=hybrid_kwargs,
    )
