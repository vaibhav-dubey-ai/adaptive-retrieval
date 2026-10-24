"""Retriever interfaces.

``BM25Retriever`` is the default. A dense-embedding retriever can be added
behind the same ``retrieve`` interface without touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from ..types import CorpusDoc, RetrievalHit


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        """Return up to ``top_k`` hits ordered by decreasing score.

        Deterministic: ties are broken by doc_id, ascending.
        """


def load_retriever(retriever_name: str, docs: Sequence[CorpusDoc], **kwargs) -> Retriever:
    from .bm25 import BM25Retriever

    if retriever_name == "bm25":
        return BM25Retriever(docs, **kwargs)
    if retriever_name == "dense":
        from .dense import DenseRetriever

        return DenseRetriever(docs, **kwargs)
    raise ValueError(f"unknown retriever {retriever_name!r}")
