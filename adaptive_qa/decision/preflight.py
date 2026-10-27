"""Cheap document-relevance preflight used as a decision signal.

Why this module exists
----------------------
The pre-retrieval decision engine originally had only two inputs: surface
features of the question and (optionally) a "does the model already know
this" familiarity probe. In the live dashboard flow no familiarity prober is
configured, so the ``memory_uncertainty`` term was a constant 1.0 and the
score collapsed to a fixed value (0.500) for every uploaded-document
question. That is a placeholder, not a decision.

This preflight supplies the missing, genuinely informative signal: *do the
active documents contain the question's terminology at all?* It performs one
lexical (BM25) scoring pass over the ACTIVE chunk collection and returns
AGGREGATE SIGNALS ONLY:

- ``corpus_coverage`` : fraction of the question's content terms that occur
  anywhere in the collection.
- ``topk_coverage``   : fraction of the question's content terms that occur in
  the union of the top-k scored chunks (are the matching terms concentrated
  in a few passages, or scattered noise?).
- ``top1_coverage``   : the same for the single best-scoring chunk.
- ``collection_relevance`` = 0.50*corpus_coverage + 0.35*topk_coverage
  + 0.15*top1_coverage  — a documented weighted mean in [0, 1].

Coverage is measured WITHOUT idf weighting on purpose: an unweighted fraction
is directly interpretable ("2 of 3 content words occur in the document"), and
idf weighting lets a single out-of-vocabulary word dominate the score (a
question that says "explain ... dbms" would be punished just for using a
synonym the slides do not use).

Wh-words and instruction verbs (``how``, ``many``, ``explain``, ...) carry no
topical content and are filtered from the term set (documented constant
``QUESTION_WORDS``). No loading of labels, answers or retrieved passages
happens here, and neither passage text nor chunk ids leave the class: the
preflight can never become Option B's evidence, which always comes from the
user-selected retrieval method.

Cost
----
Building the BM25 index over the active chunks is a one-off per collection
snapshot (measured as ``build_ms``, ``cold=True`` on the first probe after a
collection change). A probe itself is a single sparse scoring pass; its
measured ``ms`` is reported and included in the decision time, so the
dashboard never hides the price of looking at the collection before deciding.

Design note (honesty)
---------------------
This IS a bounded lexical look at the collection *before* the decision is
finalised. It is documented as a preflight, kept cheap and non-generative, and
its measured cost is part of the reported adaptive latency. It gives Option A
no advantage: the closed-book branch still receives only the question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..retrieval.bm25 import BM25Retriever, tokenize
from ..timeutil import now_ns
from ..types import CorpusDoc

# Signal combination weights (documented, fixed, not tuned per question).
CORPUS_WEIGHT = 0.50
TOPK_WEIGHT = 0.35
TOP1_WEIGHT = 0.15

# Question words / instruction verbs without topical content. Kept small and
# explicit so the coverage numbers stay interpretable.
QUESTION_WORDS = frozenset(
    """
    how why many much explain describe define tell list name give mention say
    discuss state outline show identify
    """.split()
)


def content_terms(question: str) -> List[str]:
    """Question content terms used by the preflight coverage signals."""
    return sorted({t for t in tokenize(question) if t not in QUESTION_WORDS})


@dataclass
class PreflightResult:
    """Aggregate relevance signal for one question against one collection."""

    signals: Dict[str, float]
    collection: Dict[str, Any]
    ms: float
    cold: bool
    build_ms: Optional[float] = None

    @property
    def relevance(self) -> float:
        return float(self.signals.get("collection_relevance", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": True,
            "ms": self.ms,
            "cold": self.cold,
            "build_ms": self.build_ms,
            "signals": dict(self.signals),
            "collection": dict(self.collection),
            "note": (
                "Aggregate lexical relevance of the active collection (one BM25 "
                "scoring pass). No passage text or chunk id leaves the preflight, "
                "so it can never become Option B's evidence."
            ),
        }


class CollectionPreflight:
    """Sparse, non-generative relevance probe over the ACTIVE chunks."""

    def __init__(
        self,
        docs: Sequence[CorpusDoc],
        top_k: int = 5,
        doc_names: Optional[Iterable[str]] = None,
        build_ms: Optional[float] = None,
    ) -> None:
        self.docs: List[CorpusDoc] = list(docs)
        self.top_k = max(1, int(top_k))
        t0 = now_ns()
        self._bm25 = BM25Retriever(self.docs) if self.docs else None
        self._build_ms = (
            float(build_ms) if build_ms is not None else (now_ns() - t0) / 1_000_000.0
        )
        self._vocab: set = set()
        for d in self.docs:
            self._vocab |= set(tokenize(d.text)) | set(tokenize(d.title))
        self._n_docs = len(self.docs)
        self._names = list(doc_names or [])
        self.probes = 0

    # ---------------------------------------------------------------- probe
    def probe(self, question: str) -> PreflightResult:
        """Score ``question`` against the collection; aggregate signals only."""
        t0 = now_ns()
        terms = content_terms(question)
        cold = self.probes == 0
        self.probes += 1

        if not self.docs or not terms:
            ms = round((now_ns() - t0) / 1_000_000.0, 3)
            return PreflightResult(
                signals={
                    "corpus_coverage": 0.0,
                    "topk_coverage": 0.0,
                    "top1_coverage": 0.0,
                    "collection_relevance": 0.0,
                },
                collection=self._collection_info(0, terms),
                ms=ms,
                cold=cold,
                build_ms=round(self._build_ms, 3) if cold else None,
            )

        n = float(len(terms))
        corpus_hits = [t for t in terms if t in self._vocab]
        corpus_cov = len(corpus_hits) / n

        hits = self._bm25.retrieve(question, self.top_k) if self._bm25 is not None else []
        topk_terms: set = set()
        for h in hits:
            topk_terms |= set(tokenize(h.text))
        top1_terms = set(tokenize(hits[0].text)) if hits else set()
        topk_cov = len([t for t in terms if t in topk_terms]) / n
        top1_cov = len([t for t in terms if t in top1_terms]) / n

        relevance = (
            CORPUS_WEIGHT * corpus_cov
            + TOPK_WEIGHT * topk_cov
            + TOP1_WEIGHT * top1_cov
        )
        ms = round((now_ns() - t0) / 1_000_000.0, 3)
        return PreflightResult(
            signals={
                "corpus_coverage": round(corpus_cov, 4),
                "topk_coverage": round(topk_cov, 4),
                "top1_coverage": round(top1_cov, 4),
                "collection_relevance": round(max(0.0, min(1.0, relevance)), 4),
                "n_question_terms": float(len(terms)),
                "n_terms_in_collection": float(len(corpus_hits)),
                "n_chunks_scored": float(len(hits)),
            },
            collection=self._collection_info(len(hits), terms),
            ms=ms,
            cold=cold,
            build_ms=round(self._build_ms, 3) if cold else None,
        )

    def _collection_info(self, n_scored: int, terms: Sequence[str] = ()) -> Dict[str, Any]:
        return {
            "scope": "active_uploaded_documents",
            "n_chunks": self._n_docs,
            "documents": self._names,
            "top_k_probed": self.top_k,
            "n_chunks_scored": n_scored,
            "lexical": "bm25",
            "question_terms": list(terms),
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "prober": "collection_preflight",
            "n_chunks": self._n_docs,
            "top_k": self.top_k,
            "build_ms": round(self._build_ms, 3),
            "probes": self.probes,
        }
