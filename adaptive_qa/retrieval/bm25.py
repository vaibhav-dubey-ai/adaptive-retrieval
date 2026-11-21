"""BM25 (Okapi) retrieval implemented in-repo.

Why a hand-written BM25 instead of ``rank_bm25``: it keeps the stack minimal on
Python 3.13, gives us deterministic tie-breaking and exact control for unit
tests. Scoring is the standard Robertson-Sparck-Jones formulation with the
classic k1/b parameters.

Determinism: identical corpus + query always yields identical hits in
identical order (ties broken by ``doc_id`` ascending).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Tuple

from . import Retriever
from ..common import CorpusDoc, RetrievalHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, standard English stopword list (kept constant and in-repo so result
# artifacts carry no hidden external dependency).
STOPWORDS = frozenset(
    "a an the and or but if then else when while is are was were be been being "
    "of in on at to for by with from as into about over after under before "
    "between out up down off above below what which who whom whose this that "
    "these those there here it its his her their our your my do does did done "
    "can could should would will shall may might must have has had also very "
    "s t d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma "
    "mightn mustn needn shan shouldn wasn weren won wouldn".split()
)


def tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


class BM25Retriever(Retriever):
    def __init__(
        self,
        docs: Sequence[CorpusDoc],
        k1: float = 1.5,
        b: float = 0.75,
        title_boost: float = 2.0,
    ) -> None:
        """``title_boost`` repeats title tokens so a paragraph's ``title``
        (e.g. Wikipedia article name) contributes like short-field BM25.
        """
        self.k1 = float(k1)
        self.b = float(b)
        self.title_boost = float(title_boost)
        self.docs: List[CorpusDoc] = list(docs)
        self._postings: Dict[str, Dict[int, int]] = {}
        self._doc_lens: List[int] = []
        self._avgdl = 0.0
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        doc_len_total = 0
        for idx, doc in enumerate(self.docs):
            tokens = tokenize(doc.text) + tokenize(doc.title) * int(self.title_boost)
            doc_len_total += len(tokens)
            self._doc_lens.append(len(tokens))
            counts: Dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for term, tf in counts.items():
                bucket = self._postings.get(term)
                if bucket is None:
                    bucket = {}
                    self._postings[term] = bucket
                bucket[idx] = tf
        n = max(1, len(self.docs))
        self._avgdl = doc_len_total / n

    # ----------------------------------------------------------------- search
    def retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        tokens = tokenize(query)
        term_tfs = {}
        for tok in tokens:
            term_tfs[tok] = term_tfs.get(tok, 0) + 1

        scores: Dict[int, float] = {}
        n = len(self.docs)
        for term, qtf in term_tfs.items():
            bucket = self._postings.get(term)
            if not bucket:
                continue
            df = len(bucket)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in bucket.items():
                dl = self._doc_lens[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self._avgdl))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1) / denom)
                _ = qtf  # BM25 (not BM25+); query term frequency weight is 1

        order = sorted(scores.items(), key=lambda kv: (-kv[1], self.docs[kv[0]].doc_id))
        hits: List[RetrievalHit] = []
        for idx, score in order[:top_k]:
            doc = self.docs[idx]
            hits.append(
                RetrievalHit(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=score,
                )
            )
        return hits
