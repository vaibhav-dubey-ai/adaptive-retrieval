"""Method B: grounded answer generation over retrieved passages.

The deterministic ``span_absorber`` backend reads the top-k retrieved passages
and selects the best sentence, then a short span within it. It abstains when
retrieval returned nothing usable - grounded answering is forbidden from
answering "from memory".

Optional ``openai_compatible`` backend does the same job through a hosted LLM.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..retrieval.bm25 import tokenize
from ..types import GeneratedAnswer, RetrievalHit
from . import AnsweringMethod
from .span_extraction import extract_answer_span, extract_option_candidates, question_type

GROUND_ABSTAIN_TEXT = "I could not find the answer in the provided passages."


def _idf(df: int, n_docs: int) -> float:
    """Smoothed, BOUNDED IDF within the retrieved passage set.

    Bounded to [1, 2] for present tokens: with only k retrieved docs, raw IDF
    lets once-appearing noise tokens dominate sentence scoring (measured
    failure); the bound keeps rare-token weighting directionally useful.
    Tokens not present in any passage get a tiny constant weight.
    """
    if n_docs <= 0:
        return 0.0
    if df <= 0:
        return 0.05
    return 1.0 + min(math.log(n_docs / df), 1.0)


class SpanAbsorber(AnsweringMethod):
    name = "span_absorber"

    def __init__(
        self,
        cfg: dict,
    ) -> None:
        self.min_sentence_score = float(cfg.get("min_sentence_score", 0.35))
        self.max_context_chars = int(cfg.get("max_context_chars", 6000))

    def answer(self, question: str, passages: Optional[Sequence[RetrievalHit]] = None) -> GeneratedAnswer:
        if not passages:
            return GeneratedAnswer(
                text=GROUND_ABSTAIN_TEXT, abstained=True, evidence_doc_ids=(), backend=self.name
            )

        # Either/or questions: grounded answering selects the option whose
        # evidence is better represented in the retrieved passages, choosing
        # SPANS the retrieval actually returned - no memory shortcuts.
        options = extract_option_candidates(question)
        if len(options) == 2 and question_type(question) != "comparison":
            ranked = self._rank_options_by_passages(question, options, passages)
            if ranked is not None:
                best = ranked[0]
                other = ranked[1] if len(ranked) > 1 else None
                if (
                    best["title_cov"] >= 0.5
                    and best["rel_cov"] >= self.min_sentence_score
                    and (other is None or best["title_cov"] > other["title_cov"])
                ):
                    return GeneratedAnswer(
                        text=best["option"],
                        abstained=False,
                        evidence_doc_ids=tuple(h.doc_id for h in passages),
                        backend=self.name,
                        meta={"strategy": "passage_option_ranking", **best},
                    )
                return GeneratedAnswer(
                    text=GROUND_ABSTAIN_TEXT,
                    abstained=True,
                    evidence_doc_ids=tuple(h.doc_id for h in passages),
                    backend=self.name,
                    meta={"strategy": "passage_option_ranking", "ranked": ranked},
                )

        qt = set(tokenize(question))
        # IDF over the retrieved set: tokens rare across the returned passages
        # carry answer evidence; shared-entity distractor noise is common.
        n_docs = max(1, len(passages))
        doc_freq: dict = {}
        for hit in passages:
            for tok in set(tokenize(hit.text)):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        title_cov_cache = {}
        best = None
        for rank, hit in enumerate(passages):
            title_toks = set(tokenize(hit.title))
            title_cov_cache[rank] = (
                len(title_toks & qt) / len(title_toks) if title_toks else 0.0
            )
            for s_i, sentence in enumerate(hit.text.split(". ")):
                sentence = sentence.strip()
                if not sentence:
                    continue
                stoks = set(tokenize(sentence)) | set(tokenize(hit.title))
                inter = qt & stoks
                w_inter = sum(_idf(doc_freq.get(t, 0), n_docs) for t in inter)
                w_q = sum(_idf(doc_freq.get(t, 0), n_docs) for t in qt)
                w_s = sum(_idf(doc_freq.get(t, 0), n_docs) for t in stoks)
                denom = w_q + w_s
                idf_f1 = 2.0 * w_inter / denom if denom else 0.0
                # Subject-passage bonus: the passage whose TITLE the question
                # names is the question's first hop (HotpotQA gold structure).
                # Kept SMALL: multi-hop gold docs (hop 2) share few title
                # tokens with the question, so a large bonus ranks biography-
                # style first-hop docs above the sentence that contains the
                # answer (measured failure; documented).
                subject_bonus = 0.12 * title_cov_cache[rank]
                # slight preference for highly-ranked passages
                rank_bonus = 0.08 / (1.0 + rank)
                score = 0.80 * idf_f1 + subject_bonus + rank_bonus
                if best is None or score > best[0]:
                    best = (score, sentence, hit, rank, s_i)
        if best is None or best[0] < self.min_sentence_score:
            return GeneratedAnswer(
                text=GROUND_ABSTAIN_TEXT,
                abstained=True,
                evidence_doc_ids=tuple(h.doc_id for h in passages),
                backend=self.name,
                meta={"best_sentence": best[1] if best else None,
                      "best_score": round(best[0], 4) if best else None},
            )
        score, sentence, hit, rank, s_i = best
        span = extract_answer_span(question, sentence)
        if span is None:
            return GeneratedAnswer(
                text=GROUND_ABSTAIN_TEXT,
                abstained=True,
                evidence_doc_ids=(hit.doc_id,),
                backend=self.name,
                meta={"best_sentence": sentence, "best_doc_rank": rank, "sentence_f1": round(score, 4)},
            )
        return GeneratedAnswer(
            text=span,
            abstained=False,
            evidence_doc_ids=(hit.doc_id,),
            backend=self.name,
            meta={
                "best_sentence": sentence,
                "best_doc_rank": rank,
                "sentence_f1": round(score, 4),
            },
        )

    def _rank_options_by_passages(self, question: str, options, passages):
        """Rank either/or options against retrieved passage TITLES and text.

        ``title_cov`` = overlap of option tokens with the union of retrieved
        titles; ``rel_cov`` = best coverage of option-relative tokens by a
        retrieved sentence. Deterministic, retrieval-only."""
        out = []
        for option in options:
            otoks = set(tokenize(option))
            title_union = set()
            for hit in passages:
                title_union |= set(tokenize(hit.title))
            title_cov = len(otoks & title_union) / max(1, len(otoks))
            rel_ctoks = otoks - title_union
            rel_cov = 0.0
            if not rel_ctoks:
                rel_cov = 1.0
            else:
                for hit in passages:
                    for sentence in hit.text.split(". "):
                        stoks = set(tokenize(sentence))
                        if stoks:
                            rel_cov = max(rel_cov, len(rel_ctoks & stoks) / len(rel_ctoks))
            out.append({
                "option": option,
                "exact_title": any(
                    set(tokenize(hit.title)) == otoks for hit in passages
                ),
                "title_cov": round(title_cov, 4),
                "rel_cov": round(rel_cov, 4),
            })
        out.sort(key=lambda d: (-(1 if d["exact_title"] else 0), -d["title_cov"], -d["rel_cov"]))
        return out
