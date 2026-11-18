"""Method A: closed-book answering with simulated parametric memory.

Honest role of this engine (see README § Limitations): it is NOT a
neural LLM. It emulates an LLM's closed-book behavior deterministically:
- it "knows" only facts present in its parametric store (built from disjoint
  HotpotQA train rows - the model's "pretraining data");
- it abstains rather than inventing when its memory gives weak evidence;
- identical input -> identical output (reproducibility requirement).

Its knowledge is deliberately imperfect: retrieval is NOT always the better
choice, which is what makes the adaptive decision problem non-trivial.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from ..types import GeneratedAnswer, RetrievalHit
from ..config import resolve_path
from . import AnsweringMethod
from .span_extraction import extract_answer_span, extract_option_candidates, question_type
from .parametric_store import load_parametric_store

ABSTAIN_TEXT = "I don't know (abstained from answering from memory)."


class FiniteEngine(AnsweringMethod):
    name = "finite_engine"

    def __init__(
        self,
        store: ParametricStore,
        knowledge_threshold: float = 0.5,
        relation_threshold: float = 0.35,
        min_title_frequency: int = 2,
    ) -> None:
        """``knowledge_threshold`` = minimum title-anchor coverage the store
        must show for the question's entity; ``relation_threshold`` = minimum
        coverage of the question's relation tokens by the chosen memory
        sentence. Sound like what a familiarity-controlled decision would do:
        the engine answers only when entity memory AND compatible relation
        memory both fire."""
        self.store = store
        self.knowledge_threshold = float(knowledge_threshold)
        self.relation_threshold = float(relation_threshold)
        self.min_title_frequency = int(min_title_frequency)

    def answer(self, question: str, passages: Optional[Sequence[RetrievalHit]] = None) -> GeneratedAnswer:
        if passages is not None:
            raise ValueError(
                "Method A (closed-book) must not receive retrieved passages. "
                "Call answer(question) with passages=None."
            )
        cands = self.store.top_snippets(question, top_k=12)

        # Either/or questions: closed-book answer = the option whose memory
        # anchor is better; the question already names both possibilities, so
        # no corpus access is involved (parametric-memory disambiguation).
        qtype = question_type(question)
        options = extract_option_candidates(question)
        if len(options) == 2 and qtype != "comparison":
            ranked = self.store.rank_options(options)
            best = ranked[0]
            worst = ranked[1] if len(ranked) > 1 else None
            confident = (
                best["title_cov"] >= self.knowledge_threshold
                and best["rel_cov"] >= self.relation_threshold
                and (worst is None or best["title_cov"] > worst["title_cov"])
            )
            if confident:
                return GeneratedAnswer(
                    text=best["option"],
                    abstained=False,
                    backend=self.name,
                    meta={
                        "strategy": "memory_option_ranking",
                        "memory_title_cov": best["title_cov"],
                        "memory_rel_cov": best["rel_cov"],
                    },
                )
            return GeneratedAnswer(
                text=ABSTAIN_TEXT,
                abstained=True,
                backend=self.name,
                meta={"strategy": "memory_option_ranking", "ranked": ranked},
            )

        if not cands:
            return GeneratedAnswer(text=ABSTAIN_TEXT, abstained=True, backend=self.name)
        title, sentence, title_cov, rel_cov = cands[0]
        title_freq = sum(1 for t, _s, _tc, _rc in cands if t == title)
        if (
            title_cov < self.knowledge_threshold
            or rel_cov < self.relation_threshold
            or title_freq < self.min_title_frequency
        ):
            return GeneratedAnswer(
                text=ABSTAIN_TEXT,
                abstained=True,
                backend=self.name,
                meta={
                    "memory_title_cov": round(title_cov, 4),
                    "memory_rel_cov": round(rel_cov, 4),
                    "title_freq": title_freq,
                },
            )
        span = extract_answer_span(question, sentence)
        if span is None:
            return GeneratedAnswer(
                text=ABSTAIN_TEXT,
                abstained=True,
                backend=self.name,
                meta={
                    "memory_title_cov": round(title_cov, 4),
                    "memory_rel_cov": round(rel_cov, 4),
                    "extraction": None,
                },
            )
        return GeneratedAnswer(
            text=span,
            abstained=False,
            backend=self.name,
            meta={
                "memory_title": title,
                "memory_sentence": sentence,
                "memory_title_cov": round(title_cov, 4),
                "memory_rel_cov": round(rel_cov, 4),
            },
        )


def build_finite_engine(cfg: dict, store: Optional[ParametricStore] = None) -> FiniteEngine:
    sub = cfg["answering"]["finite_engine"]
    if store is None:
        path = cfg["parametric_store"]["jsonl_path"]
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"parametric store not found at {resolved}. "
                "Run: python scripts/build_dataset.py (it builds data + store)."
            )
        store = load_parametric_store(str(resolved))
    return FiniteEngine(
        store,
        knowledge_threshold=sub.get("knowledge_threshold", 0.5),
        relation_threshold=sub.get("relation_threshold", 0.35),
        min_title_frequency=sub.get("min_title_frequency", 2),
    )
