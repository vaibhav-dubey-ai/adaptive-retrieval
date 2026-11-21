"""Offline, deterministic answerers used by the HotpotQA experiment (no network, no cost).

- span_extraction: shared answer-span heuristics
- ParametricStore: simulated model memory
- FiniteEngine: Method A (closed-book)
- SpanAbsorber: Method B (grounded extraction)
"""

from __future__ import annotations

import json
import math
import os
import re
from . import AnsweringMethod
from ..common import GeneratedAnswer, RetrievalHit
from ..common import resolve_path
from ..retrieval.bm25 import tokenize
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from typing import List, Optional
from typing import Optional, Sequence


# ======================================================================
# Deterministic surface-answer-span extraction.
# ======================================================================

_YEAR_RE = re.compile(r"\b(1[4-9]\d{2}|20[0-4]\d)\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(1[4-9]\d{2}|20[0-4]\d)\b",
    re.IGNORECASE,
)
_CAPITALIZED_SPAN_RE = re.compile(
    r"\b([A-Z][a-z']+(?:\s+(?:of|the|de|II|III|IV|V|\d))*(?:\s+[A-Z][A-Za-z']+)+)\b"
)
_SIMPLE_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)*\b")
_QUESTION_HINT_RE = re.compile(
    r"(?:what group|what genre|what movement|what era|which genre|movement as|styled|"
    r"known as|called|description)"
)
_LOCATED_RE = re.compile(
    r"\b(?:born|founded|formed|established|located|situated|based|resides)\b"
    r"[^,\.0-9]{0,60}?\b(?:in|at|near)\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)"
    r"(?=[,\.]|\s+(?:and|the)\b|\s*$)",
)


def question_type(question: str) -> str:
    q = question.strip().lower()
    if re.match(r"^(are both|is both|do both|were both|were either|are either|are all)", q):
        return "comparison"
    if re.match(r"^who\b", q):
        return "who"
    if re.match(r"^when\b", q) or "what year" in q:
        return "when"
    if re.match(r"^where\b", q) or q.startswith("in what"):
        return "where"
    if re.match(r"^(what|which)\b", q):
        return "what"
    if re.match(r"^how (many|much|long|far)", q):
        return "quantity"
    return "other"


def extract_answer_span(question: str, sentence: str) -> Optional[str]:
    """Extract a short answer span from ``sentence`` conditioned on the question.

    Deterministic and purely lexical. Returns None when extraction fails.
    """
    qtype = question_type(question)
    caps = _CAPITALIZED_SPAN_RE.findall(sentence) + _SIMPLE_CAPITALIZED_RE.findall(sentence)
    q_lower = question.lower()

    if qtype == "comparison":
        if sentence.strip().lower().startswith(("yes", "no")):
            return sentence.strip().lower().split()[0]
        # Not a yes/no sentence. Aggregation-style two-entity questions such
        # as "X and Y have which mutual occupations?" ARE evaluable against
        # a biography sentence, so fall through to the generic rules - but
        # only when the question lacks explicit compare markers (both/
        # either/or), which correctly stay None (measured failure cases).
        if not re.search(r"\b(?:both|either)\b|\s+or\s+", question.lower()):
            qtype = "other"

    if qtype == "when":
        m = _MONTH_YEAR_RE.search(sentence)
        if m:
            return m.group(0)
        years = _YEAR_RE.findall(sentence)
        if years:
            return years[0]
        return None

    if qtype == "who":
        m = re.search(
            r"\b(?:by|from)\s+([A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)*)", sentence
        )
        if m:
            return m.group(1).strip()
        # prefer a person-like span not echoed by the question
        for span in caps:
            if span.lower() not in q_lower and not span.isupper():
                return span.strip()
        return caps[0].strip() if caps else None

    if qtype == "where":
        m = _LOCATED_RE.search(sentence)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bin\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)", sentence)
        if m:
            return m.group(1).strip()
        carried = [s for s in caps if s.lower() not in q_lower]
        if carried:
            return carried[-1].strip()
        return None

    if qtype == "quantity":
        nums = re.findall(r"\b\d[\d,\.]*\b", sentence)
        if nums:
            return nums[0]
        return None

    # 'what'/'other': prefer a span not repeated from the question, and prefer
    # to echo the semantic head right after a copula ("was a <Genre>", "is a Y")
    # True compare questions keep the None contract of the graceful-failure
    # test: without a yes/no sentence and with no aggregation fall-through,
    # extraction is not trustworthy.
    if qtype == "comparison":
        return None
    # Rule (b) copular-subject: HotpotQA easy gold sentences overwhelmingly
    # define the answer at sentence head ("The Maloof family is a prominent
    # ...", "When the Good Guys Win is the ninth studio album ...") - the
    # answer is the SENTENCE SUBJECT there.
    m_subject = re.match(
        r"^(?:The\s+|A\s+|An\s+)?([A-Z][A-Za-z'\-\s]+?)\s+(?:is|was|are)\s+(?:a\s|an\s|the\s|one\s)",
        sentence,
    )
    if m_subject:
        subj = m_subject.group(1).strip()
        if subj.lower() not in q_lower:
            return subj
    # Rule (c) occupation questions: the answer is the lowercase professional
    # phrase following the copula ("is an American musician, singer and ...")
    q_toks = set(q_lower.replace("?", " ").split())
    if q_toks & {"occupation", "occupations", "profession", "jobs", "job", "famous", "known"}:
        m_occ = re.search(
            r"\bis\s+(?:an?\s+)?(?:[A-Z][a-z][a-z]+\s+|American\s+)?(([a-z][a-z\-']+(?:[ ,\-']+[a-z][a-z\-']*){1,6}))",
            sentence,
        )
        if m_occ:
            return m_occ.group(1).strip().strip(",")
    if re.search(r"\b(?:is|was|are|were)\b.{0,40}\b(?:by|in)\b", sentence, re.IGNORECASE) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower and not s.isupper()]
        return (not_in_q[-1] if not_in_q else caps[-1]).strip()
    m = re.search(r"\byears?\b|\bdynasty\b|\bstation\b|\btruth\b", sentence, re.IGNORECASE)
    if _QUESTION_HINT_RE.search(sentence.lower()) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower]
        return (not_in_q[-1] if not_in_q else caps[-1]).strip()
    if re.search(r"\b(is|was|corresponds|belongs)\b", sentence, re.IGNORECASE) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower]
        if len(caps) > 1:
            span = not_in_q[-1] if not_in_q else caps[-1]
            return span.strip()
        return caps[0].strip()
    if m and caps:
        return caps[-1].strip()
    not_in_q = [s for s in caps if s.lower() not in q_lower and not s.isupper()]
    if not_in_q:
        return not_in_q[-1].strip()
    return caps[-1].strip() if caps else None


_WH_WORDS = {
    "which", "what", "who", "whom", "whose", "where", "when", "why", "how",
    "does", "did", "do", "are", "is", "was", "were", "were", "in", "both",
    "either", "all", "a", "an", "or", "and", "the", "of",
}


def extract_option_candidates(question: str) -> List[str]:
    """Extract answer options from an either/or (or both-and) question.

    Handles the two common HotpotQA shapes:
    - "..., <Option A> or <Option B>?"
    - "both <Option A> and <Option B>" (option form only; yes/no never uses it)

    Returns up to two candidates (an empty list when the shape is absent);
    callers must decide how to rank them against their own knowledge.
    """
    q = question.strip().rstrip("?").strip()
    opts: List[str] = []

    if "," in q:
        tail = q.rsplit(",", 1)[1]
        tail = re.sub(r"\s*\bor\b\s*", "|", tail)
        for piece in tail.split("|"):
            piece = piece.strip(" .")
            first = piece.split(" ", 1)[0].lower()
            if piece and first not in _WH_WORDS and piece[0].isupper():
                opts.append(piece)
    if len(opts) < 2:
        m = re.search(r"\bboth\s+(.+?)\s+and\s+(.+?)(?:\s+have|\s+are|\s+is|\s+of\b|\s+from\b|\?|$)", q)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            cand_a = _most_capitalized_part(a)
            cand_b = _most_capitalized_part(b)
            opts = [c for c in (cand_a, cand_b) if c]
    # de-dup, case-insensitive, max 2
    out: List[str] = []
    for o in opts:
        if o and not any(o.lower() == x.lower() for x in out):
            out.append(o)
    return out[:2]


def _most_capitalized_part(text: str) -> Optional[str]:
    spans = _CAPITALIZED_SPAN_RE.findall(text) + _SIMPLE_CAPITALIZED_RE.findall(text)
    if not spans:
        return None
    # the option usually ends the clause; take the last capitalized span
    return spans[-1]


# ======================================================================
# Simulated LLM parametric memory for the closed-book answering engine.
# ======================================================================

def build_parametric_store(
    records: Sequence[Dict[str, Any]],
    max_snippet_tokens: int = 60,
    gold_only: bool = True,
) -> List[Dict[str, Any]]:
    """Build (title, sentence) memory snippets from raw HotpotQA-style rows.

    ``gold_only=True`` keeps only each row's supporting paragraphs: memory gets
    densely-packaged factual sentences rather than 8 noisy distractor
    paragraphs per row, which better matches what a pretraining corpus
    curates. Returns a list of {"title", "sentence"} dicts; deduplicates
    identical sentences.
    """
    snippets: List[Dict[str, Any]] = []
    seen = set()
    for rec in records:
        gold_titles = set(rec.get("gold_titles", []))
        for para in rec["paragraphs"]:
            if gold_only and para["title"] not in gold_titles:
                continue
            title = para["title"]
            for s in para["sentences"]:
                s = s.strip()
                if not s:
                    continue
                toks = s.split()
                if len(toks) > max_snippet_tokens:
                    s = " ".join(toks[:max_snippet_tokens])
                key = (title, s)
                if key in seen:
                    continue
                seen.add(key)
                snippets.append({"title": title, "sentence": s})
    return snippets


def save_parametric_store(snippets: List[Dict[str, Any]], jsonl_path: str) -> None:
    p = resolve_path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in snippets:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_parametric_store(jsonl_path: str) -> "ParametricStore":
    p = resolve_path(jsonl_path)
    snippets: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                snippets.append((obj["title"], obj["sentence"]))
    return ParametricStore(snippets)


class ParametricStore:
    """Simulated parametric memory: an inverted index over (title, sentence)
    snippets, probed with a two-stage entity-anchored match:

    1. *Title anchoring* - which memory entities (Wikipedia-style titles)
       overlap the question; emulates entity-level recall.
    2. *Relation matching* - which sentences of those entities best cover the
       question's remaining (relation-descriptive) tokens.

    Used by two components, with strictly separated responsibilities:
    - ``familiarity_probe``: cheap statistics (for the *decision engine*'s
      uncertainty feature). Reads only the question.
    - ``top_snippets``: candidate memory sentences (for the *Method A
      answering engine*). Also reads only the question.
    Neither exposes reference answers; the store contains memory sentences.
    """

    def __init__(self, snippets: List[Tuple[str, str]]) -> None:
        self.titles: List[str] = []
        self.snippets: List[str] = []
        self._postings: Dict[str, List[int]] = {}
        self._title_postings: Dict[str, List[int]] = {}
        self._title_idf: Dict[str, float] = {}
        self._snippets_of_title: Dict[str, List[int]] = {}
        self._title_tokens: List[frozenset] = []

        for idx, (title, sentence) in enumerate(snippets):
            self.titles.append(title)
            self.snippets.append(sentence)
            for tok in set(tokenize(sentence)):
                self._postings.setdefault(tok, []).append(idx)
            self._snippets_of_title.setdefault(title, []).append(idx)
            ttoks = frozenset(tokenize(title))
            self._title_tokens.append(ttoks)
            for tok in ttoks:
                self._title_postings.setdefault(tok, []).append(idx)

        n_titles = max(1, len(self._snippets_of_title))
        for title, idxs in self._snippets_of_title.items():
            self._title_idf[title] = math.log(1.0 + (n_titles / len(idxs)))

    # ------------------------------------------------------------------ probe
    def familiarity_probe(
        self,
        question: str,
        top_k: int = 12,
    ) -> dict:
        """Cheap pre-retrieval uncertainty signals - the "does the model know
        it" style feature (cardinality of entity memory). Reads only the
        question.

        Returns:
        ``best_title_cov``  : best coverage of any memory title by the
        question's tokens (entity-anchor strength).
        ``titles_hit``      : # distinct titles with any question token.
        ``best_rel_cov``    : best coverage of the question's relation tokens
        (tokens not used for the title anchor) by a matching snippet sentence.
        ``best_title_idf``  : IDF-style rarity of the best-matching title.
        """
        qt = set(tokenize(question))
        if not qt:
            return {
                "best_title_cov": None,
                "titles_hit": 0,
                "best_rel_cov": None,
                "best_title_idf": None,
            }

        # Stage 1: candidate titles via title postings
        cand: Dict[int, int] = {}
        for tok in qt:
            for idx in self._title_postings.get(tok, ()):
                cand[idx] = cand.get(idx, 0) + 1
        if not cand:
            return {
                "best_title_cov": 0.0,
                "titles_hit": 0,
                "best_rel_cov": 0.0,
                "best_title_idf": None,
            }
        ranked = sorted(cand, key=lambda i: (-cand[i], self.titles[i].lower()))[:top_k]

        best_title_cov = 0.0
        best_rel_cov = 0.0
        best_title_idf = 0.0
        for idx in ranked:
            ttoks = self._title_tokens[idx]
            t_cov = len(qt & ttoks) / len(ttoks) if ttoks else 0.0
            if t_cov <= 0:
                continue
            best_title_cov = max(best_title_cov, t_cov)
            # Stage 2: relative coverage of remaining tokens by the snippet
            rel_tokens = qt - ttoks
            rel_cov = 0.0
            if rel_tokens:
                for s_idx in self._snippets_of_title[self.titles[idx]]:
                    stoks = set(tokenize(self.snippets[s_idx]))
                    if stoks:
                        rel_cov = max(rel_cov, len(rel_tokens & stoks) / len(rel_tokens))
            best_rel_cov = max(best_rel_cov, rel_cov)
            best_title_idf = max(best_title_idf, self._title_idf.get(self.titles[idx], 0.0))

        titles_hit = len({
            self.titles[idx] for idx in ranked if (self._title_tokens[idx] & qt)
        })
        return {
            "best_title_cov": round(best_title_cov, 4),
            "titles_hit": titles_hit,
            "best_rel_cov": round(best_rel_cov, 4),
            "best_title_idf": round(best_title_idf, 4) if best_title_idf else None,
        }

    # ------------------------------------------------------------ rank options
    def rank_options(self, options: List[str]) -> List[Dict[str, Any]]:
        """Score candidate answer options against the memory ONLY.

        For either/or questions the model's closed-book strategy is to match
        each option (as a memory entity) and favor the better-anchored one.
        Returns options with {'option', 'exact_title', 'title_cov', 'rel_cov'}
        sorted best-first. Reads only options - no reference data involved.
        """
        out: List[Dict[str, Any]] = []
        for option in options:
            otoks = frozenset(tokenize(option))
            entity: Optional[str] = None
            if option in self._snippets_of_title:
                entity = option
            else:
                # fuzzy title match by token overlap
                best_cov = 0.0
                for title in self._snippets_of_title:
                    ttoks = frozenset(tokenize(title))
                    if not ttoks:
                        continue
                    cov = len(otoks & ttoks) / max(1, len(otoks | ttoks))
                    if cov > best_cov:
                        best_cov = cov
                        entity = title
                if best_cov < 0.5:
                    entity = None
            if entity is None:
                out.append({"option": option, "exact_title": False, "title_cov": 0.0, "rel_cov": 0.0})
                continue
            title_toks = frozenset(tokenize(entity))
            t_cov = len(otoks & title_toks) / max(1, len(otoks | title_toks))
            rel_cov = 0.0
            qt_rel = otoks - title_toks
            if not qt_rel:
                rel_cov = 1.0
            else:
                for s_idx in self._snippets_of_title[entity]:
                    stoks = set(tokenize(self.snippets[s_idx]))
                    if stoks:
                        rel_cov = max(rel_cov, len(qt_rel & stoks) / len(qt_rel))
            out.append({
                "option": option,
                "exact_title": entity == option,
                "title_cov": round(t_cov, 4),
                "rel_cov": round(rel_cov, 4),
            })
        out.sort(key=lambda d: (-(1 if d["exact_title"] else 0), -d["title_cov"], -d["rel_cov"]))
        return out

    # ------------------------------------------------------------ top snippets
    def top_snippets(
        self,
        question: str,
        top_k: int = 12,
    ) -> List[Tuple[str, str, float, float]]:
        """Candidate memory snippets ordered by (title_cov, rel_cov).

        Returns list of (title, sentence, title_cov, rel_cov). No answer text
        is produced here - the store is memory, not an answering method.
        """
        qt = set(tokenize(question))
        if not qt:
            return []
        cand: Dict[int, int] = {}
        for tok in qt:
            for idx in self._title_postings.get(tok, ()):
                cand[idx] = cand.get(idx, 0) + 1
        ranked = sorted(cand, key=lambda i: (-cand[i], self.titles[i].lower()))[: top_k * 2]

        out: List[Tuple[str, str, float, float]] = []
        for idx in ranked:
            ttoks = self._title_tokens[idx]
            t_cov = len(qt & ttoks) / len(ttoks) if ttoks else 0.0
            if t_cov <= 0:
                continue
            rel_tokens = qt - ttoks
            for s_idx in self._snippets_of_title[self.titles[idx]]:
                stoks = set(tokenize(self.snippets[s_idx]))
                rel_cov = (len(rel_tokens & stoks) / len(rel_tokens)) if rel_tokens else 1.0
                out.append(
                    (self.titles[idx], self.snippets[s_idx], t_cov, rel_cov)
                )
        out.sort(key=lambda t: (-t[2], -t[3], t[1]))
        return out[:top_k]


# ======================================================================
# Method A: closed-book answering with simulated parametric memory.
# ======================================================================

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


# ======================================================================
# Method B: grounded answer generation over retrieved passages.
# ======================================================================

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
