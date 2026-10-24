"""Simulated LLM parametric memory for the closed-book answering engine.

A real LLM answers without retrieval from weights learned at pretraining time.
This machine has no local neural model, so the deterministic *finite answer
engine* emulates memory with a store built from a HotpotQA ``train`` slice -
facts the "model would have read" during pretraining. Because HotpotQA train
and validation rows are disjoint from each other, the store is disjoint from
the eval sample: no question, answer or paragraph overlaps.

Leakage rule (enforced by tests): the store is built ONLY from train-split
rows and NEVER from the eval sample, corpus, or any reference answers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import resolve_path
from ..retrieval.bm25 import tokenize


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
