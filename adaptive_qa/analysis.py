"""Evidence and answer-difference analysis for the comparison dashboard.

What this module DOES
---------------------
* Assigns STABLE citation labels (``[1]``, ``[2]``, ... and the anchors
  ``passage-1``, ...) to the exact passages Option B received, so a citation in
  an answer can be traced back to a real retrieved chunk.
* Measures, per answer sentence, how much of the sentence's content vocabulary
  is present in the retrieved passages (lexical overlap against each passage).
* Summarises which passage's vocabulary Option B actually used that Option A
  did not (and vice versa), which is where the answers differ in *useful
  factual detail*.

What this module does NOT do
----------------------------
It does not decide whether an answer is correct, and it does not check
faithfulness in the entailment sense. Lexical overlap is blind to paraphrase
and to contradiction ("A weak entity set has its own primary key" reuses the
passage's vocabulary while reversing its meaning). Everything it returns is
labelled ``lexical_overlap_only`` and must be presented that way: a citation
is evidence of *where the wording came from*, never a correctness proof.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .retrieval.bm25 import tokenize

# Documented thresholds on lexical overlap (share of a sentence's content terms
# that appear in a passage). Status names describe OVERLAP, not entailment:
HIGH_OVERLAP_AT = 0.60
PARTIAL_OVERLAP_AT = 0.35
MIN_TERMS_TO_CHECK = 3

STATUS_HIGH = "high_overlap"
STATUS_PARTIAL = "partial_overlap"
STATUS_NONE = "no_overlap"
STATUS_UNCHECKED = "unchecked"

# Terms too generic to be interesting in a "what did B use that A did not"
# comparison.
_GENERIC = frozenset(
    "also such which would could should their there these those about using used "
    "each other into from with that this have has been were was are is".split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    for piece in _SENTENCE_SPLIT.split((text or "").strip()):
        s = piece.strip().lstrip("-*•0123456789. )(").strip()
        if s:
            out.append(s)
    return out


def term_set(text: str) -> set:
    return set(tokenize(text or ""))


def _informative(terms: Iterable[str]) -> List[str]:
    return sorted(
        t for t in set(terms) if len(t) >= 4 and t not in _GENERIC
    )


def sentence_overlap_support(
    answer_text: str,
    passages: Sequence[Dict[str, Any]],
    *,
    high_at: float = HIGH_OVERLAP_AT,
    partial_at: float = PARTIAL_OVERLAP_AT,
) -> List[Dict[str, Any]]:
    """LEXICAL OVERLAP of each answer sentence against the retrieved passages.

    Status meanings (all of them describe shared vocabulary, NOT
    entailment, NOT factual verification):

    - ``high_overlap``   : ≥ ``high_at`` of the sentence's content terms occur
      in the passage (the sentence largely reuses passage vocabulary).
    - ``partial_overlap``: between ``partial_at`` and ``high_at``.
    - ``no_overlap``     : below ``partial_at``.
    - ``unchecked``      : sentence too short to test (fewer than
      MIN_TERMS_TO_CHECK content terms).

    A sentence marked ``high_overlap`` can still be semantically WRONG
    (reversing a passage's meaning reuses its vocabulary), and a well-formed
    paraphrase can score ``no_overlap``. Use this as a pointer back to the
    passage the wording likely came from — nothing more.
    """
    passage_terms = [term_set(p.get("text") or p.get("excerpt") or "") for p in passages]
    rows: List[Dict[str, Any]] = []
    for sentence in split_sentences(answer_text):
        terms = term_set(sentence)
        if len(terms) < MIN_TERMS_TO_CHECK:
            rows.append(
                {
                    "sentence": sentence,
                    "overlap_status": STATUS_UNCHECKED,
                    "overlap": None,
                    "citations": [],
                    "terms_in_evidence": [],
                    "note": "too short to check lexically",
                }
            )
            continue
        scores = []
        for i, pt in enumerate(passage_terms):
            overlap = len(terms & pt) / max(1, len(terms))
            scores.append((overlap, i + 1))
        scores.sort(key=lambda t: (-t[0], t[1]))
        best_overlap, best_rank = scores[0]
        citing = [rank for ov, rank in scores if ov >= high_at][:3]
        if best_overlap >= high_at:
            status = STATUS_HIGH
        elif best_overlap >= partial_at:
            status = STATUS_PARTIAL
        else:
            status = STATUS_NONE
        evidence_terms = sorted(terms & passage_terms[best_rank - 1]) if passage_terms else []
        rows.append(
            {
                "sentence": sentence,
                "overlap_status": status,
                "overlap": round(best_overlap, 3),
                "best_passage": best_rank,
                "citations": [f"[{r}]" for r in citing],
                "terms_in_evidence": _informative(evidence_terms)[:10],
            }
        )
    return rows


def sentence_support(answer_text, passages, **kw):
    """Backwards-compatible alias for :func:`sentence_overlap_support`."""
    return sentence_overlap_support(answer_text, passages, **kw)


def assign_citations(passages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy passages and attach 1-based citation labels + HTML anchors."""
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(passages, start=1):
        entry = dict(p)
        entry["rank"] = i
        entry["citation"] = f"[{i}]"
        entry["anchor"] = f"passage-{i}"
        out.append(entry)
    return out


def compare_answers(
    answer_a: str,
    answer_b: str,
    passages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Lexical-overlap map for B plus a passage-by-passage answer difference summary."""
    cited = assign_citations(passages)
    b_support = sentence_overlap_support(answer_b, cited)
    a_support = sentence_overlap_support(answer_a, cited)

    a_terms = term_set(answer_a)
    b_terms = term_set(answer_b)
    shared = a_terms & b_terms

    per_passage: List[Dict[str, Any]] = []
    for p in cited:
        p_terms = term_set(p.get("text") or "")
        in_b = b_terms & p_terms
        in_a = a_terms & p_terms
        per_passage.append(
            {
                "rank": p["rank"],
                "citation": p["citation"],
                "anchor": p["anchor"],
                "source_file": p.get("source_file"),
                "page": p.get("page"),
                "terms_b_not_a": _informative(in_b - in_a)[:8],
                "terms_a_not_b": _informative(in_a - in_b)[:8],
                "coverage_a": round(len(in_a) / max(1, len(a_terms)), 3),
                "coverage_b": round(len(in_b) / max(1, len(b_terms)), 3),
            }
        )

    def _count(rows, status):
        return sum(1 for r in rows if r["overlap_status"] == status)

    return {
        "passages": cited,
        "b_sentences": b_support,
        "a_sentences": a_support,
        "per_passage": per_passage,
        "answer_diff": {
            "only_in_a": _informative(a_terms - b_terms)[:15],
            "only_in_b": _informative(b_terms - a_terms)[:15],
            "shared_terms": len(shared),
            "jaccard": round(len(shared) / max(1, len(a_terms | b_terms)), 3),
        },
        "summary": {
            "n_passages": len(cited),
            "b_high_overlap": _count(b_support, STATUS_HIGH),
            "b_partial_overlap": _count(b_support, STATUS_PARTIAL),
            "b_no_overlap": _count(b_support, STATUS_NONE),
            "b_unchecked": _count(b_support, STATUS_UNCHECKED),
            "a_high_overlap": _count(a_support, STATUS_HIGH),
            "a_no_overlap": _count(a_support, STATUS_NONE),
        },
        "method": "lexical_overlap_only",
        "status_names": {
            STATUS_HIGH: "≥ HIGH_OVERLAP_AT of the sentence's content terms occur in the passage",
            STATUS_PARTIAL: "between PARTIAL_OVERLAP_AT and HIGH_OVERLAP_AT",
            STATUS_NONE: "below PARTIAL_OVERLAP_AT",
            STATUS_UNCHECKED: "sentence too short to test",
        },
        "method_note": (
            "This map measures LEXICAL OVERLAP ONLY: the share of a sentence's "
            "content terms that occur in a retrieved passage. It is NOT "
            "entailment and NOT a correctness check — a sentence can reuse a "
            "passage's vocabulary while contradicting it, and a factual "
            "paraphrase can score no_overlap. Treat these statuses as shared- "
            "vocabulary labels, never as 'the passage proves this sentence'."
        ),
        "thresholds": {
            "high_overlap_at": HIGH_OVERLAP_AT,
            "partial_overlap_at": PARTIAL_OVERLAP_AT,
        },
    }


def citation_map(passages: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{"[1]": {anchor, source_file, page, ...}}`` for frontend validation."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in assign_citations(passages):
        out[p["citation"]] = {
            "anchor": p["anchor"],
            "source_file": p.get("source_file"),
            "page": p.get("page"),
            "score": p.get("score"),
        }
    return out


def invalid_citations(answer_text: str, passages: Sequence[Dict[str, Any]]) -> List[str]:
    """Citations in the answer that do not map to a supplied passage ([] if ok).

    Used by tests and the dashboard to catch a model citing ``[9]`` when only
    five passages were given.
    """
    supplied = {f"[{i}]" for i in range(1, len(passages) + 1)}
    found = set(re.findall(r"\[(\d+)\]", answer_text or ""))
    return sorted(f"[{f}]" for f in found if f"[{f}]" not in supplied)
