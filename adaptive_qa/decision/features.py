"""Pre-retrieval features for the adaptive decision engine.

Every feature is computed from (a) the question text and (b) the answering
model's own parametric-memory probe (an uncertainty signal, like the "does the
LM know it" feature in adaptive-retrieval literature). No retrieved passage,
candidate answer, reference answer or oracle label is accessed - the module
would have nowhere to get them even by accident.

Feature order is part of the persisted model bundle, so retraining keeps a
stable contract.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..retrieval.bm25 import tokenize
from ..answering.span_extraction import question_type

WH_TYPES = ["who", "what", "when", "where", "quantity", "comparison", "other"]

_SUPERLATIVE_RE = re.compile(
    r"\b(first|last|largest|smallest|oldest|youngest|most|best|longest|shortest|"
    r"current|oldest|prime|biggest|only)\b",
    re.IGNORECASE,
)
_YEAR_CUE_RE = re.compile(r"\b(1[4-9]\d{2}|20[0-4]\d)\b")
_QUOTES_CUE_RE = re.compile(r"\"([^\"]+)\"")


def extract_features(question: str, familiarity: Optional[dict] = None) -> List[float]:
    """Build the numeric pre-retrieval feature vector (stable order)."""
    q = question.strip()
    toks = tokenize(q)
    qtype = question_type(q)

    feat: List[float] = []
    feat.append(min(1.0, len(toks) / 20.0))                # question length (tokens)
    feat.append(min(1.0, len(q) / 120.0))                  # question length (chars)
    # mid-sentence capitalized tokens ~ named-entity density
    caps = sum(1 for t in q.split()[1:] if t[:1].isupper() and t[1:2].islower())
    feat.append(min(1.0, caps / 4.0))
    feat.append(1.0 if _YEAR_CUE_RE.search(q) else 0.0)    # asks about a specific year
    feat.append(1.0 if _SUPERLATIVE_RE.search(q) else 0.0) # superlative/ordinal cues
    feat.append(1.0 if _QUOTES_CUE_RE.search(q) else 0.0)  # quotes -> titles/works

    wh_onehot = [1.0 if qtype == t else 0.0 for t in WH_TYPES]
    feat.extend(wh_onehot)

    fam = familiarity or {}
    feat.append(float(fam.get("best_title_cov", 0.0) or 0.0))
    feat.append(min(1.0, (fam.get("titles_hit", 0) or 0) / 5.0))
    feat.append(float(fam.get("best_rel_cov", 0.0) or 0.0))
    if fam.get("best_title_idf"):
        feat.append(min(1.0, fam["best_title_idf"] / 10.0))
    else:
        feat.append(0.0)
    return feat


FEATURE_NAMES: List[str] = [
    "q_len_tokens",
    "q_len_chars",
    "q_entity_density",
    "q_year_cue",
    "q_superlative_cue",
    "q_quoted_title",
] + [f"wh_{t}" for t in WH_TYPES] + [
    "fam_best_title_cov",
    "fam_titles_hit",
    "fam_best_rel_cov",
    "fam_title_idf",
]


def features_dict(question: str, familiarity: Optional[dict] = None) -> Dict[str, float]:
    vec = extract_features(question, familiarity)
    return dict(zip(FEATURE_NAMES, vec))
