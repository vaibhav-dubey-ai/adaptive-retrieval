"""Exact-match and token-F1 (SQuAD-style normalization).

Normalization: lower-case, strip punctuation, drop articles (a/an/the) and
collapse whitespace. Documented limitation (PLAN.md): these metrics suit
HotpotQA's short-answer style and are blind to paraphrase.
"""

from __future__ import annotations

import re
import string
from typing import List, Sequence

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    if not gold_tokens or not pred_tokens:
        return 1.0 if gold_tokens == pred_tokens else 0.0
    from collections import Counter

    common = Counter(gold_tokens) & Counter(pred_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def em_score(gold: str, pred: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(pred))


def composite_score(gold: str, pred: str) -> float:
    return 0.5 * em_score(gold, pred) + 0.5 * f1_score(gold, pred)
