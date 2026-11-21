"""Answer scoring (EM/F1), per-question outcome labels, and policy-level metrics."""

from __future__ import annotations

import re
import statistics
import string
from ..common import Decision
from ..common import GeneratedAnswer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing import Any, Optional
from typing import List, Sequence


# ======================================================================
# Exact-match and token-F1 (SQuAD-style normalization).
# ======================================================================

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


def recall_score(gold: str, pred: str) -> float:
    """Fraction of the reference tokens that appear in the prediction.

    Token-F1 punishes a correct but wordy answer (low precision); recall alone
    rewards rambling. ``quality_score`` blends both for free-form LLM answers.
    """
    from collections import Counter

    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    common = Counter(gold_tokens) & Counter(pred_tokens)
    return sum(common.values()) / len(gold_tokens)


def quality_score(gold: str, pred: str) -> float:
    """Answer quality for free-form LLM output: mean of token-F1 and token-recall."""
    return 0.5 * f1_score(gold, pred) + 0.5 * recall_score(gold, pred)


# ======================================================================
# Cross-verification outcome labeling (the offline oracle).
# ======================================================================

def _answer_fields(answer):
    """Accept a GeneratedAnswer OR its serialized dict form (records)."""
    if isinstance(answer, GeneratedAnswer):
        return answer.text, bool(answer.abstained)
    if isinstance(answer, dict):
        return (answer.get("text") or ""), bool(answer.get("abstained", False))
    return "", True


@dataclass
class Outcome:
    label: str                    # one of the four class names
    quality_a: float              # composite quality of Method A
    quality_b: float              # composite quality of Method B
    delta: float                  # quality_b - quality_a
    em_a: float
    f1_a: float
    em_b: float
    f1_b: float


def label_outcome(
    question_record,
    answer_a: GeneratedAnswer,
    answer_b: GeneratedAnswer,
    gap_threshold: float = 0.10,
) -> Outcome:
    """Score both answers and classify the cross-verification outcome.

    ``question_record`` carries the reference answer. The answers must come
    from executing the respective methods, not from any cached guess.

    Notice that citations or passages attached to Method B do NOT influence
    the label: only answer quality does.
    """
    text_a, abstained_a = _answer_fields(answer_a)
    text_b, abstained_b = _answer_fields(answer_b)
    empty_a = answer_a is None or abstained_a or not text_a.strip()
    empty_b = answer_b is None or abstained_b or not text_b.strip()

    gold = question_record.answer
    em_a = em_score(gold, text_a) if (answer_a and not empty_a) else 0.0
    f1_a = f1_score(gold, text_a) if (answer_a and not empty_a) else 0.0
    em_b = em_score(gold, text_b) if (answer_b and not empty_b) else 0.0
    f1_b = f1_score(gold, text_b) if (answer_b and not empty_b) else 0.0

    quality_a = 0.5 * em_a + 0.5 * f1_a
    quality_b = 0.5 * em_b + 0.5 * f1_b
    delta = quality_b - quality_a

    if not gold.strip() or (empty_a and empty_b):
        label = "ambiguous"
    elif delta >= gap_threshold:
        label = "retrieval_better"
    elif delta <= -gap_threshold:
        label = "no_retrieval_better"
    else:
        label = "similar"
    return Outcome(
        label=label,
        quality_a=quality_a,
        quality_b=quality_b,
        delta=round(delta, 4),
        em_a=em_a,
        f1_a=f1_a,
        em_b=em_b,
        f1_b=f1_b,
    )


# ======================================================================
# Per-policy metric aggregation.
# ======================================================================

def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else 0.0


def median(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else 0.0


def stdev(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return round(statistics.stdev(xs), 4)


def _primary(rec: Dict[str, Any], policy: str) -> Dict[str, Any]:
    """Which evaluated answer should this policy be credited with?"""
    outcome = rec.get("outcome") or {}
    if policy == "never_retrieve":
        return (rec.get("answer_a") or {})
    if policy == "always_retrieve":
        return (rec.get("answer_b") or {})
    if policy == "adaptive":
        return (rec.get("answer_b") if rec.get("decision") == "RETRIEVE" else rec.get("answer_a")) or {}
    if policy == "oracle":
        if outcome.get("retrieval_better"):
            return rec.get("answer_b") or {}
        if outcome.get("no_retrieval_better"):
            return rec.get("answer_a") or {}
        return (rec.get("answer_b") if rec.get("decision") == "RETRIEVE" else rec.get("answer_a")) or {}
    raise ValueError(policy)


def policy_metrics(
    records: List[Dict[str, Any]],
    policy: str,
    cost_per_retrieval_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate one policy's answer quality, latency, and cost over records."""
    n = len(records)
    if n == 0:
        return {"policy": policy, "n_questions": 0}

    em, f1, comp = [], [], []
    abstain = 0
    retrieval_count = 0
    e2e, dec_ms, ret_ms, gen_a, gen_b = [], [], [], [], []
    for rec in records:
        outcome = rec.get("outcome") or {}
        oa = outcome.get("em_a"); ob = outcome.get("em_b")
        fa = outcome.get("f1_a"); fb = outcome.get("f1_b")
        ca = 0.5 * (oa or 0) + 0.5 * (fa or 0)
        cb = 0.5 * (ob or 0) + 0.5 * (fb or 0)
        is_b = _primary_row_is_b(rec, policy) == "B"
        primary = rec.get("answer_b") if is_b else rec.get("answer_a")
        em.append(ob if is_b else oa)
        f1.append(fb if is_b else fa)
        comp.append(cb if is_b else ca)
        abstain += 1 if primary.get("abstained") else 0
        if is_b:
            retrieval_count += 1
        e2e.append(_e2e_ms(rec, policy))
        dec_ms.append(rec.get("decision_ms"))
        ret_ms.append(rec.get("retrieval_ms") if is_b else None)
        gen_a.append(None if is_b else rec.get("gen_a_ms"))
        gen_b.append(rec.get("gen_b_ms") if is_b else None)

    total_cost = retrieval_count * (cost_per_retrieval_usd or 0.0)
    return {
        "policy": policy,
        "n_questions": n,
        "em": mean(em),
        "f1": mean(f1),
        "composite_quality": mean(comp),
        "abstain_rate": round(abstain / n, 4),
        "retrieval_rate": round(retrieval_count / n, 4),
        "retrievals": retrieval_count,
        "estimated_cost_usd": round(total_cost, 6),
        "estimated_cost_per_question_usd": round(total_cost / n, 8) if n else 0.0,
        "latency_ms": {
            "mean_e2e": mean(e2e),
            "median_e2e": median(e2e),
            "stdev_e2e": stdev(e2e),
            "mean_decision": mean(dec_ms),
            "median_decision": median(dec_ms),
            "mean_retrieval": mean(ret_ms),
            "median_retrieval": median(ret_ms),
            "mean_generation_a": mean(gen_a),
            "mean_generation_b": mean(gen_b),
            "mean_cross_verify": mean([r.get("cross_verify_ms") for r in records]),
        },
        "_latency_stdev_ms": {
            "e2e": stdev(e2e),
            "cross_verify": stdev([r.get("cross_verify_ms") for r in records]),
        },
    }


def _primary_row_is_b(rec, policy) -> str:
    # which primary method the policy chooses on this record
    outcome = rec.get("outcome") or {}
    if policy == "never_retrieve":
        return "A"
    if policy == "always_retrieve":
        return "B"
    if policy == "adaptive":
        return "B" if rec.get("decision") == Decision.RETRIEVE.value else "A"
    # oracle: prefer ledger class first
    if outcome.get("retrieval_better"):
        return "B"
    if outcome.get("no_retrieval_better"):
        return "A"
    return "B" if rec.get("decision") == Decision.RETRIEVE.value else "A"


def _e2e_ms(rec, policy) -> Optional[float]:
    """Deployed end-to-end latency: only the selected path's real work.

    For fixed policies the decision stage is a no-op (no decision engine
    call), so latency equals one decision + selected path work.
    """
    outcome = rec.get("outcome") or {}
    is_b = _primary_row_is_b(rec, policy) == "B"
    dec = rec.get("decision_ms") or 0.0
    ret = rec.get("retrieval_ms") if is_b else 0.0
    gen = rec.get("gen_b_ms") if is_b else rec.get("gen_a_ms")
    ret = ret or 0.0
    return round(dec + ret + (gen or 0.0), 3)


def comparison_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Adaptive-vs-oracle decision metrics over evaluable records."""
    usable = [r for r in records if (r.get("outcome") or {}).get("label") in ("retrieval_better", "no_retrieval_better", "similar")]
    excluded = [r for r in records if r not in usable]

    true_retrieve = 0
    pred_retrieve = 0
    tp = fp = fn = tn = 0
    for r in usable:
        outcome = r["outcome"]
        beneficial = outcome["label"] == "retrieval_better"
        pred = r.get("decision") == Decision.RETRIEVE.value
        if beneficial:
            true_retrieve += 1
        if pred:
            pred_retrieve += 1
        if beneficial and pred:
            tp += 1
        elif not beneficial and pred:
            fp += 1
        elif beneficial and not pred:
            fn += 1
        else:
            tn += 1

    accuracy = round((tp + tn) / len(usable), 4) if usable else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    outcome_counts = {
        "retrieval_better": 0,
        "no_retrieval_better": 0,
        "similar": 0,
        "ambiguous": 0,
    }
    for r in records:
        lbl = (r.get("outcome") or {}).get("label")
        if lbl in outcome_counts:
            outcome_counts[lbl] += 1

    return {
        "n_evaluable": len(usable),
        "n_excluded_ambiguous": len(excluded),
        "excluded_rate": round(len(excluded) / len(records), 4) if records else 0.0,
        "outcome_counts": outcome_counts,
        "retrieval_rate": round(true_retrieve / len(usable), 4) if usable else None,
        "decision_accuracy": accuracy,
        "retrieve_class_precision": precision,
        "retrieve_class_recall": recall,
        "retrieve_class_f1": f1,
        "confusion": {
            "true_retrieve_predicted_retrieve": tp,
            "true_no_retrieve_predicted_retrieve": fp,
            "true_retrieve_predicted_skip": fn,
            "true_no_retrieve_predicted_skip": tn,
        },
        "beneficial_retrievals_identified": tp,
        "beneficial_retrievals_missed": fn,
        "unnecessary_retrievals": fp,
    }
