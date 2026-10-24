"""Per-policy metric aggregation.

All aggregations come from the per-question run records (machine-readable
JSONL) so every number in a summary can be recomputed from the artifact.
``ambiguous`` oracle cases are excluded from binary decision accuracy and
reported separately (both options from the brief are honored).
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from ..types import Decision


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
