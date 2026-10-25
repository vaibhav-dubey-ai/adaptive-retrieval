"""Retrieval quality metrics: Recall@k, MRR@k, nDCG@k.

All three metrics use **binary relevance**: a retrieved chunk counts as
relevant when its stable chunk id appears in the question's ground-truth
``relevant_chunk_ids`` list. Labels are hand-verified against the source
document (see ``data/retrieval_eval/README.md``); they are read only by this
evaluation code and never by the retrievers, the decision engine, or the
answer prompts.

Definitions used throughout the project
---------------------------------------
Recall@k       = |relevant ∩ retrieved[:k]| / |relevant|
                 Fraction of the labelled evidence found in the top k.
MRR@k          = mean over questions of 1 / rank(first relevant hit in top k)
                 (0 when no relevant chunk appears in the top k).
nDCG@k         = DCG@k / IDCG@k with gain 1 for a relevant chunk,
                 DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
                 IDCG@k = Σ_{i=1..min(|relevant|, k)} 1 / log2(i + 1)
                 Rewards relevant chunks ranked high, and caps the ideal at the
                 number of labelled chunks (so a question with two relevant
                 chunks cannot be penalised for lacking a third).

Handling of edge cases
----------------------
* Questions with **no** relevant chunks (unanswerable questions) have no
  defined Recall/MRR/nDCG; they return ``None`` and are excluded from the
  aggregate means. Their retrieval behaviour is reported separately as
  ``no_evidence_rate`` = share of unanswerable questions where the corpus
  contains no evidence, which cannot be judged by rank metrics.
* ``k`` larger than the number of retrieved chunks is allowed: missing
  positions simply contribute nothing (they are not padded with hits).
* An empty retrieval result scores 0.0 for MRR/nDCG and 0.0 for Recall.
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


def _top(retrieved: Sequence[str], k: int) -> List[str]:
    return list(retrieved)[: max(0, int(k))]


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> Optional[float]:
    rel: Set[str] = set(relevant)
    if not rel:
        return None
    found = rel.intersection(_top(retrieved, k))
    return len(found) / len(rel)


def reciprocal_rank_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int = 10
) -> Optional[float]:
    rel: Set[str] = set(relevant)
    if not rel:
        return None
    for rank, doc_id in enumerate(_top(retrieved, k), start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int = 10
) -> Optional[float]:
    rel: Set[str] = set(relevant)
    if not rel:
        return None
    hits = _top(retrieved, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(hits, start=1)
        if doc_id in rel
    )
    ideal_n = min(len(rel), max(0, int(k)))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (0 <= q <= 100); None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Mean/median/min/max/p95 over millisecond timings."""
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean_ms": None, "median_ms": None, "min_ms": None,
                "max_ms": None, "p95_ms": None}
    return {
        "n": len(vals),
        "mean_ms": round(mean(vals), 3),
        "median_ms": round(float(median(vals)), 3),
        "min_ms": round(min(vals), 3),
        "max_ms": round(max(vals), 3),
        "p95_ms": round(percentile(vals, 95) or 0.0, 3),
    }


def aggregate(
    rows: Sequence[Dict[str, Any]],
    k_recall: int = 5,
    k_rank: int = 10,
) -> Dict[str, Any]:
    """Aggregate per-question rows into headline metrics.

    Each row must provide ``retrieved`` (ordered chunk ids) and ``relevant``
    (labelled chunk ids); ``category`` is optional and used for breakdowns.
    """
    scored = [r for r in rows if r.get("relevant")]
    unscored = [r for r in rows if not r.get("relevant")]

    def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
        ok = [v for v in vals if v is not None]
        return round(mean(ok), 4) if ok else None

    out: Dict[str, Any] = {
        "n_questions": len(rows),
        "n_scored_questions": len(scored),
        "n_unanswerable_questions": len(unscored),
        f"recall_at_{k_recall}": _mean(
            recall_at_k(r["retrieved"], r["relevant"], k_recall) for r in scored
        ),
        f"mrr_at_{k_rank}": _mean(
            reciprocal_rank_at_k(r["retrieved"], r["relevant"], k_rank) for r in scored
        ),
        f"ndcg_at_{k_rank}": _mean(
            ndcg_at_k(r["retrieved"], r["relevant"], k_rank) for r in scored
        ),
        "hit_rate_at_1": _mean(
            1.0 if (r["relevant"] and r["retrieved"] and r["retrieved"][0] in set(r["relevant"])) else 0.0
            for r in scored
        ),
        "no_evidence_rate": _mean(
            1.0 if not r.get("retrieved") else 0.0 for r in unscored
        ),
        "mean_relevant_chunks": round(
            mean(len(r["relevant"]) for r in scored), 3
        ) if scored else None,
        "metrics_note": (
            "Binary relevance against hand-verified chunk ids. Questions with no "
            "labelled evidence (unanswerable) are excluded from Recall/MRR/nDCG and "
            "reported separately via no_evidence_rate."
        ),
    }
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in sorted({r.get("category", "uncategorised") for r in rows}):
        sub = [r for r in rows if r.get("category", "uncategorised") == cat]
        by_cat[cat] = aggregate_basic(sub, k_recall, k_rank)
    out["by_category"] = by_cat
    return out


def aggregate_basic(
    rows: Sequence[Dict[str, Any]], k_recall: int = 5, k_rank: int = 10
) -> Dict[str, Any]:
    """Same metrics without the recursive category breakdown."""
    scored = [r for r in rows if r.get("relevant")]

    def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
        ok = [v for v in vals if v is not None]
        return round(mean(ok), 4) if ok else None

    return {
        "n_questions": len(rows),
        "n_scored_questions": len(scored),
        f"recall_at_{k_recall}": _mean(
            recall_at_k(r["retrieved"], r["relevant"], k_recall) for r in scored
        ),
        f"mrr_at_{k_rank}": _mean(
            reciprocal_rank_at_k(r["retrieved"], r["relevant"], k_rank) for r in scored
        ),
        f"ndcg_at_{k_rank}": _mean(
            ndcg_at_k(r["retrieved"], r["relevant"], k_rank) for r in scored
        ),
        "no_evidence_rate": _mean(
            1.0 if not r.get("retrieved") else 0.0
            for r in rows
            if not r.get("relevant")
        ),
    }
