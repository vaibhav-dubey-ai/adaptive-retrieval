"""Retrieval evaluation: ranking metrics (recall/MRR/nDCG) and the labelled-question evaluation harness."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from ..common import now_ns
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


# ======================================================================
# Retrieval quality metrics: Recall@k, MRR@k, nDCG@k.
# ======================================================================

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


# ======================================================================
# Reproducible retrieval evaluation over a labelled question set.
# ======================================================================

REQUIRED_KEYS = ("question_id", "question")


@dataclass(frozen=True)
class EvalExample:
    question_id: str
    question: str
    relevant_chunk_ids: tuple = ()
    category: str = "uncategorised"
    reference_answer: str = ""
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_chunk_ids)


def load_eval_examples(path: str | Path) -> List[EvalExample]:
    """Read a JSONL labelled set, validating the minimum schema."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evaluation set not found: {p}")
    out: List[EvalExample] = []
    seen_ids = set()
    seen_questions = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno} is not valid JSON: {exc}") from exc
        missing = [k for k in REQUIRED_KEYS if not raw.get(k)]
        if missing:
            raise ValueError(f"{p}:{lineno} missing required keys: {', '.join(missing)}")
        qid = str(raw["question_id"]).strip()
        question = str(raw["question"]).strip()
        if qid in seen_ids:
            raise ValueError(f"{p}:{lineno} duplicate question_id {qid!r}")
        if question.lower() in seen_questions:
            raise ValueError(f"{p}:{lineno} duplicate question text {question!r}")
        seen_ids.add(qid)
        seen_questions.add(question.lower())
        rel = raw.get("relevant_chunk_ids") or []
        if isinstance(rel, str):
            rel = [rel]
        out.append(
            EvalExample(
                question_id=qid,
                question=question,
                relevant_chunk_ids=tuple(str(x) for x in rel),
                category=str(raw.get("category") or "uncategorised"),
                reference_answer=str(raw.get("reference_answer") or ""),
                notes=str(raw.get("notes") or ""),
            )
        )
    if not out:
        raise ValueError(f"{p} contains no examples")
    return out


def validate_labels(
    examples: Sequence[EvalExample], available_chunk_ids: Iterable[str]
) -> Dict[str, Any]:
    """Check that every labelled chunk id exists in the corpus.

    Catches typos and stale labels after re-upload/re-chunking, which would
    silently depress every metric.
    """
    available = set(available_chunk_ids)
    unknown: List[Dict[str, str]] = []
    for ex in examples:
        for cid in ex.relevant_chunk_ids:
            if cid not in available:
                unknown.append({"question_id": ex.question_id, "chunk_id": cid})
    return {
        "ok": not unknown,
        "n_available_chunks": len(available),
        "unknown_chunk_ids": unknown,
    }


def evaluate_retriever(
    retriever,
    examples: Sequence[EvalExample],
    top_k: int = 10,
    k_recall: int = 5,
    k_rank: int = 10,
    repeat: int = 1,
) -> Dict[str, Any]:
    """Run one retriever over all examples; return rows, aggregate and latency."""
    rows: List[Dict[str, Any]] = []
    latency_ms: List[float] = []
    first_query_ms: Optional[float] = None

    for rep in range(max(1, int(repeat))):
        for i, ex in enumerate(examples):
            t0 = now_ns()
            hits = retriever.retrieve(ex.question, top_k)
            elapsed = (now_ns() - t0) / 1_000_000.0
            if rep == 0 and i == 0:
                first_query_ms = elapsed
            latency_ms.append(elapsed)
            if rep == 0:
                retrieved = [h.doc_id for h in hits]
                rows.append(
                    {
                        "question_id": ex.question_id,
                        "question": ex.question,
                        "category": ex.category,
                        "relevant": list(ex.relevant_chunk_ids),
                        "retrieved": retrieved,
                        "recall_at_k": recall_at_k(retrieved, ex.relevant_chunk_ids, k_recall),
                        "reciprocal_rank": reciprocal_rank_at_k(
                            retrieved, ex.relevant_chunk_ids, k_rank
                        ),
                        "ndcg": ndcg_at_k(retrieved, ex.relevant_chunk_ids, k_rank),
                        "retrieval_ms": round(elapsed, 3),
                    }
                )

    steady = latency_ms[1:] if len(latency_ms) > 1 else latency_ms
    return {
        "per_question": rows,
        "aggregate": aggregate(rows, k_recall=k_recall, k_rank=k_rank),
        "latency": {
            "cold_first_query_ms": round(first_query_ms, 3) if first_query_ms is not None else None,
            "steady_state": latency_stats(steady),
        },
        "settings": {
            "top_k": top_k,
            "k_recall": k_recall,
            "k_rank": k_rank,
            "repeat": repeat,
        },
    }


def compare_methods(
    cfg: Dict[str, Any],
    examples: Sequence[EvalExample],
    methods: Sequence[str],
    store,
    top_k: int = 10,
    k_recall: int = 5,
    k_rank: int = 10,
    repeat: int = 1,
) -> Dict[str, Any]:
    """Evaluate several retrieval methods over one corpus / one question set."""
    results: Dict[str, Any] = {}
    for method in methods:
        t0 = now_ns()
        retriever = store.build_retriever(
            method=method, retrieval_cfg=cfg.get("retrieval", {})
        )
        build_ms = (now_ns() - t0) / 1_000_000.0
        out = evaluate_retriever(
            retriever,
            examples,
            top_k=top_k,
            k_recall=k_recall,
            k_rank=k_rank,
            repeat=repeat,
        )
        diagnostics = retriever.diagnostics() if hasattr(retriever, "diagnostics") else {}
        out["method"] = method
        out["n_chunks"] = getattr(retriever, "n_chunks", None)
        out["build_ms"] = round(build_ms, 3)
        out["diagnostics"] = diagnostics
        results[method] = out
    return results


def _model_load_ms(diagnostics: Dict[str, Any]) -> Optional[float]:
    if diagnostics.get("model_load_ms") is not None:
        return diagnostics["model_load_ms"]
    return (diagnostics.get("components", {}).get("dense") or {}).get("model_load_ms")


def _table_row(method: str, res: Dict[str, Any], k_recall: int, k_rank: int) -> Dict[str, Any]:
    agg = res["aggregate"]
    lat = res["latency"]
    return {
        "method": method,
        "n_questions": agg["n_questions"],
        "n_scored_questions": agg["n_scored_questions"],
        f"recall_at_{k_recall}": agg.get(f"recall_at_{k_recall}"),
        f"mrr_at_{k_rank}": agg.get(f"mrr_at_{k_rank}"),
        f"ndcg_at_{k_rank}": agg.get(f"ndcg_at_{k_rank}"),
        "hit_rate_at_1": agg.get("hit_rate_at_1"),
        "no_evidence_rate": agg.get("no_evidence_rate"),
        "build_ms": res.get("build_ms"),
        "model_load_ms": _model_load_ms(res.get("diagnostics") or {}),
        "cold_first_query_ms": lat.get("cold_first_query_ms"),
        "median_query_ms": (lat.get("steady_state") or {}).get("median_ms"),
        "mean_query_ms": (lat.get("steady_state") or {}).get("mean_ms"),
        "p95_query_ms": (lat.get("steady_state") or {}).get("p95_ms"),
    }


def write_results(
    out_dir: Path | str,
    results: Dict[str, Any],
    *,
    dataset_path: Path | str,
    k_recall: int = 5,
    k_rank: int = 10,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Write ``retrieval_metrics.json``, ``summary.csv`` and ``per_question.csv``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    dataset_p = Path(dataset_path)
    digest = (
        hashlib.sha256(dataset_p.read_bytes()).hexdigest() if dataset_p.exists() else None
    )
    table = [
        _table_row(m, res, k_recall, k_rank) for m, res in results.items()
    ]
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"path": str(dataset_p), "sha256": digest},
        "metrics": {"recall_k": k_recall, "rank_k": k_rank},
        "comparison_table": table,
        "methods": results,
        **(extra or {}),
    }
    json_path = d / "retrieval_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_path = d / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(table[0].keys()) if table else ["method"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    per_q_path = d / "per_question.csv"
    with per_q_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "method", "question_id", "category", "question",
            f"recall_at_{k_recall}", f"reciprocal_rank", "ndcg",
            "retrieval_ms", "relevant_chunk_ids", "retrieved_chunk_ids",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for method, res in results.items():
            for row in res["per_question"]:
                writer.writerow(
                    {
                        "method": method,
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "question": row["question"],
                        f"recall_at_{k_recall}": row["recall_at_k"],
                        "reciprocal_rank": row["reciprocal_rank"],
                        "ndcg": row["ndcg"],
                        "retrieval_ms": row["retrieval_ms"],
                        "relevant_chunk_ids": ";".join(row["relevant"]),
                        "retrieved_chunk_ids": ";".join(row["retrieved"]),
                    }
                )
    return {
        "json": str(json_path),
        "summary_csv": str(summary_path),
        "per_question_csv": str(per_q_path),
    }
