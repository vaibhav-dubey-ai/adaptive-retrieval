"""Read saved experiment artifacts for the dashboard's Experiments view.

Nothing here re-runs or re-scores an experiment: the view is built from the
JSON/CSV files written by ``scripts/eval_retrieval.py`` and
``scripts/eval_policy.py``, so the numbers on screen are exactly the numbers in
the result files (and the files are named in the payload).

Labels (``relevant_chunk_ids``) appear only in this evaluation-only payload.
The live answering path (``/ask``, ``/retrieval/compare``) never receives them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

METRIC_HELP: Dict[str, str] = {
    "recall_at_5": (
        "Recall@5 — of the passages labelled relevant for a question, what "
        "fraction appears in the retriever's top 5? Averaged over answerable "
        "questions; unanswerable questions are excluded because the value is "
        "undefined without relevant passages."
    ),
    "mrr_at_10": (
        "MRR@10 — 1 / (rank of the first relevant passage), capped at 10, "
        "averaged over answerable questions. 1.0 means a relevant passage was "
        "always ranked first; 0.5 means it was second on average."
    ),
    "ndcg_at_10": (
        "nDCG@10 — rank-discounted gain of the top 10 against the ideal "
        "ordering. Unlike MRR it credits finding MORE than one relevant "
        "passage and penalises relevant passages that appear late."
    ),
    "hit_rate_at_1": (
        "Hit rate@1 — fraction of answerable questions whose very first "
        "passage is relevant."
    ),
    "no_evidence_rate": (
        "No-evidence rate — fraction of ALL questions (including unanswerable "
        "ones) for which the retriever returned no passage at all. A lexical "
        "retriever abstains when no query term matches; a dense retriever "
        "always returns its nearest neighbours, even for questions the corpus "
        "cannot answer."
    ),
    "median_query_ms": (
        "Median retrieval latency over queries after warm-up (milliseconds, "
        "monotonic clock). Excludes index build, model load and corpus encoding "
        "— see build_ms / model_load_ms / cold_first_query_ms."
    ),
    "cold_first_query_ms": (
        "The first query after the retriever was built, including any lazy "
        "initialisation still pending. Dense loads its model inside the build "
        "(see build_ms / model_load_ms), so its first query excludes the load; "
        "hybrid loads lazily, so its first query includes it. Full cold start = "
        "build_ms + cold_first_query_ms. Reported separately so a one-off cost is "
        "never mistaken for steady-state query latency."
    ),
    "build_ms": (
        "Time to construct the retriever over the corpus (index build; for "
        "dense, loading the sentence-embedding model and encoding the chunks "
        "when the embedding cache is empty)."
    ),
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


REQUIRED_METHODS = ("bm25", "dense", "hybrid")


def _run_scope(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Which corpus a saved run covers, and whether its own results agree with that.

    ``scope_warning`` is set when retrieved chunk ids belong to documents the
    run's recorded corpus does not contain (the recorded scope is then wrong).
    """
    corpus = payload.get("corpus") or {}
    doc_ids = list(corpus.get("document_ids") or [])
    seen = {
        cid.split(":")[0]
        for res in (payload.get("methods") or {}).values()
        for row in res.get("per_question") or []
        for cid in row.get("retrieved") or []
    }
    extra = sorted(seen - set(doc_ids)) if doc_ids else []
    return {
        "n_chunks": corpus.get("n_chunks"),
        "document_ids": doc_ids,
        "scope_warning": (
            f"Retrieved chunks come from {len(extra)} document(s) not in the recorded corpus "
            f"({', '.join(extra)}); the recorded corpus size is unreliable and the labels only cover "
            "the recorded document."
            if extra
            else None
        ),
    }


def list_retrieval_runs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        f = d / "retrieval_metrics.json"
        if not f.exists():
            continue
        payload = _read_json(f) or {}
        table = payload.get("comparison_table") or []
        methods = [row.get("method") for row in table]
        scope = _run_scope(payload)
        missing = [m for m in REQUIRED_METHODS if m not in methods]
        out.append(
            {
                "run_id": d.name,
                "generated_at": payload.get("generated_at_utc"),
                "methods": methods,
                "missing_methods": missing,
                "complete": not missing and scope["scope_warning"] is None,
                "n_chunks": scope["n_chunks"],
                "scope_warning": scope["scope_warning"],
                "n_questions": (table[0] or {}).get("n_questions") if table else None,
                "path": str(d),
            }
        )
    return out


def latest_run(root: Path) -> Optional[Path]:
    """Newest run that has BM25, dense and hybrid over a consistent corpus.

    Falls back to the newest run of any kind (the view then flags what is missing),
    so a newer partial run never silently hides the full three-method benchmark.
    """
    runs = list_retrieval_runs(root)
    pick = next((r for r in runs if r["complete"]), runs[0] if runs else None)
    return Path(pick["path"]) if pick else None


def _per_method_questions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge per-question rows of every method into one row per question."""
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for method, res in (payload.get("methods") or {}).items():
        for row in res.get("per_question") or []:
            qid = row.get("question_id")
            if qid not in merged:
                merged[qid] = {
                    "question_id": qid,
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "relevant_chunk_ids": list(row.get("relevant") or []),
                    "answerable": bool(row.get("relevant")),
                    "methods": {},
                }
                order.append(qid)
            merged[qid]["methods"][method] = {
                "retrieved": list(row.get("retrieved") or []),
                "recall_at_k": row.get("recall_at_k"),
                "reciprocal_rank": row.get("reciprocal_rank"),
                "ndcg": row.get("ndcg"),
                "retrieval_ms": row.get("retrieval_ms"),
            }
    return [merged[q] for q in order]


def build_retrieval_view(run_dir: Path) -> Dict[str, Any]:
    payload = _read_json(Path(run_dir) / "retrieval_metrics.json") or {}
    table = payload.get("comparison_table") or []
    methods = [row.get("method") for row in table]
    metrics_present = [
        k
        for k in ("recall_at_5", "mrr_at_10", "ndcg_at_10", "hit_rate_at_1")
        if any(row.get(k) is not None for row in table)
    ]
    # Absolute improvement of every method against the lexical baseline.
    base = next((row for row in table if row.get("method") == "bm25"), None)
    for row in table:
        if base is None or row is base:
            row["vs_bm25"] = {}
            continue
        row["vs_bm25"] = {
            k: round(float(row[k]) - float(base[k]), 4)
            for k in metrics_present
            if row.get(k) is not None and base.get(k) is not None
        }
    # Which method is best per metric (ties listed).
    best: Dict[str, List[str]] = {}
    for k in metrics_present:
        vals = [(row.get(k), row.get("method")) for row in table if row.get(k) is not None]
        if not vals:
            continue
        top = max(v for v, _ in vals)
        best[k] = [m for v, m in vals if v == top]
    scope = _run_scope(payload)
    return {
        "run_id": Path(run_dir).name,
        "run_dir": str(run_dir),
        "generated_at_utc": payload.get("generated_at_utc"),
        "dataset": payload.get("dataset") or {},
        "corpus": scope,
        "methods_missing": [m for m in REQUIRED_METHODS if m not in methods],
        "settings": payload.get("metrics") or {},
        "comparison_table": table,
        "best": best,
        "metric_help": METRIC_HELP,
        "questions": _per_method_questions(payload),
        "notes": [
            "Scores come from the saved result file; the dashboard does not "
            "recompute them.",
            "Recall/MRR/nDCG are computed over answerable questions only; the "
            "no-evidence rate covers all questions.",
            "Labels were fixed before the first evaluation run and are used "
            "only here, never by the live decision engine or answer prompts.",
            "Latency excludes LLM generation entirely; it is retrieval only.",
        ],
        "label_visibility": "evaluation_only",
    }


def load_retrieval_view(root: Path, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if run_id:
        path = Path(root) / run_id
    else:
        path = latest_run(root)
    if path is None or not (Path(path) / "retrieval_metrics.json").exists():
        return None
    return build_retrieval_view(Path(path))


def chunk_details(
    store, chunk_ids: List[str], max_chars: int = 600
) -> Dict[str, Dict[str, Any]]:
    """Map chunk ids to source metadata + excerpt using the document store."""
    wanted = set(chunk_ids)
    out: Dict[str, Dict[str, Any]] = {}
    if not wanted:
        return out
    by_doc: Dict[str, List[str]] = {}
    for cid in wanted:
        doc_id = cid.split(":")[0]
        by_doc.setdefault(doc_id, []).append(cid)
    for doc_id, ids in by_doc.items():
        meta = store.get(doc_id) or {}
        for chunk in store.load_chunks(doc_id):
            if chunk["chunk_id"] not in ids:
                continue
            text = chunk.get("text") or ""
            out[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "filename": meta.get("filename"),
                "page": chunk.get("page"),
                "excerpt": text[:max_chars],
                "chars": len(text),
            }
    return out


def list_policy_runs(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), reverse=True):
        f = d / "policy_results.json"
        if not f.exists():
            continue
        payload = _read_json(f) or {}
        out.append(
            {
                "run_id": d.name,
                "generated_at": payload.get("generated_at_utc"),
                "n_questions": (payload.get("settings") or {}).get("n_questions"),
                "answers_run": (payload.get("settings") or {}).get("answers_run"),
                "path": str(d),
            }
        )
    return out


def load_policy_view(root: Path, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    runs = list_policy_runs(root)
    if not runs:
        return None
    chosen = next((r for r in runs if r["run_id"] == run_id), runs[0]) if run_id else runs[0]
    payload = _read_json(Path(chosen["path"]) / "policy_results.json")
    if payload is None:
        return None
    payload["run_id"] = chosen["run_id"]
    payload["run_dir"] = chosen["path"]
    payload["policy_help"] = {
        "never_retrieve": "Always answers closed-book (Option A).",
        "always_retrieve": "Always retrieves and answers from evidence (Option B).",
        "adaptive": "Uses the decision engine's per-question choice; latency includes the decision step (and its preflight).",
        "oracle": "Evaluation upper bound: picks the better answer by reference-answer score. Never used live.",
    }
    return payload
