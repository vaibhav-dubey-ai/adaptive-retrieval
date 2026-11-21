"""Saved-results routes: /evaluate/* (HotpotQA experiment) and /experiments/* (retrieval, policy, routing views)."""

from __future__ import annotations

import io
import csv
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from adaptive_qa.common import resolve_path
from adaptive_qa.live.experiments_view import chunk_details, list_policy_runs, list_retrieval_runs, load_policy_view, load_retrieval_view
from adaptive_qa.common import read_json
from fastapi import APIRouter
from api import deps

router = APIRouter()


@router.post("/evaluate/run")
def evaluate_run(payload: Dict[str, Any] = None) -> JSONResponse:
    n = int(((payload or {}).get("limit")) or 0)
    runner = deps._runner()
    questions = runner.records_eval
    if n > 0:
        questions = questions[:n]
    artifacts = runner.evaluate_questions(questions)
    from adaptive_qa.eval.charts import write_all

    write_all(artifacts.summary, artifacts.run_dir)
    return JSONResponse({"run_dir": artifacts.run_dir, "summary": artifacts.summary})


@router.get("/evaluate/summary")
def evaluate_summary() -> JSONResponse:
    candidates = sorted(resolve_path("runs").glob("*/summary.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail="no completed runs; POST /evaluate/run first")
    return JSONResponse(read_json(str(candidates[-1])))


_CHART_NAMES = {"quality_vs_latency.png", "quality_vs_cost.png", "retrieval_rate.png"}


@router.get("/evaluate/chart/{name}")
def evaluate_chart(name: str):
    """Serve a chart PNG from the most recent completed run (whitelist)."""
    if name not in _CHART_NAMES:
        raise HTTPException(status_code=404, detail="unknown chart")
    runs = sorted(resolve_path("runs").glob("*/summary.json"))
    if not runs:
        raise HTTPException(status_code=404, detail="no completed runs")
    png = runs[-1].parent / name
    if not png.exists():
        raise HTTPException(status_code=404, detail="chart not generated in latest run")
    return FileResponse(str(png), media_type="image/png")


def _rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = [
        "question_id", "question", "split", "policy", "decision", "decision_score",
        "primary_method",
    ]
    # outsourced columns for per-method evaluation
    extra = [
        "em_a", "f1_a", "em_b", "f1_b", "outcome_label", "delta",
        "decision_ms", "retrieval_ms", "gen_a_ms", "gen_b_ms",
        "adaptive_e2e_ms", "cross_verify_ms", "evaluation_ms",
        "answer_a_text", "answer_b_text", "primary_answer_text", "retrieval_title_1",
        "retrieval_score_1", "reference_answer",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields + extra, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        out = {
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "split": row.get("split"),
            "policy": row.get("policy"),
            "decision": row.get("decision"),
            "decision_score": row.get("decision_score"),
            "primary_method": row.get("primary_method"),
        }
        outcome = row.get("outcome") or {}
        for k in ("em_a", "f1_a", "em_b", "f1_b", "delta"):
            out[k] = outcome.get(k)
        out["outcome_label"] = outcome.get("label")
        for k in ("decision_ms", "retrieval_ms", "gen_a_ms", "gen_b_ms", "adaptive_e2e_ms", "cross_verify_ms", "evaluation_ms"):
            out[k] = row.get(k)
        out["answer_a_text"] = (row.get("answer_a") or {}).get("text")
        out["answer_b_text"] = (row.get("answer_b") or {}).get("text")
        out["primary_answer_text"] = (row.get("primary_answer") or {}).get("text")
        hits = row.get("retrieval_hits") or []
        if hits:
            out["retrieval_title_1"] = hits[0].get("title")
            out["retrieval_score_1"] = hits[0].get("score")
        out["reference_answer"] = outcome.get("reference_answer")
        writer.writerow(out)
    return buf.getvalue()


@router.get("/evaluate/export")
def evaluate_export(format: str = Query(default="csv")):
    rows = deps._eval_rows()
    if not rows:
        raise HTTPException(status_code=404, detail="no per_question.jsonl run yet")
    if format == "json":
        return JSONResponse(rows)
    if format == "csv":
        csv_text = _rows_to_csv(rows)
        return StreamingResponse(
            io.StringIO(csv_text),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=per_question.csv"},
        )
    raise HTTPException(status_code=400, detail="format must be csv|json")


@router.get("/experiments/retrieval")
def experiments_retrieval(run: Optional[str] = None, dataset: str = "dbms") -> JSONResponse:
    """Saved BM25 / dense / hybrid retrieval evaluation (read-only)."""
    rdir = deps._results_dir("retrieval_eval", dataset)
    runs = list_retrieval_runs(rdir)
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                "no retrieval-evaluation results found; run "
                "scripts/eval_retrieval.py first"
            ),
        )
    view = load_retrieval_view(rdir, run_id=run)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run!r}")
    return JSONResponse({"available": True, "runs": runs, "view": view})


@router.get("/experiments/retrieval/question/{question_id}")
def experiments_retrieval_question(
    question_id: str, run: Optional[str] = None, dataset: str = "dbms"
) -> JSONResponse:
    """Per-question inspector: retrieved passages, ranks and gold labels.

    EVALUATION ONLY. This is the only place labels are exposed; it is not on
    the live answering path.
    """
    view = load_retrieval_view(deps._results_dir("retrieval_eval", dataset), run_id=run)
    if view is None:
        raise HTTPException(status_code=404, detail="no retrieval-evaluation run")
    row = next(
        (q for q in view["questions"] if q["question_id"] == question_id), None
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"question {question_id!r} not in run")
    all_ids = list(row["relevant_chunk_ids"])
    for res in row["methods"].values():
        all_ids.extend(res["retrieved"])
    details = chunk_details(deps._docs(), all_ids)
    relevant = set(row["relevant_chunk_ids"])
    methods = {}
    for method, res in row["methods"].items():
        methods[method] = {
            "recall_at_k": res["recall_at_k"],
            "reciprocal_rank": res["reciprocal_rank"],
            "ndcg": res["ndcg"],
            "retrieval_ms": res["retrieval_ms"],
            "retrieved": [
                {
                    "rank": i + 1,
                    "chunk_id": cid,
                    "is_relevant": cid in relevant,
                    "source": details.get(cid),
                }
                for i, cid in enumerate(res["retrieved"])
            ],
        }
    return JSONResponse(
        {
            "evaluation_only": True,
            "run_id": view["run_id"],
            "question": {
                "question_id": row["question_id"],
                "question": row["question"],
                "category": row["category"],
            },
            "relevant_chunk_ids": row["relevant_chunk_ids"],
            "relevant_passages": [details.get(c) for c in row["relevant_chunk_ids"]],
            "methods": methods,
            "note": "Labels come from hand labelling and are never used live.",
        }
    )


@router.get("/experiments/policy")
def experiments_policy(run: Optional[str] = None, dataset: str = "dbms") -> JSONResponse:
    """Saved adaptive-vs-never/always/oracle policy evaluation (read-only)."""
    pdir = deps._results_dir("policy_eval", dataset)
    runs = list_policy_runs(pdir) if pdir.exists() else []
    if not runs:
        raise HTTPException(
            status_code=404,
            detail="no policy-evaluation results found; run scripts/eval_policy.py first",
        )
    view = load_policy_view(pdir, run_id=run)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run!r}")
    return JSONResponse({"available": True, "runs": runs, "view": view})


@router.get("/experiments/routing")
def experiments_routing(run: Optional[str] = None, dataset: str = "dbms") -> JSONResponse:
    """Saved routing evaluation: does the router retrieve when retrieval helps? (read-only)."""
    import json as _json

    rtdir = deps._results_dir("routing_eval", dataset)
    runs = sorted((d for d in rtdir.glob("*") if (d / "routing_results.json").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True) if rtdir.exists() else []
    if not runs:
        raise HTTPException(status_code=404,
                            detail="no routing results found; run scripts/eval_routing.py first")
    chosen = next((d for d in runs if d.name == run), runs[0]) if run else runs[0]
    payload = _json.loads((chosen / "routing_results.json").read_text(encoding="utf-8"))
    rows = []
    qfile = chosen / "per_question.jsonl"
    if qfile.exists():
        for line in qfile.read_text(encoding="utf-8").splitlines():
            r = _json.loads(line)
            rows.append({k: r.get(k) for k in (
                "question_id", "category", "question", "reference_answer", "answer_a", "answer_b",
                "quality_a", "quality_b", "retrieval_helps", "retrieval_hurts", "heuristic_score",
                "decision", "b_fallback", "tokens_a", "tokens_b", "gen_a_ms", "gen_b_ms", "retrieval_ms")})

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    payload["run_id"] = chosen.name
    payload["branch_costs"] = {k: mean(k) for k in ("quality_a", "quality_b", "tokens_a", "tokens_b",
                                                     "gen_a_ms", "gen_b_ms", "retrieval_ms")}
    # Ties = neither helps nor hurts (|quality delta| < gap); fallback ties had no BM25 passage, so B == A by construction.
    scored = [r for r in rows if r.get("quality_a") is not None and r.get("quality_b") is not None]
    ties = [r for r in scored if not r.get("retrieval_helps") and not r.get("retrieval_hurts")]
    payload["outcome_counts"] = {
        "helps": sum(bool(r.get("retrieval_helps")) for r in scored),
        "hurts": sum(bool(r.get("retrieval_hurts")) for r in scored),
        "ties": len(ties),
        "ties_no_passage_fallback": sum(bool(r.get("b_fallback")) for r in ties),
    }
    payload["questions"] = rows
    return JSONResponse({"available": True, "view": payload})
