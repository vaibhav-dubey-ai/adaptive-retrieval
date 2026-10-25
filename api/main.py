"""FastAPI backend for the research dashboard.

LD-1 boundaries (requirement §10):
- ``/ask`` answers with the ADAPTIVE workflow only (decision engine ->
  selected path). Reference answers and oracle labels are never exposed to
  the decision engine and never included in the payload of the live mode.
- ``/ask?mode=cross`` performs cross-verification: both paths run, retrieved
  passages and scores are shown, and - only when offline evaluation data is
  available for the question - the decision-correctness flag is attached and
  clearly marked as evaluation-only.
- ``/evaluate/run`` executes a batch of the four-policy experiment.
- ``/evaluate/summary|export`` expose aggregates and per-question CSV/JSON.

Tests call the modules directly; this HTTP layer is a thin, optional shell
(the CLI pipeline works without it).
"""

from __future__ import annotations

import io
import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from adaptive_qa.answering.llm_backend import LLMBackend, resolve_chat_kwargs
from adaptive_qa.config import load_config, resolve_path
from adaptive_qa.documents import (
    DOCUMENT_ROOT,
    DEFAULT_MAX_UPLOAD_BYTES,
    DocumentStore,
    DocumentUploadError,
)
from adaptive_qa.experiment import ExperimentRunner
from adaptive_qa.eval.outcome import label_outcome
from adaptive_qa.pipeline import Pipeline
from adaptive_qa.serialization import read_jsonl, read_json
from adaptive_qa.serve import ServeFlow

app = FastAPI(title="Adaptive QA Research Dashboard", version="0.2.0")

_STATE: Dict[str, Any] = {
    "runner": None,
    "eval_rows": None,
    "docs": None,
    "retrievers": {},          # (method, active-doc signature) -> CollectionRetriever
}


def _runner() -> ExperimentRunner:
    if _STATE["runner"] is None:
        cfg = load_config(os.environ.get("ADAPTIVE_QA_CONFIG", "configs/default.json"))
        _STATE["runner"] = ExperimentRunner(cfg)
    runner: ExperimentRunner = _STATE["runner"]
    return runner


def _docs() -> DocumentStore:
    if _STATE["docs"] is None:
        _STATE["docs"] = DocumentStore(
            DOCUMENT_ROOT,
            max_upload_bytes=int(
                os.environ.get("ADAPTIVE_QA_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)
            ),
        )
    return _STATE["docs"]


@dataclass
class _LLMBundle:
    """Lazy LLM backend bundle; raises a CLEAR error when unconfigured."""
    cfg: Dict[str, Any]

    def backends(self) -> tuple:
        cfg = self.cfg["answering"]["llm"] if "llm" in self.cfg["answering"] else {}
        closed = LLMBackend(cfg, grounded=False, name="llm_closedbook")
        grounded = LLMBackend(cfg, grounded=True, name="llm_grounded")
        return closed, grounded


def _active_signature() -> tuple:
    """Identity of the active document collection (cache key component)."""
    return tuple(
        (m.get("doc_id"), bool(m.get("active")), int(m.get("n_chunks") or 0))
        for m in _docs().list_documents()
    )


def _default_method() -> str:
    return str(_runner().cfg.get("retrieval", {}).get("retriever", "bm25"))


def _collection_retriever(method: Optional[str] = None):
    """Cached retriever for the active collection and the requested method.

    Building a dense/hybrid retriever re-reads chunk metadata and consults the
    persistent embedding cache; caching the instance keeps repeated dashboard
    requests fast. The cache key includes the active-document signature, so
    uploading, deleting, activating or deactivating a document rebuilds it.
    """
    from adaptive_qa.retrieval import normalize_method

    runner = _runner()
    chosen = normalize_method(method or _default_method())
    if not _docs().active_documents():
        return None
    key = (chosen, _active_signature())
    if key not in _STATE["retrievers"]:
        _STATE["retrievers"][key] = _docs().build_retriever(
            method=chosen, retrieval_cfg=runner.cfg.get("retrieval", {})
        )
    return _STATE["retrievers"][key]


def _serve_flow(retrieval_method: Optional[str] = None) -> ServeFlow:
    """Interactive flow: decision engine FIRST, real LLM backends, retrievers."""
    runner = _runner()
    closed, grounded = _LLMBundle(runner.cfg).backends()
    collection_retriever = _collection_retriever(retrieval_method)
    return ServeFlow(
        decision_engine=runner.decision_engine,
        method_a=closed,
        method_b=grounded,
        benchmark_retriever=runner.retriever,
        collection_retriever=collection_retriever,
        top_k=runner.top_k,
    )


def _eval_rows():
    if _STATE["eval_rows"] is None:
        path_cfg = resolve_path("runs")
        # latest run's per_question.jsonl if present, else None
        candidates = sorted(path_cfg.glob("*/per_question.jsonl"))
        _STATE["eval_rows"] = read_jsonl(str(candidates[-1])) if candidates else []
    return _STATE["eval_rows"]


@app.get("/")
def index():
    return FileResponse(str(resolve_path("dashboard/index.html")))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/benchmark")
def ask_benchmark(payload: Dict[str, Any], mode: str = Query(default="adaptive")) -> JSONResponse:
    """Live adaptive answering; ``mode=cross`` adds cross-verification.

    In adaptive mode the ONLY executed work is decision + selected path. In
    cross mode both paths execute; retrieved passages, timings and (when a
    saved eval run contains the question) the decision-correctness flag are
    returned and marked ``evaluation_only``.
    """
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")

    runner = _runner()
    if mode not in ("adaptive", "cross"):
        raise HTTPException(status_code=400, detail="mode must be adaptive|cross")

    pipeline = Pipeline(
        retriever=runner.retriever,
        method_a=runner.method_a,
        method_b=runner.method_b,
        decision_engine=runner.decision_engine,
        top_k=runner.top_k,
    )

    if mode == "adaptive":
        rec = pipeline.run_adaptive("api-live", question)
        return JSONResponse({"mode": mode, "record": rec.to_dict()})

    rec = pipeline.run_cross_verify("api-cross", question)

    # Offline evaluation linkage (clearly labeled, never fed to the engine)
    eval_link = None
    for row in _eval_rows():
        if row.get("question", "").strip().lower() == question.lower():
            qrec = None
            for q2 in runner.records_eval:
                if q2.question.strip().lower() == question.lower():
                    qrec = q2
                    break
            if qrec is not None:
                outcome = label_outcome(qrec, rec.answer_a, rec.answer_b)
                # decision correctness vs the SAVED run's decision on the
                # same question (scored offline; consistent with the oracle
                # tie-break: 'similar' counts as correct either way, since
                # both methods are practically equivalent there)
                expected = None
                if outcome.label == "retrieval_better":
                    expected = "RETRIEVE"
                elif outcome.label == "no_retrieval_better":
                    expected = "DO_NOT_RETRIEVE"
                saved_decision = row.get("decision")
                decision_correct = (
                    (expected is None)
                    or (saved_decision is None)
                    or (saved_decision == expected)
                )
                eval_link = {
                    "evaluation_only": True,
                    "question_id": row.get("question_id"),
                    "reference_answer": qrec.answer,
                    "gold_titles": list(qrec.gold_titles),
                    "saved_run_decision": saved_decision,
                    "decision_correct": bool(decision_correct),
                    "outcome_label": outcome.label,
                    "quality_a": outcome.quality_a,
                    "quality_b": outcome.quality_b,
                    "note": "Computed offline against saved runs; never shown to the decision engine.",
                }
            break

    return JSONResponse(
        {
            "mode": mode,
            "record": rec.to_dict(),
            "evaluation_link": eval_link,
        }
    )


@app.post("/evaluate/run")
def evaluate_run(payload: Dict[str, Any] = None) -> JSONResponse:
    n = int(((payload or {}).get("limit")) or 0)
    runner = _runner()
    questions = runner.records_eval
    if n > 0:
        questions = questions[:n]
    artifacts = runner.evaluate_questions(questions)
    from adaptive_qa.eval.charts import write_all

    write_all(artifacts.summary, artifacts.run_dir)
    return JSONResponse({"run_dir": artifacts.run_dir, "summary": artifacts.summary})


@app.get("/evaluate/summary")
def evaluate_summary() -> JSONResponse:
    candidates = sorted(resolve_path("runs").glob("*/summary.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail="no completed runs; POST /evaluate/run first")
    return JSONResponse(read_json(str(candidates[-1])))


_CHART_NAMES = {"quality_vs_latency.png", "quality_vs_cost.png", "retrieval_rate.png"}


@app.get("/evaluate/chart/{name}")
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


@app.get("/evaluate/export")
def evaluate_export(format: str = Query(default="csv")):
    rows = _eval_rows()
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


# ============================================================================
# Interactive document upload + side-by-side answering (requirements 3, 4)
# ============================================================================

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a PDF/TXT/MD/DOCX, persist it, chunk + index it."""
    data = await file.read()
    try:
        meta = _docs().save_document(file.filename or "", data)
    except DocumentUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Any collection change invalidates the cached retrievers.
    _STATE["retrievers"] = {}
    return JSONResponse(
        {
            "document": meta,
            "message": (
                "Document already existed (deduplicated by content)."
                if meta.get("duplicate")
                else "Uploaded, chunked and indexed into the active collection."
            ),
        }
    )


@app.get("/documents")
def list_documents() -> JSONResponse:
    docs = _docs().list_documents()
    return JSONResponse({"documents": docs})


@app.post("/documents/{doc_id}/active")
def set_document_active(doc_id: str, payload: Dict[str, Any] = None) -> JSONResponse:
    active = bool(((payload or {}).get("active")) or ((payload or {}).get("value")))
    if not _docs().set_active(doc_id, active):
        raise HTTPException(status_code=404, detail="document not found")
    _STATE["retrievers"] = {}
    return JSONResponse({"doc_id": doc_id, "active": active})


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    if not _docs().delete(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    _STATE["retrievers"] = {}
    return JSONResponse({"deleted": doc_id})


def _collection_method(name: str) -> str:
    """Normalise a requested retrieval method (400 on unknown names)."""
    from adaptive_qa.retrieval import normalize_method

    return normalize_method(name)


@app.get("/retrieval/config")
def retrieval_config() -> JSONResponse:
    """Retrieval methods selectable from the dashboard, plus real diagnostics."""
    from adaptive_qa.retrieval import RETRIEVAL_METHODS

    cfg = _runner().cfg.get("retrieval", {})
    try:
        import sentence_transformers  # noqa: F401

        dense_available = True
        dense_reason = ""
    except Exception as exc:  # pragma: no cover - environment dependent
        dense_available = False
        dense_reason = (
            "sentence-transformers is not installed; run "
            "pip install sentence-transformers to enable dense/hybrid retrieval. "
            f"({type(exc).__name__})"
        )
    return JSONResponse(
        {
            "methods": list(RETRIEVAL_METHODS),
            "default": _default_method(),
            "dense": {
                "model_name": (cfg.get("dense") or {}).get("model_name"),
                "available": dense_available,
                "reason": dense_reason,
            },
            "hybrid": cfg.get("hybrid") or {},
            "active_collection": _active_signature(),
        }
    )


@app.get("/llm/config-status")
def llm_config_status() -> JSONResponse:
    """Whether the answering model is configured (WITHOUT echoing values)."""
    args = resolve_chat_kwargs()
    return JSONResponse(
        {
            "api_key_set": bool(args.get("api_key")),
            "model_set": bool(args.get("model")),
            "base_url_set": bool(args.get("base_url")),
        }
    )


@app.post("/ask")
def ask(payload: Dict[str, Any], mode: str = Query(default="adaptive")) -> JSONResponse:
    """Live answering with side-by-side research comparison.

    - mode=compare: decision FIRST, then BOTH branches for research
      comparison. Timing/token columns come from the actual branches.
    - mode=adaptive: deployed operational mode — decision then ONLY the
      selected path; measured wall-clock is decision + (retrieval?) +
      generation.
    """

    body = payload or {}
    question = (body.get("question", "") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    if mode not in ("adaptive", "compare"):
        raise HTTPException(status_code=400, detail="mode must be adaptive|compare")
    method = body.get("retrieval_method") or _default_method()
    try:
        method = _collection_method(method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    flow = _serve_flow(retrieval_method=method)
    try:
        result = (
            flow.run_adaptive_only(question)
            if mode == "adaptive"
            else flow.run_comparison(question)
        )
    except Exception as exc:  # configuration errors surface to users clearly
        raise HTTPException(status_code=502, detail=f"Serve flow failed: {exc}")
    return JSONResponse(result)
