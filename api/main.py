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
from adaptive_qa.decision import build_decision_engine
from adaptive_qa.decision.preflight import CollectionPreflight
from adaptive_qa.documents import (
    DOCUMENT_ROOT,
    DEFAULT_MAX_UPLOAD_BYTES,
    DocumentStore,
    DocumentUploadError,
)
from adaptive_qa.eval.retrieval_eval import load_eval_examples
from adaptive_qa.experiments_view import (
    chunk_details,
    list_policy_runs,
    list_retrieval_runs,
    load_policy_view,
    load_retrieval_view,
)
from adaptive_qa.retrieval import RETRIEVAL_METHODS
from adaptive_qa.retrieval.compare import compare_methods
from adaptive_qa.types import CorpusDoc
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
    "preflight": None,         # (active-doc signature, CollectionPreflight)
}

EVAL_SET_DIR = resolve_path("eval_data/retrieval_eval")
RESULTS_RETRIEVAL_DIR = resolve_path("results/retrieval_eval")
RESULTS_POLICY_DIR = resolve_path("results/policy_eval")
RESULTS_ROUTING_DIR = resolve_path("results/routing_eval")


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
        # Reused across requests: a backend's generation_cold flag marks ITS first call, so
        # fresh instances per request would label every live answer "cold".
        if _STATE.get("llm") is None:
            cfg = self.cfg["answering"]["llm"] if "llm" in self.cfg["answering"] else {}
            _STATE["llm"] = (
                LLMBackend(cfg, grounded=False, name="llm_closedbook"),
                LLMBackend(cfg, grounded=True, name="llm_grounded"),
            )
        return _STATE["llm"]


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


def _invalidate_collection() -> None:
    """Any change to the active collection drops cached retrievers + preflight."""
    _STATE["retrievers"] = {}
    _STATE["preflight"] = None


def _collection_preflight() -> Optional[CollectionPreflight]:
    """Aggregate document-relevance prober over the ACTIVE chunks (cached).

    One BM25 index build per collection snapshot; the first probe after a
    change is flagged ``cold`` so its cost is visible instead of hidden.
    """
    if not _docs().active_documents():
        return None
    key = _active_signature()
    cached = _STATE.get("preflight")
    if cached is not None and cached[0] == key:
        return cached[1]
    chunks = _docs().load_chunks()
    names = [m["filename"] for m in _docs().active_documents()]
    top_k = int(
        ((_runner().cfg.get("decision", {}).get("heuristic", {}) or {}).get("preflight") or {})
        .get("top_k", 5)
    )
    prober = CollectionPreflight(
        [CorpusDoc(c["chunk_id"], names[0] if names else "", c["text"], ()) for c in chunks],
        top_k=top_k,
        doc_names=names,
    )
    _STATE["preflight"] = (key, prober)
    return prober


def _collection_info(method: str, top_k: int) -> Dict[str, Any]:
    active = _docs().active_documents()
    dense_cfg = (_runner().cfg.get("retrieval", {}) or {}).get("dense", {}) or {}
    return {
        "active_documents": [
            {
                "filename": m.get("filename"),
                "n_chunks": int(m.get("n_chunks") or 0),
                "n_pages": m.get("n_pages"),
            }
            for m in active
        ],
        "n_documents": len(active),
        "n_chunks": sum(int(m.get("n_chunks") or 0) for m in active),
        "retrieval_method": method,
        "top_k": int(top_k),
        "embedding_model": (
            dense_cfg.get("model_name") if method in ("dense", "hybrid") else None
        ),
    }


def _serve_flow(retrieval_method: Optional[str] = None, top_k: Optional[int] = None) -> ServeFlow:
    """Interactive flow: decision engine FIRST, real LLM backends, retrievers.

    The decision engine is rebuilt per request with the collection preflight
    attached, so the live decision depends on the ACTIVE uploaded documents
    (and its preflight cost is measured inside the decision step).
    """
    runner = _runner()
    closed, grounded = _LLMBundle(runner.cfg).backends()
    method = retrieval_method or _default_method()
    k = int(top_k or runner.top_k)
    collection_retriever = _collection_retriever(method)
    engine = build_decision_engine(runner.cfg, None, _collection_preflight())
    return ServeFlow(
        decision_engine=engine,
        method_a=closed,
        method_b=grounded,
        benchmark_retriever=runner.retriever,
        collection_retriever=collection_retriever,
        top_k=k,
        collection_info=_collection_info(method, k),
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
    # Any collection change invalidates the cached retrievers + preflight.
    _invalidate_collection()
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
    _invalidate_collection()
    return JSONResponse({"doc_id": doc_id, "active": active})


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    if not _docs().delete(doc_id):
        raise HTTPException(status_code=404, detail="document not found")
    _invalidate_collection()
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
    # find_spec checks the package is installed without importing PyTorch (many seconds
    # on first call), which left the dashboard's retriever list empty while it loaded.
    from importlib.util import find_spec

    dense_available = find_spec("sentence_transformers") is not None
    dense_reason = "" if dense_available else (
        "sentence-transformers is not installed; run "
        "pip install sentence-transformers to enable dense/hybrid retrieval."
    )
    return JSONResponse(
        {
            "methods": list(RETRIEVAL_METHODS),
            "default": _default_method(),
            "default_top_k": int(_runner().top_k),
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
            # The model id is not a secret and the dashboard shows which model
            # answered; the API key itself is never returned.
            "model": args.get("model"),
            "local_endpoint": bool(
                (args.get("base_url") or "").startswith("http://localhost")
                or (args.get("base_url") or "").startswith("http://127.0.0.1")
            ),
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
    top_k = body.get("top_k")
    if top_k is not None:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="top_k must be an integer")
        if not 1 <= top_k <= 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    # Pass top_k only when the request overrides it, so monkeypatched flows
    # that only accept ``retrieval_method`` keep working.
    flow = (
        _serve_flow(retrieval_method=method)
        if top_k is None
        else _serve_flow(retrieval_method=method, top_k=top_k)
    )
    try:
        result = (
            flow.run_adaptive_only(question)
            if mode == "adaptive"
            else flow.run_comparison(question)
        )
    except Exception as exc:  # configuration errors surface to users clearly
        raise HTTPException(status_code=502, detail=f"Serve flow failed: {exc}")
    return JSONResponse(result)


# ============================================================================
# Demo mode, three-method retrieval comparison, and the Experiments view
# ============================================================================

def _eval_files() -> List[Any]:
    if not EVAL_SET_DIR.exists():
        return []
    return sorted(EVAL_SET_DIR.glob("*.jsonl"))


@app.get("/demo/examples")
def demo_examples() -> JSONResponse:
    """Representative questions from the labelled evaluation set.

    Reference answers and chunk labels are deliberately NOT included: the demo
    only needs the question text, and the live engine must never see labels.
    """
    files = _eval_files()
    if not files:
        return JSONResponse(
            {
                "available": False,
                "reason": f"no *.jsonl evaluation set found under {EVAL_SET_DIR}",
                "categories": [],
            }
        )
    path = files[0]
    try:
        examples = load_eval_examples(path)
    except Exception as exc:
        return JSONResponse(
            {"available": False, "reason": f"{type(exc).__name__}: {exc}", "categories": []}
        )
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        groups.setdefault(ex.category, []).append(
            {"question_id": ex.question_id, "question": ex.question}
        )
    order = [
        "exact_terminology",
        "paraphrase",
        "definition",
        "multi_passage",
        "unanswerable",
    ]
    categories = [
        {
            "name": name,
            "count": len(groups[name]),
            "answerable": name != "unanswerable",
            "examples": groups[name],
        }
        for name in order
        if name in groups
    ]
    for name in sorted(groups):
        if name not in order:
            categories.append(
                {
                    "name": name,
                    "count": len(groups[name]),
                    "answerable": True,
                    "examples": groups[name],
                }
            )
    return JSONResponse(
        {
            "available": True,
            "source": path.name,
            "n_questions": len(examples),
            "categories": categories,
            "note": (
                "Question text only. Reference answers and relevance labels stay "
                "in the evaluation harness and are never sent to the live "
                "decision engine or the answer prompts."
            ),
        }
    )


@app.post("/retrieval/compare")
def retrieval_compare(payload: Dict[str, Any] = None) -> JSONResponse:
    """Run BM25, dense and hybrid on ONE question over the same chunk set."""
    body = payload or {}
    question = (body.get("question", "") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    if not _docs().active_documents():
        raise HTTPException(
            status_code=400,
            detail="No active documents: upload and activate a document first.",
        )
    methods = body.get("methods") or list(RETRIEVAL_METHODS)
    try:
        methods = [_collection_method(m) for m in methods]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    top_k = body.get("top_k", 5)
    if top_k is None:
        top_k = 5
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="top_k must be an integer")
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    method_for_collection = _default_method()
    result = compare_methods(
        question,
        methods,
        top_k,
        retriever_provider=_collection_retriever,
        collection=_collection_info(method_for_collection, top_k),
    )
    return JSONResponse(result)


@app.get("/experiments/retrieval")
def experiments_retrieval(run: Optional[str] = None) -> JSONResponse:
    """Saved BM25 / dense / hybrid retrieval evaluation (read-only)."""
    runs = list_retrieval_runs(RESULTS_RETRIEVAL_DIR)
    if not runs:
        raise HTTPException(
            status_code=404,
            detail=(
                "no retrieval-evaluation results found; run "
                "scripts/eval_retrieval.py first"
            ),
        )
    view = load_retrieval_view(RESULTS_RETRIEVAL_DIR, run_id=run)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run!r}")
    return JSONResponse({"available": True, "runs": runs, "view": view})


@app.get("/experiments/retrieval/question/{question_id}")
def experiments_retrieval_question(
    question_id: str, run: Optional[str] = None
) -> JSONResponse:
    """Per-question inspector: retrieved passages, ranks and gold labels.

    EVALUATION ONLY. This is the only place labels are exposed; it is not on
    the live answering path.
    """
    view = load_retrieval_view(RESULTS_RETRIEVAL_DIR, run_id=run)
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
    details = chunk_details(_docs(), all_ids)
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


@app.get("/experiments/policy")
def experiments_policy(run: Optional[str] = None) -> JSONResponse:
    """Saved adaptive-vs-never/always/oracle policy evaluation (read-only)."""
    runs = list_policy_runs(RESULTS_POLICY_DIR)
    if not runs:
        raise HTTPException(
            status_code=404,
            detail="no policy-evaluation results found; run scripts/eval_policy.py first",
        )
    view = load_policy_view(RESULTS_POLICY_DIR, run_id=run)
    if view is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run!r}")
    return JSONResponse({"available": True, "runs": runs, "view": view})


@app.get("/experiments/routing")
def experiments_routing(run: Optional[str] = None) -> JSONResponse:
    """Saved routing evaluation: does the router retrieve when retrieval helps? (read-only)."""
    import json as _json

    runs = sorted((d for d in RESULTS_ROUTING_DIR.glob("*") if (d / "routing_results.json").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True) if RESULTS_ROUTING_DIR.exists() else []
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
