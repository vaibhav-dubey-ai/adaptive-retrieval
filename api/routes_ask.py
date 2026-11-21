"""Question answering routes: /ask, /ask/benchmark, demo examples, retrieval config and the three-method comparison."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from adaptive_qa.answering.llm_backend import resolve_chat_kwargs
from adaptive_qa.common import resolve_path
from adaptive_qa.eval.retrieval_eval import load_eval_examples
from adaptive_qa.retrieval import RETRIEVAL_METHODS
from adaptive_qa.retrieval.compare import compare_methods
from adaptive_qa.eval.metrics import label_outcome
from adaptive_qa.hotpot.pipeline import Pipeline
from fastapi import APIRouter
from api import deps

router = APIRouter()


@router.get("/")
def index():
    return FileResponse(str(resolve_path("dashboard/index.html")))


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask/benchmark")
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

    runner = deps._runner()
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
    for row in deps._eval_rows():
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


def _collection_method(name: str) -> str:
    """Normalise a requested retrieval method (400 on unknown names)."""
    from adaptive_qa.retrieval import normalize_method

    return normalize_method(name)


@router.get("/retrieval/config")
def retrieval_config() -> JSONResponse:
    """Retrieval methods selectable from the dashboard, plus real diagnostics."""
    from adaptive_qa.retrieval import RETRIEVAL_METHODS

    cfg = deps._runner().cfg.get("retrieval", {})
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
            "default": deps._default_method(),
            "default_top_k": int(deps._runner().top_k),
            "dense": {
                "model_name": (cfg.get("dense") or {}).get("model_name"),
                "available": dense_available,
                "reason": dense_reason,
            },
            "hybrid": cfg.get("hybrid") or {},
            "active_collection": deps._active_signature(),
        }
    )


@router.get("/llm/config-status")
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


@router.post("/ask")
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
    method = body.get("retrieval_method") or deps._default_method()
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
        deps._serve_flow(retrieval_method=method)
        if top_k is None
        else deps._serve_flow(retrieval_method=method, top_k=top_k)
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


def _eval_files() -> List[Any]:
    if not deps.EVAL_SET_DIR.exists():
        return []
    return sorted(deps.EVAL_SET_DIR.glob("*.jsonl"))


@router.get("/demo/examples")
def demo_examples(dataset: str = "dbms") -> JSONResponse:
    """Representative questions from the labelled evaluation set.

    Reference answers and chunk labels are deliberately NOT included: the demo
    only needs the question text, and the live engine must never see labels.
    """
    files = _eval_files()
    if dataset == "finance":
        files = [f for f in (deps.EVAL_SET_DIR / "jpm_final_questions.jsonl",
                             resolve_path("eval_data/routing_eval/jpm_general_questions.jsonl")) if f.exists()]
    elif dataset != "dbms":
        raise HTTPException(status_code=400, detail=f"unknown dataset {dataset!r}")
    if not files:
        return JSONResponse(
            {
                "available": False,
                "reason": f"no *.jsonl evaluation set found under {deps.EVAL_SET_DIR}",
                "categories": [],
            }
        )
    path = files[0]
    try:
        # finance = corpus questions + general/unanswerable ones; dbms keeps its original single file
        examples = [e for f in (files if dataset == "finance" else files[:1]) for e in load_eval_examples(f)]
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


@router.post("/retrieval/compare")
def retrieval_compare(payload: Dict[str, Any] = None) -> JSONResponse:
    """Run BM25, dense and hybrid on ONE question over the same chunk set."""
    body = payload or {}
    question = (body.get("question", "") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    if not deps._docs().active_documents():
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
    method_for_collection = deps._default_method()
    result = compare_methods(
        question,
        methods,
        top_k,
        retriever_provider=deps._collection_retriever,
        collection=deps._collection_info(method_for_collection, top_k),
    )
    return JSONResponse(result)
