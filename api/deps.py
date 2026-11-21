"""Shared API state and helpers: lazy singletons (experiment runner, document store, LLM bundle), collection retriever/preflight caches and result-folder paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from fastapi import HTTPException
from adaptive_qa.answering.llm_backend import LLMBackend
from adaptive_qa.common import load_config, resolve_path
from adaptive_qa.decision import build_decision_engine
from adaptive_qa.decision.preflight import CollectionPreflight
from adaptive_qa.live.documents import DOCUMENT_ROOT, DEFAULT_MAX_UPLOAD_BYTES, DocumentStore
from adaptive_qa.common import CorpusDoc
from adaptive_qa.hotpot.experiment import ExperimentRunner
from adaptive_qa.common import read_jsonl
from adaptive_qa.live.serve import ServeFlow


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


def _results_dir(kind: str, dataset: str):
    """Saved-results folder for one dataset view: "dbms" (default, original runs) or "finance"."""
    if dataset not in ("dbms", "finance"):
        raise HTTPException(status_code=400, detail=f"unknown dataset {dataset!r}")
    return resolve_path(f"results/finance/{kind}" if dataset == "finance" else f"results/{kind}")


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
