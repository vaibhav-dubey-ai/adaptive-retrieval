"""Downstream QA evaluation: does the retrieval method change the final answer?

Four conditions are compared on the SAME questions, corpus, LLM and generation
settings:

    closed_book   question -> LLM (no passages at all)
    bm25          question -> BM25 passages -> LLM
    dense         question -> dense passages -> LLM
    hybrid        question -> RRF(bm25, dense) passages -> LLM

Measured separately per condition
---------------------------------
* retrieval latency (0 for closed book, measured for the others)
* LLM generation latency
* total condition latency
* input / output / total tokens, labelled provider-reported vs estimated
* answer quality against the reference answer: token-F1 and exact match
* ``evidence_coverage``: share of the answer's content tokens that occur in the
  retrieved passages. This is a *lexical grounding proxy*, NOT an entailment or
  faithfulness measure - it can be high for a wrong answer that copies words.
  Report it as such.

This script calls the configured LLM, so it needs a working provider. Nothing
here is mocked: if the provider is missing or fails, the failure is printed and
recorded, never replaced by a fabricated answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.answering.llm_backend import (  # noqa: E402
    AnswerBackendError,
    LLMBackend,
    ModelNotConfiguredError,
    resolve_chat_kwargs,
)
from adaptive_qa.config import load_config  # noqa: E402
from adaptive_qa.documents import DocumentStore  # noqa: E402
from adaptive_qa.eval.em_f1 import em_score, f1_score  # noqa: E402
from adaptive_qa.eval.retrieval_eval import load_eval_examples  # noqa: E402
from adaptive_qa.retrieval import RETRIEVAL_METHODS  # noqa: E402
from adaptive_qa.timeutil import now_ns  # noqa: E402

CONDITIONS = ("closed_book",) + RETRIEVAL_METHODS

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but",
    "it", "its", "this", "that", "these", "those", "there", "here", "which",
    "who", "whom", "whose", "what", "when", "where", "why", "how", "can",
    "could", "should", "would", "may", "might", "must", "will", "shall", "not",
    "no", "if", "then", "than", "so", "such", "also", "into", "over", "under",
    "between", "each", "any", "all", "some", "we", "they", "their", "them",
}


def content_tokens(text: str) -> List[str]:
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def evidence_coverage(answer: str, passages: Sequence[str]) -> Optional[float]:
    """Lexical overlap between answer content tokens and retrieved passages."""
    answer_tokens = content_tokens(answer)
    if not answer_tokens or not passages:
        return None
    pool = set(content_tokens(" ".join(passages)))
    if not pool:
        return None
    return round(sum(1 for t in answer_tokens if t in pool) / len(answer_tokens), 4)


def _ms(ns: int) -> float:
    return round(ns / 1_000_000.0, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare closed-book vs retrieval-augmented answers.")
    ap.add_argument("--dataset", default="eval_data/retrieval_eval/ch6_er_model_questions.jsonl")
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=5, help="number of questions (0 = all)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--out", default="results/downstream_eval")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    llm_cfg = cfg["answering"].get("llm", {})
    args_llm = resolve_chat_kwargs()
    missing = [k for k in ("api_key", "model", "base_url") if not args_llm.get(k)]
    if missing:
        print(
            "The answering model is not configured (missing: "
            + ", ".join(missing)
            + "). Set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in .env and retry. "
            "No answer is fabricated in this situation.",
            file=sys.stderr,
        )
        return 3
    print(f"LLM provider configured: model={args_llm.get('model')} base_url={args_llm.get('base_url')}")

    store = DocumentStore()
    chunks = store.load_chunks(args.doc_id)
    if not chunks:
        print("No indexed chunks found. Upload a document first.", file=sys.stderr)
        return 2

    examples = load_eval_examples(args.dataset)
    answerable = [e for e in examples if e.answerable and e.reference_answer]
    if args.limit:
        answerable = answerable[: args.limit]
    print(f"questions: {len(answerable)} | corpus chunks: {len(chunks)} | top_k: {args.top_k}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            print(f"unknown condition {c!r}; choose from {', '.join(CONDITIONS)}", file=sys.stderr)
            return 2

    closed = LLMBackend(llm_cfg, grounded=False, name="llm_closedbook")
    grounded = LLMBackend(llm_cfg, grounded=True, name="llm_grounded")

    retrievers: Dict[str, Any] = {}
    build_ms: Dict[str, float] = {}
    warmup_ms: Dict[str, float] = {}
    warmup_question = answerable[0].question if answerable else "warm up query"
    for method in conditions:
        if method == "closed_book":
            continue
        t0 = now_ns()
        retrievers[method] = store.build_retriever(
            method=method, retrieval_cfg=cfg.get("retrieval", {})
        )
        build_ms[method] = _ms(now_ns() - t0)
        # The dense model is loaded lazily on first use; without a warm-up that
        # one-off cost would be charged to whichever question happened to go
        # first. Warm up once and report it separately.
        t0 = now_ns()
        retrievers[method].retrieve(warmup_question, 1)
        warmup_ms[method] = _ms(now_ns() - t0)

    rows: List[Dict[str, Any]] = []
    started = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / started
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(answerable, start=1):
        print(f"[{i}/{len(answerable)}] {ex.question_id}: {ex.question}")
        for method in conditions:
            row: Dict[str, Any] = {
                "question_id": ex.question_id,
                "question": ex.question,
                "condition": method,
                "reference_answer": ex.reference_answer,
            }
            passages: List[str] = []
            t_all = now_ns()
            if method == "closed_book":
                row["retrieval_ms"] = None
                row["retrieved_chunk_ids"] = []
            else:
                t0 = now_ns()
                hits = retrievers[method].retrieve(ex.question, args.top_k)
                row["retrieval_ms"] = _ms(now_ns() - t0)
                row["retrieved_chunk_ids"] = [h.doc_id for h in hits]
                row["retrieval_scores"] = [round(float(h.score), 4) for h in hits]
                passages = [h.text for h in hits]
            try:
                t0 = now_ns()
                answer = closed.answer(ex.question) if method == "closed_book" else grounded.answer(ex.question, hits)
                gen_ms = _ms(now_ns() - t0)
                row.update(
                    {
                        "ok": True,
                        "answer": answer.text,
                        "backend": answer.backend,
                        "model": (answer.meta or {}).get("model"),
                        "routed_model": (answer.meta or {}).get("routed_model"),
                        "generation_ms": gen_ms,
                        "total_ms": _ms(now_ns() - t_all),
                        "tokens": (answer.meta or {}).get("tokens"),
                        "f1": round(f1_score(ex.reference_answer, answer.text), 4),
                        "em": round(em_score(ex.reference_answer, answer.text), 4),
                        "evidence_coverage": evidence_coverage(answer.text, passages),
                    }
                )
            except (AnswerBackendError, ModelNotConfiguredError, Exception) as exc:  # noqa: BLE001
                row.update(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "generation_ms": _ms(now_ns() - t_all),
                        "total_ms": _ms(now_ns() - t_all),
                    }
                )
            print(
                f"    {method:<12} ok={row['ok']} "
                f"f1={row.get('f1')} tokens={((row.get('tokens') or {}).get('total_tokens'))} "
                f"latency={row.get('total_ms')}ms"
                + ("" if row["ok"] else f" error={row['error'][:120]}")
            )
            rows.append(row)

    summary: Dict[str, Any] = {}
    for method in conditions:
        sub = [r for r in rows if r["condition"] == method]
        ok = [r for r in sub if r.get("ok")]
        tokens = [(r.get("tokens") or {}).get("total_tokens") for r in ok]
        tokens = [t for t in tokens if isinstance(t, (int, float))]
        summary[method] = {
            "n_questions": len(sub),
            "n_succeeded": len(ok),
            "n_failed": len(sub) - len(ok),
            "mean_f1": round(sum(r["f1"] for r in ok) / len(ok), 4) if ok else None,
            "mean_em": round(sum(r["em"] for r in ok) / len(ok), 4) if ok else None,
            "mean_evidence_coverage": (
                round(sum(r["evidence_coverage"] for r in ok if r.get("evidence_coverage") is not None)
                      / max(1, len([r for r in ok if r.get("evidence_coverage") is not None])), 4)
                if any(r.get("evidence_coverage") is not None for r in ok) else None
            ),
            "mean_retrieval_ms": (
                round(sum(r["retrieval_ms"] for r in ok if r.get("retrieval_ms") is not None)
                      / max(1, len([r for r in ok if r.get("retrieval_ms") is not None])), 3)
                if any(r.get("retrieval_ms") is not None for r in ok) else None
            ),
            "mean_generation_ms": round(sum(r["generation_ms"] for r in ok) / len(ok), 3) if ok else None,
            "mean_total_ms": round(sum(r["total_ms"] for r in ok) / len(ok), 3) if ok else None,
            "mean_total_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "token_sources": sorted({(r.get("tokens") or {}).get("source") for r in ok if r.get("tokens")}),
            "retriever_build_ms": build_ms.get(method),
            "retriever_warmup_ms": warmup_ms.get(method),
        }

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args_llm.get("model"),
        "base_url": args_llm.get("base_url"),
        "dataset": args.dataset,
        "top_k": args.top_k,
        "retriever_build_ms": build_ms,
        "retriever_warmup_ms": warmup_ms,
        "notes": [
            "retriever_warmup_ms covers lazy model load / thread-pool init and is excluded from per-question retrieval latency.",
            "F1 and EM are lexical (SQuAD normalization) and blind to paraphrase; low EM is expected for free-form answers.",
            "evidence_coverage is lexical overlap with retrieved passages, NOT entailment or faithfulness.",
            "retrieval latency excludes generation; total_ms includes retrieval for retrieval conditions.",
        ],
        "summary": summary,
        "per_question": rows,
    }
    (out_dir / "downstream_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "question_id", "condition", "ok", "f1", "em", "evidence_coverage",
            "retrieval_ms", "generation_ms", "total_ms", "total_tokens", "token_source",
            "model", "routed_model", "answer", "error", "question",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **r,
                    "total_tokens": (r.get("tokens") or {}).get("total_tokens"),
                    "token_source": (r.get("tokens") or {}).get("source"),
                }
            )

    print("\n=== downstream summary ===")
    print(f"{'condition':<13} {'ok/n':>7} {'F1':>7} {'EM':>7} {'retr_ms':>9} {'gen_ms':>9} {'gen_tok':>9}")
    for method, s in summary.items():
        def f(v, nd=3):
            return "—" if v is None else f"{v:.{nd}f}"
        print(
            f"{method:<13} {s['n_succeeded']}/{s['n_questions']:<5} {f(s['mean_f1'])} {f(s['mean_em'])} "
            f"{f(s['mean_retrieval_ms'], 1):>9} {f(s['mean_generation_ms'], 1):>9} {f(s['mean_total_tokens'], 1):>9}"
        )
    print(f"\nresults written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
