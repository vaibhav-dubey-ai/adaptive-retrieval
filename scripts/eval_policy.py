"""Evaluate the adaptive retrieval policy against never/always/oracle.

Runs on the hand-labelled retrieval set (default:
``eval_data/retrieval_eval/ch6_er_model_questions.jsonl``) over the ACTIVE
uploaded document collection.

Two layers, reported separately:

1. Evidence layer (always computed, no LLM):
   one decision per question (with the collection preflight) and one BM25/dense/
   hybrid retrieval. From these, evidence recall@k, unanswerable-probe rate and
   retrieval latency are derived for all four policies.

2. Answer layer (optional, ``--answers N``):
   for N questions spread across categories, run BOTH answering branches once
   each with the configured LLM. Each policy's primary answer is then scored
   (EM/F1/composite) and its latency/token usage reported. ``oracle`` picks the
   higher-scoring answer and is clearly an evaluation upper bound only.

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\eval_policy.py
    .\\.venv\\Scripts\\python.exe scripts\\eval_policy.py --answers 10 --method hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.common import load_config, resolve_path  # noqa: E402
from adaptive_qa.decision import build_decision_engine  # noqa: E402
from adaptive_qa.decision.preflight import CollectionPreflight  # noqa: E402
from adaptive_qa.live.documents import DocumentStore  # noqa: E402
from adaptive_qa.eval.policy import evaluate_policies  # noqa: E402
from adaptive_qa.eval.retrieval_eval import (  # noqa: E402
    EvalExample,
    load_eval_examples,
    validate_labels,
)
from adaptive_qa.retrieval import normalize_method  # noqa: E402
from adaptive_qa.common import CorpusDoc  # noqa: E402


def _spread_by_category(examples: Sequence[EvalExample], n: int) -> List[EvalExample]:
    """Pick ``n`` examples round-robin across categories (no cherry-picking)."""
    buckets: Dict[str, List[EvalExample]] = {}
    for ex in examples:
        buckets.setdefault(ex.category, []).append(ex)
    for bucket in buckets.values():
        bucket.sort(key=lambda e: e.question_id)
    picked: List[EvalExample] = []
    idx = 0
    while len(picked) < n:
        added = False
        for name in sorted(buckets):
            bucket = buckets[name]
            if idx < len(bucket):
                picked.append(bucket[idx])
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        idx += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", default="eval_data/retrieval_eval/ch6_er_model_questions.jsonl"
    )
    ap.add_argument("--method", default=None, help="retrieval method for Option B")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--answers",
        type=int,
        default=0,
        help="number of questions to send to the LLM (0 = evidence layer only)",
    )
    ap.add_argument("--no-cache", action="store_true", help="bypass cached LLM answers")
    ap.add_argument("--limit", type=int, default=0, help="cap total questions (0 = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    method = normalize_method(args.method or (cfg["retrieval"].get("retriever") or "bm25"))
    store = DocumentStore()
    if not store.active_documents():
        print("No active documents. Upload one through the dashboard first.")
        return 2
    examples = load_eval_examples(resolve_path(args.dataset))
    if args.limit:
        examples = examples[: args.limit]

    chunks = store.load_chunks()
    docs = [CorpusDoc(c["chunk_id"], c["doc_id"], c["text"], ()) for c in chunks]
    check = validate_labels(examples, [d.doc_id for d in docs])
    if not check["ok"]:
        print(
            f"WARNING: {len(check['unknown_chunk_ids'])} labelled chunk ids do not "
            f"exist in the active collection ({check['n_available_chunks']} chunks). "
            "The corpus differs from the one the labels were made for; metrics "
            "would be meaningless."
        )
        for miss in check["unknown_chunk_ids"][:5]:
            print("  missing:", miss["question_id"], miss["chunk_id"])

    retriever = store.build_retriever(method=method, retrieval_cfg=cfg.get("retrieval", {}))
    preflight_k = int(
        (((cfg.get("decision", {}).get("heuristic") or {}).get("preflight") or {}).get("top_k"))
        or 5
    )
    prober = CollectionPreflight(
        docs, top_k=preflight_k, doc_names=[m["filename"] for m in store.active_documents()]
    )
    engine = build_decision_engine(cfg, None, prober)

    from adaptive_qa.answering.cache import AnswerCache
    cache = None if args.no_cache else AnswerCache()

    method_a = method_b = None
    answer_examples = examples
    wanted_qids = None
    if args.answers > 0:
        from adaptive_qa.answering.llm_backend import LLMBackend

        llm_cfg = (cfg["answering"].get("llm") or {})
        method_a = LLMBackend(llm_cfg, grounded=False, name="llm_closedbook")
        method_b = LLMBackend(llm_cfg, grounded=True, name="llm_grounded")
        answer_examples = _spread_by_category(examples, args.answers)
        wanted_qids = [ex.question_id for ex in answer_examples]
        print(f"Running the answering model on {len(answer_examples)} questions (cache size: {cache.size() if cache else 0}) …")

    payload = evaluate_policies(
        examples,
        decide_fn=engine.decide,
        retriever=retriever,
        top_k=args.top_k,
        k_recall=args.top_k,
        method_a=method_a,
        method_b=method_b,
        max_answers=args.answers if args.answers > 0 else None,
        answer_question_ids=wanted_qids,
        cache=cache,
        gap_threshold=float(cfg["evaluation"].get("outcome_gap_threshold", 0.10)),
    )
    payload["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["dataset"] = {
        "path": str(resolve_path(args.dataset)),
        "n_examples": len(examples),
        "label_check": check,
    }
    payload["settings"]["method"] = method
    payload["settings"]["answer_question_ids"] = [
        ex.question_id for ex in answer_examples
    ]
    payload["decision_config"] = {
        "engine": getattr(engine, "name", None),
        "threshold": getattr(engine, "threshold", None),
        "weights": getattr(engine, "weights", None),
        "cue_weights": getattr(engine, "cue_weights", None),
        "preflight_top_k": preflight_k,
    }

    out_dir = (
        resolve_path(args.out)
        if args.out
        else resolve_path("results/policy_eval") / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (out_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "question_id", "question", "category", "answerable", "decision",
            "decision_score", "decision_ms", "retrieval_ms", "evidence_recall",
            "retrieved_chunk_ids", "relevant_chunk_ids", "answer_a", "answer_b",
            "f1_a", "f1_b", "composite_a", "composite_b", "oracle_prefers",
            "gen_a_ms", "gen_b_ms", "error_a", "error_b",
        ]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in payload["per_question"]:
            out = dict(row)
            out["retrieved_chunk_ids"] = ";".join(row.get("retrieved_chunk_ids") or [])
            out["relevant_chunk_ids"] = ";".join(row.get("relevant_chunk_ids") or [])
            w.writerow(out)

    print(f"\nResults written to {out_dir}")
    print(f"questions: {payload['settings']['n_questions']} "
          f"(answerable {payload['settings']['n_answerable']}, "
          f"unanswerable {payload['settings']['n_unanswerable']}), method={method}, "
          f"top_k={args.top_k}")
    k = args.top_k
    print(f"\n{'policy':<17}{'recall@'+str(k):>10}{'unans.probe':>13}"
          f"{'calls':>7}{'retr_ms':>9}{'path_ms':>10}{'tokens':>9}")
    for name, p in payload["policies"].items():
        print(
            f"{name:<17}{(p.get(f'evidence_recall_at_{k}') or 0):>10.3f}"
            f"{(p.get('unanswerable_probe_rate') or 0):>13.3f}"
            f"{p['retrieval_calls']:>7}"
            f"{(p.get('mean_retrieval_ms') or 0):>9.2f}"
            f"{(p.get('mean_latency_ms') or 0):>10.0f}"
            f"{(p.get('mean_total_tokens') or 0):>9.0f}"
        )
    print("path_ms / tokens average only over questions that actually produced an "
          "answer (see n_latency_samples in policy_results.json).")
    dq = payload["decision_quality"]
    print(f"\nadaptive decision accuracy: {dq['accuracy']} "
          f"({dq['n_correct']}/{dq['n_questions']})")
    print("skipped but answerable:", ", ".join(dq["skipped_but_answerable"]) or "none")
    print("retrieved but unanswerable:", ", ".join(dq["retrieved_but_unanswerable"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
