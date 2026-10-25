"""Retrieval evaluation CLI: BM25 vs dense vs hybrid on a labelled question set.

Example
-------
    python scripts/eval_retrieval.py \
        --dataset eval_data/retrieval_eval/ch6_er_model_questions.jsonl \
        --methods bm25,dense,hybrid --top-k 10 --repeat 2

Writes ``retrieval_metrics.json``, ``summary.csv`` and ``per_question.csv`` under
``results/retrieval_eval/<timestamp>/`` (override with ``--out``).

The script evaluates retrieval ONLY. Answer generation and its latency are
measured by the serve/experiment pipeline and are deliberately excluded here so
that a "retrieval is cheaper/slower" claim is not contaminated by LLM time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.config import load_config  # noqa: E402
from adaptive_qa.documents import DocumentStore  # noqa: E402
from adaptive_qa.eval.retrieval_eval import (  # noqa: E402
    compare_methods,
    load_eval_examples,
    validate_labels,
    write_results,
)
from adaptive_qa.retrieval import RETRIEVAL_METHODS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval methods on labelled questions.")
    ap.add_argument(
        "--dataset",
        default="eval_data/retrieval_eval/ch6_er_model_questions.jsonl",
    )
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--methods", default=",".join(RETRIEVAL_METHODS))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--k-recall", type=int, default=5)
    ap.add_argument("--k-rank", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1, help="repeats for latency stability")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument("--doc-id", default=None, help="restrict the corpus to one stored document")
    ap.add_argument("--out", default="results/retrieval_eval")
    ap.add_argument("--clear-embedding-cache", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    store = DocumentStore()

    if args.clear_embedding_cache:
        cache_dir = Path(cfg["retrieval"]["dense"].get("cache_dir") or "data/embeddings")
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"cleared embedding cache: {cache_dir}")

    chunks = store.load_chunks(args.doc_id)
    if not chunks:
        print("No indexed chunks found. Upload a document first.", file=sys.stderr)
        return 2

    examples = load_eval_examples(PROJECT_ROOT / args.dataset if not Path(args.dataset).is_absolute() else args.dataset)
    if args.limit:
        examples = examples[: args.limit]

    validation = validate_labels(examples, [c["chunk_id"] for c in chunks])
    if not validation["ok"]:
        print(
            "WARNING: labelled chunk ids not found in the corpus "
            f"({len(validation['unknown_chunk_ids'])} of them). Metrics would be wrong.",
            file=sys.stderr,
        )
        for item in validation["unknown_chunk_ids"][:5]:
            print(f"  - {item['question_id']}: {item['chunk_id']}", file=sys.stderr)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in RETRIEVAL_METHODS:
            print(f"unknown method {m!r}; choose from {', '.join(RETRIEVAL_METHODS)}", file=sys.stderr)
            return 2

    print(f"corpus: {len(chunks)} chunks | questions: {len(examples)} | methods: {', '.join(methods)}")
    started = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / started

    results = compare_methods(
        cfg,
        examples,
        methods,
        store,
        top_k=args.top_k,
        k_recall=args.k_recall,
        k_rank=args.k_rank,
        repeat=max(1, args.repeat),
    )

    table = []
    for method, res in results.items():
        agg, lat = res["aggregate"], res["latency"]["steady_state"]
        diag = res.get("diagnostics") or {}
        table.append(
            {
                "method": method,
                "recall@5": agg.get(f"recall_at_{args.k_recall}"),
                "mrr@10": agg.get(f"mrr_at_{args.k_rank}"),
                "ndcg@10": agg.get(f"ndcg_at_{args.k_rank}"),
                "median_ms": lat.get("median_ms"),
                "build_ms": res.get("build_ms"),
                "model_load_ms": diag.get("model_load_ms"),
                "model": (diag.get("model_name") or diag.get("components", {}).get("dense", {}).get("model_name")),
            }
        )

    paths = write_results(
        out_dir,
        results,
        dataset_path=Path(args.dataset),
        k_recall=args.k_recall,
        k_rank=args.k_rank,
        extra={
            "corpus": {
                "n_chunks": len(chunks),
                "document_ids": sorted({c["doc_id"] for c in chunks}),
            },
            "label_validation": validation,
            "config_retrieval": cfg.get("retrieval", {}),
        },
    )

    print("\n=== retrieval quality (higher is better) ===")
    header = (
        f"{'method':<8} {'recall@5':>9} {'mrr@10':>8} {'ndcg@10':>8} "
        f"{'median_ms':>10} {'build_ms':>9} {'model_load_ms':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in table:
        def fmt(v, nd=3):
            return "—" if v is None else f"{v:.{nd}f}"
        print(
            f"{row['method']:<8} {fmt(row['recall@5']):>9} {fmt(row['mrr@10']):>8} "
            f"{fmt(row['ndcg@10']):>8} {fmt(row['median_ms'], 1):>10} "
            f"{fmt(row['build_ms'], 1):>9} {fmt(row['model_load_ms'], 1):>14}"
        )
    print(f"\nresults written to: {out_dir}")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    models = {row["model"] for row in table if row["model"]}
    if models:
        print(f"\nembedding model(s) actually used: {', '.join(sorted(models))}")
    print(json.dumps({"label_validation_ok": validation["ok"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
