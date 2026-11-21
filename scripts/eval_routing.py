"""Routing evaluation: does the router retrieve exactly when retrieval helps?

Phase 1 (needs the LLM, slow, resumable): for every question run the closed-book
and the grounded branch once and cache both answers (data/eval_answer_cache.jsonl).
Phase 2 (instant, no LLM): label each question by whether retrieval improved the
answer, then compare never / always / heuristic / tuned-heuristic / learned / oracle
routers on held-out random splits.

    python scripts/eval_routing.py                 # collect (resumes) + analyse
    python scripts/eval_routing.py --no-llm        # analyse cached answers only
    python scripts/eval_routing.py --limit 5       # quick smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.answering.cache import AnswerCache  # noqa: E402
from adaptive_qa.answering.llm_backend import LLMBackend, resolve_chat_kwargs  # noqa: E402
from adaptive_qa.common import load_config, resolve_path  # noqa: E402
from adaptive_qa.decision import build_decision_engine  # noqa: E402
from adaptive_qa.decision.preflight import CollectionPreflight  # noqa: E402
from adaptive_qa.live.documents import DocumentStore  # noqa: E402
from adaptive_qa.eval.retrieval_eval import load_eval_examples, validate_labels  # noqa: E402
from adaptive_qa.eval.routing import label_rows, repeated_splits  # noqa: E402
from adaptive_qa.retrieval import normalize_method  # noqa: E402
from adaptive_qa.common import CorpusDoc  # noqa: E402

DATASETS = [
    "eval_data/retrieval_eval/ch6_er_model_questions.jsonl",
    "eval_data/routing_eval/ch7_normalization_questions.jsonl",
    "eval_data/routing_eval/general_questions.jsonl",
]


def _tag(llm_cfg: dict, model: str) -> str:
    """Cache namespace: model + prompts, so changing a prompt never reuses stale answers."""
    blob = json.dumps([llm_cfg.get("closedbook_system_prompt"), llm_cfg.get("grounded_system_prompt"),
                       llm_cfg.get("max_tokens"), llm_cfg.get("temperature")], sort_keys=True)
    return f"{model}|{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


def _call(cache, backend, tag, q, grounded, hits, ids):
    got = cache.get(q, grounded, tag, ids)
    if got and got.get("text"):
        return got["text"], (got.get("tokens") or {}).get("total_tokens"), got.get("gen_ms") or 0.0, None
    t0 = time.perf_counter()
    try:
        ans = backend.answer(q, hits) if grounded else backend.answer(q)
    except Exception as exc:  # recorded, never counted as an answer
        return None, None, 0.0, f"{type(exc).__name__}: {exc}"
    ms = round((time.perf_counter() - t0) * 1000, 1)
    tokens = (ans.meta or {}).get("tokens")
    cache.put(q, grounded, tag, ids, ans.text, tokens, ms, "llm")
    return ans.text, (tokens or {}).get("total_tokens"), ms, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--method", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="analyse cached answers only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gap", type=float, default=0.10)
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    method = normalize_method(args.method or cfg["retrieval"].get("retriever") or "bm25")
    store = DocumentStore()
    if not store.active_documents():
        print("No active documents; upload ch6.pdf / ch7.pdf first.")
        return 2
    examples = []
    for p in args.datasets:
        examples += load_eval_examples(resolve_path(p))
    if args.limit:
        examples = examples[: args.limit]
    chunks = store.load_chunks()
    docs = [CorpusDoc(c["chunk_id"], c["doc_id"], c["text"], ()) for c in chunks]
    check = validate_labels(examples, [d.doc_id for d in docs])
    if not check["ok"]:
        print("label check FAILED:", check["unknown_chunk_ids"][:3])
        return 2

    retriever = store.build_retriever(method=method, retrieval_cfg=cfg.get("retrieval", {}))
    prober = CollectionPreflight(docs, top_k=5, doc_names=[m["filename"] for m in store.active_documents()])
    engine = build_decision_engine(cfg, None, prober)
    llm_cfg = cfg["answering"]["llm"]
    model = resolve_chat_kwargs().get("model") or "unknown-model"
    tag = _tag(llm_cfg, model)
    cache = AnswerCache()
    backend_a = LLMBackend(llm_cfg, grounded=False, name="llm_closedbook")
    backend_b = LLMBackend(llm_cfg, grounded=True, name="llm_grounded")

    out_dir = resolve_path(args.out) if args.out else resolve_path("results/routing_eval") / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    for i, ex in enumerate(examples, 1):
        t0 = time.perf_counter()
        decision, score, meta = engine.decide(ex.question)
        decision_ms = round((time.perf_counter() - t0) * 1000, 3)
        t1 = time.perf_counter()
        hits = retriever.retrieve(ex.question, args.top_k)
        retrieval_ms = round((time.perf_counter() - t1) * 1000, 3)
        ids = [h.doc_id for h in hits]
        if args.no_llm:
            a = cache.get(ex.question, False, tag, []) or {}
            b = cache.get(ex.question, True, tag, ids) or {}
            ans_a, tok_a, ms_a, err_a = a.get("text"), (a.get("tokens") or {}).get("total_tokens"), a.get("gen_ms") or 0.0, None
            ans_b, tok_b, ms_b, err_b = b.get("text"), (b.get("tokens") or {}).get("total_tokens"), b.get("gen_ms") or 0.0, None
        else:
            ans_a, tok_a, ms_a, err_a = _call(cache, backend_a, tag, ex.question, False, None, [])
            ans_b, tok_b, ms_b, err_b = _call(cache, backend_b, tag, ex.question, True, hits, ids)
            print(f"[{i}/{len(examples)}] {ex.question_id} A={ms_a/1000:.1f}s B={ms_b/1000:.1f}s", flush=True)
        # No passage matched at all: the grounded branch cannot answer (the backend returns a
        # canned refusal at zero cost). A deployed system would fall back to the closed-book
        # answer, so that is what "retrieve" means here. Flagged per question.
        b_fallback = not ids
        if b_fallback and ans_a:
            ans_b, tok_b, ms_b = ans_a, tok_a, ms_a
        for e in (err_a, err_b):
            if e:
                errors.append({"question_id": ex.question_id, "error": e})
        rows.append({
            "question_id": ex.question_id, "category": ex.category, "question": ex.question,
            "reference_answer": ex.reference_answer, "heuristic_score": score,
            "decision": getattr(decision, "value", str(decision)), "decision_ms": decision_ms,
            "preflight": (meta.get("preflight") or {}).get("signals") or {},
            "features": meta.get("features") or {},
            "wh_open": ((meta.get("components") or {}).get("question_cue") or {}).get("detail", {}).get("wh_open", 0.0),
            "retrieved_chunk_ids": ids, "relevant_chunk_ids": list(ex.relevant_chunk_ids),
            "retrieval_ms": retrieval_ms, "answer_a": ans_a, "answer_b": ans_b,
            "b_fallback": b_fallback, "tokens_a": tok_a, "tokens_b": tok_b, "gen_a_ms": ms_a, "gen_b_ms": ms_b,
        })

    labelled = label_rows(rows, args.gap)
    if len(labelled) < 10:
        print(f"only {len(labelled)} scorable questions; nothing to analyse yet")
        return 1
    summary = repeated_splits(labelled, n_splits=args.splits)
    by_cat = {}
    for r in labelled:
        c = by_cat.setdefault(r["category"], {"n": 0, "helps": 0, "hurts": 0, "qa": 0.0, "qb": 0.0})
        c["n"] += 1; c["helps"] += r["retrieval_helps"]; c["hurts"] += r["retrieval_hurts"]
        c["qa"] += r["quality_a"]; c["qb"] += r["quality_b"]
    for c in by_cat.values():
        c["mean_quality_closed_book"] = round(c.pop("qa") / c["n"], 3)
        c["mean_quality_grounded"] = round(c.pop("qb") / c["n"], 3)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {"retriever": method, "top_k": args.top_k, "gap": args.gap, "model": model,
                     "n_splits": args.splits, "test_frac": 0.3, "cache_tag": tag,
                     "n_questions": len(examples), "n_scored": len(labelled),
                     "n_retrieval_helps": sum(r["retrieval_helps"] for r in labelled),
                     "n_retrieval_hurts": sum(r["retrieval_hurts"] for r in labelled),
                     "n_empty_retrieval_fallback": sum(bool(r.get("b_fallback")) for r in labelled),
                     "datasets": args.datasets},
        "policies": summary, "by_category": by_cat, "errors": errors,
        "notes": ["Policies are scored on held-out random splits (30% test); mean and sd over splits.",
                  "oracle is an evaluation upper bound. Quality = mean(token-F1, token-recall) vs the reference.",
                  "The question mix (corpus vs general) is a constructed workload, not natural traffic.",
                  "When retrieval returns no passage the retrieve path falls back to the closed-book answer (same cost as closed-book)."],
    }
    (out_dir / "routing_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for r in labelled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    s = payload["settings"]
    print(f"\n{s['n_scored']}/{s['n_questions']} scored | retrieval helps on {s['n_retrieval_helps']}, hurts on {s['n_retrieval_hurts']}")
    print(f"\n{'policy':<17}{'quality':>9}{'retr.rate':>10}{'tokens':>8}{'latency_s':>10}{'bal.acc':>9}")
    for name, m in summary.items():
        f = lambda k, d=1: (m[k]['mean'] if m[k]['mean'] is not None else float('nan'))
        print(f"{name:<17}{f('quality'):>9.3f}{f('retrieval_rate'):>10.2f}{f('tokens'):>8.0f}{f('latency_ms')/1000:>10.1f}{f('balanced_accuracy'):>9.3f}")
    print("\nby category:")
    for k, c in sorted(by_cat.items()):
        print(f"  {k:<18} n={c['n']:<3} A={c['mean_quality_closed_book']:.2f} B={c['mean_quality_grounded']:.2f} helps={c['helps']} hurts={c['hurts']}")
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
