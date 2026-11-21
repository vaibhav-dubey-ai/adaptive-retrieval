"""Fair A/B latency benchmark: interleaved trials, medians, cold/warm split.

Why this exists
---------------
The dashboard comparison always runs Option A first, so its generation may
include a one-off model load (verified live on Ollama 0.40.10 +
qwen3:4b-instruct: first chat-completion 23.8 s vs 7.2 s warm). ONE sequential
run therefore cannot support any claim that retrieving is faster or slower.

This script runs the two branches INTERLEAVED with the starting order swapped
every trial (A,B then B,A ...), so position bias cancels, and summarizes with
medians (robust to the occasional outlier).

What is measured, kept separate (ms, monotonic clock):
* cold      : each BACKEND's very first call, which may pay model loading.
* warm gen  : remaining generation calls (median over trials).
* retrieval : retrieval only (Option B), median.
* decision  : adaptive decision incl. preflight (no LLM), median.
* tokens    : mean total tokens per branch; source labelled (provider vs est).

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\bench_latency.py                # 3 trials
    .\\.venv\\Scripts\\python.exe scripts\\bench_latency.py --trials 5 --question "explain er model"
Output: results/bench_latency/<timestamp>/bench.json + bench.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.common import load_config, resolve_path  # noqa: E402


def median(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def summarize(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate interleaved trial rows into per-branch medians.

    ``trials``: one row per trial with
    ``{order, a_ms, b_ms, a_cold, b_cold, a_tokens, b_tokens,
      retrieval_ms, generation_a_ms, generation_b_ms, decision_ms, ok_a, ok_b}``.
    Cold samples (first call per backend) are reported separately from warm
    medians — they measure model loading, not steady-state answering.
    """
    ok = [t for t in trials if t.get("ok_a") and t.get("ok_b")]
    failed = len(trials) - len(ok)
    out: Dict[str, Any] = {"n_trials": len(trials), "n_complete_pairs": len(ok), "n_failed": failed}
    for label, tok_key, cold_key, ms_key, gen_key in (
        ("A", "a_tokens", "a_cold", "a_ms", "generation_a_ms"),
        ("B", "b_tokens", "b_cold", "b_ms", "generation_b_ms"),
    ):
        cold = [
            t.get(ms_key) for t in ok
            if t.get(cold_key)
        ]
        warm_gen = [t.get(gen_key) for t in ok if not t.get(cold_key)]
        warm_all = [t.get(ms_key) for t in ok if not t.get(cold_key)]
        out[f"branch_{label}"] = {
            "median_path_ms_warm": median(warm_all),
            "median_generation_ms_warm": median(warm_gen),
            "cold_first_call_ms": cold[0] if cold else None,
            "n_warm_samples": len(warm_all),
            "mean_total_tokens": _mean_tok([t.get(tok_key) for t in ok]),
        }
    out["median_retrieval_ms"] = median([t.get("retrieval_ms") for t in ok])
    out["median_decision_ms"] = median([t.get("decision_ms") for t in ok])
    out["median_delta_warm_ms"] = None
    a, b = out.get("branch_A"), out.get("branch_B")
    if a["median_path_ms_warm"] is not None and b["median_path_ms_warm"] is not None:
        out["median_delta_warm_ms"] = round(
            b["median_path_ms_warm"] - a["median_path_ms_warm"], 3
        )
    out["read_note"] = (
        "median_delta_warm_ms = warm B − warm A path latency (positive ⇒ the "
        "retrieval path cost more). Cold first calls are excluded from these "
        "medians and reported separately; they measure model loading, not "
        "answering. Medians over interleaved trials with swapped order are the "
        "only latency comparison this project reports as evidence."
    )
    return out


def _mean_tok(samples: List[Optional[Dict[str, Any]]]) -> Optional[float]:
    vals = [s.get("total_tokens") for s in samples if s and s.get("total_tokens") is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _tok(result) -> Optional[Dict[str, Any]]:
    meta = getattr(result.answer, "meta", None) or {}
    return meta.get("tokens")


def _cold(result) -> Optional[bool]:
    meta = getattr(result.answer, "meta", None) or {}
    return meta.get("generation_cold")


def build_flow():
    from adaptive_qa.live.serve import ServeFlow  # noqa: F401 (import check only)
    from api.main import _serve_flow  # reuses the exact dashboard flow

    return _serve_flow()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", default="Explain ER model basics in the example document.")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_config()  # validates env/config the same way the app does
    flow = build_flow()
    if args.top_k:
        flow.top_k = int(args.top_k)

    trials: List[Dict[str, Any]] = []
    for i in range(max(1, args.trials)):
        b_first = i % 2 == 1  # swap the running order every trial
        row: Dict[str, Any] = {"trial": i + 1, "order": "B,A" if b_first else "A,B"}
        print(f"trial {i + 1}/{args.trials} order {row['order']} …", flush=True)

        a = b = None
        try:
            d = flow.decide_first(args.question)
            row["decision_ms"] = d.get("decision_ms")
            if b_first:
                b = flow._branch_b(args.question)
                a = flow._branch_a(args.question)
            else:
                a = flow._branch_a(args.question)
                b = flow._branch_b(args.question)
        except Exception as exc:  # keep the run going; failures are reported
            row["error"] = f"{type(exc).__name__}: {exc}"
            trials.append(row)
            continue

        # Path totals measured here directly (branch methods are called in
        # isolation, so the flow's own path clock is not set):
        #   A = generation only · B = retrieval + generation
        a_ms = a.generation_ms if a.ok else None
        b_ms = (
            round((b.retrieval_ms or 0.0) + (b.generation_ms or 0.0), 3)
            if b.ok else None
        )
        row.update(
            ok_a=bool(a.ok),
            ok_b=bool(b.ok),
            a_ms=a_ms,
            b_ms=b_ms,
            a_cold=_cold(a),
            b_cold=_cold(b),
            a_tokens=_tok(a),
            b_tokens=_tok(b),
            retrieval_ms=b.retrieval_ms if b.ok else None,
            generation_a_ms=a.generation_ms if a.ok else None,
            generation_b_ms=b.generation_ms if b.ok else None,
        )
        if not a.ok:
            row["error_a"] = a.error
        if not b.ok:
            row["error_b"] = b.error
        trials.append(row)

    summary = summarize(trials)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": args.question,
        "top_k": getattr(flow, "top_k", None),
        "order_policy": "swapped every trial (A,B ⇄ B,A)",
        "summary": summary,
        "trials": trials,
        "notes": [
            "Latency uses timeutil.now_ns (monotonic clock), milliseconds.",
            "A failed generation is never counted as a valid timing sample; "
            "failures are listed per trial under error_a / error_b.",
            "The cold sample is one call per backend and describes load cost, "
            "not answering speed.",
            "keep_alive keeps the local model resident between calls "
            "(verified honored by Ollama via the OpenAI-compatible endpoint).",
        ],
    }

    out_dir = (
        resolve_path(args.out)
        if args.out
        else resolve_path("results/bench_latency") / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "bench.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "order", "a_ms", "b_ms", "a_cold", "b_cold",
                    "a_tokens", "b_tokens", "retrieval_ms", "decision_ms",
                    "error_a", "error_b"])
        for t in trials:
            w.writerow([
                t.get("trial"), t.get("order"), t.get("a_ms"), t.get("b_ms"),
                t.get("a_cold"), t.get("b_cold"),
                (t.get("a_tokens") or {}).get("total_tokens") if t.get("a_tokens") else None,
                (t.get("b_tokens") or {}).get("total_tokens") if t.get("b_tokens") else None,
                t.get("retrieval_ms"), t.get("decision_ms"),
                t.get("error_a"), t.get("error_b"),
            ])

    print(f"\nResults written to {out_dir}")
    s = summary
    print(f"trials={s['n_trials']} complete_pairs={s['n_complete_pairs']} failed={s['n_failed']}")
    for lbl in ("A", "B"):
        br = s[f"branch_{lbl}"]
        print(f"  branch {lbl}: warm median {br['median_path_ms_warm']} ms "
              f"(n={br['n_warm_samples']}), cold first call {br['cold_first_call_ms']} ms, "
              f"mean tokens {br['mean_total_tokens']}")
    print(f"  median retrieval: {s['median_retrieval_ms']} ms · median decision: "
          f"{s['median_decision_ms']} ms · warm Δ(B−A): {s['median_delta_warm_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
