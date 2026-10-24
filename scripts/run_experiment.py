#!/usr/bin/env python
"""Run the four-policy batch experiment on held-out questions.

Evaluates never_retrieve / always_retrieve / adaptive / oracle over the
test-split questions (or all eval questions with --split all), persists
``per_question.jsonl`` + ``summary.json`` into ``runs/<run_id>`` and prints
the headline comparison.

Usage:
    python scripts/run_experiment.py --config configs/default.json
    python scripts/run_experiment.py --split validation --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_qa.config import load_config  # noqa: E402
from adaptive_qa.experiment import ExperimentRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = ExperimentRunner(cfg)

    if args.split == "all":
        questions = runner.records_eval
    else:
        questions = [q for q in runner.records_eval if q.split == args.split]
    if args.limit:
        questions = questions[: args.limit]
    print(f"[experiment] {args.split} questions: n={len(questions)}")

    artifacts = runner.evaluate_questions(questions, run_id=args.run_id)
    summary = artifacts.summary

    print("\n=== degradation overview ===")
    for policy, m in summary["policies"].items():
        lat = m.get("latency_ms", {})
        print(
            f"{policy:>16}: EM={m.get('em', 0):.3f} F1={m.get('f1', 0):.3f} "
            f"retrieval_rate={m.get('retrieval_rate', 0):.2f} "
            f"abstain_rate={m.get('abstain_rate', 0):.2f} "
            f"e2e_ms(mean/median)={lat.get('mean_e2e', 0):.1f}/{lat.get('median_e2e', 0):.1f}"
        )
    dm = summary["decision_metrics"]
    print(
        "\n=== decision quality (adaptive vs oracle) ===\n"
        f"evaluable={dm['n_evaluable']} excluded_ambiguous={dm['n_excluded_ambiguous']} "
        f"accuracy={dm['decision_accuracy']} P={dm['retrieve_class_precision']} "
        f"R={dm['retrieve_class_recall']} F1={dm['retrieve_class_f1']}"
    )
    # Policy-trade-off charts next to the summary (best-effort: charts must
    # never make a successful experiment look failed)
    try:
        from adaptive_qa.eval.charts import write_all

        write_all(summary, artifacts.run_dir)
        print(f"[charts] quality_vs_latency / quality_vs_cost / retrieval_rate -> {artifacts.run_dir}")
    except Exception as exc:  # noqa: BLE001 - charting is auxiliary
        print(f"[charts] skipped ({exc.__class__.__name__}: {exc})")

    print(f"\n[artifact] {artifacts.run_dir}")


if __name__ == "__main__":
    main()
