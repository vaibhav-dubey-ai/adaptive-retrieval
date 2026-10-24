#!/usr/bin/env python
"""Train the adaptive decision engine with oracle supervision.

STRICT data discipline (brief §4):
- labels come ONLY from train-split questions (oracle labeling on train data
  is legitimate supervised learning; test data is never touched);
- the decision threshold is tuned ONLY on the validation split;
- the learned classifier is fit on pre-retrieval features only;
- the resulting bundle is saved under runs/decision_model/ and is what
  ``run_experiment.py`` loads for the ADAPTIVE policy.

For the learned engine the script first runs the full cross-verification
pipeline on the train+validation questions (both methods + oracle labeling),
derives labels from the outcome, extracts pre-retrieval features, then fits.

Usage:
    python scripts/train_decision.py --engine heuristic
    python scripts/train_decision.py --engine learned [--config configs/default.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_qa.config import load_config, resolve_path  # noqa: E402
from adaptive_qa.dataset import load_records_jsonl  # noqa: E402
from adaptive_qa.experiment import ExperimentRunner  # noqa: E402
from adaptive_qa.answering.parametric_store import load_parametric_store  # noqa: E402
from adaptive_qa.decision.features import extract_features, FEATURE_NAMES  # noqa: E402
from adaptive_qa.types import Decision  # noqa: E402
from adaptive_qa.pipeline import Pipeline  # noqa: E402
from adaptive_qa.eval.outcome import label_outcome  # noqa: E402


def derive_labels(runner: ExperimentRunner, questions) -> dict:
    """Run cross-verification on the given (train/val) questions, label the
    oracle outcome, and return lists for training the decision engine."""
    pipeline = Pipeline(
        retriever=runner.retriever,
        method_a=runner.method_a,
        method_b=runner.method_b,
        decision_engine=None,  # no decisions needed for labeling (saves time)
        top_k=runner.top_k,
    )
    store = runner.familiarity_store()
    out = {"questions": [], "labels": [], "features": [], "familiarity": []}
    counts = {"retrieval_better": 0, "no_retrieval_better": 0, "similar": 0, "ambiguous": 0}
    for r in questions:
        rec = pipeline.run_cross_verify(r.question_id, r.question, split=r.split, include_decision=False)
        outcome = label_outcome(r, rec.answer_a, rec.answer_b, gap_threshold=runner.gap_threshold)
        counts[outcome.label] = counts.get(outcome.label, 0) + 1
        if outcome.label == "ambiguous":
            continue  # not reliably evaluable -> excluded from training
        # Binary label: RETRIEVE iff retrieval clearly helps. 'similar' counts
        # as DO_NOT_RETRIEVE (produces the most useful decision policy; the
        # modeling choice is documented and reported).
        y = 1 if outcome.label == "retrieval_better" else 0
        fam = store.familiarity_probe(r.question)
        out["questions"].append(r.question)
        out["labels"].append(y)
        out["features"].append(extract_features(r.question, fam))
        out["familiarity"].append(fam)
    out["label_counts"] = counts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--engine", choices=["heuristic", "learned"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap train questions (smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["decision"]["engine"] = args.engine

    runner = ExperimentRunner(cfg)
    train_q = [q for q in runner.records_eval if q.split == "train"]
    val_q = [q for q in runner.records_eval if q.split == "validation"]
    if args.limit:
        train_q = train_q[: args.limit]
        val_q = val_q[: args.limit]
    assert train_q, "no train-split questions"
    assert val_q, "no validation-split questions"

    print(f"[labels] deriving oracle labels on {len(train_q)} train + {len(val_q)} validation questions")
    t0 = time.time()
    train_data = derive_labels(runner, train_q)
    val_data = derive_labels(runner, val_q)
    print(f"[labels] done in {time.time() - t0:.1f}s; train counts: {train_data['label_counts']}")

    if args.engine == "heuristic":
        from adaptive_qa.decision.heuristic import HeuristicDecisionEngine

        store = runner.familiarity_store()
        eng = HeuristicDecisionEngine(cfg["decision"].get("heuristic", {}), None)
        # tune threshold on validation
        best_f1 = -1.0
        best_thr = eng.threshold
        for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            eng.threshold = thr
            tp = fp = fn = 0
            for feat_q, y, fam in zip(val_data["questions"], val_data["labels"], val_data["familiarity"]):
                d = eng.decide(feat_q, fam)
                pred = d[0] == Decision.RETRIEVE
                tp += pred and y
                fp += pred and not y
                fn += (not pred) and y
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        eng.threshold = best_thr
        out_path = resolve_path(cfg["decision"]["model_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "engine": "heuristic",
            "threshold": best_thr,
            "weights": cfg["decision"]["heuristic"]["weights"],
            "feature_names": FEATURE_NAMES,
            "valid_f1_retrieve": round(best_f1, 4),
            "train_label_counts": train_data["label_counts"],
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"[saved] heuristic bundle -> {out_path}")
        print(f"    tuned threshold {best_thr}; validation RETRIEVE-F1 {best_f1:.3f}")

    else:
        from adaptive_qa.decision.learned import LearnedDecisionEngine

        eng = LearnedDecisionEngine(cfg.get("decision", {}))
        tuning = eng.fit(
            train_data["questions"], train_data["labels"], train_data["familiarity"],
            val_data["questions"], val_data["labels"], val_data["familiarity"],
        )
        # engine-specific bundle path so heuristic and learned training do not
        # clobber each other's artifacts (the base path is the heuristic's)
        learned_path = str(cfg["decision"]["model_path"]).replace(
            "decision_bundle.json", "decision_bundle_learned.json"
        )
        cfg["decision"]["model_path"] = learned_path
        eng.save(
            learned_path,
            train_meta={
                "n_train": len(train_data["labels"]),
                "n_valid": len(val_data["labels"]),
                "train_label_counts": train_data["label_counts"],
            },
        )
        print(f"[saved] learned bundle under {learned_path}(.json/.model.pkl)")
        print(f"    {tuning}")


if __name__ == "__main__":
    main()
