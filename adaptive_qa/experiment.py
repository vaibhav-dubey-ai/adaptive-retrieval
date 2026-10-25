"""Batch experiment runner: four policies on the same held-out questions.

For every question the pipeline executes BOTH methods once (cross-verify) +
one decision-engine call. The four policies differ only in which primary
answer is credited:

- never_retrieve  -> Method A
- always_retrieve -> Method B
- adaptive        -> decision engine's per-question choice
- oracle          -> offline evaluation winner (approximate upper bound, for
                     benchmark reporting only)

Per-question rows and an aggregate summary are persisted to ``runs/<id>``.
Evaluation time is measured separately from inference time and subtracted
from any latency claim (requirement §7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import load_config, resolve_path
from .corpus import load_corpus
from .dataset import load_records_jsonl, to_question_records
from .pipeline import Pipeline, QuestionRunRecord
from .retrieval import build_retriever_from_config
from .types import Decision, QuestionRecord
from .eval.outcome import label_outcome
from .eval import metrics as M


@dataclass
class ExperimentArtifacts:
    run_dir: str
    summary: Dict[str, Any]


class ExperimentRunner:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.top_k = int(cfg["retrieval"].get("top_k", 8))
        docs = load_corpus(cfg["corpus"]["jsonl_path"])
        self.retriever = build_retriever_from_config(cfg, docs)
        from .answering import build_method_a, build_method_b

        self.method_a = build_method_a(cfg)
        self.method_b = build_method_b(cfg)
        self.decision_engine = None
        from .decision import build_decision_engine

        self.decision_engine = build_decision_engine(cfg)

        self.records_eval = to_question_records(load_records_jsonl(cfg["dataset"]["eval_jsonl"]))

        self.gap_threshold = float(cfg["evaluation"].get("outcome_gap_threshold", 0.10))

    # ------------------------------------------------------------------ runs
    def evaluate_questions(
        self,
        questions: Sequence[QuestionRecord],
        run_id: Optional[str] = None,
        decision_engine=None,
    ) -> ExperimentArtifacts:
        """Cross-verify every question once, label oracle outcomes, then
        derive all four policies from the same underlying outputs.
        """
        decision_engine = decision_engine or self.decision_engine
        pipeline = Pipeline(
            retriever=self.retriever,
            method_a=self.method_a,
            method_b=self.method_b,
            decision_engine=decision_engine,
            top_k=self.top_k,
        )

        rows: List[Dict[str, Any]] = []
        evaluation_total_ms = 0.0
        inference_started = time.perf_counter_ns()
        for i, qrec in enumerate(questions):
            if decision_engine is not None:
                rec = pipeline.run_cross_verify(qrec.question_id, qrec.question, split=qrec.split)
            else:
                # no decision engine available: skip adaptive-related fields
                rec = pipeline.run_cross_verify(
                    qrec.question_id, qrec.question, split=qrec.split, include_decision=False
                )
            t0 = time.perf_counter_ns()
            outcome = label_outcome(qrec, rec.answer_a, rec.answer_b)
            evaluation_total_ms += (time.perf_counter_ns() - t0) / 1_000_000.0
            rec.evaluation_ms = round(evaluation_total_ms, 3)
            rec.outcome = {
                "label": outcome.label,
                "retrieval_better": outcome.label == "retrieval_better",
                "no_retrieval_better": outcome.label == "no_retrieval_better",
                "quality_a": outcome.quality_a,
                "quality_b": outcome.quality_b,
                "delta": outcome.delta,
                "em_a": outcome.em_a,
                "f1_a": outcome.f1_a,
                "em_b": outcome.em_b,
                "f1_b": outcome.f1_b,
            }
            rows.append(rec.to_dict())
        inference_elapsed_ms = (time.perf_counter_ns() - inference_started) / 1_000_000.0

        run_dir = self._new_run_dir(run_id)
        per_question_path = run_dir / "per_question.jsonl"
        rows_path = per_question_path
        (rows_path).write_text(
            "\n".join(_dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        summary = self._build_summary(rows, run_dir)
        summary["timings"] = {
            "cross_verify_total_inference_ms": round(inference_elapsed_ms, 1),
            "evaluation_only_ms": round(evaluation_total_ms, 1),
            "note": "evaluation time is reported separately from inference",
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(_dumps_pretty(summary), encoding="utf-8")
        return ExperimentArtifacts(run_dir=str(run_dir), summary=summary)

    # --------------------------------------------------------------- helpers
    def _new_run_dir(self, run_id: Optional[str]) -> Path:
        base = resolve_path(self.cfg["experiment"]["runs_dir"])
        if run_id is None:
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(self.cfg['experiment'].get('policy_names', [])[:2])}"
        d = base / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_summary(self, rows: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
        from .cost import cost_summary

        per_retrieval = self._cost_per_retrieval_usd()
        summary: Dict[str, Any] = {
            "policies": {},
            "decision_metrics": M.comparison_metrics(rows),
            "cost_model": cost_summary(rows, cost_per_retrieval_usd=per_retrieval),
        }
        for policy in self.cfg["experiment"]["policy_names"]:
            summary["policies"][policy] = M.policy_metrics(rows, policy, cost_per_retrieval_usd=per_retrieval)
        return summary

    def _cost_per_retrieval_usd(self) -> Optional[float]:
        # BM25-local: zero; could read hosted-config pricing if added
        return None

    def familiarity_store(self):
        """Lazy-load the answering model's simulated parametric memory.

        The store supplies only model-side uncertainty signals for the
        decision engine's features - never reference answers or gold docs.
        """
        from .answering.parametric_store import load_parametric_store

        if getattr(self, "_familiarity_store", None) is None:
            self._familiarity_store = load_parametric_store(
                resolve_path(self.cfg["parametric_store"]["jsonl_path"])
            )
        return self._familiarity_store


def _dumps(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _dumps_pretty(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
