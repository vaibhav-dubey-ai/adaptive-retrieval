"""Policy evaluation on a labelled retrieval set: adaptive vs never/always/oracle.

What is compared
----------------
For every labelled question the SAME underlying work is done once (one
decision call, one retrieval, and - when an answering model is supplied - one
closed-book answer and one grounded answer). Four policies are then derived
from those outputs, exactly like ``ExperimentRunner`` does for the HotpotQA
benchmark:

- ``never_retrieve`` : never retrieves; the closed-book answer is primary.
- ``always_retrieve``: always retrieves; the grounded answer is primary.
- ``adaptive``       : the decision engine's per-question choice.
- ``oracle``         : the better of the two answers by reference-answer score.
  EVALUATION ONLY - it is an upper bound, never a live policy, and the live
  decision engine can never see it.

Metrics reported (kept separate, never blended)
-----------------------------------------------
* ``evidence_recall_at_k`` over answerable questions ("did the evidence the
  policy would have used contain a labelled relevant passage"). ``None`` when
  no answerable question exists; unanswerable questions are excluded because
  recall is undefined without relevant documents.
* ``unanswerable_probe_rate``: fraction of unanswerable questions on which the
  policy still retrieved (lower is better; a policy that always retrieves
  scores 1.0).
* ``retrieval_calls`` and measured retrieval latency (per policy).
* decision quality for the adaptive policy: accuracy against the
  oracle-answerable label, plus the actual inappropriate decisions
  (skipped-but-answerable, retrieved-but-unanswerable) listed by question id.
* answer quality (EM / F1, only when an LLM was run), generation latency and
  token usage of the primary answer of each policy.

Honesty notes
-------------
* If the answering model is unavailable or a call fails, the row records the
  error and is excluded from answer-quality means; a failure is never counted
  as an answer or as a valid latency sample.
* Labels are used only here and in the experiments view. ``decide_fn`` receives
  the question string only.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..timeutil import now_ns
from .em_f1 import composite_score, em_score, f1_score

POLICIES = ("never_retrieve", "always_retrieve", "adaptive", "oracle")


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _recall(retrieved: Sequence[str], relevant: Sequence[str]) -> Optional[float]:
    if not relevant:
        return None
    rel = set(relevant)
    return len(rel & set(retrieved)) / len(rel)


def evaluate_policies(
    examples: Sequence[Any],
    decide_fn: Callable[[str], Any],
    retriever,
    *,
    top_k: int = 5,
    k_recall: int = 5,
    method_a=None,
    method_b=None,
    max_answers: Optional[int] = None,
    gap_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Run the four-policy experiment over ``examples``.

    ``decide_fn(question)`` must return ``(decision, score, meta)`` where
    ``decision`` is a :class:`~adaptive_qa.types.Decision` (or its string
    value). ``retriever`` has ``retrieve(question, top_k)``.
    """
    from ..types import Decision

    rows: List[Dict[str, Any]] = []
    answers_used = 0
    for ex in examples:
        relevant = list(getattr(ex, "relevant_chunk_ids", ()) or ())
        answerable = bool(relevant)
        q = ex.question

        # ---- decision (question only; never sees labels) ------------------
        t0 = now_ns()
        try:
            decision, score, meta = decide_fn(q)
            decision_ms = round((now_ns() - t0) / 1_000_000.0, 3)
            decision_str = getattr(decision, "value", decision)
        except Exception as exc:  # decision failure must be visible, not fatal
            decision, score, meta = None, None, {}
            decision_ms = round((now_ns() - t0) / 1_000_000.0, 3)
            decision_str = f"ERROR: {type(exc).__name__}: {exc}"

        # ---- one retrieval (shared by every policy that retrieves) -------
        t1 = now_ns()
        try:
            hits = retriever.retrieve(q, top_k) if retriever is not None else []
            retrieval_ms = round((now_ns() - t1) / 1_000_000.0, 3)
            retrieved = [h.doc_id for h in hits][:k_recall]
            retrieval_error = None
        except Exception as exc:
            hits, retrieval_ms, retrieved = [], None, []
            retrieval_error = f"{type(exc).__name__}: {exc}"

        row: Dict[str, Any] = {
            "question_id": getattr(ex, "question_id", ""),
            "question": q,
            "category": getattr(ex, "category", ""),
            "answerable": answerable,
            "relevant_chunk_ids": relevant,
            "decision": decision_str,
            "decision_score": score,
            "decision_ms": decision_ms,
            "preflight_ms": (meta or {}).get("preflight", {}).get("ms"),
            "retrieval_ms": retrieval_ms,
            "retrieved_chunk_ids": retrieved,
            "retrieval_error": retrieval_error,
            "evidence_recall": _recall(retrieved, relevant),
        }

        # ---- optional answering model (both branches, once) --------------
        want_answers = (
            method_a is not None
            and method_b is not None
            and (max_answers is None or answers_used < max_answers)
        )
        if want_answers:
            answers_used += 1
            t2 = now_ns()
            try:
                ans_a = method_a.answer(q)
                row["gen_a_ms"] = round((now_ns() - t2) / 1_000_000.0, 3)
                row["answer_a"] = ans_a.text
                row["tokens_a"] = (ans_a.meta or {}).get("tokens")
            except Exception as exc:
                row["gen_a_ms"] = round((now_ns() - t2) / 1_000_000.0, 3)
                row["answer_a"] = None
                row["error_a"] = f"{type(exc).__name__}: {exc}"
            t3 = now_ns()
            try:
                ans_b = method_b.answer(q, hits)
                row["gen_b_ms"] = round((now_ns() - t3) / 1_000_000.0, 3)
                row["answer_b"] = ans_b.text
                row["tokens_b"] = (ans_b.meta or {}).get("tokens")
            except Exception as exc:
                row["gen_b_ms"] = round((now_ns() - t3) / 1_000_000.0, 3)
                row["answer_b"] = None
                row["error_b"] = f"{type(exc).__name__}: {exc}"
            reference = getattr(ex, "reference_answer", "") or ""
            if reference:
                for label in ("a", "b"):
                    text = row.get(f"answer_{label}")
                    row[f"em_{label}"] = em_score(reference, text) if text else None
                    row[f"f1_{label}"] = f1_score(reference, text) if text else None
                    row[f"composite_{label}"] = (
                        composite_score(reference, text) if text else None
                    )
                row["oracle_prefers"] = _oracle_preference(row, gap_threshold)

        rows.append(row)

    policies = _aggregate(rows, k_recall=k_recall)
    decision_quality = _decision_quality(rows)
    return {
        "settings": {
            "top_k": top_k,
            "k_recall": k_recall,
            "n_questions": len(rows),
            "n_answerable": sum(1 for r in rows if r["answerable"]),
            "n_unanswerable": sum(1 for r in rows if not r["answerable"]),
            "answers_run": answers_used,
            "retriever": getattr(retriever, "method", None),
            "gap_threshold": gap_threshold,
        },
        "policies": policies,
        "decision_quality": decision_quality,
        "per_question": rows,
        "notes": [
            "evidence_recall_at_k measures EVIDENCE availability, not answer "
            "correctness; it is computed over answerable questions only.",
            "unanswerable_probe_rate counts how often a policy still retrieves "
            "when the corpus cannot answer the question.",
            "oracle is an evaluation upper bound and is never a live policy.",
            "answer-quality means exclude failed generations; failures are "
            "reported separately and never counted as answers.",
            "Routing metrics (evidence recall, retrieval calls, probe rate, "
            "confusion matrix) cover EVERY labelled question; answer-quality, "
            "latency and token means cover only the questions with a "
            "successful live LLM call — the two sample sizes are reported "
            "separately and never blended.",
        ],
    }


def _oracle_preference(row: Dict[str, Any], gap_threshold: float) -> Optional[str]:
    a, b = row.get("composite_a"), row.get("composite_b")
    if a is None or b is None:
        return None
    if abs(a - b) < gap_threshold:
        return "similar"
    return "B" if b > a else "A"


def _policy_primary(row: Dict[str, Any], policy: str) -> Optional[str]:
    """Which answer the policy would have shown ('A'/'B'), or None."""
    if policy == "never_retrieve":
        return "A"
    if policy == "always_retrieve":
        return "B"
    if policy == "adaptive":
        return "B" if row["decision"] == "RETRIEVE" else "A"
    pref = row.get("oracle_prefers")
    if pref in ("A", "B"):
        return pref
    return "B" if row["decision"] == "RETRIEVE" else "A"


def _primary_latency(row: Dict[str, Any], policy: str) -> Optional[float]:
    """Measured latency of the answer this policy would actually have shown.

    ``None`` when that answer never ran (no LLM in this experiment, or the
    generation failed) — a failure is never counted as a latency sample.
    """
    primary = _policy_primary(row, policy)
    if row.get(f"answer_{primary.lower()}") is None:
        return None
    if primary == "A":
        base = row.get("gen_a_ms")
        if base is None:
            return None
    else:
        base = (row.get("retrieval_ms") or 0.0) + (row.get("gen_b_ms") or 0.0)
    if policy == "adaptive":
        base += row.get("decision_ms") or 0.0
    return base


def _confusion(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-row decision confusion matrix for ONE policy.

    Positive class (definition, not guessed): a question IS answerable from
    the corpus (``answerable`` label). The predicted-positive action is
    RETRIEVE. So:

        TP = RETRIEVE on an answerable question
        FN = DO_NOT_RETRIEVE on an answerable question  (skipped evidence)
        FP = RETRIEVE on an unanswerable question       (wasted retrieval)
        TN = DO_NOT_RETRIEVE on an unanswerable question

    A decision that raised an error (``ERROR: ...``) is counted as
    *predicted negative* and listed separately so the matrix stays complete
    without hiding the failure.
    """
    tp = fn = fp = tn = error_decisions = 0
    for r in rows:
        actual_pos = bool(r["answerable"])
        d = r["decision"]
        pred_pos = d == "RETRIEVE"
        if d is None or (isinstance(d, str) and d.startswith("ERROR")):
            error_decisions += 1
        if actual_pos and pred_pos:
            tp += 1
        elif actual_pos and not pred_pos:
            fn += 1
        elif not actual_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    return {
        "positive_class": "answerable (the corpus contains a labelled relevant passage)",
        "predicted_positive_action": "RETRIEVE",
        "tp_retrieve_answerable": tp,
        "fn_skip_answerable": fn,
        "fp_retrieve_unanswerable": fp,
        "tn_skip_unanswerable": tn,
        "n_error_decisions": error_decisions,
        "n_rows": len(rows),
    }


def _per_class_metrics(cm: Dict[str, Any]) -> Dict[str, Any]:
    """Precision/recall/F1 for the RETRIEVE class, specificity and balanced
    accuracy, computed straight from the confusion matrix.

    Formulas (standard binary-classification definitions with
    retrieve = positive):
        precision   = TP / (TP + FP)
        recall      = TP / (TP + FN)      (sensitivity; = accuracy on
                                           answerable questions here)
        F1          = 2 P R / (P + R)
        specificity = TN / (TN + FP)      (recall of the DO_NOT_RETRIEVE /
                                           unanswerable class)
        balanced    = (recall + specificity) / 2
        accuracy    = (TP + TN) / (TP + FN + FP + TN)
    An undefined ratio (zero denominator) is reported as None, never as 0.
    """
    tp, fn, fp, tn = (
        cm["tp_retrieve_answerable"],
        cm["fn_skip_answerable"],
        cm["fp_retrieve_unanswerable"],
        cm["tn_skip_unanswerable"],
    )

    def ratio(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    specificity = ratio(tn, tn + fp)
    balanced = (
        round((recall + specificity) / 2, 4)
        if recall is not None and specificity is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "accuracy": ratio(tp + tn, cm["n_rows"]),
        "definitions": {
            "precision": "TP / (TP + FP) — how often RETRIEVE was justified by the corpus",
            "recall": "TP / (TP + FN) — how often an answerable question got retrieved",
            "f1": "2·precision·recall / (precision + recall)",
            "specificity": "TN / (TN + FP) — how often an unanswerable question was correctly skipped",
            "balanced_accuracy": "(recall + specificity) / 2 — insensitive to class imbalance",
            "accuracy": "(TP + TN) / n — optimistic on this 67/8 imbalanced set",
        },
        "positive_class": cm["positive_class"],
    }


def _policy_confusion(rows: Sequence[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    """Confusion matrix of a FIXED policy, derived from the same rows.

    ``always_retrieve`` predicts RETRIEVE for every question, so
    TP = n_answerable, FP = n_unanswerable, FN = TN = 0; ``never_retrieve``
    is the mirror image. Numbers are computed from the data at evaluation
    time, never copied from a saved run.
    """
    n_ans = sum(1 for r in rows if r["answerable"])
    n_unans = len(rows) - n_ans
    if policy == "always_retrieve":
        tp, fn, fp, tn = n_ans, 0, n_unans, 0
    elif policy == "never_retrieve":
        tp, fn, fp, tn = 0, n_ans, 0, n_unans
    else:
        raise ValueError(policy)
    cm = {
        "positive_class": "answerable (the corpus contains a labelled relevant passage)",
        "predicted_positive_action": "RETRIEVE" if policy == "always_retrieve" else "DO_NOT_RETRIEVE",
        "tp_retrieve_answerable": tp,
        "fn_skip_answerable": fn,
        "fp_retrieve_unanswerable": fp,
        "tn_skip_unanswerable": tn,
        "n_error_decisions": 0,
        "n_rows": len(rows),
    }
    out = {"confusion_matrix": cm, "metrics": _per_class_metrics(cm)}
    out["note"] = (
        f"Fixed-policy baseline computed from the same {len(rows)} labelled "
        "questions (not copied from any saved run). It shows what accuracy "
        '“just always retrieve” (or never) scores on this imbalanced '
        "label set, against which the adaptive policy must be read."
    )
    return out


def _aggregate(rows: Sequence[Dict[str, Any]], k_recall: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    retrievals = [r["retrieval_ms"] for r in rows if r.get("retrieval_ms") is not None]
    for policy in POLICIES:
        if policy == "never_retrieve":
            recalls = [0.0 for _ in answerable]
            probes = 0
            calls = 0
            retrieval_lat: List[Optional[float]] = []
        elif policy == "always_retrieve":
            recalls = [r.get("evidence_recall") for r in answerable]
            probes = len(unanswerable)
            calls = len(rows)
            retrieval_lat = retrievals
        elif policy == "adaptive":
            recalls = [
                r.get("evidence_recall") if r["decision"] == "RETRIEVE" else 0.0
                for r in answerable
            ]
            probes = sum(1 for r in unanswerable if r["decision"] == "RETRIEVE")
            calls = sum(1 for r in rows if r["decision"] == "RETRIEVE")
            retrieval_lat = [
                r["retrieval_ms"] for r in rows if r["decision"] == "RETRIEVE"
            ]
        else:  # oracle
            recalls = [r.get("evidence_recall") for r in answerable]
            probes = 0
            calls = len(answerable)
            retrieval_lat = [r["retrieval_ms"] for r in answerable]

        lat: List[Optional[float]] = []
        tokens: List[Optional[int]] = []
        quality: List[Optional[float]] = []
        f1: List[Optional[float]] = []
        em: List[Optional[float]] = []
        n_ok = 0
        attempts = 0
        failures = 0
        for r in rows:
            primary = _policy_primary(r, policy)
            key = f"answer_{primary.lower()}"
            if key in r:  # the LLM layer ran for this policy's primary answer
                attempts += 1
                if r.get(key) is None:
                    failures += 1
                else:
                    n_ok += 1
            lat.append(_primary_latency(r, policy))
            tokens.append(
                ((r.get("tokens_b") if primary == "B" else r.get("tokens_a")) or {}).get(
                    "total_tokens"
                )
            )
            quality.append(r.get(f"composite_{primary.lower()}"))
            f1.append(r.get(f"f1_{primary.lower()}"))
            em.append(r.get(f"em_{primary.lower()}"))

        n_quality = sum(1 for v in quality if v is not None)
        n_tokens = sum(1 for v in tokens if v is not None)
        out[policy] = {
            f"evidence_recall_at_{k_recall}": _mean(recalls),
            "unanswerable_questions": len(unanswerable),
            "unanswerable_probe_rate": (
                round(probes / len(unanswerable), 4) if unanswerable else None
            ),
            "retrieval_calls": calls,
            "mean_retrieval_ms": _mean(retrieval_lat),
            "mean_latency_ms": _mean(lat),
            "n_latency_samples": sum(1 for v in lat if v is not None),
            "mean_total_tokens": _mean(tokens),
            "n_token_samples": n_tokens,
            "mean_composite": _mean(quality),
            "n_quality_samples": n_quality,
            # Answer-quality numbers come only from successful live LLM calls;
            # nquestions_total << n_quality_samples means the quality means
            # cover a fraction of the labelled set and are PRELIMINARY, not a
            # validation over all questions.
            "n_answers_attempted": attempts,
            "n_failed_generations": failures,
            "small_sample_warning": 0 < n_quality < 10,
            "mean_f1": _mean(f1),
            "mean_em": _mean(em),
            "n_rows_with_answer": n_ok,
            "sample_note": (
                "mean_composite / mean_f1 / mean_em / mean_total_tokens are "
                "averaged ONLY over successful live generations "
                "(n_quality_samples / n_token_samples); the decision-quality "
                "metadata covering all questions is decision_quality. When "
                "1 <= n_quality_samples < 10 the quality means are flagged "
                "small_sample_warning and must be read as preliminary."
            ),
            "latency_definition": {
                "never_retrieve": "answer generation only (A), over questions with an answer",
                "always_retrieve": "retrieval + grounded generation (B), over questions with an answer",
                "adaptive": "decision (incl. preflight) + the selected path, over questions with an answer",
                "oracle": "retrieval + the higher-scoring answer; evaluation upper bound only",
            }[policy],
        }
    return out


def _decision_quality(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How good were the adaptive policy's retrieve/skip decisions?

    Reports three layers, explicitly separated:

    1. the simple accuracy/error lists (what the earlier view showed),
    2. the full confusion matrix with the positive class DEFINED
       (positive = answerable; predicted-positive action = RETRIEVE),
    3. per-class precision / recall / F1 / specificity / balanced accuracy
       computed from that matrix with standard binary formulas.

    The fixed always-retrieve and never-retrieve baselines are computed from
    the SAME rows at evaluation time — on this imbalanced label set
    always-retrieve scores the answerable share as accuracy, so raw accuracy
    alone must never be reported without that baseline alongside.
    """
    skipped_but_answerable = [
        r["question_id"]
        for r in rows
        if r["answerable"] and r["decision"] != "RETRIEVE"
    ]
    retrieved_but_unanswerable = [
        r["question_id"]
        for r in rows
        if (not r["answerable"]) and r["decision"] == "RETRIEVE"
    ]
    correct = len(rows) - len(skipped_but_answerable) - len(retrieved_but_unanswerable)
    cm = _confusion(rows)
    out = {
        "n_questions": len(rows),
        "n_correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "skipped_but_answerable": skipped_but_answerable,
        "retrieved_but_unanswerable": retrieved_but_unanswerable,
        "note": (
            "Correct = RETRIEVE on an answerable question, or DO_NOT_RETRIEVE on "
            "an unanswerable one. This is a decision-level metric computed with "
            "the labels, never fed back to the engine."
        ),
    }
    if rows:
        out["confusion_matrix"] = cm
        out["metrics"] = _per_class_metrics(cm)
        out["baseline_always_retrieve"] = _policy_confusion(rows, "always_retrieve")
        out["baseline_never_retrieve"] = _policy_confusion(rows, "never_retrieve")
        out["metric_note"] = (
            "Positive class = answerable-from-corpus. Per-class metrics follow "
            "the standard binary definitions with RETRIEVE as the predicted "
            "positive. The baselines are derived from the label counts of this "
            "same evaluation, not hardcoded. Accuracy alone rewards "
            "always-retrieve on this 67/8 imbalanced set — compare the "
            "baselines too (that is what balanced accuracy is for)."
        )
    return out
