"""Per-question execution pipeline.

Two operation modes, strictly separated (requirement §2):

- ``run_adaptive``: the DECISION ENGINE chooses one path and ONLY that path
  executes. Nothing about the alternative path runs or is measured.
- ``run_cross_verify``: BOTH paths execute for the same question (the
  decision engine still runs and is recorded, but it may not influence the
  cross-verification outcome), enabling oracle labeling afterwards.

Every stage timing uses the monotonic high-resolution clock. Decision latency
and retrieval latency are recorded separately from answer generation, and
cross-verification total wall-clock is recorded separately from the deployed
adaptive end-to-end latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..common import now_ns
from ..common import Decision, GeneratedAnswer, RetrievalHit

ABSTAIN_TEXT = "I don't know (abstained from answering from memory)."


@dataclass
class QuestionRunRecord:
    question_id: str
    question: str
    split: str = ""
    policy: str = ""

    decision: Optional[str] = None
    decision_score: Optional[float] = None
    decision_meta: Dict[str, Any] = field(default_factory=dict)

    retrieval_hits: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_ms: Optional[float] = None

    answer_a: Optional[Dict[str, Any]] = None
    answer_b: Optional[Dict[str, Any]] = None

    gen_a_ms: Optional[float] = None
    gen_b_ms: Optional[float] = None
    decision_ms: Optional[float] = None

    adaptive_e2e_ms: Optional[float] = None      # decision + selected path only
    cross_verify_ms: Optional[float] = None      # BOTH paths (evaluation only)

    primary_method: Optional[str] = None         # "A" | "B" (adaptive/oracle)
    primary_answer: Optional[Dict[str, Any]] = None

    outcome: Optional[Dict[str, Any]] = None     # oracle labeling (evaluation)
    evaluation_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _answer_dict(answer: Optional[GeneratedAnswer]) -> Optional[Dict[str, Any]]:
    if answer is None:
        return None
    return {
        "text": answer.text,
        "abstained": answer.abstained,
        "evidence_doc_ids": list(answer.evidence_doc_ids),
        "backend": answer.backend,
        "meta": answer.meta,
    }


def _hits_dict(hits: Optional[Sequence[RetrievalHit]]) -> Optional[List[Dict[str, Any]]]:
    if hits is None:
        return None
    return [
        {"doc_id": h.doc_id, "title": h.title, "score": round(h.score, 4), "text": h.text}
        for h in hits
    ]


class Pipeline:
    def __init__(
        self,
        retriever,
        method_a,
        method_b,
        decision_engine=None,
        top_k: int = 8,
    ) -> None:
        self.retriever = retriever
        self.method_a = method_a
        self.method_b = method_b
        self.decision_engine = decision_engine
        self.top_k = int(top_k)

    # ------------------------------------------------------------ components
    def decide(self, question: str) -> tuple:
        """Run the decision engine (with its own latency measurement).

        In fixed policies (never/always) the pipeline does NOT call this.
        """
        t0 = now_ns()
        decision, score, meta = self.decision_engine.decide(question)
        ms = (now_ns() - t0) / 1_000_000.0
        return decision, score, meta, ms

    def retrieve_only(self, question: str) -> tuple:
        t0 = now_ns()
        hits = self.retriever.retrieve(question, self.top_k)
        ms = (now_ns() - t0) / 1_000_000.0
        return hits, ms

    def _gen_a(self, question: str) -> tuple:
        t0 = now_ns()
        answer = self.method_a.answer(question)
        ms = (now_ns() - t0) / 1_000_000.0
        return answer, ms

    def _gen_b(self, question: str, hits: Sequence[RetrievalHit]) -> tuple:
        t0 = now_ns()
        answer = self.method_b.answer(question, hits)
        ms = (now_ns() - t0) / 1_000_000.0
        return answer, ms

    # -------------------------------------------------------------- modes
    def run_adaptive(
        self,
        question_id: str,
        question: str,
        split: str = "",
    ) -> QuestionRunRecord:
        """Live adaptive operation: only the selected path executes."""
        rec = QuestionRunRecord(question_id=question_id, question=question, split=split)
        t_all = now_ns()
        decision, score, meta, ms_dec = self.decide(question)
        rec.decision = decision.value
        rec.decision_score = score
        rec.decision_meta = {**meta, **({"mode": "adaptive_live"})}
        rec.decision_ms = round(ms_dec, 3)

        if decision == Decision.RETRIEVE:
            hits, ms_ret = self.retrieve_only(question)
            answer, ms_gen = self._gen_b(question, hits)
            rec.retrieval_hits = _hits_dict(hits) or []
            rec.retrieval_ms = round(ms_ret, 3)
            rec.gen_b_ms = round(ms_gen, 3)
            rec.answer_b = _answer_dict(answer)
            rec.primary_method = "B"
            rec.primary_answer = _answer_dict(answer)
            # deployed wall-clock includes decision + retrieval + generation
            # and EXCLUDES the unselected path (it never ran)
            rec.adaptive_e2e_ms = round((now_ns() - t_all) / 1_000_000.0, 3)
        else:
            answer, ms_gen = self._gen_a(question)
            rec.gen_a_ms = round(ms_gen, 3)
            rec.answer_a = _answer_dict(answer)
            rec.primary_method = "A"
            rec.primary_answer = _answer_dict(answer)
            # decision + generation; the retrieval path never ran
            rec.adaptive_e2e_ms = round((now_ns() - t_all) / 1_000_000.0, 3)
        return rec

    def run_fixed(
        self,
        question_id: str,
        question: str,
        split: str = "",
        policy: str = "never_retrieve",
    ) -> QuestionRunRecord:
        """Fixed policies: never retrieve (A) or always retrieve (B).

        The decision engine is NOT consulted (those policies do not use it).
        """
        assert policy in ("never_retrieve", "always_retrieve")
        rec = QuestionRunRecord(
            question_id=question_id, question=question, split=split, policy=policy
        )
        t_all = now_ns()
        if policy == "never_retrieve":
            answer, ms_gen = self._gen_a(question)
            rec.gen_a_ms = round(ms_gen, 3)
            rec.answer_a = _answer_dict(answer)
            rec.primary_method = "A"
            rec.primary_answer = _answer_dict(answer)
            rec.decision = Decision.DO_NOT_RETRIEVE.value
            rec.decision_score = 0.0
            rec.decision_meta = {"mode": "fixed_policy", "policy": policy}
            rec.adaptive_e2e_ms = round((now_ns() - t_all) / 1_000_000.0, 3)
        else:
            hits, ms_ret = self.retrieve_only(question)
            answer, ms_gen = self._gen_b(question, hits)
            rec.retrieval_hits = _hits_dict(hits) or []
            rec.retrieval_ms = round(ms_ret, 3)
            rec.gen_b_ms = round(ms_gen, 3)
            rec.answer_b = _answer_dict(answer)
            rec.primary_method = "B"
            rec.primary_answer = _answer_dict(answer)
            rec.decision = Decision.RETRIEVE.value
            rec.decision_score = 1.0
            rec.decision_meta = {"mode": "fixed_policy", "policy": policy}
            rec.adaptive_e2e_ms = round((now_ns() - t_all) / 1_000_000.0, 3)
        return rec

    def run_cross_verify(
        self,
        question_id: str,
        question: str,
        split: str = "",
        include_decision: bool = True,
    ) -> QuestionRunRecord:
        """Cross-verification: BOTH paths execute for the same question.

        The decision engine (if present) runs once and is recorded - it does
        *not* prevent either path from executing, and nothing it says can
        influence the oracle labeling performed later from the two answers.
        """
        rec = QuestionRunRecord(question_id=question_id, question=question, split=split)
        t_all = now_ns()
        if include_decision and self.decision_engine is not None:
            decision, score, meta, ms_dec = self.decide(question)
            rec.decision = decision.value
            rec.decision_score = score
            rec.decision_meta = {**meta, "mode": "cross_verify"}
            rec.decision_ms = round(ms_dec, 3)
        else:
            rec.decision = None
            t0 = now_ns()
            rec.decision_ms = round((now_ns() - t0) / 1_000_000.0, 3)

        # Path A: closed book (never retrieves)
        answer_a, ms_gen_a = self._gen_a(question)
        rec.gen_a_ms = round(ms_gen_a, 3)
        rec.answer_a = _answer_dict(answer_a)

        # Path B: retrieval + grounded generation
        hits, ms_ret = self.retrieve_only(question)
        answer_b, ms_gen_b = self._gen_b(question, hits)
        rec.retrieval_hits = _hits_dict(hits) or []
        rec.retrieval_ms = round(ms_ret, 3)
        rec.gen_b_ms = round(ms_gen_b, 3)
        rec.answer_b = _answer_dict(answer_b)

        rec.cross_verify_ms = round((now_ns() - t_all) / 1_000_000.0, 3)
        return rec

    def apply_policy_primary(self, rec: QuestionRunRecord, policy: str, gap_threshold: float = 0.10) -> None:
        """Derive the policy's primary answer(s) on a cross-verify record.

        Four experimental policies on the SAME underlying outputs:
        never/always/adaptive/oracle. The oracle picks by score, using
        ``rec.outcome`` (evaluation only).
        """
        assert rec.outcome is not None, "oracle labeling must run before apply_policy_primary"
        assert policy in ("never_retrieve", "always_retrieve", "adaptive", "oracle")
        rec.policy = policy
        if policy == "never_retrieve":
            rec.primary_method = "A"
        elif policy == "always_retrieve":
            rec.primary_method = "B"
        elif policy == "adaptive":
            rec.primary_method = "B" if rec.decision == Decision.RETRIEVE.value else "A"
        else:  # oracle (approximate upper bound, evaluation only)
            if rec.outcome["retrieval_better"]:
                rec.primary_method = "B"
            elif rec.outcome["no_retrieval_better"]:
                rec.primary_method = "A"
            else:
                # similar / ambiguous: the two methods are practically tied;
                # the oracle falls back to the decision engine's choice when
                # one exists (documented), otherwise to NOT retrieving
                rec.primary_method = (
                    "B" if rec.decision == Decision.RETRIEVE.value else "A"
                )
        rec.primary_answer = (
            rec.answer_b if rec.primary_method == "B" else rec.answer_a
        )
