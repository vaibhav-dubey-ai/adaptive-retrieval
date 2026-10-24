import types

import pytest

from adaptive_qa.pipeline import Pipeline, QuestionRunRecord
from adaptive_qa.eval.outcome import label_outcome
from adaptive_qa.types import Decision, GeneratedAnswer, RetrievalHit, QuestionRecord


class SpyRetriever:
    def __init__(self, hits=None):
        self.calls = 0
        self._hits = hits or [RetrievalHit("d1", "Title", "Some evidence sentence.", 5.0)]

    def retrieve(self, query, top_k):
        self.calls += 1
        self.calls_args = (query, top_k)
        return list(self._hits)


class SpyMethod:
    def __init__(self, name, text="answer"):
        self.calls = 0
        self.name = name
        self._text = text

    def answer(self, question, passages=None):
        self.calls += 1
        return GeneratedAnswer(text=self._text, abstained=False, backend=self.name)


class SpyDecision:
    def __init__(self, decision):
        self.calls = 0
        self._decision = decision

    def decide(self, question):
        self.calls += 1
        score = 0.9 if self._decision == Decision.RETRIEVE else 0.1
        return self._decision, score, {"threshold": 0.5}


def test_adaptive_executes_only_selected_path_retrieve():
    ret = SpyRetriever()
    a = SpyMethod("A")
    b = SpyMethod("B")
    eng = SpyDecision(Decision.RETRIEVE)
    pipe = Pipeline(ret, a, b, eng, top_k=8)
    rec = pipe.run_adaptive("q1", "What about this hard question?")
    assert a.calls == 0, "unselected path must not run in live adaptive mode"
    assert b.calls == 1
    assert ret.calls == 1
    assert rec.primary_method == "B"
    assert rec.adaptive_e2e_ms is not None and rec.adaptive_e2e_ms >= 0
    assert rec.cross_verify_ms is None


def test_adaptive_executes_only_selected_path_no_retrieve():
    ret = SpyRetriever()
    a = SpyMethod("A")
    b = SpyMethod("B")
    eng = SpyDecision(Decision.DO_NOT_RETRIEVE)
    pipe = Pipeline(ret, a, b, eng, top_k=8)
    rec = pipe.run_adaptive("q2", "easy")
    assert b.calls == 0
    assert ret.calls == 0
    assert a.calls == 1
    assert rec.primary_method == "A"
    assert rec.retrieval_ms is None


def test_cross_verify_runs_both_paths_even_when_choice_exists():
    ret = SpyRetriever()
    a = SpyMethod("A")
    b = SpyMethod("B")
    pipe = Pipeline(ret, a, b, SpyDecision(Decision.RETRIEVE))
    rec = pipe.run_cross_verify("q3", "What is it?")
    assert a.calls == 1 and b.calls == 1 and ret.calls == 1, "cross-verify runs BOTH paths"
    assert rec.answer_a and rec.answer_b
    assert rec.cross_verify_ms > 0
    assert rec.decision_ms is not None


def test_timing_categories_all_recorded():
    pipe = Pipeline(SpyRetriever(), SpyMethod("A"), SpyMethod("B"), SpyDecision(Decision.RETRIEVE))
    rec = pipe.run_cross_verify("q4", "What?")
    assert rec.decision_ms is not None
    assert rec.retrieval_ms is not None
    assert rec.gen_a_ms is not None
    assert rec.gen_b_ms is not None
    assert rec.cross_verify_ms > (rec.retrieval_ms + rec.gen_a_ms + rec.gen_b_ms) * 0.5
    not_decided = pipe.run_cross_verify("q5", "What else?", include_decision=False)
    assert not_decided.decision is None
    assert not_decided.decision_ms is not None  # still recorded (skipped call)


def _stats_spy(answer_a_text, answer_b_text, abstain_a=False, abstain_b=False):
    return SpyMethod("A", answer_a_text), SpyMethod("B", answer_b_text)


def _mk_question(qid="q", answer="Paris", gold=("Paris",)):
    return QuestionRecord(qid, "Where were the?", answer, gold, "bridge", "easy", "test")


def test_oracle_labeling_four_classes():
    ra, rb = _stats_spy("Paris", "Paris", ), ("", "")
    q = _mk_question()

    o = label_outcome(q, GeneratedAnswer("Paris"), GeneratedAnswer("Paris"))
    assert o.label == "similar"
    o = label_outcome(q, GeneratedAnswer("Lyon"), GeneratedAnswer("Paris"))
    assert o.label == "retrieval_better"
    o = label_outcome(q, GeneratedAnswer("Paris"), GeneratedAnswer("I could not find"))
    assert o.label == "no_retrieval_better"
    o = label_outcome(q, GeneratedAnswer(""), GeneratedAnswer(""))
    assert o.label == "ambiguous"


def test_policy_primary_selection():
    q = _mk_question(answer="Paris")
    pipe = Pipeline(
        SpyRetriever(),
        SpyMethod("A", "Paris"),
        SpyMethod("B", "Lyon"),
        SpyDecision(Decision.RETRIEVE),
    )
    rec = pipe.run_cross_verify("q6", "What?")
    outcome = label_outcome(
        q,
        GeneratedAnswer(rec.answer_a["text"]),
        GeneratedAnswer(rec.answer_b["text"]),
    )
    rec.outcome = {
        "label": outcome.label,
        "retrieval_better": outcome.label == "retrieval_better",
        "no_retrieval_better": outcome.label == "no_retrieval_better",
    }

    pipe.apply_policy_primary(rec, "never_retrieve")
    assert rec.primary_answer["text"] == "Paris"

    pipe.apply_policy_primary(rec, "always_retrieve")
    assert rec.primary_answer["text"] == "Lyon"

    pipe.apply_policy_primary(rec, "adaptive")
    assert rec.primary_answer["text"] == "Lyon"  # decision was RETRIEVE

    pipe.apply_policy_primary(rec, "oracle")
    assert rec.primary_answer["text"] == "Paris"  # oracle picks the better one


def test_fixed_policies_do_not_call_decision_engine():
    pipe = Pipeline(SpyRetriever(), SpyMethod("A"), SpyMethod("B"), None)
    rec = pipe.run_fixed("q7", "What?", policy="never_retrieve")
    assert rec.primary_method == "A"
    rec2 = pipe.run_fixed("q8", "What?", policy="always_retrieve")
    assert rec2.primary_method == "B"
    assert rec2.retrieval_ms is not None


def test_oracle_evaluation_time_not_in_adaptive_latency():
    """The deployed adaptive latency must reflect selection only."""
    pipe = Pipeline(SpyRetriever(), SpyMethod("A"), SpyMethod("B"), SpyDecision(Decision.RETRIEVE))
    rec = pipe.run_adaptive("q9", "What?")
    # evaluation fields are not even produced by the live mode
    assert rec.outcome is None
    assert rec.evaluation_ms is None
    assert rec.cross_verify_ms is None
