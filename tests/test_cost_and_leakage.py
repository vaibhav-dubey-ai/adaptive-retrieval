"""Per-policy cost aggregation fix + parametric-store leakage guard + API."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.common import cost_summary
from adaptive_qa.common import Decision


def _row(decision, retrieval_better, no_retrieval_better):
    return {
        "decision": decision,
        "outcome": {
            "retrieval_better": retrieval_better,
            "no_retrieval_better": no_retrieval_better,
        },
    }


def test_retrieval_frequency_computed_separately_per_policy():
    rows = [
        _row("RETRIEVE", True, False),
        _row("RETRIEVE", False, False),
        _row("DO_NOT_RETRIEVE", False, True),
        _row("DO_NOT_RETRIEVE", False, False),
    ]
    out = cost_summary(rows)
    assert out["never_retrieve"]["retrieval_frequency_per_question"] == 0.0
    assert out["always_retrieve"]["retrieval_frequency_per_question"] == 1.0
    assert out["adaptive"]["retrieval_frequency_per_question"] == 0.5
    # oracle: row1 better-with-retrieval, row3 better-without; the two TIED
    # rows follow the decision engine (row2 RETRIEVE, row4 DO_NOT) => 2 of 4
    assert out["oracle"]["retrieval_frequency_per_question"] == 0.5
    # the old bug: all four identical values — can never happen now
    freqs = {k: v["retrieval_frequency_per_question"] for k, v in out.items()}
    assert len(set(freqs.values())) > 1


def test_llm_calls_counted_per_policy():
    rows = [_row("RETRIEVE", True, False), _row("DO_NOT_RETRIEVE", False, False)]
    out = cost_summary(rows, cost_per_retrieval_usd=1.0, cost_per_llm_call_usd=1.0)
    # every policy still generates once per question
    for policy, m in out.items():
        assert m["llm_calls_per_question"] == 1.0
    # per-question: 1 llm call + retrieval rate, both priced at 1.0
    assert out["never_retrieve"]["estimated_cost_per_question_usd"] == 1.0
    assert out["always_retrieve"]["estimated_cost_per_question_usd"] == 2.0
    assert out["adaptive"]["estimated_cost_per_question_usd"] == 1.5


# ---------------------------------------------------------------- leakage
def test_parametric_store_built_without_eval_gold_paragraphs():
    """The store must not contain eval-question gold evidence (requirement 10)."""
    from adaptive_qa.common import load_config
    from adaptive_qa.common import read_jsonl

    cfg = load_config("configs/default.json")
    assert cfg["parametric_store"]["include_eval_gold_paragraphs"] is False

    eval_records = read_jsonl(cfg["dataset"]["eval_jsonl"])
    pretrain = read_jsonl(cfg["dataset"]["pretrain_jsonl"])
    pretrain_sentences = {
        s.strip()
        for rec in pretrain
        for para in rec.get("paragraphs", [])
        if para["title"] in set(rec.get("gold_titles", []))
        for s in para.get("sentences", [])
        if s.strip()
    }
    store_sentences = {
        (snip["title"], snip["sentence"])
        for snip in read_jsonl(cfg["parametric_store"]["jsonl_path"])
    }
    leaked = 0
    for rec in eval_records:
        gold_set = set(rec.get("gold_titles", []))
        for para in rec.get("paragraphs", []):
            if para["title"] not in gold_set:
                continue
            for s in para.get("sentences", []):
                s = s.strip()
                if not s:
                    continue
                if (para["title"], s) in store_sentences and s not in pretrain_sentences:
                    leaked += 1
    assert leaked == 0, f"{leaked} eval gold paragraph sentences leaked into the store"


# --------------------------------------------------------------------- API
def test_api_document_endpoints_end_to_end(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps, routes_ask

    monkeypatch.setattr(deps, "DOCUMENT_ROOT", tmp_path / "documents")
    deps._STATE["docs"] = None

    client = TestClient(api_main.app)
    payload = ("Persistently stored custom doc. " * 30).encode()

    r = client.post(
        "/documents/upload",
        files={"file": ("mydoc.txt", payload, "text/plain")},
    )
    assert r.status_code == 200
    doc = r.json()["document"]
    assert doc["kind"] == "text" and doc["n_chunks"] > 0 and doc["active"] is True

    # duplicate content is detected, not double-indexed
    r2 = client.post("/documents/upload", files={"file": ("copy.txt", payload, "text/plain")})
    assert r2.status_code == 200 and r2.json()["document"]["duplicate"] is True

    listing = client.get("/documents").json()["documents"]
    assert len(listing) == 1

    doc_id = doc["doc_id"]
    r3 = client.post(f"/documents/{doc_id}/active", json={"active": False})
    assert r3.status_code == 200 and r3.json()["active"] is False

    r4 = client.delete(f"/documents/{doc_id}")
    assert r4.status_code == 200
    assert client.get("/documents").json()["documents"] == []

    # unsupported type -> 400 with message
    r5 = client.post("/documents/upload", files={"file": ("x.exe", b"nope", "application/octet-stream")})
    assert r5.status_code == 400 and "Unsupported" in r5.json()["detail"]


def test_api_compare_and_adaptive_modes_with_mocks(monkeypatch):
    """Decision must precede branches; compare returns both; adaptive returns one."""
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps, routes_ask
    from adaptive_qa.live.serve import ServeFlow
    from adaptive_qa.common import Decision, GeneratedAnswer, RetrievalHit

    order = []
    engine_calls = []

    class Engine:
        def decide(self, q):
            order.append("decision")
            engine_calls.append(q)
            return Decision.RETRIEVE, 0.9, {}

    class MethodA:
        def answer(self, q):
            order.append("A")
            return GeneratedAnswer(text="A answer", backend="A", meta={"tokens": {
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "source": "estimated"}})

    class MethodB:
        def answer(self, q, passages):
            assert passages, "grounded branch must receive retrieved passages"
            order.append("B")
            return GeneratedAnswer(text="B answer", backend="B", meta={"tokens": {
                "input_tokens": 40, "output_tokens": 8, "total_tokens": 48, "source": "provider-reported"}})

    class Retr:
        def retrieve(self, q, top_k):
            order.append("retrieve")
            return [RetrievalHit("d", "t", "passage", 1.0)]

    class FakeFlow(ServeFlow):
        def __init__(self, **kw):
            super().__init__(decision_engine=Engine(), method_a=MethodA(),
                             method_b=MethodB(), benchmark_retriever=Retr())
        def decide_first(self, question): return super().decide_first(question)

    # the flow factory now takes an optional retrieval_method; the fake ignores it
    monkeypatch.setattr(deps, "_serve_flow", lambda **kw: FakeFlow())
    client = TestClient(api_main.app)

    rec = client.post("/ask?mode=compare", json={"question": "Where is the lab?"})
    assert rec.status_code == 200
    j = rec.json()
    # strict order: decision BEFORE retrieve, A, and B
    assert order == ["decision", "retrieve", "A", "B"] or order == ["decision", "A", "retrieve", "B"]
    assert order[0] == "decision"
    assert j["branch_a"]["ok"] and j["branch_b"]["ok"]
    assert j["comparison"]["token_difference"] == 33
    assert j["decision"]["decision"] == "RETRIEVE"

    # adaptive mode: only selected branch runs
    order.clear()
    j2 = client.post("/ask?mode=adaptive", json={"question": "Where is the lab?"}).json()
    assert j2["selected_branch"] == "B"
    assert j2["executed_branch"]["ok"] is True
    executed_dash = [x for x in j2 if isinstance(j2[x], dict)]
    assert j2["adaptive_total_ms"] > 0
    assert j2.get("executed_branch", {}).get("text") == "B answer"
    assert "A" not in order  # closed-book never ran in adaptive mode


def test_api_benchmark_endpoint_preserved(monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    from api import deps, routes_ask

    class FakePipeline:
        def __init__(self, **kw):
            pass
        def run_cross_verify(self, qid, question, split="", include_decision=True):
            from adaptive_qa.hotpot.pipeline import QuestionRunRecord

            return QuestionRunRecord(question_id=qid, question=question, answer_a={"text": "a", "abstained": False, "backend": "x", "meta": {}}, answer_b={"text": "b", "abstained": False, "backend": "y", "meta": {}})
        def run_adaptive(self, qid, question):
            from adaptive_qa.hotpot.pipeline import QuestionRunRecord

            return QuestionRunRecord(question_id=qid, question=question, answer_a={"text": "a", "abstained": False, "backend": "x", "meta": {}}, primary_method="A")

    class FakeRunner:
        retriever = method_a = method_b = decision_engine = top_k = None
        records_eval = []

    monkeypatch.setattr(deps, "_runner", lambda: FakeRunner())
    class MonkeyPipeline:
        """api.main.builds Pipeline directly; patch the Pipeline symbol."""
    monkeypatch.setattr(routes_ask, "Pipeline", FakePipeline)

    client = TestClient(api_main.app)
    j = client.post("/ask/benchmark?mode=cross", json={"question": "benchmark q?"}).json()
    assert j["mode"] == "cross" and j["record"]["answer_a"]["text"] == "a"


def test_llm_backends_reused_so_cold_flag_is_not_always_true(monkeypatch):
    """A fresh backend per request made generation_cold True for every live answer."""
    import api.main as api_main
    from api import deps, routes_ask

    monkeypatch.setitem(deps._STATE, "llm", None)
    bundle = deps._LLMBundle({"answering": {"llm": {}}})
    assert bundle.backends() is bundle.backends()
