"""Decision trace: real signals, real rationale, no placeholder score.

Regression target: the dashboard used to show ``DO_NOT_RETRIEVE / 0.500`` with
an empty features object, because ``decide()`` returned only ``{"threshold"}``
and the memory component was a constant 1.0 while no familiarity probe was
configured. These tests pin the fixed behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.decision.heuristic import HeuristicDecisionEngine  # noqa: E402
from adaptive_qa.decision.preflight import CollectionPreflight, content_terms  # noqa: E402
from adaptive_qa.types import CorpusDoc, Decision  # noqa: E402


class FakeProber:
    """Stands in for CollectionPreflight: deterministic, no BM25 needed."""

    def __init__(self, relevance: float, n_chunks: int = 12):
        self.relevance = relevance
        self.n_chunks = n_chunks
        self.calls = 0

    def probe(self, question):
        self.calls += 1
        return {
            "available": True,
            "ms": 0.42,
            "cold": self.calls == 1,
            "build_ms": 3.5 if self.calls == 1 else None,
            "signals": {
                "corpus_coverage": self.relevance,
                "topk_coverage": self.relevance * 0.9,
                "top1_coverage": self.relevance * 0.8,
                "collection_relevance": self.relevance,
                "n_terms_in_collection": 2.0,
                "n_question_terms": 3.0,
            },
            "collection": {"n_chunks": self.n_chunks, "top_k_probed": 5, "scope": "test"},
        }


def _engine(relevance=None, threshold=0.5, weights=None):
    cfg = {"threshold": threshold}
    if weights:
        cfg["weights"] = weights
    prober = FakeProber(relevance) if relevance is not None else None
    return HeuristicDecisionEngine(cfg, None, prober), prober


def test_trace_reports_every_component_with_values_and_explanations():
    eng, _ = _engine(relevance=0.8)
    decision, score, meta = eng.decide("What is an entity set?")
    assert decision == Decision.RETRIEVE
    assert 0.0 <= score <= 1.0
    comps = meta["components"]
    assert set(comps) == {"question_cue", "memory_uncertainty", "collection_relevance"}
    for c in comps.values():
        assert "explanation" in c and c["explanation"]
        assert "weight" in c and "contribution" in c
    assert meta["collection_available"] is True
    assert meta["preflight"]["ms"] == 0.42
    assert "retrieval-need score" in meta["score_definition"]
    assert meta["decision_rule"].startswith("RETRIEVE if score")


def test_memory_component_is_excluded_not_faked_as_one():
    eng, _ = _engine(relevance=0.8)
    _, _, meta = eng.decide("What is an entity set?")
    mem = meta["components"]["memory_uncertainty"]
    assert mem["available"] is False
    assert mem["value"] is None  # NOT 1.0
    assert meta["memory_available"] is False
    assert "Excluded" in meta["rationale"]


def test_score_is_not_a_constant_placeholder():
    """Different collections must produce different scores."""
    hi, _ = _engine(relevance=0.9)
    lo, _ = _engine(relevance=0.1)
    _, s_hi, _ = hi.decide("What is an entity set?")
    _, s_lo, _ = lo.decide("What is an entity set?")
    assert s_hi != s_lo
    assert s_hi > s_lo
    # and a fixed 0.5 is no longer what an unconfigured collection yields
    eng_none, _ = _engine(relevance=None)
    _, s_none, meta_none = eng_none.decide("What is an entity set?")
    assert meta_none["collection_available"] is False
    assert s_none != 0.5 or meta_none["components"]["question_cue"]["value"] == 0.5


def test_rationale_states_the_actual_score_and_threshold():
    eng, _ = _engine(relevance=0.8, threshold=0.55)
    _, score, meta = eng.decide("What is an entity set?")
    assert f"{score:.3f}" in meta["rationale"]
    assert "0.55" in meta["rationale"]
    assert meta["threshold"] == 0.55


def test_threshold_decides_and_is_reported():
    eng_low, _ = _engine(relevance=0.35)
    d, s, meta = eng_low.decide("What is an entity set?")
    assert d == Decision.DO_NOT_RETRIEVE and s < meta["threshold"]


def test_component_weights_come_from_config():
    strict, _ = _engine(relevance=0.9, weights={"collection_relevance": 1.0, "question_cue": 10.0})
    _, score, meta = strict.decide("What is an entity set?")
    assert meta["components"]["collection_relevance"]["weight"] == 1.0
    assert meta["components"]["question_cue"]["weight"] == 10.0
    assert score < 0.2  # heavy, always-zero cue weight dominates


def test_legacy_weight_keys_are_still_accepted():
    eng = HeuristicDecisionEngine({"weights": {"temporal_cue": 0.9, "unknown": 0.1}, "threshold": 0.5})
    assert eng.cue_weights["temporal"] == 0.9
    assert eng.weights["memory_uncertainty"] == 0.1


def test_engineering_specific_question_now_retrieves_against_real_corpus():
    docs = [
        CorpusDoc("c1", "ch6", "Complex attributes can be divided into component attributes.", ()),
        CorpusDoc("c2", "ch6", "A weak entity set uses a discriminator and identifying entity.", ()),
        CorpusDoc("c3", "ch6", "Aggregation removes redundancy by treating a relationship as an entity.", ()),
    ]
    prober = CollectionPreflight(docs, top_k=2, doc_names=["ch6.pdf"])
    eng = HeuristicDecisionEngine({"threshold": 0.5}, None, prober)
    d_doc, score_doc, _ = eng.decide("explain about complex attributes in dbms")
    d_off, score_off, _ = eng.decide("Who won the 2022 FIFA World Cup?")
    assert d_doc == Decision.RETRIEVE and d_off == Decision.DO_NOT_RETRIEVE
    assert score_doc > score_off


# --------------------------------------------------------------- preflight
def test_preflight_signals_are_aggregate_only():
    docs = [CorpusDoc("c1", "d", "Composite attributes have component attributes.", ())]
    pre = CollectionPreflight(docs, top_k=1)
    out = pre.probe("What are composite attributes?")
    payload = out.to_dict()
    flat = str(payload)
    assert "Composite attributes have component" not in flat  # no passage text
    assert "c1" not in flat  # no chunk ids
    assert 0.0 <= payload["signals"]["collection_relevance"] <= 1.0
    assert payload["signals"]["corpus_coverage"] == 1.0


def test_preflight_cold_flag_and_build_cost_only_on_first_probe():
    docs = [CorpusDoc("c1", "d", "Weak entity sets use discriminating attributes.", ())]
    pre = CollectionPreflight(docs, top_k=1)
    first = pre.probe("weak entity")
    second = pre.probe("weak entity")
    assert first.cold is True and first.build_ms is not None
    assert second.cold is False and second.build_ms is None
    assert first.ms >= 0 and second.ms >= 0


def test_preflight_absent_terms_score_zero():
    docs = [CorpusDoc("c1", "d", "Normalization removes redundancy.", ())]
    pre = CollectionPreflight(docs, top_k=1)
    out = pre.probe("What is the capital of France?")
    assert out.signals["collection_relevance"] == 0.0
    assert out.signals["n_terms_in_collection"] == 0


def test_question_content_terms_drop_instruction_words():
    assert content_terms("Explain how many complex attributes exist") == ["attributes", "complex", "exist"]
