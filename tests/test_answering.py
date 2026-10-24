import pytest

from adaptive_qa.answering.finite_engine import FiniteEngine
from adaptive_qa.answering.parametric_store import ParametricStore
from adaptive_qa.answering.span_absorber import SpanAbsorber
from adaptive_qa.types import RetrievalHit


def _store() -> ParametricStore:
    snippets = [
        ("George Washington", "George Washington was the first president of the United States."),
        ("George Washington", "George Washington served from seventeen eighty-nine to seventeen ninety-seven."),
        ("The Beatles", "The Beatles were an English rock band formed in Liverpool."),
        ("Pixar", "Pixar Animation Studios was founded in nineteen eighty-six."),
        ("France", "France has a wide variety of wine regions."),
    ]
    return ParametricStore([(t, s) for t, s in snippets])


def test_closed_book_refuses_passages():
    eng = FiniteEngine(_store(), knowledge_threshold=0.5, relation_threshold=0.1)
    with pytest.raises(ValueError):
        eng.answer("Who was president?", passages=[RetrievalHit("d", "T", "t", 1.0)])


def test_closed_book_answers_from_memory_when_anchored():
    eng = FiniteEngine(_store(), knowledge_threshold=0.5, relation_threshold=0.1)
    out = eng.answer("George Washington first president of United States?")
    assert not out.abstained
    assert out.meta["memory_title"].startswith("George Washington")


def test_closed_book_abstains_when_memory_misses():
    eng = FiniteEngine(_store(), knowledge_threshold=0.9, relation_threshold=0.9)
    out = eng.answer("What is the capital of Australia?")
    assert out.abstained


def test_closed_book_deterministic():
    eng = FiniteEngine(_store(), knowledge_threshold=0.5, relation_threshold=0.1)
    a1 = eng.answer("The Beatles band formed where?")
    a2 = eng.answer("The Beatles band formed where?")
    assert a1.text == a2.text and a1.abstained == a2.abstained


def test_span_absorber_grounds_on_retrieved_text():
    absorber = SpanAbsorber({"min_sentence_score": 0.05})
    hits = [
        RetrievalHit("d1", "The Beatles", "The Beatles were an English rock band formed in Liverpool.", 10.0),
        RetrievalHit("d2", "France", "France has wine regions.", 3.0),
    ]
    out = absorber.answer("form english rock band where beatles", hits)
    assert not out.abstained
    assert "Beatles" in out.meta["best_sentence"]
    assert out.evidence_doc_ids == ("d1",)


def test_span_absorber_abstains_on_empty_retrieval():
    absorber = SpanAbsorber({"min_sentence_score": 100})
    out = absorber.answer("anything?", [RetrievalHit("d", "x", "completely unrelated note text", 1.0)])
    assert out.abstained
    out2 = absorber.answer("anything?", [])
    assert out2.abstained
