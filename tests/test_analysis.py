"""Citations, lexical support map and answer-difference analysis."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.live.serve import (  # noqa: E402
    assign_citations,
    citation_map,
    compare_answers,
    invalid_citations,
    sentence_overlap_support,
    sentence_support,
    split_sentences,
    term_set,
)

PASSAGES = [
    {
        "rank": 1,
        "doc_id": "udoc_x:0010",
        "source_file": "ch6.pdf",
        "page": 19,
        "score": 0.81,
        "text": "Representing Relationship Sets: diamonds represent relationship sets in an ER diagram.",
    },
    {
        "rank": 2,
        "doc_id": "udoc_x:0017",
        "source_file": "ch6.pdf",
        "page": 17,
        "score": 0.42,
        "text": "Composite attributes allow attributes to be divided into subparts called component attributes.",
    },
]


def test_assign_citations_is_one_based_and_anchored():
    cited = assign_citations(PASSAGES)
    assert [c["citation"] for c in cited] == ["[1]", "[2]"]
    assert [c["anchor"] for c in cited] == ["passage-1", "passage-2"]
    # original passages are not mutated
    assert PASSAGES[0]["rank"] == 1 and "citation" not in PASSAGES[0]


def test_citation_map_points_at_the_right_source():
    m = citation_map(PASSAGES)
    assert m["[1]"]["anchor"] == "passage-1"
    assert m["[1]"]["page"] == 19
    assert m["[2]"]["anchor"] == "passage-2"
    assert m["[2]"]["source_file"] == "ch6.pdf"


def test_invalid_citations_detects_out_of_range_reference():
    assert invalid_citations("See [1] and [2].", PASSAGES) == []
    assert invalid_citations("See [1] and [7].", PASSAGES) == ["[7]"]


def test_sentence_from_the_passage_is_high_overlap_with_the_right_citation():
    answer = "Diamonds represent relationship sets in an ER diagram."
    rows = sentence_overlap_support(answer, PASSAGES)
    assert len(rows) == 1
    assert rows[0]["overlap_status"] == "high_overlap"
    assert rows[0]["citations"] == ["[1]"]
    assert rows[0]["best_passage"] == 1


def test_unrelated_sentence_shows_low_overlap():
    rows = sentence_overlap_support("Diamonds are a compressed form of carbon crystals.", PASSAGES)
    assert rows[0]["overlap_status"] in ("partial_overlap", "no_overlap")
    assert rows[0]["citations"] == []


def test_very_short_sentence_is_reported_as_unchecked():
    rows = sentence_overlap_support("Yes.", PASSAGES)
    assert rows[0]["overlap_status"] == "unchecked"
    assert rows[0]["overlap"] is None


def test_overlap_status_describes_lexical_terms_not_entailment():
    # Reuses the passage's TEXT while changing its meaning: all terms overlap,
    # so the map may call it high_overlap — and must never call it verified.
    answer = "Relationship sets are NEVER drawn as diamonds in ER diagrams."
    rows = sentence_overlap_support(answer, PASSAGES)
    assert set(r["overlap_status"] for r in rows) <= {
        "high_overlap", "partial_overlap", "no_overlap", "unchecked"
    }
    assert not any(r["overlap_status"] in ("supported", "entailed", "verified") for r in rows)


def test_compare_answers_reports_passage_specific_terms_and_counts():
    a = "Complex attributes hold structured data such as objects."
    b = (
        "Composite attributes are divided into subparts called component attributes. "
        "Diamonds represent relationship sets in an ER diagram."
    )
    an = compare_answers(a, b, PASSAGES)
    assert an["method"] == "lexical_overlap_only"
    assert "not entailment" in an["method_note"].lower()
    assert an["summary"]["n_passages"] == 2
    assert an["summary"]["b_high_overlap"] >= 1
    # renamed keys: no legacy 'supported' key remains
    assert "b_supported" not in an["summary"]
    assert an["thresholds"]["high_overlap_at"] == 0.60
    # passage 1 vocabulary (diamonds / relationship / diagram) is used by B only
    p1 = an["per_passage"][0]
    assert "diamonds" in p1["terms_b_not_a"]
    assert p1["coverage_b"] > p1["coverage_a"]
    assert an["b_sentences"][0]["citations"]
    # answer-level vocabulary diff
    assert "composite" in an["answer_diff"]["only_in_b"]
    assert an["answer_diff"]["jaccard"] <= 1.0


def test_split_sentences_handles_bullets_and_newlines():
    parts = split_sentences("- First point. Second point!\n\nThird?")
    assert parts == ["First point.", "Second point!", "Third?"]


def test_term_set_drops_stopwords():
    assert term_set("What is the entity set?") == {"entity", "set"}
