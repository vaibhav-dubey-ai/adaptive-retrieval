import pytest

from adaptive_qa.eval.metrics import em_score, f1_score


def test_em_perfect_and_empty():
    assert em_score("Paris", "Paris") == 1.0
    assert em_score("Paris", "Lyon") == 0.0


def test_em_ignores_case_and_articles_and_punctuation():
    assert em_score("The Beatles", "beatles") == 1.0
    assert em_score("U.S.A.", "usa") == 1.0
    assert em_score("the - top", "top") == 1.0


def test_f1_partial_overlap():
    # normalized: {beatles} vs {beatles, rock, band}
    f1 = f1_score("The Beatles", "Beatles rock band")
    p = 1 / 3
    r = 1.0
    assert f1 == pytest.approx(2 * p * r / (p + r), rel=1e-6)  # = 0.5


def test_f1_order_independent_and_symmetric():
    assert f1_score("a b c", "b c d") == f1_score("b c d", "a b c")
    assert f1_score("a b c", "c b a") == 1.0
