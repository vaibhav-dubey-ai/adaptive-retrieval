import inspect
from pathlib import Path

import numpy as np

from adaptive_qa.decision.heuristic import HeuristicDecisionEngine
from adaptive_qa.decision.learned import LearnedDecisionEngine
from adaptive_qa.decision.features import extract_features, FEATURE_NAMES
from adaptive_qa.types import Decision


def title_cov_fam(cov: float, titles: int = 3, rel: float = 0.8) -> dict:
    return {"best_title_cov": cov, "titles_hit": titles, "best_rel_cov": rel, "best_title_idf": 9.0}


def test_feature_vector_shape_and_familiarity_effect():
    q = "Who won the first cricket world cup in 1975?"
    f0 = extract_features(q, title_cov_fam(0.9))
    assert len(f0) == len(FEATURE_NAMES)
    f_no_mem = extract_features(q, {"best_title_cov": 0.0, "titles_hit": 0, "best_rel_cov": 0.0})
    # familiarity changes only the memory features (last four)
    assert f0[:-4] == f_no_mem[:-4]
    assert f0[-4] > f_no_mem[-4]


def test_heuristic_retrieves_for_uncertain_obscure_question():
    eng = HeuristicDecisionEngine({"weights": {"temporal": 0.35, "entity": 0.25, "unknown": 0.5, "wh_factor": 0.4}, "threshold": 0.5})
    fam = title_cov_fam(0.0)
    decision, score, meta = eng.decide(
        "Which battle in autumn eighteen twelve led General Ney to retreat near during a first victory of the war?",
        fam,
    )
    assert decision == Decision.RETRIEVE
    assert 0.0 <= score <= 1.0


def test_heuristic_skips_for_familiar_memory_hit():
    eng = HeuristicDecisionEngine({"threshold": 0.5})
    decision, score, meta = eng.decide(
        "George Washington was the first president of the United States, is that right?",
        title_cov_fam(1.0, titles=0, rel=0.9),
    )
    assert decision == Decision.DO_NOT_RETRIEVE


def test_learned_engine_learns_familiarity_from_synthetic_labels():
    """Synthetic task: unknown memory (low familiarity) => retrieve label 1."""
    eng = LearnedDecisionEngine({"learned": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 2, "random_state": 0}})

    def synth(q_suffix: str, fam_cov: float) -> str:
        return f"Who is the obscure person {q_suffix}?" if fam_cov < 0.5 else (
            f"George Washington first president united states {q_suffix}?"
        )

    rng = np.random.RandomState(0)
    train_q, train_y, train_fam = [], [], []
    for i in range(60):
        y = i % 2
        cov = 0.0 if y == 1 else 0.9 + rng.rand() * 0.1
        q = synth(str(i), cov)
        train_q.append(q)
        train_y.append(y)
        train_fam.append(title_cov_fam(cov))
    val_q, val_y, val_fam = [], [], []
    for i in range(20):
        y = i % 2
        cov = 0.05 if y == 1 else 0.95
        q = synth(f"v{i}", cov)
        val_q.append(q)
        val_y.append(y)
        val_fam.append(title_cov_fam(cov))

    tuning = eng.fit(train_q, train_y, train_fam, val_q, val_y, val_fam)
    assert tuning["valid_f1_retrieve"] == pytest_stub_f1()

    test_q = [synth("t1", 0.0), synth("t2", 1.0)]
    d0 = eng.decide(test_q[0], title_cov_fam(0.0))
    d1 = eng.decide(test_q[1], title_cov_fam(1.0))
    assert d0[0] == Decision.RETRIEVE
    assert d1[0] == Decision.DO_NOT_RETRIEVE


def pytest_stub_f1():
    # with this separable synthetic task, tuned F1 on validation must be perfect
    return 1.0


def test_save_and_load_roundtrip(tmp_path: Path):
    eng = LearnedDecisionEngine({"learned": {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 2, "random_state": 0}})
    q = ["a obscure unknown question", "b known memory question"] * 10
    y = [1, 0] * 10
    fams = [title_cov_fam(0.0), title_cov_fam(1.0)] * 10
    eng.fit(q, y, fams, q, y, fams)
    path = str(tmp_path / "bundle.pkl")
    eng.save(path, train_meta={"n": len(q)})

    eng2 = LearnedDecisionEngine({})
    eng2.load(path)
    # Roundtrip fidelity is checked on exact training rows (OOD generalization
    # is not this test's subject).
    d_retrieve = eng2.decide(q[0], fams[0])
    d_skip = eng2.decide(q[1], fams[1])
    assert d_retrieve[0] == Decision.RETRIEVE
    assert d_skip[0] == Decision.DO_NOT_RETRIEVE


def test_decision_signature_never_accepts_gold_or_answers():
    """Structural leakage gate: the decide() API must take only question +
    familiarity probe; no reference/gold/oracle/answer parameter exists."""
    for cls in (HeuristicDecisionEngine, LearnedDecisionEngine):
        sig = inspect.signature(cls.decide)
        params = set(sig.parameters)
        assert params <= {"self", "question", "familiarity"}, params


def test_decision_source_never_touches_reference_data():
    """Source-level leakage gate: decision modules must not reference gold
    answers, oracle outcomes, or evaluation artifacts in any form."""
    forbidden_symbols = ["gold_titles", "gold_doc_ids", "em_score(", "f1_score(",
                         "evaluation_meta", "oracle_label", "reference_answer",
                         "GeneratedAnswer", "try_pagination"]
    forbidden = forbidden_symbols
    decision_dir = Path(__file__).resolve().parents[1] / "adaptive_qa" / "decision"
    assert decision_dir.is_dir()
    for f in decision_dir.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in src, f"{f.name} mentions forbidden symbol {token!r}"
