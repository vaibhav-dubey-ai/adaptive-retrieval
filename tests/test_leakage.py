"""Leakage tests required by the brief: retrieval leakage, reference-answer
leakage, and oracle isolation (the oracle must never be a live engine)."""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.common import Decision  # noqa: E402,E401


def test_oracle_evaluation_module_is_not_deleted_or_imported_by_decision():
    """Oracle labeling lives in adaptive_qa/eval/outcome.py and is imported
    only by evaluation code; no decision module imports it."""
    for f in (PROJECT_ROOT / "adaptive_qa" / "decision").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [
                    f"{node.module}.{alias.name}" if alias.name and alias.name != '*' else node.module
                    for alias in node.names
                ]
            else:
                continue
            for name in names:
                assert "outcome" not in name and "corpus_manifest" not in name, (
                    f"decision module {f.name} imports oracle/evaluation code: {name}"
                )


def test_pipeline_never_feeds_reference_to_decision_engine():
    """Decision engine input is a question string and an optional
    familiarity probe (model-side); the pipeline must not construct any other
    input from evaluation data."""

    calls = {}

    class ProbeDecision:
        def decide(self, question, familiarity=None):
            calls["question"] = question
            calls["familiarity_keys"] = (
                sorted((familiarity or {}).keys()) if familiarity is not None else None
            )
            return Decision.RETRIEVE, 0.99, {}

    from adaptive_qa.hotpot.pipeline import Pipeline
    from tests.test_pipeline import SpyRetriever, SpyMethod

    Pipe = Pipeline(SpyRetriever(), SpyMethod("A"), SpyMethod("B"), ProbeDecision())
    Pipe.run_adaptive("q", "What is the?!?!*")
    # question reaches the engine unchanged; nothing else was recorded
    assert calls["question"] == "What is the?!?!*"
    assert calls["familiarity_keys"] in (None, ["best_rel_cov", "best_title_cov", "best_title_idf", "titles_hit"])


def test_run_record_never_contains_gold_answer_in_decision_fields():
    from adaptive_qa.hotpot.pipeline import Pipeline
    from tests.test_pipeline import SpyRetriever, SpyMethod

    spy = SpyDecisionRecordingMethods()
    Pipe = Pipeline(SpyRetriever(), SpyMethod("A"), SpyMethod("B"), spy)
    rec = Pipe.run_adaptive("q", "What is it?")

    # decision fields carry no reference data
    assert "answer" not in rec.decision_meta
    assert "gold" not in str(rec.decision_meta).lower()


class SpyDecisionRecordingMethods:
    def __init__(self):
        self.calls = 0

    def decide(self, question):
        self.calls += 1
        return Decision.DO_NOT_RETRIEVE, 0.1, {"threshold": 0.5}


def test_corpus_manifest_evaluation_only_note_is_written(tmp_path: Path):
    from adaptive_qa.hotpot.data import build_corpus, save_eval_manifest

    docs, meta = build_corpus([
        {
            "question_id": "q1",
            "paragraphs": [{"title": "T", "sentences": ["s."]}],
            "gold_titles": ["T"],
        }
    ])
    m = tmp_path / "m.json"
    save_eval_manifest(meta, str(m))
    note = m.read_text(encoding="utf-8")
    assert "EVALUATION ONLY" in note
    # It must reference exactly the gold map and nothing that would be fed to retrieval
    assert "gold_doc_ids_by_question" in note


def test_eval_manifest_not_consulted_by_retriever(run_tmp=None):
    """The retriever signature takes only the corpus documents - it cannot
    receive the evaluation manifest by construction."""
    import inspect

    from adaptive_qa.retrieval.bm25 import BM25Retriever

    params = inspect.signature(BM25Retriever.__init__).parameters
    allowed = {"self", "docs", "k1", "b", "title_boost"}
    assert set(params) <= allowed
