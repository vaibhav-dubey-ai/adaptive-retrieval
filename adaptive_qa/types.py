"""Shared dataclass definitions used across the pipeline.

Leakage note (enforced by tests): ``gold_titles`` is reference/evaluation
information. It is only ever read by the evaluation/oracle modules, never by
the decision engine or by the answering methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Decision(str, Enum):
    RETRIEVE = "RETRIEVE"
    DO_NOT_RETRIEVE = "DO_NOT_RETRIEVE"


@dataclass(frozen=True)
class QuestionRecord:
    """One question with its reference answer (evaluation-only fields)."""

    question_id: str
    question: str
    answer: str
    gold_titles: Tuple[str, ...]
    row_type: str = ""
    level: str = ""
    split: str = ""


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    sentences: Tuple[str, ...]
    source_questions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    title: str
    text: str
    score: float


@dataclass
class GeneratedAnswer:
    text: str
    abstained: bool = False
    evidence_doc_ids: Tuple[str, ...] = ()
    backend: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTimings:
    """Latency of one pipeline stage, in milliseconds."""

    decision_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_method_a_ms: Optional[float] = None
    generation_method_b_ms: Optional[float] = None
    adaptive_e2e_ms: Optional[float] = None
    cross_verify_ms: Optional[float] = None
    evaluation_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "decision_ms": self.decision_ms,
            "retrieval_ms": self.retrieval_ms,
            "generation_method_a_ms": self.generation_method_a_ms,
            "generation_method_b_ms": self.generation_method_b_ms,
            "adaptive_e2e_ms": self.adaptive_e2e_ms,
            "cross_verify_ms": self.cross_verify_ms,
            "evaluation_ms": self.evaluation_ms,
        }


PRIMITIVE_STR = Any
ListLike = List
