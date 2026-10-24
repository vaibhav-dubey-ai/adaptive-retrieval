"""Cross-verification outcome labeling (the offline oracle).

For each evaluation question both answering methods run independently; their
answers are scored against the reference answer, and the quality delta
decides the outcome class:

- retrieval_better       : quality(B) - quality(A) >= threshold
- no_retrieval_better    : quality(A) - quality(B) >= threshold
- similar                : |delta| < threshold
- ambiguous              : reference information missing, or BOTH methods
                           abstained (no scorable comparison exists).
                           A one-sided abstention IS evaluable: the abstaining
                           method's quality is 0, so a correct answer on the
                           other side is a clear win (candidate-missing is the
                           essence of the retrieval/decision question).

The oracle label is an EVALUATION artifact produced AFTER inference; it is
never an input to the deployed decision engine (enforced structurally by
tests: this module is not imported by decision code paths).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..types import GeneratedAnswer
from .em_f1 import em_score, f1_score


def _answer_fields(answer):
    """Accept a GeneratedAnswer OR its serialized dict form (records)."""
    if isinstance(answer, GeneratedAnswer):
        return answer.text, bool(answer.abstained)
    if isinstance(answer, dict):
        return (answer.get("text") or ""), bool(answer.get("abstained", False))
    return "", True


@dataclass
class Outcome:
    label: str                    # one of the four class names
    quality_a: float              # composite quality of Method A
    quality_b: float              # composite quality of Method B
    delta: float                  # quality_b - quality_a
    em_a: float
    f1_a: float
    em_b: float
    f1_b: float


def label_outcome(
    question_record,
    answer_a: GeneratedAnswer,
    answer_b: GeneratedAnswer,
    gap_threshold: float = 0.10,
) -> Outcome:
    """Score both answers and classify the cross-verification outcome.

    ``question_record`` carries the reference answer. The answers must come
    from executing the respective methods, not from any cached guess.

    Notice that citations or passages attached to Method B do NOT influence
    the label: only answer quality does.
    """
    text_a, abstained_a = _answer_fields(answer_a)
    text_b, abstained_b = _answer_fields(answer_b)
    empty_a = answer_a is None or abstained_a or not text_a.strip()
    empty_b = answer_b is None or abstained_b or not text_b.strip()

    gold = question_record.answer
    em_a = em_score(gold, text_a) if (answer_a and not empty_a) else 0.0
    f1_a = f1_score(gold, text_a) if (answer_a and not empty_a) else 0.0
    em_b = em_score(gold, text_b) if (answer_b and not empty_b) else 0.0
    f1_b = f1_score(gold, text_b) if (answer_b and not empty_b) else 0.0

    quality_a = 0.5 * em_a + 0.5 * f1_a
    quality_b = 0.5 * em_b + 0.5 * f1_b
    delta = quality_b - quality_a

    if not gold.strip() or (empty_a and empty_b):
        label = "ambiguous"
    elif delta >= gap_threshold:
        label = "retrieval_better"
    elif delta <= -gap_threshold:
        label = "no_retrieval_better"
    else:
        label = "similar"
    return Outcome(
        label=label,
        quality_a=quality_a,
        quality_b=quality_b,
        delta=round(delta, 4),
        em_a=em_a,
        f1_a=f1_a,
        em_b=em_b,
        f1_b=f1_b,
    )
