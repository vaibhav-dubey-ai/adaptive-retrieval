"""Adaptive pre-retrieval decision engine.

Policies predict RETRIEVE / DO_NOT_RETRIEVE from the question ALONE plus
signals derived from the answering model's own parametric memory (an
uncertainty/familiarity signal, in the spirit of "does the model know it"
research). The decision never sees retrieved passages, candidate answers,
reference answers, or oracle labels - tests enforce this structurally.
"""

from __future__ import annotations

from typing import Optional

from . import learned  # noqa: F401
from . import heuristic as _heuristic  # noqa: F401


def build_decision_engine(cfg: dict, familiarity_prober=None):
    """Factory. ``familiarity_prober`` is an object with a
    ``familiarity_probe(question) -> dict`` method (the answering model's
    simulated memory); it supplies the model-side uncertainty feature.
    """
    engine_name = cfg["decision"]["engine"]
    if engine_name == "heuristic":
        from .heuristic import HeuristicDecisionEngine

        return HeuristicDecisionEngine(
            cfg["decision"].get("heuristic", {}), familiarity_prober
        )
    if engine_name == "learned":
        from .learned import LearnedDecisionEngine

        return LearnedDecisionEngine(cfg, familiarity_prober)
    raise ValueError(f"unknown decision engine {engine_name!r}")
