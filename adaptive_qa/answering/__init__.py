"""Answering methods.

- ``FiniteEngine`` (Method A): closed-book answering from simulated parametric
  memory; deterministic, free, no network.
- ``SpanAbsorber`` (Method B): grounded extractive answering over retrieved
  passages; deterministic, free, no network.
- ``OpenAICompatBackend``: optional hosted adapter for either method.

Both deterministic backends share ``span_extraction`` so that quality differs
only through *what evidence is available*, which is exactly what the research
question manipulates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from ..types import GeneratedAnswer, RetrievalHit


class AnsweringMethod(ABC):
    name: str = "abstract"

    @abstractmethod
    def answer(self, question: str, passages: Optional[Sequence[RetrievalHit]] = None) -> GeneratedAnswer:
        ...


def build_method_a(cfg: "dict", **kwargs) -> "AnsweringMethod":
    from .finite_engine import build_finite_engine

    backend = cfg["answering"]["method_a_backend"]
    if backend == "finite_engine":
        return build_finite_engine(cfg, **kwargs)
    if backend == "openai_compatible":
        from .openai_compat import OpenAICompatBackend

        sub = cfg["answering"]["openai_compatible"]
        return OpenAICompatBackend(grounded=False, cfg=sub)
    raise ValueError(f"unknown method A backend {backend!r}")


def build_method_b(cfg: dict) -> "AnsweringMethod":
    backend = cfg["answering"]["method_b_backend"]
    if backend == "span_absorber":
        from .span_absorber import SpanAbsorber

        return SpanAbsorber(cfg["answering"]["span_absorber"])
    if backend == "openai_compatible":
        from .openai_compat import OpenAICompatBackend

        sub = cfg["answering"]["openai_compatible"]
        return OpenAICompatBackend(grounded=True, cfg=sub)
    raise ValueError(f"unknown method B backend {backend!r}")
