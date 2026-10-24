"""Interpretable heuristic pre-retrieval policy (the required baseline).

retrieve_score = weighted sum of interpretable cues:
- temporal/precision cues (years, superlatives, quotes)
- named-entity density
- model-side familiarity: how well the answering model's own memory matches
  the question entity (1 - known => retrieval more likely)

Threshold is a single interpretable scalar, tuned on validation data in
``scripts/train_decision.py`` (never on test).
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any, Tuple

from ..types import Decision
from ..answering.span_extraction import question_type
from .features import extract_features


class HeuristicDecisionEngine:
    name = "heuristic"

    def __init__(self, cfg: dict = None, familiarity_prober: Optional[Callable] = None) -> None:
        cfg = cfg or {}
        w = cfg.get("weights", {})
        defaults = {"temporal": 0.35, "entity": 0.25, "unknown": 0.5, "wh_factor": 0.4}
        self.w_temporal = float(w.get("temporal", defaults["temporal"]))
        self.w_entity = float(w.get("entity", defaults["entity"]))
        self.w_unknown = float(w.get("unknown", defaults["unknown"]))
        self.w_wh = float(w.get("wh_factor", defaults["wh_factor"]))
        self.threshold = float(cfg.get("threshold", 0.5))
        self.prober = familiarity_prober

    def predict_score(self, question: str, familiarity: Optional[dict]) -> float:
        f = extract_features(question, familiarity)
        names = [
            "q_len_tokens", "q_len_chars", "q_entity_density", "q_year_cue",
            "q_superlative_cue", "q_quoted_title",
            "wh_who", "wh_what", "wh_when", "wh_where", "wh_quantity",
            "wh_comparison", "wh_other",
            "fam_best_title_cov", "fam_titles_hit", "fam_best_rel_cov", "fam_title_idf",
        ]
        vals = dict(zip(names, f))

        temporal = max(vals["q_year_cue"], vals["q_superlative_cue"], vals["q_quoted_title"])
        entity = vals["q_entity_density"]
        known = vals["fam_best_title_cov"]
        qtype = question_type(question)
        wh_openness = 0.0
        if qtype in ("what", "when", "who", "quantity"):
            wh_openness = 1.0  # open factual WH question; retrieval tends to help

        base = self.w_temporal * temporal + self.w_entity * entity + \
            self.w_unknown * (1.0 - known) + self.w_wh * wh_openness
        # Interpretability gate: when the model's own memory strongly anchors
        # the question's entity, ANY retrieval need multiplies down.
        score = base * (1.0 - 0.9 * known)
        return min(1.0, score)

    def decide(self, question: str, familiarity: Optional[dict] = None) -> Tuple[Decision, float, Dict[str, Any]]:
        if self.prober is not None and familiarity is None:
            familiarity = self.prober.familiarity_probe(question)
        score = self.predict_score(question, familiarity)
        decision = Decision.RETRIEVE if score >= self.threshold else Decision.DO_NOT_RETRIEVE
        return decision, round(score, 4), {"threshold": self.threshold}
