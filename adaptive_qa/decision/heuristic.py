"""Interpretable heuristic pre-retrieval policy.

The policy answers one question: *how much does this question look like it
needs evidence from the active collection?* It is deliberately a transparent
weighted mean of three bounded components, so the dashboard can show exactly
why a decision was taken:

1. ``question_cue`` (always available)
   Surface cues that a question wants a specific external fact rather than a
   general definition: a year, a superlative/ordinal, a quoted title, a high
   named-entity density, or an open WH form (what/when/who/how many).

2. ``memory_uncertainty`` (available only when a familiarity probe exists)
   ``1 - best_title_cov`` from the answering model's own parametric-memory
   probe. When no prober is configured this component is EXCLUDED and the
   score is re-normalised over the components that do exist. (The earlier
   implementation always counted it as 1.0 in the dashboard flow, which is
   why every uploaded-document question scored exactly 0.500.)

3. ``collection_relevance`` (available when a collection preflight exists)
   Aggregate lexical relevance of the ACTIVE document collection — the
   fraction of the question's content terms that the collection actually
   contains — from :mod:`adaptive_qa.decision.preflight`. This is what makes
   the decision depend on the documents the user actually uploaded. It gets
   the largest weight because, with a local corpus, "does the corpus contain
   this question's terminology at all" is the strongest available pre-retrieval
   signal; the surface cues only act as a weak prior.

    score = sum_i(w_i * v_i) / sum_i(w_i)      over AVAILABLE components

``score`` is a bounded retrieval-need score in [0, 1]. It is NOT a calibrated
probability: no calibration study exists, and the dashboard must not present
it as one. ``RETRIEVE`` is issued when ``score >= threshold``. The threshold
is a single configurable operating point (``decision.heuristic.threshold``);
it was not tuned per question.

Leakage contract (enforced by tests): the engine receives ONLY the question
string plus optional model-memory / collection-relevance signals. It never
sees retrieved passages, reference answers, gold evidence or oracle labels.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..types import Decision
from ..answering.span_extraction import question_type
from .features import extract_features, FEATURE_NAMES

# Component weights: the relative say each available signal has.
DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "question_cue": 0.15,
    "memory_uncertainty": 0.20,
    "collection_relevance": 0.70,
}

# Sub-weights of the composite question_cue component (normalised to [0, 1]).
DEFAULT_CUE_WEIGHTS: Dict[str, float] = {
    "temporal": 0.35,
    "superlative": 0.15,
    "quoted": 0.25,
    "entity": 0.15,
    "wh_open": 0.10,
}

# Legacy config keys map onto the same sub-weights so older config files keep
# working instead of being silently ignored.
_LEGACY_CUE_ALIASES = {
    "temporal": "temporal",
    "temporal_cue": "temporal",
    "superlative": "superlative",
    "superlative_cue": "superlative",
    "quoted": "quoted",
    "quoted_title": "quoted",
    "entity": "entity",
    "entity_count": "entity",
    "wh_factor": "wh_open",
    "wh_open": "wh_open",
}
_COMPONENT_KEYS = tuple(DEFAULT_COMPONENT_WEIGHTS)

_OPEN_WH = ("what", "when", "who", "quantity")

SCORE_NOTE = (
    "Weighted mean of the AVAILABLE signals in [0, 1] — a retrieval-need "
    "score, not a calibrated probability."
)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class HeuristicDecisionEngine:
    name = "heuristic"

    def __init__(
        self,
        cfg: dict = None,
        familiarity_prober: Optional[Callable] = None,
        collection_prober: Any = None,
    ) -> None:
        cfg = cfg or {}
        raw_weights = dict(cfg.get("weights", {}) or {})
        self.weights: Dict[str, float] = dict(DEFAULT_COMPONENT_WEIGHTS)
        # A legacy ``unknown`` weight is the memory component's weight.
        if "unknown" in raw_weights or "unknown_familiarity" in raw_weights:
            self.weights["memory_uncertainty"] = float(
                raw_weights.get("unknown", raw_weights.get("unknown_familiarity"))
            )
        for key in _COMPONENT_KEYS:
            if key in raw_weights:
                self.weights[key] = float(raw_weights[key])

        self.cue_weights: Dict[str, float] = dict(DEFAULT_CUE_WEIGHTS)
        cue_cfg = dict(cfg.get("cue_weights", {}) or {})
        for legacy_key, target in _LEGACY_CUE_ALIASES.items():
            if legacy_key in raw_weights:
                cue_cfg.setdefault(target, raw_weights[legacy_key])
        for key, value in cue_cfg.items():
            if key in self.cue_weights:
                self.cue_weights[key] = float(value)

        self.threshold = float(cfg.get("threshold", 0.5))
        self.prober = familiarity_prober
        self.collection_prober = collection_prober
        self._last_preflight: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------- components
    def _cue(self, question: str, feats: Dict[str, float]) -> Dict[str, Any]:
        qtype = question_type(question)
        triggers = {
            "temporal": _clip(feats["q_year_cue"]),
            "superlative": _clip(feats["q_superlative_cue"]),
            "quoted": _clip(feats["q_quoted_title"]),
            "entity": _clip(feats["q_entity_density"]),
            "wh_open": 1.0 if qtype in _OPEN_WH else 0.0,
        }
        total_w = sum(self.cue_weights.values()) or 1.0
        value = sum(self.cue_weights[k] * v for k, v in triggers.items()) / total_w
        fired = [k for k, v in triggers.items() if v > 0]
        explanation = (
            "question-level cues: "
            + ", ".join(
                f"{k}={triggers[k]:.2f}" for k in ("temporal", "superlative", "quoted", "entity")
            )
            + f", open-WH={triggers['wh_open']:.0f} (question type: {qtype})"
            + (f"; triggers: {', '.join(fired)}" if fired else "; no specificity cue fired")
        )
        return {
            "value": round(_clip(value), 4),
            "weight": self.weights["question_cue"],
            "available": True,
            "explanation": explanation,
            "detail": {k: round(v, 4) for k, v in triggers.items()},
        }

    def _memory(self, familiarity: Optional[dict]) -> Dict[str, Any]:
        if not familiarity:
            return {
                "value": None,
                "weight": self.weights["memory_uncertainty"],
                "available": False,
                "explanation": (
                    "no model-memory familiarity probe is configured for this flow, so "
                    "this component is excluded and the score re-normalised "
                    "(it must not be reported as a constant 1.0)"
                ),
            }
        known = _clip(float(familiarity.get("best_title_cov", 0.0) or 0.0))
        return {
            "value": round(1.0 - known, 4),
            "weight": self.weights["memory_uncertainty"],
            "available": True,
            "explanation": (
                f"model-memory familiarity: best title coverage {known:.2f} "
                f"=> uncertainty {1.0 - known:.2f}"
            ),
            "detail": {
                "best_title_cov": known,
                "titles_hit": familiarity.get("titles_hit"),
                "best_rel_cov": familiarity.get("best_rel_cov"),
            },
        }

    def _collection(self, preflight: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not preflight or not preflight.get("available"):
            return {
                "value": None,
                "weight": self.weights["collection_relevance"],
                "available": False,
                "explanation": (
                    "no active document collection to preflight (upload and activate a "
                    "document to make document relevance part of the decision)"
                ),
            }
        signals = preflight.get("signals") or {}
        value = _clip(float(signals.get("collection_relevance", 0.0)))
        n_chunks = (preflight.get("collection") or {}).get("n_chunks")
        return {
            "value": round(value, 4),
            "weight": self.weights["collection_relevance"],
            "available": True,
            "explanation": (
                f"document relevance: fraction of the question's content terms that "
                f"occur in the {n_chunks} active chunks — corpus "
                f"{signals.get('corpus_coverage', 0):.2f}, top-"
                f"{(preflight.get('collection') or {}).get('top_k_probed')} "
                f"{signals.get('topk_coverage', 0):.2f}, best chunk "
                f"{signals.get('top1_coverage', 0):.2f} => {value:.2f}"
            ),
            "detail": dict(signals),
        }

    # -------------------------------------------------------------- scoring
    def trace(
        self,
        question: str,
        familiarity: Optional[dict] = None,
        collection: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Full, human-readable decision trace (used by the dashboard)."""
        feats_list = extract_features(question, familiarity)
        feats = dict(zip(FEATURE_NAMES, feats_list))
        components = {
            "question_cue": self._cue(question, feats),
            "memory_uncertainty": self._memory(familiarity),
            "collection_relevance": self._collection(collection),
        }
        available = {k: c for k, c in components.items() if c["available"]}
        w_total = sum(c["weight"] for c in available.values())
        score = (
            sum(c["weight"] * float(c["value"]) for c in available.values()) / w_total
            if w_total > 0
            else 0.0
        )
        score = round(_clip(score), 4)
        decision = (
            Decision.RETRIEVE if score >= self.threshold else Decision.DO_NOT_RETRIEVE
        )
        for c in components.values():
            c["contribution"] = (
                round(c["weight"] * float(c["value"]) / w_total, 4)
                if c["available"] and w_total > 0
                else 0.0
            )
        ranked = sorted(
            available.items(), key=lambda kv: kv[1]["contribution"], reverse=True
        )
        rationale = (
            f"{decision.value}: score {score:.3f} "
            f"{'≥' if decision == Decision.RETRIEVE else '<'} threshold "
            f"{self.threshold:.2f}, computed as a weighted mean over "
            f"{len(available)} available signal(s) — "
            + "; ".join(
                f"{k} {float(c['value']):.2f}×w{c['weight']:.2f}" for k, c in ranked
            )
            + "."
        )
        excluded = [k for k, c in components.items() if not c["available"]]
        if excluded:
            rationale += " Excluded (unavailable): " + ", ".join(excluded) + "."

        return {
            "engine": self.name,
            "decision": decision.value,
            "score": score,
            "threshold": self.threshold,
            "decision_rule": "RETRIEVE if score >= threshold, else DO_NOT_RETRIEVE",
            "score_definition": SCORE_NOTE,
            "components": components,
            "features": {k: round(float(v), 4) for k, v in feats.items()},
            "rationale": rationale,
            "memory_available": bool(familiarity),
            "collection_available": bool(
                collection and collection.get("available")
            ),
            "preflight": collection or {"available": False},
        }

    def predict_score(
        self,
        question: str,
        familiarity: Optional[dict],
        collection: Optional[Dict[str, Any]] = None,
    ) -> float:
        return float(self.trace(question, familiarity, collection)["score"])

    # ------------------------------------------------------------ inference
    def decide(
        self, question: str, familiarity: Optional[dict] = None
    ) -> Tuple[Decision, float, Dict[str, Any]]:
        if self.prober is not None and familiarity is None:
            familiarity = self.prober.familiarity_probe(question)
        collection = None
        if self.collection_prober is not None:
            result = self.collection_prober.probe(question)
            collection = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            self._last_preflight = collection
        meta = self.trace(question, familiarity, collection)
        return Decision(meta["decision"]), meta["score"], meta
