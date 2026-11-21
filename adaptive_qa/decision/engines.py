"""Decision engines: question features, the heuristic engine and the learned (scikit-learn) engine."""

from __future__ import annotations

import json
import numpy as np
import pickle
import re
import time
from ..answering.simulated import question_type
from ..common import Decision
from ..common import resolve_path
from ..retrieval.bm25 import tokenize
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import Dict, List, Optional


# ======================================================================
# Pre-retrieval features for the adaptive decision engine.
# ======================================================================

WH_TYPES = ["who", "what", "when", "where", "quantity", "comparison", "other"]

_SUPERLATIVE_RE = re.compile(
    r"\b(first|last|largest|smallest|oldest|youngest|most|best|longest|shortest|"
    r"current|oldest|prime|biggest|only)\b",
    re.IGNORECASE,
)
_YEAR_CUE_RE = re.compile(r"\b(1[4-9]\d{2}|20[0-4]\d)\b")
_QUOTES_CUE_RE = re.compile(r"\"([^\"]+)\"")


def extract_features(question: str, familiarity: Optional[dict] = None) -> List[float]:
    """Build the numeric pre-retrieval feature vector (stable order)."""
    q = question.strip()
    toks = tokenize(q)
    qtype = question_type(q)

    feat: List[float] = []
    feat.append(min(1.0, len(toks) / 20.0))                # question length (tokens)
    feat.append(min(1.0, len(q) / 120.0))                  # question length (chars)
    # mid-sentence capitalized tokens ~ named-entity density
    caps = sum(1 for t in q.split()[1:] if t[:1].isupper() and t[1:2].islower())
    feat.append(min(1.0, caps / 4.0))
    feat.append(1.0 if _YEAR_CUE_RE.search(q) else 0.0)    # asks about a specific year
    feat.append(1.0 if _SUPERLATIVE_RE.search(q) else 0.0) # superlative/ordinal cues
    feat.append(1.0 if _QUOTES_CUE_RE.search(q) else 0.0)  # quotes -> titles/works

    wh_onehot = [1.0 if qtype == t else 0.0 for t in WH_TYPES]
    feat.extend(wh_onehot)

    fam = familiarity or {}
    feat.append(float(fam.get("best_title_cov", 0.0) or 0.0))
    feat.append(min(1.0, (fam.get("titles_hit", 0) or 0) / 5.0))
    feat.append(float(fam.get("best_rel_cov", 0.0) or 0.0))
    if fam.get("best_title_idf"):
        feat.append(min(1.0, fam["best_title_idf"] / 10.0))
    else:
        feat.append(0.0)
    return feat


FEATURE_NAMES: List[str] = [
    "q_len_tokens",
    "q_len_chars",
    "q_entity_density",
    "q_year_cue",
    "q_superlative_cue",
    "q_quoted_title",
] + [f"wh_{t}" for t in WH_TYPES] + [
    "fam_best_title_cov",
    "fam_titles_hit",
    "fam_best_rel_cov",
    "fam_title_idf",
]


def features_dict(question: str, familiarity: Optional[dict] = None) -> Dict[str, float]:
    vec = extract_features(question, familiarity)
    return dict(zip(FEATURE_NAMES, vec))


# ======================================================================
# Interpretable heuristic pre-retrieval policy.
# ======================================================================

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


# ======================================================================
# Learned classifier pre-retrieval policy (scikit-learn gradient boosting).
# ======================================================================

class LearnedDecisionEngine:
    name = "learned"

    def __init__(self, cfg: dict = None, familiarity_prober: Optional[Callable] = None) -> None:
        cfg = cfg or {}
        self.hyper = cfg.get("learned", {})
        self.threshold = float(cfg.get("retrieve_class_threshold", 0.5))
        self.model = None
        self.prober = familiarity_prober
        self.trained_at: Optional[str] = None

    # ------------------------------------------------------------- training
    def fit(
        self,
        questions: Sequence[str],
        labels: Sequence[int],  # 1 = retrieve beneficial, 0 = not
        familiarities: Optional[Sequence[dict]] = None,
        valid_questions: Optional[Sequence[str]] = None,
        valid_labels: Optional[Sequence[int]] = None,
        valid_familiarities: Optional[Sequence[dict]] = None,
    ) -> Dict[str, Any]:
        """Train gradient boosting; tune the decision threshold on validation."""
        from sklearn.ensemble import GradientBoostingClassifier

        X = np.array([
            extract_features(q, fam) for q, fam in zip(
                questions,
                familiarities or [None] * len(questions),
            )
        ])
        y = np.array(labels)
        assert len(X) == len(y) and len(X) > 0, "empty training set"
        model = GradientBoostingClassifier(
            n_estimators=int(self.hyper.get("n_estimators", 200)),
            learning_rate=float(self.hyper.get("learning_rate", 0.05)),
            max_depth=int(self.hyper.get("max_depth", 3)),
            random_state=self.hyper.get("random_state", 13),
        )
        model.fit(X, y)
        self.model = model
        self.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        tuning: Dict[str, Any] = {}
        if valid_questions and valid_labels is not None and len(valid_questions) > 0:
            Xv = np.array([
                extract_features(q, fam) for q, fam in zip(
                    valid_questions,
                    valid_familiarities or [None] * len(valid_questions),
                )
            ])
            yv = np.array(valid_labels)
            best_f1 = -1.0
            best_thresh = self.threshold
            probs = model.predict_proba(Xv)[:, 1]
            for threshold in np.arange(0.05, 0.96, 0.05):
                pred = (probs >= threshold).astype(int)
                tp = int(((pred == 1) & (yv == 1)).sum())
                fp = int(((pred == 1) & (yv == 0)).sum())
                fn = int(((pred == 0) & (yv == 1)).sum())
                f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
                # on F1 ties prefer the threshold closest to a neutral 0.5
                if f1 > best_f1 or (
                    f1 == best_f1 and abs(threshold - 0.5) < abs(best_thresh - 0.5)
                ):
                    best_f1 = f1
                    best_thresh = round(float(threshold), 2)
            self.threshold = best_thresh
            tuning = {"valid_f1_retrieve": round(best_f1, 4), "tuned_threshold": self.threshold}
        return tuning

    # ------------------------------------------------------------ inference
    def decide(self, question: str, familiarity: Optional[dict] = None) -> Tuple[Decision, float, Dict[str, Any]]:
        assert self.model is not None, "learned decision engine not trained/loaded"
        if self.prober is not None and familiarity is None:
            familiarity = self.prober.familiarity_probe(question)
        x = np.array([extract_features(question, familiarity)])
        proba = float(self.model.predict_proba(x)[0, 1])
        decision = Decision.RETRIEVE if proba >= self.threshold else Decision.DO_NOT_RETRIEVE
        meta = {
            "engine": self.name,
            "threshold": self.threshold,
            "decision_rule": "RETRIEVE if probability >= threshold, else DO_NOT_RETRIEVE",
            "score_definition": (
                "predict_proba of the trained gradient-boosting classifier — a model "
                "score, not a calibrated probability"
            ),
            "features": dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in x[0]])),
            "rationale": (
                f"{decision.value}: learned classifier score {proba:.3f} "
                f"{'≥' if decision == Decision.RETRIEVE else '<'} threshold {self.threshold:.2f} "
                f"over {len(FEATURE_NAMES)} question/memory features"
            ),
            "preflight": {"available": False},
            "note": (
                "the learned engine was trained offline on pre-retrieval features and "
                "does not consume the collection preflight signal"
            ),
        }
        return decision, round(proba, 4), meta

    # ---------------------------------------------------------- persistence
    def save(self, path: str, train_meta: Optional[dict] = None) -> None:
        p = resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "adaptive-qa-decision-bundle-v1",
            "feature_names": FEATURE_NAMES,
            "threshold": self.threshold,
            "trained_at": self.trained_at,
            "hyperparameters": self.hyper,
            "train_meta": train_meta or {},
        }
        p.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with open(str(p) + ".model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        p = resolve_path(path)
        blob = p.with_suffix(".pkl.model.pkl") if False else Path(str(p) + ".model.pkl")
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta.get("format") == "adaptive-qa-decision-bundle-v1"
        self.threshold = float(meta["threshold"])
        self.trained_at = meta.get("trained_at")
        with open(str(blob), "rb") as f:
            self.model = pickle.load(f)
