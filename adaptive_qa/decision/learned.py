"""Learned classifier pre-retrieval policy (scikit-learn gradient boosting).

Trained OFFLINE on train-split questions whose labels come from the
cross-verification oracle on those same train questions (supervised learning
on training data only). Threshold is tuned for the RETRIEVE-class F1 on the
validation split. This engine then runs online BEFORE retrieval; it never
sees reference answers or oracle labels at inference time.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import resolve_path
from ..types import Decision
from .features import FEATURE_NAMES, extract_features


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
        return decision, round(proba, 4), {"threshold": self.threshold}

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
