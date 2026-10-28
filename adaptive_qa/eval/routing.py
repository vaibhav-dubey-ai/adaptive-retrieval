"""Routing evaluation: does the router retrieve exactly when retrieval helps?

The label here is *behavioural*, not "is the topic in the corpus": for every
question both answering branches run (closed-book A, grounded B), each answer
is scored against the reference with ``quality_score``, and

    retrieval_helps = quality(B) - quality(A) >= gap

Routers are then compared on questions they were NOT tuned on (repeated random
train/test splits) by what actually matters: the answer quality obtained, how
often retrieval was paid for, and the tokens/latency spent.

Policies
--------
never / always / oracle     fixed baselines (oracle = evaluation upper bound)
heuristic@0.5               the shipped engine, untuned threshold
heuristic-tuned             same score, threshold chosen on the train split
learned                     logistic regression on preflight + question cues,
                            trained on the train split

The deployed routers only see the question and the collection preflight; labels
and answers are used only to fit/score them here.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from .em_f1 import quality_score

FEATURES = (
    "corpus_coverage",
    "topk_coverage",
    "top1_coverage",
    "collection_relevance",
    "n_question_terms",
    "q_len_tokens",
    "q_entity_density",
    "wh_open",
)


def label_rows(rows: Sequence[Dict[str, Any]], gap: float = 0.10) -> List[Dict[str, Any]]:
    """Add quality_a/quality_b/delta/retrieval_helps; drop rows that cannot be scored."""
    out = []
    for r in rows:
        ref = (r.get("reference_answer") or "").strip()
        if not ref or not r.get("answer_a") or not r.get("answer_b"):
            continue
        qa = quality_score(ref, r["answer_a"])
        qb = quality_score(ref, r["answer_b"])
        out.append({**r, "quality_a": qa, "quality_b": qb, "delta": qb - qa,
                    "retrieval_helps": (qb - qa) >= gap,
                    "retrieval_hurts": (qa - qb) >= gap})
    return out


def feature_vector(row: Dict[str, Any]) -> List[float]:
    sig = row["preflight"]
    cue = row["features"]
    return [
        sig.get("corpus_coverage", 0.0), sig.get("topk_coverage", 0.0),
        sig.get("top1_coverage", 0.0), sig.get("collection_relevance", 0.0),
        min(1.0, sig.get("n_question_terms", 0.0) / 8.0),
        cue.get("q_len_tokens", 0.0), cue.get("q_entity_density", 0.0),
        float(row.get("wh_open", 0.0)),
    ]


# ---------------------------------------------------------------- policies
def _tune_threshold(scores: Sequence[float], targets: Sequence[bool]) -> float:
    """Pick the score threshold with the best balanced accuracy on the train split."""
    best_t, best = 0.5, -1.0
    for t in sorted(set(round(s, 4) for s in scores)) + [1.01]:
        pred = [s >= t for s in scores]
        pos = [p for p, y in zip(pred, targets) if y]
        neg = [p for p, y in zip(pred, targets) if not y]
        tpr = sum(pos) / len(pos) if pos else 0.0
        tnr = (len(neg) - sum(neg)) / len(neg) if neg else 0.0
        bal = (tpr + tnr) / 2
        if bal > best + 1e-9:
            best, best_t = bal, t
    return best_t


def _fit_logreg(X: List[List[float]], y: List[bool]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if len(set(y)) < 2:
        return None
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(sc.transform(X), y)
    return sc, clf


def _outcome(rows: Sequence[Dict[str, Any]], retrieve: Sequence[bool]) -> Dict[str, Any]:
    """Quality / cost of a routing choice over ``rows`` (uses measured per-branch costs)."""
    n = len(rows)
    q = [r["quality_b"] if d else r["quality_a"] for r, d in zip(rows, retrieve)]
    tok = [(r["tokens_b"] if d else r["tokens_a"]) or 0 for r, d in zip(rows, retrieve)]
    ms = [(r["gen_b_ms"] + r["retrieval_ms"] if d else r["gen_a_ms"]) for r, d in zip(rows, retrieve)]
    y = [r["retrieval_helps"] for r in rows]
    tp = sum(1 for d, t in zip(retrieve, y) if d and t)
    tn = sum(1 for d, t in zip(retrieve, y) if (not d) and (not t))
    npos, nneg = sum(y), n - sum(y)
    tpr = tp / npos if npos else float("nan")
    tnr = tn / nneg if nneg else float("nan")
    return {
        "quality": sum(q) / n,
        "retrieval_rate": sum(retrieve) / n,
        "tokens": sum(tok) / n,
        "latency_ms": sum(ms) / n,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (tpr + tnr) / 2 if npos and nneg else float("nan"),
        "recall_of_helpful": tpr,
    }


def evaluate_split(train, test, threshold_default: float = 0.5) -> Dict[str, Dict[str, float]]:
    s_tr = [r["heuristic_score"] for r in train]
    t_tune = _tune_threshold(s_tr, [r["retrieval_helps"] for r in train])
    model = _fit_logreg([feature_vector(r) for r in train], [r["retrieval_helps"] for r in train])
    if model is None:
        learned = [r["heuristic_score"] >= threshold_default for r in test]
    else:
        sc, clf = model
        learned = list(clf.predict(sc.transform([feature_vector(r) for r in test])))
    return {
        "never": _outcome(test, [False] * len(test)),
        "always": _outcome(test, [True] * len(test)),
        "heuristic@0.5": _outcome(test, [r["heuristic_score"] >= threshold_default for r in test]),
        "heuristic-tuned": _outcome(test, [r["heuristic_score"] >= t_tune for r in test]),
        "learned": _outcome(test, [bool(x) for x in learned]),
        "oracle": _outcome(test, [r["quality_b"] > r["quality_a"] for r in test]),
    }


def repeated_splits(rows, n_splits: int = 30, test_frac: float = 0.3, seed: int = 13) -> Dict[str, Any]:
    """Mean ± sd of every policy metric over stratified random splits."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["retrieval_helps"]]
    neg = [r for r in rows if not r["retrieval_helps"]]
    acc: Dict[str, Dict[str, List[float]]] = {}
    for _ in range(n_splits):
        train, test = [], []
        for grp in (pos, neg):
            g = grp[:]
            rng.shuffle(g)
            k = max(1, int(round(len(g) * test_frac))) if len(g) > 1 else 0
            test += g[:k]
            train += g[k:]
        if not test or not train:
            continue
        for name, m in evaluate_split(train, test).items():
            for key, v in m.items():
                acc.setdefault(name, {}).setdefault(key, []).append(v)
    summary = {}
    for name, d in acc.items():
        summary[name] = {}
        for key, vals in d.items():
            vals = [v for v in vals if not math.isnan(v)]
            if not vals:
                summary[name][key] = {"mean": None, "sd": None}
                continue
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary[name][key] = {"mean": round(mean, 4), "sd": round(sd, 4)}
    return summary
