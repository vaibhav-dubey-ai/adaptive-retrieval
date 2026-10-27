"""Live, side-by-side comparison of the retrieval methods on ONE question.

The three methods are run over the *same* question, the *same* active chunk
collection and the *same* top-k, so their top-k lists are directly comparable.
Nothing here selects or filters results: every list is exactly what the
retriever returned.

Score scales differ by construction and are labelled, not blended:

- ``bm25``   : raw Okapi BM25 score, unbounded
- ``dense``  : cosine similarity in [-1, 1]
- ``hybrid`` : Reciprocal Rank Fusion score (sum of 1/(60+rank))

Only *ranks* and *set membership* are compared across methods.

The comparison never reads evaluation labels: relevance information lives in
the separate experiments view, so a live request cannot see ground truth.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..timeutil import now_ns
from ..types import RetrievalHit

SCORE_KIND = {
    "bm25": "BM25 score (unbounded)",
    "dense": "cosine similarity in [-1, 1]",
    "hybrid": "RRF score (sum of 1/(60+rank))",
}


def _entry(
    hit: RetrievalHit, rank: int, method: str, retriever: Any = None
) -> Dict[str, Any]:
    title = getattr(hit, "title", "") or ""
    entry: Dict[str, Any] = {
        "rank": rank,
        "doc_id": hit.doc_id,
        "score": round(float(hit.score), 4),
        "score_kind": SCORE_KIND.get(method, "score"),
        "title": title,
        "excerpt": (hit.text or "")[:500],
        "text": hit.text or "",
    }
    source = getattr(hit, "source_file", None)
    page = getattr(hit, "page", None)
    # Upload provenance lives in the document store, not on the frozen hit.
    if retriever is not None and hasattr(retriever, "source_meta"):
        try:
            sm = retriever.source_meta(hit) or {}
            source = sm.get("filename", source)
            page = sm.get("page", page)
        except Exception:
            pass
    if source is not None:
        entry["source_file"] = source
        entry["page"] = page if page is not None else None
    ranks = getattr(hit, "component_ranks", None)
    if ranks:
        entry["component_ranks"] = dict(ranks)
    scores = getattr(hit, "component_scores", None)
    if scores:
        entry["component_scores"] = {
            k: round(float(v), 4) for k, v in scores.items()
        }
    return entry


def compare_methods(
    question: str,
    methods: Sequence[str],
    top_k: int,
    retriever_provider: Callable[[str], Any],
    collection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every method on ``question`` and annotate overlap between rankings.

    ``retriever_provider(method)`` returns a ready retriever (the API caches
    them per collection snapshot). Its construction time is measured
    separately as ``setup_ms`` so a first-request model load is never reported
    as query latency.
    """
    out_methods: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        t0 = now_ns()
        try:
            retriever = retriever_provider(method)
        except Exception as exc:  # e.g. embeddings unavailable
            out_methods[method] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hits": [],
                "setup_ms": None,
                "query_ms": None,
            }
            continue
        setup_ms = round((now_ns() - t0) / 1_000_000.0, 3)
        if retriever is None:
            out_methods[method] = {
                "ok": False,
                "error": "No active document collection to search.",
                "hits": [],
                "setup_ms": setup_ms,
                "query_ms": None,
            }
            continue
        t1 = now_ns()
        try:
            hits = retriever.retrieve(question, top_k)
        except Exception as exc:
            out_methods[method] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hits": [],
                "setup_ms": setup_ms,
                "query_ms": round((now_ns() - t1) / 1_000_000.0, 3),
            }
            continue
        query_ms = round((now_ns() - t1) / 1_000_000.0, 3)
        diagnostics = None
        if hasattr(retriever, "diagnostics"):
            try:
                diagnostics = retriever.diagnostics()
            except Exception:
                diagnostics = None
        out_methods[method] = {
            "ok": True,
            "error": None,
            "setup_ms": setup_ms,
            "query_ms": query_ms,
            "hits": [_entry(h, i + 1, method, retriever) for i, h in enumerate(hits)],
            "diagnostics": diagnostics,
            "score_kind": SCORE_KIND.get(method),
        }

    # -------------------------------------------------- cross-method overlap
    ids_by_method: Dict[str, List[str]] = {
        m: [h["doc_id"] for h in r["hits"]] for m, r in out_methods.items() if r["ok"]
    }
    counts = Counter(cid for lst in ids_by_method.values() for cid in lst)
    for method, res in out_methods.items():
        for hit in res["hits"]:
            found_by = sorted(m for m, lst in ids_by_method.items() if hit["doc_id"] in lst)
            hit["found_by"] = found_by
            hit["shared"] = len(found_by) > 1
            if len(found_by) > 1:
                hit["ranks_by_method"] = {
                    m: ids_by_method[m].index(hit["doc_id"]) + 1 for m in found_by
                }
                hit["rank_note"] = " / ".join(
                    f"{m} #{ids_by_method[m].index(hit['doc_id']) + 1}" for m in found_by
                )

    ok_methods = [m for m, r in out_methods.items() if r["ok"]]
    shared_by_all = (
        sorted(cid for cid, c in counts.items() if c == len(ok_methods)) if ok_methods else []
    )
    unique_to = {
        m: [cid for cid in lst if counts[cid] == 1] for m, lst in ids_by_method.items()
    }
    pairwise: Dict[str, float] = {}
    for i, a in enumerate(ok_methods):
        for b in ok_methods[i + 1:]:
            sa, sb = set(ids_by_method[a]), set(ids_by_method[b])
            union = sa | sb
            pairwise[f"{a}|{b}"] = (
                round(len(sa & sb) / len(union), 3) if union else 0.0
            )

    return {
        "question": question,
        "top_k": int(top_k),
        "methods": out_methods,
        "method_order": list(methods),
        "collection": collection or {},
        "overlap": {
            "shared_by_all": shared_by_all,
            "shared_by_two_or_more": sorted(cid for cid, c in counts.items() if c > 1),
            "unique_to": unique_to,
            "pairwise_jaccard": pairwise,
            "n_unique": {m: len(v) for m, v in unique_to.items()},
        },
        "note": (
            "Same question, same active chunks, same top-k for every method. "
            "Scores are on different scales (see score_kind) and are not "
            "comparable across methods; only ranks and set membership are."
        ),
        "labels_included": False,
    }
