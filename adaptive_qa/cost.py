"""Cost model for measured/estimated expense.

The brief requires estimated cost per question where measurable. With the
deterministic local backends there is no API spend; the measurable cost
dimension is retrieval frequency (retrievals per question) plus CPU time.
If a hosted answering or embedding backend is configured, pass its published
prices as ``cost_per_llm_call_usd`` / ``cost_per_retrieval_usd`` and the
cost fields become real estimates (documented in the report).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _policy_uses_retrieval(policy: str, rec: Dict[str, Any]) -> bool:
    """Does THIS policy retrieve on THIS question? Computed per policy —
    retrieval frequency must NEVER be copied across the four policies
    (bug fix: the previous implementation inferred is_b from ``decision``
    for every policy, making never/always/oracle identical)."""
    if policy == "always_retrieve":
        return True
    if policy == "never_retrieve":
        return False
    if policy == "adaptive":
        return rec.get("decision") == "RETRIEVE"
    if policy == "oracle":
        outcome = rec.get("outcome") or {}
        if outcome.get("retrieval_better"):
            return True
        if outcome.get("no_retrieval_better"):
            return False
        # tied / ambiguous: follow the decision engine when available
        return rec.get("decision") == "RETRIEVE"
    # fallback: explicit per-question policy when present
    return rec.get("primary_method") == "B"


def cost_summary(
    records: List[Dict[str, Any]],
    cost_per_retrieval_usd: Optional[float] = None,
    cost_per_llm_call_usd: Optional[float] = None,
) -> Dict[str, Any]:
    n = max(1, len(records))

    def for_policy(policy: str) -> Dict[str, Any]:
        retrievals = 0
        llm_calls = 0
        for rec in records:
            is_b = _policy_uses_retrieval(policy, rec)
            if is_b:
                retrievals += 1
                llm_calls += 1
            else:
                llm_calls += 1  # closed-book generation is still a model call
        return {
            "policy": policy,
            "retrieval_frequency_per_question": round(retrievals / n, 4),
            "llm_calls_per_question": round(llm_calls / n, 4),
            "estimated_cost_per_question_usd": round(
                (retrievals * (cost_per_retrieval_usd or 0.0))
                + (llm_calls * (cost_per_llm_call_usd or 0.0)) / n,
                8,
            ),
            "note": (
                "0.0 unless hosted per-call prices are configured; local "
                "deterministic backends have zero API spend."
            ),
        }

    out = {}
    for policy in ("never_retrieve", "always_retrieve", "adaptive", "oracle"):
        out[policy] = for_policy(policy)
    return out
