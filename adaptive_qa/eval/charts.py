"""Run-artifact charts (matplotlib, saved into the run directory).

Charts: quality-vs-latency scatter per policy, quality-vs-cost bars, and the
adaptive decision confusion. The dashboard serves equivalent data from
``summary.json`` for the browser to render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402


def _policy_points(summary: Dict[str, Any]):
    pts = []
    for policy, m in (summary.get("policies") or {}).items():
        if not m or not m.get("n_questions"):
            continue
        lat = m.get("latency_ms", {})
        pts.append(
            {
                "policy": policy,
                "quality": m.get("composite_quality", 0.0),
                "em": m.get("em", 0.0),
                "f1": m.get("f1", 0.0),
                "mean_e2e_ms": lat.get("mean_e2e", 0.0),
                "retrieval_rate": m.get("retrieval_rate", 0.0),
                "cost": m.get("estimated_cost_per_question_usd", 0.0),
            }
        )
    return pts


def plot_quality_latency(summary: Dict[str, Any], out_dir: Path) -> None:
    pts = _policy_points(summary)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in pts:
        ax.scatter(p["mean_e2e_ms"], p["quality"], s=80)
        ax.annotate(
            p["policy"],
            (p["mean_e2e_ms"], p["quality"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("mean end-to-end latency (ms)")
    ax.set_ylabel("composite quality (0.5*EM + 0.5*F1)")
    ax.set_title("Answer quality vs latency per policy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_quality_cost(summary: Dict[str, Any], out_dir: Path) -> None:
    pts = _policy_points(summary)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    policies = [p["policy"] for p in pts]
    ax.bar(policies, [p["quality"] for p in pts], color="#4a76a8", alpha=0.8)
    for i, p in enumerate(pts):
        ax.text(
            i,
            p["quality"] + 0.01,
            f"retr={p['retrieval_rate']:.0%}\ncost={p['cost']:.2e}$",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("composite quality")
    ax.set_title("Quality, retrieval rate and cost per policy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_cost.png", dpi=150)
    plt.close(fig)


def plot_retrieval_rate(summary: Dict[str, Any], out_dir: Path) -> None:
    pts = _policy_points(summary)
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    policies = [p["policy"] for p in pts]
    ax.bar(policies, [p["retrieval_rate"] for p in pts], color="#76a84a", alpha=0.8)
    ax.set_ylabel("retrieval rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval frequency per policy")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_rate.png", dpi=150)
    plt.close(fig)


def write_all(summary: Dict[str, Any], run_dir: str) -> None:
    out_dir = Path(run_dir)
    plot_quality_latency(summary, out_dir)
    plot_quality_cost(summary, out_dir)
    plot_retrieval_rate(summary, out_dir)
