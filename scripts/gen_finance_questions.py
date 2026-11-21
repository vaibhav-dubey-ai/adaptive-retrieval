"""Draft labelled retrieval questions from the ACTIVE documents with the configured LLM.

Each question is written against one chunk (or two adjacent chunks for
``multi_passage``), so the label is correct by construction. Output is a DRAFT:
every line has ``"needs_review": true``; read the file and delete or fix weak
questions before running the evaluations.

    python scripts/gen_finance_questions.py --n 80
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_qa.answering.llm_backend import resolve_chat_kwargs  # noqa: E402
from adaptive_qa.live.documents import DocumentStore  # noqa: E402

BOILERPLATE = re.compile(
    r"table of contents|exhibit|signature|pursuant to|commission file|incorporated by reference|"
    r"forward-looking|\.{5,}|www\.|@",
    re.I,
)
STYLE = {
    "exact_terminology": "Reuse the exact key terms and wording found in the passage.",
    "paraphrase": "Do NOT reuse the passage's distinctive words; ask in different everyday wording.",
    "definition": "Ask what a specific term, measure or concept named in the passage means or how it is defined.",
    "multi_passage": "The question needs facts from BOTH passages to be answered fully.",
}
BAD_Q = re.compile(r"\b(this|the following) (passage|section|text|excerpt|paragraph|document)\b|according to the passage", re.I)


def usable(c: dict) -> bool:
    t = c["text"]
    alpha = sum(ch.isalpha() for ch in t) / max(1, len(t))
    return 450 <= len(t) <= 1400 and alpha >= 0.7 and not BOILERPLATE.search(t)


def ask(kw: dict, passages: list[str], style: str) -> dict | None:
    body = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(passages))
    prompt = (
        f"{body}\n\nWrite ONE self-contained question about a company's financial report that these "
        f"passages answer. {STYLE[style]} Name the company/period if needed so the question stands alone; "
        'never say "this passage". Reply with JSON only: {"question": "...", "answer": "..."} where '
        "answer is a short exact span copied from the passages."
    )
    r = requests.post(
        kw["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {kw['api_key']}"},
        json={"model": kw["model"], "temperature": 0.3, "max_tokens": 200,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
    )
    r.raise_for_status()
    m = re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    q, a = str(d.get("question", "")).strip(), str(d.get("answer", "")).strip()
    if len(q) < 25 or not q.endswith("?") or BAD_Q.search(q) or not a:
        return None
    return {"question": q, "answer": a}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="eval_data/retrieval_eval/jpm_questions.jsonl")
    args = ap.parse_args(argv)

    kw = resolve_chat_kwargs()
    if not all(kw.get(k) for k in ("base_url", "api_key", "model")):
        print("LLM not configured (.env: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)", file=sys.stderr)
        return 2
    chunks = DocumentStore().load_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    pool = [c for c in chunks if usable(c)]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    print(f"{len(chunks)} active chunks, {len(pool)} usable; drafting {args.n}")

    styles = ["exact_terminology", "paraphrase", "definition", "multi_passage"]
    weights = [0.3, 0.3, 0.25, 0.15]
    rows, seen, fails = [], set(), 0
    for c in pool:
        if len(rows) >= args.n:
            break
        style = rng.choices(styles, weights)[0]
        chosen = [c]
        if style == "multi_passage":
            nxt = by_id.get(f"{c['doc_id']}:{c['chunk_index'] + 1:04d}")
            if not nxt or not usable(nxt):
                style, chosen = "definition", [c]
            else:
                chosen = [c, nxt]
        try:
            d = ask(kw, [x["text"] for x in chosen], style)
        except requests.RequestException as exc:
            print(f"LLM call failed, skipping chunk: {exc}", file=sys.stderr)
            fails += 1
            if fails >= 5:
                break
            continue
        fails = 0
        if not d or d["question"].lower() in seen:
            continue
        seen.add(d["question"].lower())
        rows.append({
            "question_id": f"f{len(rows) + 1:02d}", "category": style, "question": d["question"],
            "relevant_chunk_ids": [x["chunk_id"] for x in chosen], "reference_answer": d["answer"],
            "notes": f"page {c.get('page')}", "needs_review": True,
        })
        print(f"[{len(rows)}/{args.n}] {style}: {d['question']}", flush=True)

    out = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    header = ("# DRAFT questions generated by scripts/gen_finance_questions.py from the active JPMorgan 10-K/10-Q chunks.\n"
              "# Label = the chunk(s) each question was written from. REVIEW every line (delete vague ones) before use.\n")
    out.write_text(header + "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} questions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
