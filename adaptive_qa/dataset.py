"""Dataset interface for HotpotQA (distractor configuration).

- ``fetch_hotpotqa_rows`` pulls paged rows from the Hugging Face
  ``datasets-server`` REST API. Sampling is deterministic: a fixed row offset
  (derived from a seed) and ``n`` consecutive rows are taken, so re-running the
  download script reproduces the same dataset byte-for-byte.
- ``load_records_jsonl`` reads normalized question JSONL files.
- ``assign_splits`` performs question-level train/validation/test splitting
  with a seeded shuffle, so one question, its answer or its paragraphs never
  appear in more than one split.

The default sample uses 200 questions from the ``validation`` split (the eval
sample that drives corpus construction and all experiments). A separate slice
of ``train``-split rows is the source of the answering engine's simulated
parametric knowledge (Stage 2: ``build_parametric_store``); because HotpotQA
train and validation rows are disjoint, the closed-book engine's memory is
disjoint from the eval sample, mirroring how a real LLM was pretrained on text
that is disjoint from benchmark questions.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

import requests

from .config import resolve_path
from .types import QuestionRecord


def fetch_hotpotqa_rows(
    hf_rows_api: str,
    hf_dataset: str,
    hf_config: str,
    split: str,
    offset: int,
    n: int,
    row_page_size: int = 100,
    timeout_s: int = 30,
    filter_fn=None,
    max_raw_rows: int = None,
) -> List[Dict[str, Any]]:
    """Download ``n`` rows starting at ``offset`` (post-``filter_fn``) from
    the HF datasets-server. Passing ``filter_fn`` keeps only rows matching a
    documented criterion (e.g. ``level == 'easy'``), fetching pages until the
    filtered quota is reached or ``max_raw_rows`` raw rows were consumed."""
    dataset_q = quote(hf_dataset, safe="")
    config_q = quote(hf_config, safe="")
    split_q = quote(split, safe="")
    rows: List[Dict[str, Any]] = []
    remaining = n
    fetch_offset = offset
    raw_consumed = 0
    while remaining > 0:
        length = min(row_page_size, remaining)
        if filter_fn is not None:
            length = row_page_size
        url = (
            f"{hf_rows_api}?dataset={dataset_q}&config={config_q}&split={split_q}"
            f"&offset={fetch_offset}&length={length}"
        )
        payload = None
        last_err = "unknown"
        for attempt in range(8):
            try:
                response = requests.get(url, timeout=timeout_s)
                if response.status_code in (429, 500, 502, 503, 504):
                    last_err = f"transient HTTP {response.status_code}"
                    wait = min(2.0 * (2 ** attempt), 60.0)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as exc:
                # Remote-host resets killed earlier builds; retry within-page.
                last_err = f"connection error: {exc.__class__.__name__}"
                wait = min(2.0 * (2 ** attempt), 60.0)
                time.sleep(wait)
                continue
        if payload is None:
            raise RuntimeError(
                f"datasets-server request failed after 8 attempts at offset "
                f"{fetch_offset} (last error: {last_err})"
            )
        batch = payload.get("rows", [])
        if not batch:
            raise ValueError(
                f"datasets-server returned no rows at offset {fetch_offset} "
                f"(dataset={hf_dataset!r} config={hf_config!r} split={split!r})"
            )
        for item in batch:
            raw_consumed += 1
            if filter_fn is None or filter_fn(item["row"]):
                rows.append(item["row"])
        if filter_fn is None:
            remaining -= len(batch)
        else:
            remaining = n - len(rows)
        fetch_offset += len(batch)
        if max_raw_rows is not None and raw_consumed >= max_raw_rows:
            break
        time.sleep(0.5)  # be polite to the public API
    return rows[:n]


def normalize_hotpot_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one raw HotpotQA row into our question JSONL schema.

    ``gold_titles`` (supporting-fact paragraph titles) are evaluation data and
    travel with the record, but tests enforce that decision-engine code never
    reads them.
    """
    supporting = row.get("supporting_facts", {}) or {}
    st_titles = supporting.get("title", []) or []

    # titles list may arrive as a {"title": [...], "sent_id": [...]} map or
    # list-of-lists depending on the API serialization mode
    if isinstance(st_titles, dict):
        titles = list(st_titles.get("title", []))
    elif isinstance(st_titles, list):
        first = st_titles[0] if st_titles else None
        if isinstance(first, dict):
            titles = [t.get("title") for t in st_titles]
        else:
            titles = list(st_titles)
    else:
        titles = []

    context = row.get("context", []) or []
    paragraphs = []
    if isinstance(context, dict):
        titles_col = context.get("title", [])
        sents_col = context.get("sentences", [])
        paragraphs = [
            {"title": t, "sentences": s}
            for t, s in zip(titles_col, sents_col)
        ]
    elif isinstance(context, list):
        for para in context:
            if isinstance(para, dict):
                paragraphs.append(
                    {"title": para.get("title", ""), "sentences": para.get("sentences", [])}
                )
            else:
                # [title, [sents]]
                paragraphs.append({"title": para[0], "sentences": para[1]})

    gold_titles = sorted({t for t in titles if t})
    return {
        "question_id": row.get("id", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "gold_titles": gold_titles,
        "row_type": row.get("type", ""),
        "level": row.get("level", ""),
        "paragraphs": paragraphs,
    }


def save_records_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records_jsonl(path: str) -> List[Dict[str, Any]]:
    p = resolve_path(path)
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_question_records(records: List[Dict[str, Any]]) -> List[QuestionRecord]:
    out = []
    for rec in records:
        qr = QuestionRecord(
            question_id=rec["question_id"],
            question=rec["question"],
            answer=rec["answer"],
            gold_titles=tuple(rec.get("gold_titles", ())),
            row_type=rec.get("row_type", ""),
            level=rec.get("level", ""),
            split=rec.get("split", ""),
        )
        out.append(qr)
    return out


def choose_offset(total_available_hint: int, n: int, seed: int, margin: int = 500) -> int:
    """Deterministic random start offset for sampling a consecutive slice."""
    rng = random.Random(seed)
    upper = max(0, total_available_hint - margin - n)
    return rng.randint(0, upper)


def assign_splits(
    records: List[Dict[str, Any]],
    train_frac: float,
    validation_frac: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Question-level split; mutates the record dicts and returns them.

    Splitting at the *question* level means a question, its reference answer,
    and all of its supporting paragraphs stay within exactly one split. The
    shared corpus intentionally spans all questions (a real deployment shares
    one index), but decision-engine training material (features and labels)
    derives only from train-split questions.
    """
    assert 0 < train_frac < 1 and 0 < validation_frac < 1 and train_frac + validation_frac < 1
    ids = [r["question_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate question ids: leakage"
    order = sorted(range(len(records)), key=lambda i: ids[i])
    shuffled = list(order)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(records) * train_frac)
    n_val = int(len(records) * validation_frac)
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    for i in order:
        if i in train_ids:
            records[i]["split"] = "train"
        elif i in val_ids:
            records[i]["split"] = "validation"
        else:
            records[i]["split"] = "test"
    return records


def save_gold_titles_to_positions(records: List[Dict[str, Any]]) -> None:
    """Helper: records keep supporting-fact titles inside gold_titles already."""
    return None
