#!/usr/bin/env python
"""Build the local research data: eval sample, corpus, parametric store.

Downloads a deterministic HotpotQA (distractor) sample via the HF
datasets-server REST API, assigns question-level splits, and builds:

1. ``data/hotpotqa_eval_sample.jsonl``     - the sampled questions
2. ``data/corpus.jsonl``                   - shared retrieval corpus
3. ``data/corpus_manifest.json``           - EVALUATION ONLY gold-doc map
4. ``data/hotpotqa_pretrain_sample.jsonl`` - train rows for simulated memory
5. ``data/parametric_store.jsonl``         - Method A's "LLM memory"

Run:  python scripts/build_dataset.py [--config configs/default.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_qa.config import load_config, resolve_path  # noqa: E402
from adaptive_qa.corpus import build_corpus, save_corpus, save_eval_manifest  # noqa: E402
from adaptive_qa.dataset import (  # noqa: E402
    assign_splits,
    choose_offset,
    fetch_hotpotqa_rows,
    normalize_hotpot_row,
    save_records_jsonl,
)
from adaptive_qa.answering.parametric_store import (  # noqa: E402
    build_parametric_store,
    save_parametric_store,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/default.json", help="config JSON (merged over defaults)"
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    eval_path = resolve_path(ds["eval_jsonl"])
    pretrain_path = resolve_path(ds["pretrain_jsonl"])

    if eval_path.exists() and pretrain_path.exists() and not args.force:
        print(f"[skip] {eval_path} and {pretrain_path} exist; pass --force to rebuild")
    else:
        # Disjoint windows inside the big 'train' split:
        #  - eval rows: easy-level filter (the distractor VALIDATION split is
        #    uniformly level='hard', verified empirically; the easy/medium
        #    distribution lives in the train split), anchored to offset base
        #    40k so the eval window and the memory window stay disjoint.
        #  - memory rows: an unfiltered disjunct window (offset base 15k).
        # Both are deterministic given the seeds. Disjointness of the
        # resulting question ids is asserted below (leakage guard).
        print(f"[fetch] {ds['eval_sample_n']} easy questions from split="
              f"{ds['eval_split']!r} (base {ds['eval_offset_base']})")
        eval_offset = ds["eval_offset_base"] + choose_offset(
            total_available_hint=ds["eval_sample_offset_span"],
            n=ds["eval_sample_n"],
            seed=ds["eval_sample_offset_seed"],
        )
        level_filter = ds.get("eval_level_filter")
        if level_filter:
            def _level(row, _lvl=level_filter):
                return row.get("level") == _lvl
            raw_eval = fetch_hotpotqa_rows(
                ds["hf_rows_api"], ds["hf_dataset"], ds["hf_config"], ds["eval_split"],
                eval_offset, ds["eval_sample_n"], filter_fn=_level,
                max_raw_rows=30_000,  # filter density ~20%: needs ~1000 raw rows
            )
        else:
            raw_eval = fetch_hotpotqa_rows(
                ds["hf_rows_api"], ds["hf_dataset"], ds["hf_config"], ds["eval_split"],
                eval_offset, ds["eval_sample_n"],
            )
        eval_records = [normalize_hotpot_row(r) for r in raw_eval]
        eval_records = assign_splits(
            eval_records,
            cfg["splits"]["train_frac"],
            cfg["splits"]["validation_frac"],
            cfg["splits"]["seed"],
        )
        save_records_jsonl(eval_records, ds["eval_jsonl"])
        print(f"[ok] {len(eval_records)} eval questions -> {eval_path} "
              f"(offset {eval_offset})")

        pretrain_offset = ds["pretrain_offset_base"] + choose_offset(
            total_available_hint=ds["pretrain_offset_span"],
            n=ds["pretrain_rows_n"],
            seed=ds.get("pretrain_offset_seed", 202),
        )
        upper = ds["pretrain_offset_base"] + ds["pretrain_offset_span"]
        assert ds["pretrain_offset_base"] <= pretrain_offset < upper, (
            "seeded jitter must keep the memory window inside its base span"
        )
        assert pretrain_offset + ds["pretrain_rows_n"] < ds["eval_offset_base"], (
            "memory window overlaps eval window: disallowed leakage of "
            "evaluation questions into simulated pretraining"
        )
        print(f"[fetch] {ds['pretrain_rows_n']} memory rows from split="
              f"{ds['pretrain_split']!r} (base {ds['pretrain_offset_base']})")
        raw_pretrain = fetch_hotpotqa_rows(
            ds["hf_rows_api"], ds["hf_dataset"], ds["hf_config"], ds["pretrain_split"],
            pretrain_offset, ds["pretrain_rows_n"],
        )
        # leakage guard: memory must never contain an eval question id or gold title
        eval_ids = {r["question_id"] for r in eval_records}
        pretrain_records = [normalize_hotpot_row(r) for r in raw_pretrain]
        pretrain_records = [
            r for r in pretrain_records if r["question_id"] not in eval_ids
        ]
        assert not eval_ids & {r["question_id"] for r in pretrain_records}, (
            "eval question leaked into the parametric-memory source rows"
        )
        save_records_jsonl(pretrain_records, ds["pretrain_jsonl"])
        print(f"[ok] {len(pretrain_records)} pretrain rows -> {pretrain_path}")

    # ------------------------------------------------------------------ corpus
    def read_jsonl(path_val: Path) -> list:
        records = []
        with path_val.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    records = read_jsonl(eval_path)
    eval_records = records
    docs, eval_meta = build_corpus(records)
    corpus_path = cfg["corpus"]["jsonl_path"]
    manifest_path = cfg["corpus"]["manifest_path"]
    save_corpus(docs, corpus_path)
    save_eval_manifest(eval_meta, manifest_path)
    print(f"[ok] corpus built: {len(docs)} docs -> {corpus_path}")
    print(f"[ok] evaluation-only gold manifest -> {manifest_path}")

    # --------------------------------------------------------- parametric store
    pretrain_records = read_jsonl(pretrain_path)
    # LEAKAGE RULE (requirement 10): the simulated closed-book model must NOT
    # receive held-out evaluation answers or gold evidence. The previous
    # behavior (include_eval_gold_paragraphs=true) put the eval questions'
    # gold paragraphs into memory - a data leak; it is now OFF by default and
    # the store is built solely from the disjoint pretrain window.
    include_eval_gold = bool(cfg["parametric_store"].get("include_eval_gold_paragraphs", False))
    store_source = list(eval_records) + list(pretrain_records) if include_eval_gold else list(pretrain_records)
    snippets = build_parametric_store(
        store_source,
        max_snippet_tokens=cfg["parametric_store"]["max_snippet_tokens"],
        gold_only=True,
    )
    store_path = cfg["parametric_store"]["jsonl_path"]
    save_parametric_store(snippets, store_path)
    print(f"[ok] parametric store: {len(snippets)} memory snippets -> {store_path}")


if __name__ == "__main__":
    main()
