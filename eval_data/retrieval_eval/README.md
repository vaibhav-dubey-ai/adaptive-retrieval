# Labelled retrieval-evaluation set (chapter 6, ER model)

`ch6_er_model_questions.jsonl` — **75 questions** (67 answerable, 8 unanswerable)
used by `scripts/eval_retrieval.py` to compare BM25, dense and hybrid retrieval
on identical questions and an identical corpus.

## Corpus this set is bound to

| Property | Value |
|---|---|
| Document | `ch6.pdf` (`data/documents/udoc_151a82bc49d3`) |
| Source | *Database System Concepts*, 7th ed., Chapter 6 slide deck (80 slides) |
| SHA-256 of the uploaded file | `058d724620527963e888f095e6b45fe8c8bd6afadc2354e95022b925c25156a1` |
| Chunks | 80 (one per slide), chunking `target_chars=900`, `overlap_chars=150` |
| Chunk ids | `<doc_id>:<n>` with `n` a document-wide counter (`…:0000` … `…:0079`) |

The PDF itself is **not** committed (it is third-party course material and
`data/` is gitignored). To reproduce the evaluation you need the same file:
upload it through the dashboard (`POST /documents/upload`), which stores it under
`data/documents/` and re-creates the same chunk ids deterministically.

If a different document, a different chunk size, or a re-chunking with different
parameters is used, the labelled chunk ids no longer resolve. The harness detects
this: `validate_labels()` lists every unknown chunk id and
`scripts/eval_retrieval.py` prints a warning, so a broken corpus can never
silently depress all three methods' scores.

## How the labels were produced

1. All 80 chunk texts were read in full (not just titles).
2. Each question was written against the passage(s) that actually contain the
   answer, and the answering sentence was copied into `reference_answer`.
3. When two passages genuinely answer a question (definition on one slide,
   representation rule on another), **both** chunk ids are labelled and the
   question is tagged `multi_passage`.
4. Questions whose answer is absent from the deck got an empty label list and
   the tag `unanswerable`. They are excluded from Recall/MRR/nDCG (those metrics
   are undefined without relevant documents) and reported separately through
   `no_evidence_rate`.
5. No label was derived from what any retriever returned: labels were fixed
   before the first evaluation run.

### Category distribution

| Category | Count | What it stresses |
|---|---|---|
| `exact_terminology` | 19 | wording overlap with the slide text (BM25-friendly) |
| `paraphrase` | 18 | question wording differs from the source (dense-friendly) |
| `definition` | 22 | "what is X" questions, mostly single-passage |
| `multi_passage` | 8 | evidence split across two slides |
| `unanswerable` | 8 | answer absent from the corpus entirely |

## Labelling limitations (honest)

- Labels are **partial**: when a third passage also mentions the topic but does
  not answer the question, it is not labelled. Partial labels depress Recall for
  every method equally; they do not favour one retriever.
- The set is small (67 scored questions) and single-document, so differences of
  a few points are within noise. No significance test is claimed; the ranking
  metrics are descriptive.
- Slides are terse and contain OCR-style artefacts (page footers, figure labels
  that did not extract). Questions were written only against text that
  actually survived extraction.

## Usage

```powershell
.\.venv\Scripts\python.exe scripts\eval_retrieval.py --methods bm25,dense,hybrid --top-k 10 --repeat 3
```

Results land in `results/retrieval_eval/<timestamp>/` (`retrieval_metrics.json`,
`summary.csv`, `per_question.csv`). See the project README for the measured
numbers and their interpretation.
