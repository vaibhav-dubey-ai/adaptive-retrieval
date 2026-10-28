# Should the AI Search Before Answering? — Adaptive Retrieval for Question Answering

**Academic Project Report — NLP Coursework**
*Evaluation Date: October 2026*

---

## Abstract

This report presents a reproducible experimental evaluation of an **adaptive retrieval decision engine** for question answering. The central research question is: *Can a lightweight pre-retrieval decision engine predict when a question needs document retrieval, and achieve a useful trade-off between answer quality, evidence availability, response latency, and token usage, compared with always retrieving or never retrieving?*

The system implements three interchangeable retrievers (BM25, Dense, Hybrid RRF) and a heuristic decision engine that runs before any retrieval. A hand-labelled set of 75 questions (67 answerable, 8 unanswerable) over an 80-chunk DBMS PDF corpus drives all experiments.

> **Update, 28 Oct 2026 (read first).** The routing result in the key findings below (72/75 correct, balanced accuracy 0.92)
> measures whether a question's *topic appears in the slides*, not whether retrieval *improves the answer*. Almost every
> question in that set was written from the slides, so it cannot show the second thing. Section 9.0 replaces it with an
> evaluation that runs both answering paths on every question and compares routers on held-out splits. Its conclusions are
> more modest and are the ones to cite.

**Key verified findings (from saved result files):**
- Hybrid RRF is best on all ranking metrics: Recall@5=0.9478, MRR@10=0.8112, nDCG@10=0.8461, Hit@1=0.7313 (run 20261025_180311)
- Adaptive routing: 72/75 correct (96%), F1=0.977, balanced accuracy=0.922 (run 20261026_011553)
- Answer quality: n=4 only — PRELIMINARY. Retrieval-augmented F1=0.421 vs closed-book F1=0.106 on 4 questions
- Latency: warm-path A (closed-book) 14,300 ms median vs B (retrieval-augmented) 21,469 ms median; retrieval adds <1 ms (run 20261026_011132)

---

## 1. Introduction and Motivation

Modern question-answering systems face a fundamental trade-off: closed-book answering uses only the LLM's parametric memory (fast, but limited to trained knowledge), while retrieval-augmented answering grounds answers in external documents (more reliable for domain-specific facts, but slower and more token-intensive).

An *adaptive* policy retrieves only when evidence is likely to help, avoiding wasted retrieval on questions the LLM can already answer well. This project designs, implements, and evaluates such a policy over a real corpus, using fully local inference.

---

## 2. Problem Statement

Given a fixed document corpus (80 DBMS slide chunks) and a local LLM (Ollama qwen3:4b-instruct), the system must decide — before calling the retriever or LLM — whether to answer closed-book (A) or retrieve evidence first (B). This decision uses only the question text and lightweight corpus statistics; it never sees retrieved passages, reference answers, or oracle labels.

---

## 3. Research Question and Objectives

**Research question:** Can an adaptive retrieval policy decide when a question needs document retrieval, and achieve a useful trade-off between answer quality, evidence availability, response latency, and token usage, compared with always retrieving or never retrieving?

**Objectives:**
1. Build and evaluate BM25, Dense, and Hybrid retrievers on a labelled question set.
2. Implement a heuristic pre-retrieval decision engine.
3. Evaluate routing quality against fixed-policy baselines with proper metrics.
4. Measure latency and tokens for both answering paths.
5. Compare answer quality across policies (subject to LLM availability constraints).
6. Identify and report failure modes honestly.

---

## 4. Scope and Contributions

**In scope:** BM25, dense (all-MiniLM-L6-v2), hybrid RRF; heuristic decision engine; 75-question labelled eval set; local Ollama inference.

**Out of scope:** cross-encoder reranking; RAGAS; trained decision model on DBMS corpus; multiple corpora; statistical significance testing (sample too small for answer quality).

**Contributions:**
- Reproducible local evaluation harness with labelled data, result files, and CLI scripts.
- Honest separation of four measurement types: retrieval ranking, routing correctness, answer quality, efficiency.
- Documented heuristic engine with traceable weights and explainable per-question decisions.

---

## 5. Related Work and Background

**RAG (Lewis et al., 2020):** Conditions a generative model on retrieved documents at inference time, improving factuality on knowledge-intensive tasks at the cost of retrieval latency and increased context length.

**BM25 (Robertson & Zaragoza, 2009):** TF-IDF ranking function; no neural model; sub-millisecond queries; 12.5% no-evidence rate on this corpus (abstains when no query term matches).

**Dense retrieval (Karpukhin et al., 2020; DPR):** Encodes questions and passages into shared vector spaces. Uses all-MiniLM-L6-v2 (Reimers & Gurevych, 2019); captures semantic paraphrase but requires ~16 s model load and ~30 ms/query on CPU.

**Reciprocal Rank Fusion (Cormack et al., 2009):** Combines ranked lists without score normalisation: score = sum(1/(k+rank_i)). Robust to scale differences; frequently outperforms components.

**Pre-retrieval decision (Arabzadeh et al., 2021):** Lightweight features predict whether a query requires external evidence, avoiding retrieval cost for questions the model can answer directly.

---

## 6. System Architecture

### Architecture Overview

```
Question
   |
   v
[Adaptive Decision Engine]
   heuristic: question + collection preflight
   outputs: RETRIEVE | DO_NOT_RETRIEVE
   |
   +-------(RETRIEVE)--------+
   |                         |
   v                         v
[Retriever]           [Closed-book LLM]
 BM25/Dense/Hybrid     qwen3:4b-instruct
   |
   v
[Grounded LLM]
 qwen3:4b-instruct
 + retrieved passages
```

### Chunking and Indexing

Documents are split into ~500-character overlapping chunks. Each chunk gets a stable ID `<doc_id>:<n>`. Active corpus: `udoc_151a82bc49d3` (ch6.pdf, 80 chunks). The BM25 index builds in ~49 ms; dense index builds in ~23 s (one-time, cached afterward).

### Heuristic Decision Engine

Computes a retrieval-need score in [0, 1]:

```
score = sum(w_i * v_i) / sum(w_i)   over available components
```

| Signal | Weight | Description |
|--------|--------|-------------|
| collection_relevance | 0.70 | Fraction of question content-terms in active corpus |
| memory_uncertainty | 0.20 | 1 - best_title_coverage (EXCLUDED: familiarity probe not configured) |
| question_cue | 0.15 | Surface cues: temporal, superlative, quoted, entity density, open-WH |

**Critical limitations:**
- Score is a retrieval-need score, NOT a calibrated probability.
- memory_uncertainty is excluded in the evaluation flow; engine re-normalises over 2 components.
- Threshold 0.5 is a default, not tuned on a held-out split.

**RETRIEVE if score >= 0.5, else DO_NOT_RETRIEVE.**

### Leakage Contract

The decision engine receives ONLY the question string and collection signals. It NEVER receives reference answers, gold chunk IDs, or oracle labels. Enforced by tests: `test_decision.py::test_decision_signature_never_accepts_gold_or_answers` and `test_decision.py::test_decision_source_never_touches_reference_data`.

---

## 7. Experimental Dataset

**Corpus:** Chapter 6 of *Database System Concepts* (7th ed.), 80 slides/chunks; doc_id=`udoc_151a82bc49d3`.

> **Note on Active Corpus vs. Benchmark Corpus:**
> **Corpus by experiment.** Main 125-question answer-quality experiment (9.0): chapters 6 + 7, 173 chunks (80 + 93), 2 documents. Retrieval-ranking benchmark (9.1) and the historical corpus-coverage experiment (9.2/9.3): chapter 6 only, 80 chunks. The section below describes the 75-question Chapter 6 set.
>
> The saved retrieval and policy evaluation runs (run IDs: `20261025_180311`, `20261026_011553`) were conducted exclusively over the Chapter 6 benchmark corpus (80 chunks, doc_id: `udoc_151a82bc49d3`). The active live collection in the system currently contains two uploaded chapters: Chapter 6 and Chapter 7 (total 173 chunks, doc_ids: `udoc_151a82bc49d3` and `udoc_8897c006f6f9`). The older benchmark results are reported faithfully as measurements over Chapter 6 and are not presented as measurements over the combined two-chapter corpus.

**Question set:** `eval_data/retrieval_eval/ch6_er_model_questions.jsonl`
- 75 questions total (67 answerable, 8 unanswerable)
- Categories: exact_terminology, definition, paraphrase, multi_passage, unanswerable
- Labels produced manually by reading every chunk and writing questions against actual passage content.

**Models:** qwen3:4b-instruct (Ollama), all-MiniLM-L6-v2 (sentence-transformers), BM25 (k1=1.5, b=0.75), RRF (k=60), top-k=5 (evaluation), top-k=8 (bench).

---

## 8. Evaluation Metrics

### Retrieval Ranking (over 67 answerable questions)

| Metric | Formula | Meaning |
|--------|---------|---------|
| Recall@5 | |{relevant} n top-5| / |relevant|, averaged | What fraction of labelled-relevant passages appear in top 5? |
| MRR@10 | mean(1/rank of first relevant, cap 10) | How early does first relevant passage appear? |
| nDCG@10 | rank-discounted cumulative gain | Credits multiple relevant passages; penalises late ranking |
| Hit@1 | fraction with relevant at rank 1 | Does best result answer the question? |

### Routing Quality (over all 75 questions)

Positive class: answerable. Predicted-positive action: RETRIEVE.

| Metric | Formula |
|--------|---------|
| Accuracy | (TP+TN)/n — misleading on 67/8 imbalanced set |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| F1 | 2PR/(P+R) |
| Specificity | TN/(TN+FP) |
| Balanced accuracy | (recall+specificity)/2 — fair on imbalanced classes |

**Why accuracy misleads:** Always-retrieve scores 89.3% by answering the majority class. Balanced accuracy (0.5 for both fixed-policy baselines) is the appropriate primary metric.

### Answer Quality

Token-F1 and Exact Match (SQuAD normalization): lexical; blind to paraphrase; EM near-zero for free-form answers. Evidence coverage: lexical grounding proxy ONLY — not entailment, not faithfulness, not correctness.

### Latency / Tokens

Cold-start (model load) reported separately from warm medians. Comparison-mode wall-clock (both answers) is NOT deployed latency — adaptive-only mode is.

---

## 9. Experimental Results

> All numbers verified against saved result files. Run IDs cited.

### 9.0 Routing evaluation: does the router retrieve when retrieval helps? (run `results/routing_eval/main`)

**Main experiment.** Source: `results/routing_eval/main/routing_results.json` (generated 2026-10-27T22:57:50Z) and `per_question.jsonl`. Everything in this subsection comes from that one run; sections 9.1-9.4 are separate, earlier experiments that measure different things and are not comparable with it.

**Design.** 133 questions: the 75 chapter-6 questions, 25 new chapter-7 questions, and 33 general questions with no labelled
chunk (27 unrelated to the slides, 6 textbook definitions the slides also cover). Corpus: chapters 6 and 7 (173 chunks), BM25,
top-5. Both branches answer every question (temperature 0, one-sentence prompts, `max_tokens` 100). The run's result file records the model name
`qwen3-4b-capped`: an Ollama model name set through `LLM_MODEL` at run time (the code takes it from the environment, not from any file in the repo),
described by the author as `qwen3:4b-instruct` run with Ollama capped at 3 threads and a 2048-token window (same weights, no prompt exceeded the
window). The live app uses `qwen3:4b-instruct` with the same temperature and prompts. The `qwen3-4b-capped` alias is no longer installed, so the
same-weights claim and the cap values are as reported by the author and could not be re-checked; the name and the generation settings in the
results file and `configs/default.json` were checked. Answers are scored against short reference
answers with `quality = mean(token-F1, token-recall)`. A question is labelled *retrieval helps* when the grounded answer
scores at least 0.10 higher. 125 of 133 questions are scorable (8 chapter-6 "unanswerable" questions have no reference):
retrieval **helps on 86**, **hurts on 17**, ties on 22 (a tie means the quality difference is under 0.10). On 11 questions BM25 found no passage at all; there the retrieve path
falls back to the closed-book answer at closed-book cost. **All 11 fallback questions are ties** (all `general_knowledge`, quality difference exactly 0 by construction),
so **11 of the 22 ties come from the fallback** and 11 are genuine ties with passages retrieved (checked against `per_question.jsonl`: 0 fallbacks among the 86 helps and the 17 hurts).

**Not the same experiment as 9.1.** This 125-question answer-quality experiment (133 questions, 125 scored) is separate from the 75-question Chapter 6
retrieval-ranking benchmark in 9.1 (80 chunks, ranking metrics only, no LLM answers). Corpus size: the 80 + 93 = 173 chunks are reported here from the
run's setup; the results file does not store the chunk count, Chapter 7's source is not in the repo, and the retrieved chunk ids in `per_question.jsonl` are
consistent with it (Chapter 6 ids up to 79, Chapter 7 ids up to 92). Question counts were re-verified with the project loader: 75 + 25 + 33 = 133.

Routers are compared on 30 repeated stratified 70/30 splits (`adaptive_qa/eval/routing.py:repeated_splits`: the scored questions are divided into retrieval-helps and not-helps groups, each group is shuffled with a fixed seed and 30% goes to the held-out part; the 30 draws overlap). The tuned threshold and the learned logistic regression are fitted on the 70% training part only, and every policy is scored on the 30% held-out part. Router inputs are the question text and the collection-preflight signals (`feature_vector`); reference answers, answer scores and the helps/hurts labels are used only to fit and score, never as router inputs. The oracle uses the labels by construction and is an evaluation-only upper bound. Answer quality is `mean(token-F1, token-recall)` against a short reference: a lexical-overlap score, not a measure of factual correctness, entailment or hallucination rate. Routing balanced accuracy is a separate measure of whether retrieval was chosen exactly when it helped.
Mean ± sd over splits:

| Router | Answer quality | Retrieves on | Tokens / question | Latency / question | Balanced accuracy |
|---|---|---|---|---|---|
| Never retrieve | 0.334 ± 0.022 | 0% | 96 | 3.1 s | 0.500 |
| Always retrieve | 0.516 ± 0.023 | 100% | 818 | 31.5 s | 0.500 |
| Heuristic, threshold 0.5 (shipped, untuned) | **0.570 ± 0.023** | 78% | 732 | 28.6 s | 0.833 ± 0.057 |
| Heuristic, threshold tuned on train split | 0.560 ± 0.023 | 73% | 693 | 27.1 s | 0.822 ± 0.056 |
| Learned (logistic regression) | 0.559 ± 0.030 | 69% | 661 | 26.1 s | 0.836 ± 0.060 |
| Oracle (upper bound, not deployable) | 0.589 ± 0.022 | 71% | 672 | 26.2 s | 0.964 ± 0.026 |

| Question type | n | Closed-book | Grounded | Helps / hurts |
|---|---|---|---|---|
| Definition | 32 | 0.26 | 0.57 | 28 / 0 |
| Exact terminology | 26 | 0.19 | 0.61 | 23 / 1 |
| Paraphrase | 24 | 0.18 | 0.47 | 22 / 1 |
| Needs two passages | 10 | 0.23 | 0.54 | 10 / 0 |
| Textbook definition, also in slides | 6 | 0.46 | 0.50 | 3 / 1 |
| General knowledge, not in slides | 27 | 0.68 | 0.38 | 0 / 14 |

**What the evidence supports**
1. Retrieval is essential for questions about the documents (quality roughly 0.2 → 0.55), as expected: a 4B model cannot know these slides.
2. Routing beats both fixed policies on quality (0.570 vs 0.516 and 0.334) and lands within about 0.02 of the oracle. The gain over
   always-retrieve exists because retrieval *hurts* on general questions: when the retrieved passages are irrelevant the model often
   answers "the provided passages do not address this" instead of using its own knowledge, even though the prompt allows that fallback.
3. The cost saving is modest: 732 tokens against 818 (about 10% fewer). Even the oracle only reaches 672 (about 18% fewer), because
   about three quarters of this workload is questions about the documents.
4. **The tuned and learned routers did not beat the shipped heuristic.** Their quality (0.559–0.560) is within noise of 0.570. The learned
   router saves about 10% more tokens at about 0.01 lower quality, which is a trade-off rather than an improvement.
5. The heuristic works here largely because lexical overlap with the slides separates "about the slides" from "general". Cases it
   cannot separate, such as textbook definitions the model also knows, remain (3 corpus questions skipped, 9 general ones retrieved).

**Limitations specific to this experiment.** n = 125 and a single model, corpus and retriever (BM25; hybrid was not run for answers).
The splits are repeated draws from the same 125 questions, so the sd describes split-to-split variation, not an independent
confidence interval. The 58 new questions and their reference answers were written for this evaluation and have not been independently
reviewed. The corpus-versus-general ratio was chosen by the author, so the savings describe such a mix, not real traffic. The quality
metric is lexical. Latency is for a 4-core CPU and includes no GPU. Three closed-book timings distorted by process pauses were
re-measured; answers were unaffected. The analysis reproduces from the cached answers (`--no-llm`), up to measured BM25 time.

### 9.1 Retrieval Ranking Quality (Run: 20261025_180311)

Source: `results/retrieval_eval/20261025_180311/retrieval_metrics.json`. **Scope: DBMS Chapter 6 only, 80 chunks (one document).** It does not describe the 173-chunk Chapter 6 + 7 collection that the routing experiment (9.0) and the live app can use.

Dataset: eval_data/retrieval_eval/ch6_er_model_questions.jsonl (sha256: 734a82f4...)
Settings: Recall@k=5, Rank@k=10; n_questions=75, n_scored=67 (answerable only)

| Method | Recall@5 | MRR@10 | nDCG@10 | Hit@1 | No-evidence | Median ms | p95 ms | 1st query after build ms | Build ms |
|--------|----------|--------|---------|-------|-------------|-----------|--------|-------------|----------|
| BM25 | 0.9179 | 0.7326 | 0.7871 | 0.5821 | 0.125 | 0.117 | 0.235 | 0.143 | 49.3 |
| Dense | 0.9254 | 0.7184 | 0.7704 | 0.5970 | 0.000 | 28.293 | 81.461 | 57.2 | 22,999 |
| **Hybrid** | **0.9478** | **0.8112** | **0.8461** | **0.7313** | 0.000 | **18.787** | 25.028 | 1,593 | 118 |

Bold = best per metric. Hybrid is best on all four ranking metrics and also outperforms dense on query latency (19 ms vs 28 ms warm median). Dense build time (23.0 s) includes loading the embedding model (16.5 s); hybrid's first query (1,593 ms) is dominated by its own model load (1,461 ms). The first-query and build columns are one-off costs, not per-query latency.

**Cold-start timing, checked against the code and `retrieval_metrics.json`.** The 57.2 ms dense figure is correct (57.239 ms, in milliseconds, identical in `summary.csv`, `retrieval_metrics.json` and `per_question.csv` q01) but it is *not* the full cold start: `DenseRetriever` loads the embedding model inside its constructor (`dense.py:_build`, 16,548 ms, part of the 22,999 ms build), and the first query is timed only afterwards. It was previously labelled "Cold 1st", which read as if it included the load. Full cold start = build + first query: BM25 49.4 ms, **dense 23,055.9 ms (23.06 s)**, hybrid 1,711.0 ms (hybrid loads lazily, so its model load falls inside the timed first query; its 118 ms build excludes it). The column is renamed "1st query after build" here and in the dashboard.

A newer run, `20261028_021811`, exists but is **partial and not cited**: it contains BM25 only, and 178 of its retrieved chunk ids belong to a second document although it records a single-document, 80-chunk corpus, so its recorded scope is unreliable. It is kept as a historical result file; the dashboard flags it and defaults to the newest complete run.

**These metrics measure retrieval ranking quality only — they do not indicate whether the generated answer is correct.**

### 9.2 Corpus-coverage check (Run: 20261026_011553) — earlier experiment

Source: `results/policy_eval/20261026_011553/policy_results.json`. Historical and preliminary; superseded by 9.0 for conclusions about routing.

> This measures whether the topic appears in the slides, not whether retrieval improves the answer; see 9.0 for that.


Settings: n_questions=75, retriever=bm25, top_k=5, decision_threshold=0.5

**Confusion matrix:**

|  | RETRIEVE (predicted) | DO_NOT_RETRIEVE (predicted) |
|--|---------------------|-----------------------------|
| Answerable (67) | TP = 65 | FN = 2 (q52, q66) |
| Unanswerable (8) | FP = 1 (q68) | TN = 7 |

**Routing metrics:**

| Metric | Adaptive | Always-retrieve | Never-retrieve |
|--------|---------|----------------|----------------|
| Accuracy | 0.960 | 0.893 | 0.107 |
| Precision | 0.985 | 0.893 | n/a |
| Recall | 0.970 | 1.000 | 0.000 |
| F1 | 0.977 | 0.943 | n/a |
| Specificity | 0.875 | 0.000 | 1.000 |
| **Balanced accuracy** | **0.922** | **0.500** | **0.500** |

Balanced accuracy for both fixed-policy baselines is 0.500 — they provide no discriminating information. The adaptive policy's 0.922 demonstrates genuine discrimination between answerable and unanswerable questions.

**Routing errors:**
- FN q52: "What is the difference between a total and partial participation constraint?" — engine underestimated collection relevance; corpus does contain relevant information.
- FN q66: "What does it mean for a relationship to be one-to-many?" — similar cause.
- FP q68: "Who coined the term entity-relationship model?" — historical attribution not in corpus; retrieval was wasted.

**Evidence recall@5:** adaptive=0.903, always-retrieve=0.918. Adaptive saves 9/75 retrieval calls (12%) at the cost of 2 missed answers.

### 9.3 Answer Quality (Run: 20261026_011553) — earlier experiment, 4 answers only

Source: same file as 9.2. These four-answer figures are not comparable with the 125-question results in 9.0.

**PRELIMINARY — n=4 questions only. No statistical claim is made.**

LLM: qwen3:4b-instruct via Ollama. Questions q01, q03, q18, q60.

| Policy | Mean F1 (n=4) | Mean tokens (n=4) | Warning |
|--------|-------------|------------------|---------|
| never_retrieve | 0.106 | 165.25 | PRELIMINARY |
| always_retrieve | 0.421 | 740.75 | PRELIMINARY |
| adaptive | 0.421 | 740.75 | PRELIMINARY |
| oracle | 0.421 | 740.75 | PRELIMINARY |

For these 4 questions, adaptive chose B for all 4 (same path as always-retrieve). Closed-book scored much lower F1. The identical adaptive/always numbers are expected when the engine correctly decides to retrieve for every sample question.

The downstream_eval/20261025_180359 run (6 questions via OpenRouter) failed with HTTP 429 rate limit errors. Only q01/bm25, q03/hybrid, q04/bm25, q05/closed_book, q05/hybrid, q06/closed_book succeeded. No fabricated results; all errors are recorded in the per_question.csv.

**Resumable Evaluation Harness:**
To address provider rate limits and enable robust local batch evaluation across the full 75 questions, the evaluation harness was upgraded with an append-only, disk-backed cache (`adaptive_qa/answering/cache.py`, persisting to `data/eval_answer_cache.jsonl`). The harness:
- Checks the cache by question, condition (closed-book vs grounded), model name, and retrieved chunk IDs before querying the LLM.
- Saves successful generations immediately to disk to prevent data loss on timeout or crash.
- Supports incremental resumption via `scripts/eval_policy.py --answers N` or `scripts/eval_downstream.py`.
- Correctly samples across question categories using `_spread_by_category` and records successes, timeouts, and sample sizes transparently without inventing results.

### 9.4 Latency and Token Usage (Run: 20261026_011132)

Benchmark question: "What is an entity set?"
Trials: 2 interleaved (Trial 1: A,B; Trial 2: B,A)

| Measure | Option A (closed-book) | Option B (retrieval-augmented) |
|---------|----------------------|-------------------------------|
| Cold first call | 14,360 ms | 24,300 ms |
| Warm median generation | 14,300 ms (n=1) | 21,469 ms (n=1) |
| Mean total tokens | 169.5 | 1,387.5 |
| Median retrieval (BM25) | — | 0.477 ms |
| Median decision engine | 1.549 ms | — |

**Interpretation:** Retrieval adds ~7,170 ms to the warm path. This overhead comes from larger input context (8× more tokens) increasing generation time, not from retrieval itself (< 0.5 ms BM25). With n=1 warm sample, these numbers describe one question only.

**How the dashboard reports timing.** Live single-question timings are one measurement on one machine, shown in seconds from 1 s upward and in milliseconds below. The `cold start` flag means the first model call since the dashboard server started (it may include a one-off model load by the provider). An earlier version created a fresh model client for every request, so the flag was always set; the client is now reused. Retrieval time is the search itself; embedding-model load and corpus encoding for dense and hybrid are separate one-off costs. Aggregated, repeated numbers come only from saved runs such as this one (n = 1 warm sample here, so no median or p95 is claimed).

**Retrieval does not reduce latency or tokens — it adds both.** The trade-off: more context may improve answer quality on corpus-specific questions at the cost of more tokens and longer generation.

---

## 10. Failure Analysis

**OpenRouter rate limits:** The 2025-10-25 downstream eval used OpenRouter free tier (50 req/day limit). Most conditions returned HTTP 429 after 2 successful completions. Resolution: project reconfigured for local Ollama.

**Routing errors (q52, q66):** Heuristic underestimated collection relevance. Lowering threshold would fix FNs but increase FPs. Heuristic nature of the engine limits its ability to distinguish borderline cases.

**Routing error (q68):** Historical attribution question; corpus has ER model content but not the specific attribution fact. FP rate 1/8 = 12.5% on the unanswerable subset.

**Limited latency samples:** 2 trials gives n=1 warm sample per branch. 3-5 trials recommended for robust medians.

**Dense cold start:** Model load (~16 s) affects first-query latency in any fresh session. Reported separately from warm query latency.

---

## 11. Limitations and Threats to Validity

1. **Small answer-quality sample (n=4):** Flagged `small_sample_warning: true` in all result files. No population estimate.
2. **Single corpus/domain:** 80-chunk DBMS chapter. Findings may not generalize.
3. **Threshold not tuned:** Default 0.5; overfitting risk if tuned on same 75 questions.
4. **Heuristic (not trained) decision engine:** Score is not a calibrated probability; collection_relevance dominates (70%).
5. **Memory-uncertainty excluded:** No familiarity prober configured; 2-component re-normalisation is used.
6. **Lexical quality metrics:** Token-F1/EM are blind to paraphrase and do not measure faithfulness or hallucination.
7. **Single LLM, no prompt tuning:** Different model, temperature, or prompt would yield different scores.
8. **Hardware-specific latency:** All timing on Windows 11 CPU; GPU inference would differ materially.
9. **No held-out test split:** 75 questions used for both retrieval and routing evaluation.

---

## 12. Conclusion

**Can an adaptive retrieval policy achieve a useful trade-off?**

**Yes, with caveats:**

- **Routing helps, modestly (section 9.0):** on 125 scored questions the shipped heuristic reaches answer quality 0.570 against 0.516 for always-retrieve and 0.334 for never-retrieve, with about 10% fewer tokens than always-retrieve. A learned router did not improve on it. The earlier 96% figure measured topic overlap with the slides and should not be cited as routing quality.
- **Retrieval quality is strong (evidence solid):** Hybrid RRF Recall@5=0.9478, Hit@1=0.731 over 67 answerable questions.
- **Answer quality is inconclusive (evidence preliminary, n=4):** Direction is consistent with expectations (retrieval-augmented F1 0.421 vs closed-book 0.106) but n=4 has no statistical power.
- **Latency trade-off is confirmed:** Retrieval-augmented path takes longer (~7 s more warm) and uses ~8× more tokens; retrieval itself adds negligible time.

**The adaptive policy correctly routes 97% of answerable and 87.5% of unanswerable questions, matching the retrieval quality of always-retrieve while avoiding retrieval for 12% of questions. Answer quality must be re-evaluated with a full 75-question LLM run before aggregate claims can be made.**

### Future Work

1. Run `eval_policy.py --answers 75` for complete answer-quality evidence.
2. Tune decision threshold on held-out split.
3. Add cross-encoder reranking.
4. Train LearnedDecisionEngine on DBMS corpus.
5. Evaluate on additional corpora.
6. Run 5+ latency trials.
7. Add LLM-as-judge faithfulness evaluation.

---

## 13. Reproducibility

### Result File to Table Mapping

| Section | Result file | Run ID |
|---------|------------|--------|
| 9.0 Main routing experiment | results/routing_eval/main/routing_results.json (+ per_question.jsonl) | main |
| 9.1 Retrieval ranking | results/retrieval_eval/20261025_180311/retrieval_metrics.json | 20261025_180311 |
| 9.1 Retrieval ranking (summary) | results/retrieval_eval/20261025_180311/summary.csv | 20261025_180311 |
| 9.2 Corpus-coverage (earlier) | results/policy_eval/20261026_011553/policy_results.json | 20261026_011553 |
| 9.2 Per-question routing | results/policy_eval/20261026_011553/per_question.csv | 20261026_011553 |
| 9.3 Answer quality | results/policy_eval/20261026_011553/policy_results.json | 20261026_011553 |
| 9.4 Latency/tokens | results/bench_latency/20261026_011132/bench.json | 20261026_011132 |
| Downstream eval (incomplete) | results/downstream_eval/20261025_180359/downstream_results.json | 20261025_180359 |

### Commands to Reproduce

```powershell
# Prerequisites: Python 3.10+, Ollama running with qwen3:4b-instruct

# Main routing experiment (needs Ollama; resumes from the answer cache; add --no-llm to re-analyse cached answers)
.\.venv\Scripts\python.exe scripts\eval_routing.py

# Retrieval evaluation (no LLM, ~5 min)
.\.venv\Scripts\python.exe scripts\eval_retrieval.py --methods bm25,dense,hybrid --top-k 10 --repeat 3

# Policy evaluation — routing only (no LLM, < 1 min)
.\.venv\Scripts\python.exe scripts\eval_policy.py --method hybrid --top-k 5

# Policy evaluation — with answer quality (Ollama required, ~60 min for 75)
.\.venv\Scripts\python.exe scripts\eval_policy.py --method hybrid --top-k 5 --answers 75

# Latency benchmark (Ollama required, ~10 min for 3 trials)
.\.venv\Scripts\python.exe scripts\bench_latency.py --trials 3 --question "What is an entity set?"

# Downstream eval (Ollama required, ~30 min for 10 questions)
.\.venv\Scripts\python.exe scripts\eval_downstream.py --limit 10 --top-k 5

# All tests
.\.venv\Scripts\python.exe -m pytest tests\ -q

# Dashboard
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
# -> http://127.0.0.1:8000 -> Research Results tab (it reads saved files; "Reload saved results" does not run benchmarks)
```

---

**References:**
- Lewis, P. et al. (2020). RAG for Knowledge-Intensive NLP Tasks. NeurIPS.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT. EMNLP.
- Cormack, G.V. et al. (2009). Reciprocal Rank Fusion. SIGIR.
- Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25. FnTIR.

---

*Honesty statement: All numerical results are loaded from saved result files. Small-sample results are labelled PRELIMINARY. No result has been fabricated. All result files exist in the repository and can be independently verified.*
