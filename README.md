# Should the AI Search Before Answering? — Adaptive Retrieval Decision Engine

A reproducible NLP research project (CSE coursework) that answers one question
experimentally: **can a lightweight pre-retrieval decision engine predict when
external retrieval will help a QA system, and does adaptive retrieval beat
always/never retrieving on the quality–latency–cost trade-off?**

The system is a *deterministic, locally reproducible* QA research pipeline, not
a chat demo: three interchangeable retrievers (BM25, dense sentence embeddings,
and hybrid Reciprocal Rank Fusion), two answering methods (a closed-book
span/extraction engine over a simulated parametric memory, and a
grounded/extractive reader), a pre-retrieval decision engine (heuristic policy +
learned scikit-learn classifier), and a cross-verification evaluation harness
with an offline oracle.

Retrieval quality is measured separately from answer quality: a hand-labelled
75-question set over a real 80-chunk document corpus drives Recall@5, MRR@10 and
nDCG@10 for all three retrievers, with machine-readable results under
`results/retrieval_eval/`. Measured outcome on that corpus: **hybrid RRF is best
on all three ranking metrics** (see [Measured retrieval results](#measured-retrieval-results)),
at the cost of a dense query per search.

## Quick start

```powershell
# 0) environment (Python 3.10+; verified here on 3.13, Windows 11)
.\.venv\Scripts\python.exe -m pip install requests scikit-learn fastapi uvicorn `
    matplotlib pytest httpx pypdf python-docx python-multipart `
    sentence-transformers
# sentence-transformers pulls PyTorch (~2.5 GB on CPU wheels). It is only needed
# for the dense and hybrid retrievers; BM25 works without it.

# 1) (optional) HotpotQA benchmark dataset + corpus + simulated memory
.\.venv\Scripts\python.exe scripts\build_dataset.py --force

# 2) (optional) train + tune the learned decision engine
.\.venv\Scripts\python.exe scripts\train_decision.py --config configs\default.json

# 3) (optional) four-policy experiment on the held-out TEST split
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\default.json --limit 60

# 4) retrieval evaluation (this is the new experiment)
.\.venv\Scripts\python.exe scripts\eval_retrieval.py --methods bm25,dense,hybrid --top-k 10 --repeat 3

# 5) downstream QA evaluation (needs a configured LLM, costs provider calls)
.\.venv\Scripts\python.exe scripts\eval_downstream.py --limit 6 --top-k 5

# 6) tests
.\.venv\Scripts\python.exe -m pytest tests -q

# 7) dashboard
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
#    http://127.0.0.1:8000
```

Every step works from the CLI; the web UI is optional. The retrieval evaluation
needs no API key and no paid service. (Bash users: identical commands with
forward slashes and no backticks.)

## Linux / macOS setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Ubuntu: sudo apt install python3-venv first
pip install requests scikit-learn fastapi uvicorn matplotlib pytest httpx pypdf python-docx python-multipart numpy pandas
# optional, ~2.5 GB, only for dense/hybrid retrieval:  pip install sentence-transformers

# local model (any OpenAI-compatible endpoint works; Ollama shown)
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen3:4b-instruct
cp .env.example .env     # then set LLM_BASE_URL=http://localhost:11434/v1, LLM_MODEL=qwen3:4b-instruct, LLM_API_KEY=ollama

# ingest the course PDFs (same hashing/chunking as the dashboard upload)
python - <<'PY'
from pathlib import Path
from adaptive_qa.documents import DocumentStore
for n in ("ch6.pdf", "ch7.pdf"):
    DocumentStore().save_document(n, Path(n).read_bytes())
PY

python -m pytest tests -q
uvicorn api.main:app --port 8000
```

## Routing evaluation: does the router retrieve exactly when retrieval helps?

This is the primary experiment. The older "corpus coverage" check only asks whether a
question's topic appears in the documents; almost every question in that set was written
from the slides, so even "always retrieve" scores about 89%.

`scripts/eval_routing.py` instead runs **both** answering paths for every question
(closed-book and grounded), scores each answer against a short reference, and labels the
question *retrieval helps* when the grounded answer scores at least 0.10 higher. Routers are
then compared on **held-out random splits** (mean ± sd over 30 splits) by the quality they
achieve and the tokens, latency and retrieval calls they spend:

| Router | What it is |
|---|---|
| never / always | fixed baselines |
| heuristic@0.5 | the shipped weighted-signal engine, untuned threshold |
| heuristic-tuned | same score, threshold chosen on the training split only |
| learned | logistic regression on preflight and question cues, fit on the training split only |
| oracle | the better answer per question; an upper bound, never a live policy |

```bash
python scripts/eval_routing.py              # collect answers (resumable, slow on CPU) then analyse
python scripts/eval_routing.py --no-llm     # re-analyse cached answers in seconds
```

Questions: the 75 chapter-6 questions, 25 new chapter-7 questions (`eval_data/routing_eval/ch7_normalization_questions.jsonl`)
and 33 general questions with no labelled chunk (`general_questions.jsonl`). The corpus-versus-general mix is a
**constructed workload**, so the savings it shows describe such a mix, not real traffic.
Answer quality is `mean(token-F1, token-recall)`; it is still lexical and blind to paraphrase.
Results land in `results/routing_eval/<run>/` and are shown on the dashboard's *Research Results* tab.

## Interactive Research Dashboard (Three Primary Sections)

The web dashboard is organized into three concise, uncluttered primary sections:

1. **Ask a Question** (a live experiment on one question; it never changes saved results):
   - The adaptive decision engine runs **first** (action, features, decision latency) before any retrieval occurs.
   - `POST /ask?mode=compare` executes both branches independently: Option A (closed-book parametric memory) vs. Option B (grounded with retrieved evidence chunks).
   - Default view: decision and reason, both answers, citation chips, and a compact timing/token table (one live measurement, with a `cold start` flag for the first model call since the server started). Raw signals, full passages, word-overlap tables, the full benchmark question list and the raw JSON are in expandable sections.
   - `POST /ask?mode=adaptive` executes only the selected path to measure real deployed operational latency.
   - An expandable tool compares BM25, Dense and Hybrid on the question currently typed, only when you click its button.

2. **Research Results** (saved aggregates read from `results/`; independent of the question typed; *Reload saved results* only re-reads files and runs nothing):
   - Main experiment first: summary, then one table of the six routers (quality, retrieval rate, tokens, latency, routing balanced accuracy), with the shipped heuristic marked as deployed, the tuned and learned routers as experimental and the oracle as an evaluation-only upper bound.
   - Cost of each answering path from the same run.
   - Retrieval ranking benchmark (Recall@5, MRR@10, nDCG@10, Hit@1, median/cold latency, build time) for BM25, Dense and Hybrid from one run, with its corpus scope. Partial or scope-mismatched runs are flagged and not combined with others.
   - Collapsed: the earlier 75-question corpus-coverage experiment (confusion matrix; preliminary, 4 answers only), the per-question inspector and metric definitions.

3. **Documents**:
   - Active document collection table, active/inactive toggles, chunk counts, and deletion.
   - Upload form supporting PDF, TXT, Markdown, DOCX (with page tracking and deduplication).

### Model Configuration & Resumable Answer Cache
The model backing both branches is a general-purpose LLM (configured for local Ollama `qwen3:4b-instruct`):

```bash
copy .env.example .env      # then verify local settings:
#   LLM_API_KEY=ollama
#   LLM_BASE_URL=http://localhost:11434/v1
#   LLM_MODEL=qwen3:4b-instruct
```

Evaluation runs (`scripts/eval_policy.py` and `scripts/eval_downstream.py`) are backed by `adaptive_qa/answering/cache.py` (`data/eval_answer_cache.jsonl`), saving successful model answers incrementally to prevent data loss and support resumable batch runs.

## Retrieval methods: BM25, dense and hybrid

Option B's retriever is selectable per request from the dashboard, and defaults
to the configured value (`retrieval.retriever`, `bm25` for backward
compatibility, so existing API clients behave exactly as before).

| Method | What it does | Implementation |
|---|---|---|
| `bm25` | lexical Okapi BM25 (k1=1.5, b=0.75, title boost) — the baseline | `adaptive_qa/retrieval/bm25.py` (in-repo, no dependency) |
| `dense` | cosine similarity between the query embedding and every chunk embedding | `adaptive_qa/retrieval/dense.py` |
| `hybrid` | Reciprocal Rank Fusion of both rankings, `RRF(d) = Σ 1/(60 + rank_i(d))` | `adaptive_qa/retrieval/hybrid.py` |

**Pretrained model actually used:**
`sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, CPU). It is loaded
through the real `sentence_transformers.SentenceTransformer` API — there is no
simulated or hashed embedding path in production code. Verify it yourself:

```powershell
.\.venv\Scripts\python.exe -c "import sentence_transformers as st; m=st.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); print(m.get_embedding_dimension(), m.encode(['complex attributes']).shape)"
# -> 384 (1, 384)
```

`GET /retrieval/config` reports the embedding model name, whether the package is
installed, and the method list; each `/ask` response reports the method that
Option B actually used together with retrieval diagnostics (dimension, RRF k,
model, cache statistics).

Selection rules:

- **Failure is loud.** If `sentence-transformers` is missing or the weights cannot
  be loaded, `EmbeddingModelError` is raised with an actionable message. There is
  deliberately **no silent fallback to BM25** — that would invalidate every
  BM25-vs-dense comparison.
- **Scope.** Dense and hybrid search exactly the same chunk set as BM25: only the
  chunks of *active* uploaded documents. Inactive or deleted documents cannot be
  retrieved by any method.
- **Dense never returns "nothing".** It ranks all chunks, so an unanswerable
  question still gets k passages (some with near-zero similarity). BM25 can and
  does return an empty list. This asymmetry explains the `no_evidence_rate`
  column in the results below.

### Embedding cache (why the second run is fast)

Vectors are cached at `data/embeddings/<model>/<namespace>.npz` (float32,
L2-normalised, keyed by `sha256(model | include_title | chunk_text)`):

- an **unchanged** chunk is never re-encoded — a rebuild with the same corpus
  makes zero embedding calls (asserted by `tests/test_dense_retrieval.py`);
- an **edited or re-chunked** document produces different chunk text, therefore
  different keys, therefore fresh vectors — stale vectors can never be served;
- switching the model uses a different cache file, so vectors from two models are
  never mixed;
- a corrupt cache file is ignored and re-encoded instead of crashing;
- `--clear-embedding-cache` (or deleting `data/embeddings/`) forces a cold
  rebuild. `data/` is gitignored: no model weights or vectors are committed.

## Retrieval evaluation (BM25 vs dense vs hybrid)

`scripts/eval_retrieval.py` runs all three retrievers over the *same* corpus and
the *same* labelled questions, and writes JSON + CSV. Retrieval latency is
measured here and **nowhere else in the pipeline**: answer generation is excluded
so a "retrieval costs X ms" claim cannot be contaminated by LLM time.

### Labelled set

`eval_data/retrieval_eval/ch6_er_model_questions.jsonl` — 75 questions
(67 answerable, 8 unanswerable) over the 80 extracted slides of
*Database System Concepts* ch. 6 (ER model), with 19 exact-terminology,
18 paraphrase, 22 definition, 8 multi-passage and 8 unanswerable questions.
Every chunk was read in full before a question was written; a question answered
by two passages carries both chunk ids. See
[`eval_data/retrieval_eval/README.md`](eval_data/retrieval_eval/README.md) for the
full labelling protocol, the corpus fingerprint, and the honest caveats.

### Metrics (definitions used in code)

With binary relevance (a chunk is relevant iff its id is in the label list):

```
Recall@5 = |relevant ∩ retrieved[:5]| / |relevant|
MRR@10   = mean over questions of 1/rank(first relevant hit in top 10), 0 if none
nDCG@10  = DCG@10 / IDCG@10,
           DCG  = Σ rel_i / log2(i+1),
           IDCG = Σ_{i=1..min(|relevant|,10)} 1 / log2(i+1)
```

`hit_rate_at_1` (top-1 is relevant) and `no_evidence_rate` (share of unanswerable
questions for which nothing was returned) are reported too. Questions with no
labelled evidence have undefined Recall/MRR/nDCG and are excluded from those
means, exactly as documented in `adaptive_qa/eval/retrieval_metrics.py`.

### Measured retrieval results

Corpus: 80 chunks. Questions: 75 (67 scored). `--top-k 10 --repeat 3`, CPU,
Windows 11, Python 3.13, `all-MiniLM-L6-v2` fetched from the Hugging Face hub.

| Method | Recall@5 | MRR@10 | nDCG@10 | hit@1 | no-evidence | median query | cold build |
|---|---|---|---|---|---|---|---|
| bm25 | 0.918 | 0.733 | 0.787 | 0.582 | 0.125 | 0.12 ms | 0.05 s |
| dense | 0.925 | 0.718 | 0.770 | 0.597 | 0.000 | 28.3 ms | 23.0 s (16.5 s model load) |
| hybrid | **0.948** | **0.811** | **0.846** | **0.731** | 0.000 | 18.8 ms | 0.12 s (1.5 s model load) |

Source: `results/retrieval_eval/20261025_180311/retrieval_metrics.json` (the same run as REPORT section 9.1). Corpus scope: Chapter 6 only, 80 chunks.

Reading these numbers honestly:

- **Lexical BM25 is already strong here** (slides reuse the question's wording),
  so this corpus cannot show the usual dense-retrieval advantage. That is a
  property of the corpus, not of the methods.
- **Dense is slightly better on Recall@5 but worse on MRR@10 and nDCG@10** than
  BM25: it finds the evidence somewhere in the top 10, but ranks it less
  precisely. It also *always* returns k passages, so it never signals "nothing
  found".
- **Hybrid RRF is best on every ranking metric** (Recall +0.03, nDCG +0.06 over
  BM25), which is the expected effect of fusing a precise lexical ranking with a
  broader semantic one.
- **A dense query costs ~28 ms on CPU versus ~0.1 ms for BM25** — roughly 240×.
  Whether the ranking gain justifies that is an application decision, not a
  research claim.
- **67 questions is a small sample.** The gaps between BM25 and dense are within
  noise; only the hybrid-vs-BM25 nDCG/MRR gap looks systematic, and even that is
  descriptive. No p-value or confidence interval is claimed here.

Per-question rows, per-category breakdowns and the full settings are in
`results/retrieval_eval/<timestamp>/retrieval_metrics.json`.

### Output files

```
results/retrieval_eval/<timestamp>/
  retrieval_metrics.json       aggregate + per-question + settings + label validation
  summary.csv                  one row per method (the table above)
  per_question.csv             method × question: retrieved ids, Recall, RR, nDCG, latency
```

## Downstream QA evaluation (does retrieval *help the answer*?)

`scripts/eval_downstream.py` compares four conditions on the same questions with
the same LLM and settings: `closed_book` (no passages at all), plus `bm25`,
`dense` and `hybrid` retrieval feeding the grounded prompt. It records retrieval
latency, generation latency, total latency, tokens (provider-reported or
estimated), token-F1 / EM against the reference answer, and `evidence_coverage`
(lexical overlap between the answer and the retrieved passages).

`evidence_coverage` is a **grounding proxy, not a faithfulness or entailment
measure** — a wrong answer that copies passage words can score high. EM is
normalized exact match and is ~0 for free-form answers; token-F1 is the usable
signal, and it is paraphrase-blind. A manual rubric review of a sample is the
honest way to grade this set, and that is listed as follow-up work.

A six-question live run (real provider: OpenRouter with the `openrouter/free`
routing alias, so the backend model varies per call) gave:

| Condition | succeeded | mean token-F1 | mean total tokens |
|---|---|---|---|
| closed_book | 6/6 | 0.218 | 589 |
| bm25 | 6/6 | **0.554** | 1055 |
| hybrid | 6/6 | 0.443 | 1133 |
| dense | 6/6 | 0.413 | 1159 |

**n = 6 is a smoke test, not a result**, and this is the weakest evidence in the
project. Two later runs of the same script were dominated by HTTP 429 rate
limits from the free tier (14 of 24 and 15 of 16 calls failed); the script
recorded each failure with its provider message and kept going instead of
fabricating an answer, which is the behaviour we want but leaves the numbers
unusable. For a reportable table: pin a concrete `LLM_MODEL`, wait out the rate
limit, and run `--limit 0` (all 67 answerable questions).

Retrieval latency in a downstream run is reported after a one-off warm-up
(`retriever_warmup_ms` in the JSON), because the dense model is loaded lazily;
without it, the first question absorbs ~37 s of model loading.

Comparison math shown in the UI (all times per branch measured separately,
monotonic clock):

- `time difference = Option B total latency − Option A total latency`
- `token difference = Option B total tokens − Option A total tokens`

Positive => the retrieval branch cost more (no-retrieval "saved" that
amount); negative => increase. Percentages are only shown when the baseline
is nonzero.

### Latency fairness (cold start, warm-up, and how to compare honestly)

The dashboard comparison always executes Option A first, so A's generation
absorbs any one-off provider model-load cost while warm Option B does not.
The UI therefore labels each branch's generation with a cold/warm badge
(`execution first-call model load` vs warm) states that a single sequential
run must not be used to say a branch is faster. Verified on the local stack
(Ollama 0.40.10, `qwen3:4b-instruct`, one chat completion at max_tokens 64):

| call | measured |
|---|---|
| first call after Ollama restart (model load) | 23.8 s |
| same request, model already resident | 7.2 s |

`answering.llm.keep_alive` (default `"30m"`) is sent with every chat request;
Ollama honours it on its OpenAI-compatible endpoint and keeps the model
resident between calls (`/api/ps` showed a live `expires_at`), which removes
most dead-time reloads during a demo. Thinking/`<think>` behaviour is not an
issue with this model: `qwen3:4b-instruct` answered without a reasoning block,
and Ollama accepted `think: false` on both the OpenAI-compatible and native
endpoints).

For actual speed comparisons use the interleaved benchmark:

```
.\\.venv\\Scripts\\python.exe scripts\\bench_latency.py --trials 3
```

It alternates the branch order every trial (A,B then B,A …), reports each
backend's cold first call separately from warmed-up medians, keeps retrieval
(≤ ~1 ms) / decision (~1–5 ms incl. preflight) / generation separate, and
writes `results/bench_latency/<timestamp>/bench.json` + `bench.csv`. A real
2-trial measured run on this machine (short question, model kept warm):
Option A warm median 14.3 s, Option B warm median 21.5 s (warm Δ B−A ≈ 7.2 s,
driven by B's larger prompt: 1 388 vs 170 total tokens) — i.e. on this corpus
the grounded path costs roughly 50% more warm generation time; the cold first
call of the whole app (≈24 s) must never be charged to one of the branches
alone.

### Adaptive-policy decision metrics (confusion matrix, sample sizes)

`scripts/eval_policy.py` reports the adaptive router's decision quality against
the labelled answerability of each question, with
*positive class = answerable (a labelled relevant passage exists)* and
*predicted-positive action = RETRIEVE*:

```
TP = retrieval on an answerable question        FN = skip on an answerable question
FP = retrieval on an unanswerable question      TN = skip on an unanswerable question
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)                    (= sensitivity)
specificity = TN / (TN + FP)
F1          = 2·precision·recall / (precision + recall)
balanced acc = (recall + specificity) / 2
accuracy    = (TP + TN) / n
```

always-retrieve and never-retrieve grids are computed from the same rows at
evaluation time, so raw accuracy is never read without the imbalance-aware
baselines beside it. On the shipped 75-question set (67 answerable / 8
unanswerable, fresh run 20261026_011553): adaptive TP=65 FN=2 (q52, q66) FP=1
(q68) TN=7 → accuracy 96.0%, precision 98.5%, recall 97.0%, F1 97.7%,
specificity 87.5%, balanced accuracy 92.25%; always-retrieve scores 89.3%
accuracy (= 67/75, i.e. the class imbalance) with 0.5 balanced accuracy and
specificity 0 — confirming that raw accuracy must be read next to these
numbers, not instead of them.

Routing metrics (evidence recall, probe rate, decision matrix) are measured
over **all 75 questions**; answer-quality, latency and token means are
measured separately over **the questions with a successful live LLM call**
(n = `answers_run`, reported per policy as `n_latency_samples` /
`n_token_samples` / `n_quality_samples`). With fewer than 10 LLM answers the
dashboard flags those means PRELIMINARY; failed generations are listed
separately and never counted as answers. The oracle policy exists only inside the
evaluation scripts; the live decision path never receives labels, reference
answers or oracle choices.

## Architecture

```
                        ┌───────────────────────────────┐
 question ──────────────►  ADAPTIVE DECISION ENGINE     │   pre-retrieval: sees ONLY
                        │  ✦ heuristic policy            │   the question (+ cheap
                        │  ✦ learned classifier (sklearn)│   lexical memory-coverage
                        └──────────────┬────────────────┘   features from simulated memory)
                             RETRIEVE / DO_NOT_RETRIEVE
                             │                          │
              selected path  │        cross-verify:     │
            (deployed mode)  │        BOTH paths run    │
                             ▼                          ▼
        ┌────────────────────────┐          ┌──────────────────────────────┐
        │ METHOD A (no retrieval)│          │ METHOD B (with retrieval)     │
        │ parametric memory +    │          │ BM25 over local corpus →      │
        │ span/or-option extract │          │ reciprocal-rank-fusion extract│
        └────────────────────────┘          └──────────────────────────────┘
                             └─────── per-question record (timings, answers,
                                          methods, snippets, decision score)
                                        ▼
                          OFFLINE EVALUATION (cross-verify mode only):
                          EM/F1 vs reference answer → outcome label
                          {retrieval_better, no_retrieval_better,
                           similar, ambiguous} → ORACLE benchmark
```

Key modules:

| Path | Role |
|---|---|
| `adaptive_qa/dataset.py` | deterministic HotpotQA sampling via HF datasets-server, level filter, question-level splits |
| `adaptive_qa/corpus.py` | local BM25-indexed corpus (distractor paragraphs of eval questions; disjoint from memory) |
| `adaptive_qa/retrieval/__init__.py` | retriever factory: `bm25` / `dense` / `hybrid`, config-driven with per-request override |
| `adaptive_qa/retrieval/dense.py` | dense semantic retrieval + persistent content-hashed embedding cache |
| `adaptive_qa/retrieval/hybrid.py` | Reciprocal Rank Fusion of BM25 and dense rankings |
| `adaptive_qa/eval/retrieval_metrics.py` | Recall@5, MRR@10, nDCG@10, latency statistics |
| `adaptive_qa/eval/retrieval_eval.py` | labelled-set loader, label validation, multi-method harness, CSV/JSON writer |
| `scripts/eval_retrieval.py` | CLI: retrieval evaluation over the labelled set |
| `scripts/eval_downstream.py` | CLI: closed-book vs bm25/dense/hybrid answer generation |
| `adaptive_qa/answering/parametric_store.py` | gold-paragraph-only "simulated LLM memory" (train-split snippets, never eval gold docs) |
| `adaptive_qa/answering/finite_engine.py` | Method A — closed-book answering over the parametric store |
| `adaptive_qa/answering/span_absorber.py` | Method B — retrieval-grounded answering (BM25 + rank-fusion extraction) |
| `adaptive_qa/decision/features.py` | pre-retrieval features: question type, length, entropy, option-like patterns, memory-coverage probe |
| `adaptive_qa/decision/heuristic.py` | interpretable baseline policy (reference for calibration) |
| `adaptive_qa/decision/learned.py` | logistic-regression classifier over features; tuned on validation only |
| `adaptive_qa/pipeline.py` | `run_adaptive` (selected path ONLY) vs `run_cross_verify` (BOTH paths) — the two workflows |
| `adaptive_qa/eval/outcome.py` | offline oracle labeling from EM/F1 (never an input to live decisions) |
| `adaptive_qa/eval/metrics.py` | four-policy aggregate: EM/F1, P/R/F1 for RETRIEVE, latency percentiles, cost |
| `adaptive_qa/eval/charts.py` | quality-vs-latency, quality-vs-cost, decision quality charts (matplotlib) |
| `adaptive_qa/experiment.py` | batch runner; per-question JSONL + summary.json + run artifacts |
| `api/main.py` | FastAPI backend: `/ask` (compare/adaptive, `retrieval_method`), `/retrieval/config`, `/ask/benchmark` (legacy cross-verify), `/documents/*`, `/evaluate/*` |
| `dashboard/index.html` | single-page research dashboard with a retrieval-method selector |
| `tests/` | 223 pytest tests: leakage, timings, policy selection, BM25, dense, hybrid, metrics, documents, decision trace, evidence-overlap map, policy confusion matrix, latency-benchmark math |

## Experimental design (what makes the comparison fair)

- **Two workflows are structurally distinct.** Deployed adaptive mode executes
  *only* the decision engine's chosen path. Cross-verification executes *both*
  methods per question; that dual execution is a cost of evaluation, reported
  separately and never counted in the deployed latency claims.
- **The decision engine sees only the question** (plus a cheap lexical probe of
  the *simulated memory store*, which is part of the closed-book model itself,
  not the corpus — no retrieval is performed for it). It never sees the
  reference answer, gold titles, outcome labels, the retrieved passages or any
  post-decision result. `tests/test_leakage.py` enforces this import- and
  AST-level.
- **Splits are question-level**: train/validation/test of the eval sample, with
  the classifier trained only on train rows and threshold-tuned only on
  validation rows. The held-out test split is evaluated exactly once for the
  report. No eval question id (or gold paragraph) appears in the memory
  source rows.
- **Outcome classes** come from `quality = 0.5·EM + 0.5·F1` versus the
  reference answer with gap threshold 0.10: `retrieval_better`,
  `no_retrieval_better`, `similar`, and `ambiguous` (abstentions/missing
  reference). Ambiguous rows are excluded from binary decision metrics and
  reported separately. Bad *retrieval evidence is not a free pass*: citations
  do not influence labeling — only answer quality does.
- **Four policies** (`never_retrieve`, `always_retrieve`, `adaptive`, `oracle`)
  are computed from the same per-question records, so they share identical
  evaluation questions and metrics computation.
- **Timings** use `time.perf_counter()` (monotonic) and are saved per stage:
  `decision_ms`, `retrieval_ms`, `gen_a_ms`, `gen_b_ms`, the deployed
  `adaptive_e2e_ms`, the cross-verification `cross_verify_ms`, and separate
  `evaluation_ms`. Oracle evaluation time is never included in deployed
  latency. Mean *and* median are reported; stdev is included per-policy.
- **Cost model** is documented in `adaptive_qa/cost.py`: retrieval count per
  100 questions × cost-per-retrieval (BM25 is local compute, so the cost model
  is expressed in retrievals + estimated wall-clock; an API retriever would
  substitute its price).
- **Reproducibility**: dataset sampling is a seeded offset + consecutive rows;
  splits are seeded; BM25 and branding-free extraction are deterministic;
  sklearn is seeded. Re-running `build_dataset.py` then the experiment
  reproduces the same numbers.

## Dataset & expansion

HotpotQA (distractor config) via the public `datasets-server` rows API:

- Eval sample: 200 questions level-filtered to `easy`, sampled from the
  **`train` split** at a deterministic offset (base 40,000, seeded jitter).
  Empirical finding recorded: the distractor config's `validation` split is
  **uniformly level `hard`** (verified across the split), so the easy/medium
  distribution only exists in the train split — eval is sampled there and kept
  disjoint from the simulated pretraining rows by separate offset windows
  (base 40k for eval, base 15k for memory) plus an explicit disjointness
  assertion at build time.
- Corpus: distractor paragraphs of the 200 eval questions (~2.0k docs), i.e.
  retrieval-at-test-time works on docs that include *some* gold paragraphs but
  also distractors, exactly the HotpotQA setting.
- Parametric memory: disjunct train-split rows only (base-15k window,
  disjoint from the eval window by construction and asserted), gold
  paragraphs only → ~11k memory snippets for 6,000 pretrain rows. This is the
  "LLM's pretraining" stand-in.
- Expansion: raise `dataset.pretrain_rows_n` and `dataset.eval_sample_n` in
  `configs/default.json` (e.g. `eval_sample_n: 1000`) and re-run
  `scripts/build_dataset.py --force`. The datasets-server is rate-limited; the
  fetch loop retries transient HTTP and connection resets with exponential
  backoff, and partial JSONL is append-tolerant (re-run resumes).

## Deliverable files when a run finishes

```
runs/<timestamp>/
  per_question.jsonl   machine-readable per-question records (all four policies
                       + timings + answers + retrieval hits + outcome labels)
  summary.json         aggregate table (quality, P/R/F1 for RETRIEVE, latency
                       mean/median/stdev, retrieval rate, cost per question)
  charts/*.png         quality vs latency, quality vs cost, decision confusion
  experiment_meta.json configs snapshot for reproducibility
models/                trained decision classifier + validation report
```

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

**168 tests, all passing** (105 before this retrieval work + 63 new). No test
needs network access or API credits: dense-model inference is replaced by an
injected deterministic embedder, and every LLM call is mocked.

Beyond the mocked suite, the real pretrained model was exercised live through
the HTTP layer: `POST /ask?mode=compare` with `retrieval_method=dense` over the
80-chunk corpus returned `sentence-transformers/all-MiniLM-L6-v2`, 384 dims, 80/80
vectors reused from cache, and top hits `ch6.pdf (page 18)` (the *Composite
Attributes* slide), `page 43`, `page 17` for "what is a composite attribute?" —
while `Option A` produced its own closed-book answer. In the same call the
provider rate-limited Option B (HTTP 429): Branch B showed the provider error and
Branch A's answer stayed visible, confirming live branch isolation.

The new suite adds:

| File | Covers |
|---|---|
| `tests/test_dense_retrieval.py` (13) | vector dimensions/normalisation, cosine ranking, metadata, deterministic tie-break, cache write/reuse (zero re-encoding), invalidation on changed text, corrupt cache, missing package / unloadable model raising `EmbeddingModelError` instead of falling back |
| `tests/test_hybrid_retrieval.py` (12) | RRF formula against hand-computed scores, dedupe by chunk id, single-ranking members, tie-break, component ranks/scores, fused score reproducible from components, empty corpus, determinism |
| `tests/test_retrieval_metrics.py` (18) | Recall@k, MRR@k, nDCG@k against hand-calculated values, k truncation, undefined-without-labels, unanswerable exclusion and `no_evidence_rate`, category breakdown, latency percentiles |
| `tests/test_retrieval_eval.py` (9) | shipped dataset schema/balance, loader validation errors, label/corpus mismatch detection, harness metrics and latency sampling, CSV/JSON artifacts |
| `tests/test_retrieval_selection.py` (11) | method normalisation, config default/override, BM25 kwargs never leaking into the dense constructor, dense & hybrid scoped to active documents, unique document-wide chunk ids, reindex preserving activation, diagnostics collected after the query, `/retrieval/config`, unknown method → 400, requested method reaching Option B with both answers still returned |

The pre-existing suite covers:
tokenizer/BM25 correctness, EM/F1 edge cases, span extraction, heuristic +
learned decision engine (including persistence round-trip), pipeline
adaptive-vs-cross semantics, plan/decision boundaries, timing-category
sanity, and two leakage test families (decision code never imports/reads
references or outcome labels; source-scan asserts no banned strings in
decision modules).

The new suite adds: document extraction (TXT/MD/DOCX/PDF with hand-built
minimal PDFs), scanned-PDF warning, chunking overlap/size and page
attribution, persistence across store instances, duplicate detection,
delete/active toggles, retrieval scoped to the active collection, decision-
before-branches ordering (serve flow), both-branches in compare vs
selected-only in adaptive mode, branch-failure isolation, timing/token
formula signs and zero-baseline handling, provider-reported vs estimated
token accounting, LLM misconfiguration and HTTP/timeout error paths (all
HTTP mocked — no paid API), per-policy cost aggregation regression, the
parametric-store leakage guard, and FastAPI endpoint tests for upload/list/
active/delete plus compare/adaptive/benchmark routes.

## Limitations (candid)

- The HotpotQA benchmark "LLM" is a deterministic extraction engine over a
  simulated parametric store — it is *not* a neural closed-book model.
  Conclusions about absolute answer quality do not transfer to real LLMs; the
  *experimental design* (adaptive decision, four-policy comparison, leakage
  isolation, timing categories) is the transferable contribution. The
  interactive dashboard uses a real configured LLM instead.
- **Leakage note (fixed)**: `parametric_store.include_eval_gold_paragraphs`
  previously copied eval gold paragraphs into the simulated memory (a data
  leak). It is now **false by default**, the store is rebuilt from the
  disjoint pretrain window only, and `tests/test_cost_and_leakage.py` asserts
  eval gold sentences are absent. Benchmark numbers from runs saved BEFORE
  this fix are not comparable to new ones.
- `easy`-filtered HotpotQA skews toward answerable closed-book questions; the
  hard subset would push `always_retrieve` relatively higher. Reported as a
  sampling constraint, not hidden.
- Outcome labels are EM/F1-based. Semantic equivalence beyond token matching
  (paraphrase) is penalized as wrong for both methods equally; an optional LLM
  judge exists (`rubric` documented inline) but is label-supplied only if
  explicitly enabled — flagged as weak labels in that case.
- Cost is modeled in retrievals, not dollars, because retrieval is local.
  Substituting a hosted LLM/QA API would give real costs via `cost.py`.
- **Routing aliases hurt reproducibility.** Setting `LLM_MODEL` to a routing
  alias (e.g. OpenRouter's `openrouter/free`) makes every call land on a
  *different* backend model, so answer quality, latency and token usage vary
  run-to-run; some routed models return only internal reasoning or truncate
  early. The backend records the concrete `routed_model` in each answer's
  metadata for transparency and tightens the prompt to suppress
  chain-of-thought leakage, but for publishable experiments pin a concrete
  model id in `.env`. The OpenAI-compatible parser reads
  `choices[0].message.content` and falls back to `message.reasoning` /
  `reasoning_content` / `text` so reasoning-only replies are surfaced rather
  than dropped.
- **Non-answering replies are rejected, not shown.** A routing alias can land
  on a specialist model whose `content` is a classification rather than prose
  (observed live: `nvidia/nemotron-3.5-content-safety:free` returns the literal
  `"User Safety: safe"`). The grounded backend detects empty replies,
  reasoning-only replies and short classification markers, retries them, then
  raises a clear `AnswerBackendError` naming the routed model — it never
  fabricates a fallback answer and never displays a classification as the
  answer. `answering.llm.max_tokens` defaults to 2048 so reasoning models do
  not exhaust the budget before emitting an answer.
- The pre-retrieval memory probe is a lexical feature, not true model
  uncertainty; a calibrated conformal or MC-dropout signal would be stronger.
- **The retrieval evaluation is one corpus, 67 scored questions.** Chapter 6
  slides paraphrase their own figures, so BM25 is unusually strong. Do not
  generalise "hybrid beats dense" or "BM25 is enough" beyond this corpus. The
  numbers are descriptive; no significance test is claimed.
- **Labels are partial by construction** (a passage that mentions the topic but
  does not answer the question is not labelled). This depresses recall for every
  method equally, so it does not bias the comparison, but it does mean absolute
  Recall@5 values are pessimistic.
- **Dense and hybrid always return k passages**, so they cannot express "nothing
  relevant found"; only BM25 can return an empty list. A score threshold would be
  needed to use similarity as an abstention signal, and it is not tuned here.
- **CPU-only embedding latency.** `all-MiniLM-L6-v2` costs ~37 s to load cold and
  ~45 ms per query on this CPU. A GPU or a warmed long-lived process changes the
  picture; both cold-start and steady-state are reported separately for exactly
  this reason.
- **Chunk ids changed format** while adding retrieval (`<doc_id>:<n>` now counts
  across the whole document instead of restarting per page, which made ids from
  different pages collide). Stored documents include `source.bin`, so
  `DocumentStore.reindex_all()` rebuilds chunks in place without re-uploading.
  Label sets and any chunk ids captured before this change no longer resolve.
- **Downstream QA on the labelled set is not a paper result.** The script exists
  and runs end to end, but the live demonstration used n = 3 with a routing
  alias, and token-F1 is paraphrase-blind. Grading needs a pinned model and a
  manual rubric pass.

## Report template

See `docs/report_template.md` for the written report skeleton (abstract,
method, dataset, decision engine, four-policy table, trade-off charts,
limitations, contributions).
