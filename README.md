# Should the AI Search Before Answering? — Adaptive Retrieval Decision Engine

A reproducible NLP research project (CSE coursework) that answers one question
experimentally: **can a lightweight pre-retrieval decision engine predict when
external retrieval will help a QA system, and does adaptive retrieval beat
always/never retrieving on the quality–latency–cost trade-off?**

The system is a *deterministic, locally reproducible* QA research pipeline, not
a chat demo: BM25 retrieval over a local HotpotQA corpus, two answering methods
(a closed-book span/extraction engine over a simulated parametric memory, and a
retrieval-grounded extraction engine), a pre-retrieval decision engine
(heuristic policy + learned scikit-learn classifier), and a cross-verification
evaluation harness with an offline oracle.

## Quick start

```bash
# 0) environment (Python 3.10+; tested with Anaconda on Windows)
pip install requests scikit-learn fastapi uvicorn matplotlib pytest httpx \
    pypdf python-docx python-multipart

# 1) build the dataset + corpus + parametric memory (~180 MB downloads, HF datasets-server)
python scripts/build_dataset.py --force            # cached/partial rebuilds survive interruptions

# 2) train + tune the learned decision engine (train split → tune on validation)
python scripts/train_decision.py --config configs/default.json
#    writes models/decision_classifier.json + models/decision_report.json

# 3) run the four-policy experiment on the held-out TEST split
python scripts/run_experiment.py --config configs/default.json --limit 60
#    writes runs/<timestamp>/  per_question.jsonl, summary.json, charts/*.png

# 4) optional dashboard
uvicorn api.main:app --port 8000
#    http://127.0.0.1:8000            (side-by-side comparison + documents + batch + export)
```

Every step works from the CLI; the web UI is optional. No paid API is required.

## Interactive comparison + uploaded documents (dashboard v2)

The dashboard's primary flow is a **decide-first, side-by-side experiment**:

1. The adaptive decision engine runs FIRST (recorded: action, features,
   decision latency) — no retrieval happens before it.
2. `POST /ask?mode=compare` then runs BOTH branches independently. Option A
   (closed book) gets the question only — never the uploaded documents or
   retrieved passages. Option B searches ONLY the active uploaded document
   collection and grounds its answer in retrieved chunks (filenames, PDF page
   numbers and retrieval scores are returned). Both branches use the same
   configured LLM, so context/retrieval is the only experimental difference.
3. `POST /ask?mode=adaptive` measures the deployed path: decision + only the
   selected branch. Its wall-clock is real operational latency and is
   structurally separate from the comparison request's total.

The model backing both branches is a general-purpose LLM behind the
OpenAI-compatible chat API, configured ONLY via environment (never hardcoded):

```bash
copy .env.example .env      # then fill in real values; .env is gitignored
#   LLM_API_KEY / LLM_BASE_URL / LLM_MODEL   (OPENAI_* names also accepted)
```

If the model is not configured, both branches return a clear
`ModelNotConfiguredError` message in the UI — there is NO silent fallback to
the deterministic extractors (those remain available as labeled baselines in
the HotpotQA benchmark pipeline). Token usage is provider-reported when the
API returns `usage`; otherwise estimated from text length and labeled
`"source": "estimated"`. The decision engine does not call the model, so it
has no LLM tokens.

Document upload supports **PDF / TXT / Markdown / DOCX** (via the dashboard
section 3 or `POST /documents/upload`): type/size validation, text extraction
with PDF page numbers, overlapping chunking, persistent storage under
`data/documents/` (survives restarts), content-hash deduplication, a clear
warning for text-less scanned PDFs (OCR is future work), and a BM25 index scoped
to the toggle-selected active collection. Uploaded documents never mix with
the HotpotQA benchmark corpus.

Comparison math shown in the UI (all times per branch measured separately,
monotonic clock):

- `time difference = Option B total latency − Option A total latency`
- `token difference = Option B total tokens − Option A total tokens`

Positive => the retrieval branch cost more (no-retrieval "saved" that
amount); negative => increase. Percentages are only shown when the baseline
is nonzero.

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
| `api/main.py` | FastAPI dashboard backend: `/ask` (compare/adaptive), `/ask/benchmark` (legacy cross-verify), `/documents/*`, `/evaluate/*` |
| `dashboard/index.html` | single-page research dashboard |
| `tests/` | 50+ pytest tests: leakage, boundaries, timings, policy selection, BM25, EM/F1, dataset |

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

```bash
python -m pytest tests/ -q
```

92 tests (50 pre-existing + 42 new). The pre-existing suite covers:
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
- The pre-retrieval memory probe is a lexical feature, not true model
  uncertainty; a calibrated conformal or MC-dropout signal would be stronger.

## Report template

See `docs/report_template.md` for the written report skeleton (abstract,
method, dataset, decision engine, four-policy table, trade-off charts,
limitations, contributions).
