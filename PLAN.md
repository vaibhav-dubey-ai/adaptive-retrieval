# Should the AI Search Before Answering? — Research Project Plan

Course project: adaptive retrieval decision-making for question answering.
This document was produced in **Stage 1** (repository and environment inspection,
architecture, dependencies, dataset choice, experimental plan). Later stages
implement and extend this design; deviations from the plan will be documented in
the README.

---

## 1. Environment findings (Stage 1 inspection)

| Item | Finding |
|---|---|
| Workspace | Empty (no existing repository, no prior work to preserve) |
| Python | 3.13 via Anaconda (`C:/Users/HP/anaconda3/python.exe`): `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pytest`, `requests` already installed |
| Node | v24.13.0 (available, but the dashboard will be a lightweight Python-served page) |
| GPU / Ollama | None — no local neural LLM is feasible on this machine |
| Internet | Working; Hugging Face `datasets-server` API reachable (HTTP 200) |

Implication: the answering model is behind a **configurable adapter**. The
default backend is a free, deterministic, CPU-friendly *finite answer engine*
(no network, no paid API) so the whole pipeline and the basic demonstration run
locally. An OpenAI-compatible hosted adapter (e.g. the Freebuff API) is wired in
the same interface and is activated purely via configuration.

## 2. Research question

> Can an adaptive decision engine predict when retrieval is beneficial, and can
> it improve the balance between answer quality, latency, and retrieval cost
> compared with always retrieving or never retrieving?

## 3. Dataset choice

**Primary dataset: HotpotQA (distractor setting), validation split**, sampled
via the Hugging Face `datasets-server` REST API (100 rows per request, paged —
no library install needed).

Each HotpotQA record provides:
- `question`, `answer` (reference answer string),
- `context`: 10 paragraphs (2 gold supporting paragraphs + 8 distractor
  paragraphs), each with a title and 3–8 sentences.

This gives, from a single source, everything the brief requires:
- reference answers **and** supporting evidence (gold paragraphs),
- a realistic, *uncontrolled* retrieval task: gold paragraphs must be found
  among distractors,
- a natural mixture of easy single-fact questions (answerable without
  retrieval) and obscure/multi-hop facts (retrieval substantially helps).

**Corpus construction (hostile to leakage):** the local corpus is the union of
all paragraphs of *every sampled question* (~2,000 paragraphs for 200 questions,
document IDs preserved, e.g. `hotpotqa_validation_00042# GeForce#0`).
Paragraphs that are distractors for question X are distractors for X but may be
gold evidence for question Y — this is exactly what makes retrieval non-trivial.
The eval pipeline restricts retrieval to the eval-time corpus; because all
questions share one corpus, no per-question advantage exists.

**Dataset subsets (mixture required by the brief):** the sampled questions are
not re-labeled by factories; instead, "kind" is a *derived evaluation artifact*:
`parametric_easy` (Method-A top-1 candidates judged correct without evidence),
`evidence_needed` (Method-A weak, Method-B strong), `outdated_or_precise`
(Queens/titles/dates in question text — HotpotQA is full of these), and
`poor_evidence` (retrieval returns no gold paragraph in the retrieved batch).
These definitions and their exact computable criteria are documented and
reported separately; they are evaluation summaries, not training inputs.

**Splits:** train / validation / test, stratified by nothing fancy — questions
are randomly split at the *question level* (parity check on question length and
entity distributions reported), so no question, answer text, or document is
shared across splits. The corpus is built from train+validation+test questions
together, but the *decision-engine classifier is only ever shown training-split
questions* and validation for tuning.

**Dataset-expansion path (documented in README):** any number of rows can be
requested via the sampler (`--n` / `--sample-slices`), and the loader accepts
any HotpotQA-JSONL file; a config flag points to a larger local slice. The
pipeline is corpus-size agnostic.

## 4. Answering methods

Two answering methods share one interface (`AnsweringMethod`):
`answer(question, passages=None) -> GeneratedAnswer`.

### Method A — No retrieval
- Receives only the question.
- **Default backend (`finite_answer_engine`):** a deterministic CPU answer
  engine with a finite, honest "closed-book" behavior — it either commits or
  abstains, never invents, and its behavior is fully reproducible (documented
  limitation: it is *not* a neural model, so it under-estimates the true
  parametric-quality of an LLM; see §10 limitations).
- **Optional backend (`openai_compatible`):** chat-completions against any
  OpenAI-compatible endpoint (e.g. Freebuff API), temperature 0, fixed prompt,
  closed-book system instruction.

### Method B — With retrieval
- Retrieves top-k passages from the local corpus (BM25), builds a grounded
  prompt, and generates an answer.
- **Default backend (`span_absorber`):** a deterministic extractive reader that
  selects the best-matching sentence-span from the retrieved passages via
  scoring overlap and keyword coverage (documented as a non-neural reader).
- **Optional backend (`openai_compatible`):** grounded chat-completions over
  retrieved passages, with a strong instruction to abstain unless evidence is
  present.

Both backends are tested behind the same interface, so quality numbers are
comparable across backends without re-implementing the pipeline.

## 5. Retriever

- **Default: BM25 (implemented in-repo**, not `rank_bm25`**,)** — OKAPI BM25
  (k1, b) over corpus paragraphs, with title-headered documents; tokenization
  lowercases, strips punctuation, stops on English stopwords. Rationale: 3.13 +
  `rank_bm25` compat, no dependency, full latency control, easy unit tests.
- **Optional: dense-embedding retriever** — sentence-embeddings if installed;
  interface shared with BM25 (`Retriever.retrieve(query, top_k)`) so the runner
  can switch via config. Default ON only if the package is present without cost
  increase; otherwise BM25-only.

## 6. Adaptive decision engine

`DecisionEngine.decide(question) -> Decision.RETRIEVE | DO_NOT_RETRIEVE`.

**Stage 3 baseline (interpretable):** keyword/temporal cue scoring — flags
questions needing precise or entity-specific facts (Queens-of / titles / years
/ "who won" style) vs. common-sense style. Produces a `retrieve_score` in [0,1],
tunable threshold.

**Learned classifier (if features support it):** scikit-learn `GradientBoosted
Trees` (or LogisticRegression) over interpretable features computed at
*pre-retrieval* time: question length, token counts, WH-type, named-entity
signal words, presence of temporal/dated cues, named-entity-entity count,
question "specification" via string overlap to the corpus vocabulary
(BM25-free!), optional dense question embedding features, and — critically — a
**knowledge-freshness signal**: whether the question entity or topic string is
missing from the *model's* parametric surface form heuristics (weak proxies are
documented as such, calibrated via validation set).

**Squash the calibration illusion:** the learned model's `retrieve_score` is
temperature-adjusted on *validation* data so that reported probabilities
(F1/P/R) reflect the actual working point; no self-reported LLM confidence is
used at all (research-brief requirement; documented).

**Post-retrieval policy (optional, Stage 5+, clearly separated):** a second,
labeled `post_retrieval` decision engine may revise the pre-retrieval call
after seeing retrieval results (e.g. weak-retrieval abort → fall back to Method
A). This is *never* treated as the pre-retrieval policy in experiments; it is an
extra variant reported under its own name.

Strict gates: the engine never sees the reference answer, oracle label, or any
evaluation output. The oracle's labels are consumed only by the evaluation
pipeline, after inference completes. A test enforces this (§9 tests).

## 7. Four experimental policies

All four share the *same* answering-method implementations and the same
corpus+retriever; they differ only in the retrieval decision:

| Policy | Rule |
|---|---|
| `never_retrieve` | Method A for every question |
| `always_retrieve` | Method B for every question |
| `adaptive` | Decision engine picks A or B per question |
| `oracle` | Runs *both* A and B, then picks the offline-eval winner — **posted benchmark only, never a live engine** |

## 8. Cross-verification and outcome labeling

For every *evaluation* question (validation + test):
1. Run Method A and Method B independently (same corpus, same backends).
2. Score both against the reference answer + gold paragraphs.
3. Score gap Δ = metric(B) − metric(A); classify:
   - `retrieval_better` (Δ ≥ +0.10),
   - `no_retrieval_better` (Δ ≤ −0.10),
   - `similar` (|Δ| < 0.10),
   - `ambiguous` — either answer was an abstain/empty, or either answer is
     un-evaluable (documented in detail).
   Metric gap thresholds are fixed in one config place and reported.
   **Citations / passage-presence alone never decides** the label — only the
   scored answer quality does.

**Evaluation metrics (per policy):**
- EM and token-F1 (standard SQuAD-style normalization: lowercase, strip
  punctuation/articles) against the reference answer.
- Evidence overlap: token-F1 of the scored answer against gold *paragraph
  titles* (as a proxy for correct-source) — reported, not used for EM/F1.
- Abstain rate, retrieval rate.
- Latency: decision-engine, retrieval, generation-A, generation-B, adaptive
  end-to-end, and cross-verification (both-paths) — all measured with
  `time.perf_counter_ns()` (monotonic, high-resolution). Model-generation time
  stored separately from wall-clock overhead.
- Retrieval frequency + estimated cost (BM25 = 0 API cost; documented cost
  model is per-call CPU seconds for embeddings, or per-request USD if a hosted
  retriever/LLM is configured).
- Aggregate quality-vs-latency and quality-vs-cost trade-off tables + charts.

**Decision accuracy metrics (adaptive vs oracle on evaluable non-`ambiguous`
questions):** accuracy; precision / recall / F1 for the *RETRIEVE* class;
confusion matrix (`beneficial retrievals caught`, `beneficial retrievals
missed`, `unnecessary retrievals`, `skipped retrievals that would have hurt`).
`ambiguous` cases are excluded and are also reported separately (both options
in the brief are honored).

## 9. Persistence, reproducibility, tests

- **Run artifacts:** `runs/<timestamp+run_id>/` per experiment, containing:
  `per_question.jsonl` (machine-readable, one line per question), `summary.json`
  (policy-aggregated metrics), `timings.json}`, `config_snapshot.json`,
  `corpus_manifest.json`, `run_manifest.json`, decision-engine metrics, charts
  PNG charts.
- **Reproducibility:** every run re-loads the raw sampled data (deterministic
  sample seed), rebuilds BM25 deterministically, seeds sklearn, and stores
  library versions in `run_manifest.json`. Reruns with the same config produce
  byte-identical per-question answers for the deterministic backends
  (assertors in tests).
- **Tests (`pytest`)**, mapped to brief requirements:
  1. `test_leakage.py` — the decision engine's input-fingerprint never contains
     the reference answer/gold paragraphs/oracle label; decision accuracy on a
     leak-detector canary set (questions deliberately duplicated with and
     without oracle labels must produce identical decisions).
  2. `test_oracle_isolation` — the oracle outcome-labeler is import-*separable*
     from the adaptive runtime (import-graph assertion).
  3. `test_cross_verification.py` — both paths run in cross-verification mode
     even when only one is deployed; deployed-adaptive mode never executes the
     unchosen path (operation-log assertion).
  4. `test_timing.py` — all required latency categories are recorded,
     `perf_counter_ns`-monotonic, decision time excluded/computed correctly,
     oracle-evaluation time never reported inside adaptive latency.
  5. `test_metrics.py` — EM/F1 against known cases, threshold boundary
     behavior, `RETRIEVE`-class P/R/F1 correctness.
  6. `test_retrieval.py` — BM25 correctness on toy corpora (known-best doc
     ranked first), no leaking of gold passages outside the requested top-k.
  7. `test_split.py` — no question duplicate across splits, no document leakage
     between train and test question sets.

## 10. Dashboard (Stage 6)

FastAPI backend (`api/`) serving a single-page dashboard (`dashboard/`, pure
HTML+JS, no build step) at `/dashboard`. Endpoints:
- `POST /ask` — question → decision, primary answer, timing breakdown.
- `POST /ask?mode=cross` — adds the alternative-method answer and the retrieved
  passages (relevance scores), plus — when evaluation data is available — the
  "was the decision right?" flag computed *against offline reference data*,
  clearly marked as evaluation-only (never fed to the decision engine).
- `POST /evaluate/run` — run a batch experiment; `GET /evaluate/summary`.
- `GET /evaluate/export?format=csv|json` — per-question results export.

Charts (quality-vs-latency, quality-vs-cost, retrieval rate) are rendered
client-side from `/evaluate/summary` (small self-contained JS chart code, no
CDN requirement).

Reference answers are *never* part of `/ask` payloads unless the caller
explicitly sets `mode=cross` and the evaluation store has a reference for that
question; the decision engine's input remains the same either way.

## 11. Layout

```
.
├── README.md                     # exact commands, limitations, dataset note
├── PLAN.md                       # this document (Stage 1 artifact)
├── requirements.txt              # pinned, minimal (all but one already present)
├── configs/
│   └── default.yaml              # answering backend, retriever, dataset, classifier, run
├── adaptive_qa/
│   ├── __init__.py
│   ├── config.py                 # dataclass config, single-settings-merge
│   ├── timeutil.py               # perf_counter_ns helpers (monotonic timer)
│   ├── corpus.py                 # corpus build / load, doc IDs, manifest
│   ├── dataset.py                # HotpotQA sampler + loader + splits
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py               # in-repo BM25
│   │   ├── dense.py              # optional embeddings retriever
│   │   └── base.py               # Retriever interface
│   ├── answering/
│   │   ├── __init__.py
│   │   ├── base.py               # AnsweringMethod interface
│   │   ├── finite_engine.py      # deterministic closed-book default (Method A)
│   │   ├── span_absorber.py      # deterministic extractive reader (Method B)
│   │   └── openai_compat.py      # optional hosted adapter
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── base.py               # DecisionEngine interface
│   │   ├── heuristic.py          # interpretable cue-score policy
│   │   ├── learned.py            # sklearn classifier
│   │   ├── features.py           # pre-retrieval feature extractor
│   │   └── post_retrieval.py     # optional clearly separated variant
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── em_f1.py              # normalization + EM/F1
│   │   ├── outcome.py            # cross-verification labeling
│   │   ├── metrics.py            # per-policy aggregation, P/R/F1, timings
│   │   └── charts.py             # matplotlib charts for runs
│   ├── pipeline.py               # single-question runtime (adaptive-only or cross)
│   ├── experiment.py             # batch runner, four policies, artifacts
│   ├── cost.py                   # estimated cost model
│   └── serialization.py          # run persistence (json/jsonl, safe writes)
├── api/
│   └── main.py                   # FastAPI app (dashboard + /ask + /evaluate)
├── dashboard/
│   └── index.html                # single-page research dashboard
├── scripts/
│   ├── download_sample.py        # reproducible dataset sample downloader
│   ├── build_corpus.py           # corpus builder (sample → corpus + manifest)
│   ├── train_decision.py         # train/tune decision engine (train/val; NO test)
│   ├── run_experiment.py         # batch experiment: four policies + artifacts
│   └── run_api.py                # start dashboard server
├── tests/                        # pytest suite mapped in §9
├── data/                         # downloaded sample JSONL, corpus (gitignored by default)
├── runs/                         # run artifacts (gitignored by default)
└── reports/
    ├── REPORT_TEMPLATE.md        # research report template
    └── LIMITATIONS.md            # candid limitations & potential contributions
```

## 12. Stage plan (implemented in order)

| Stage | Deliverable |
|---|---|
| ✅ 1 | This plan, environment findings, architecture |
| 2 | Dataset interface, corpus, BM25 retriever, both answering methods (+ tests) |
| 3 | Decision engine (heuristic + learned), pre-retrieval features (+ leakage tests) |
| 4 | Cross-verification, outcome labeling, four-policy experiment pipeline (+ tests) |
| 5 | Timing/cost/metrics hardening, persistence, charts, reproducible batch runs |
| 6 | Dashboard, README, REPORT_TEMPLATE, LIMITATIONS, full test pass |

## 13. Limitations to keep visible from Day 1 (also in LIMITATIONS.md)

1. **The finite answer engine is not an LLM.** It bounds Method-A quality by
   design (it abstains unless it can answer, honestly). All "parametric
   knowledge" claims in the final report will be framed for this engine
   specifically; running the same pipeline with the OpenAI-compatible backend
   is the intended follow-up experiment and requires only configuration.
2. **Corpus is small and HotpotQA-coupled.** Real-world open-domain corpora can
   be orders of magnitude larger; BM25/dense results and latencies are not
   transferable claims.
3. **Oracle labeling thresholds** (Δ = 0.10) are a modeling choice, not truth;
   sensitivity analysis will be part of Stage 5.
4. **Weak evaluation of open-ended answers.** EM/F1 are robust for HotpotQA's
   short-answer style but blind to paraphrases; no LLM judge is used by default
   (rubric criteria documented if one is needed).
5. **Cost model** is estimated, not measured in USD, unless a hosted backend is
   configured.
6. **Single-run statistics.** With deterministic backends, repeat variance is
   zero; with hosted backends, repeats matter and are supported by config.
