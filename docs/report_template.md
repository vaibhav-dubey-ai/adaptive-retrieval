# Research Report — Should the AI Search Before Answering?

> Fill this in from `runs/<timestamp>/summary.json` and `charts/*.png`.
> Every number below must come from a real run — never invent values.

## 1. Abstract
(3–5 sentences.) State the research question, the system (deterministic QA
pipeline + pre-retrieval decision engine), the dataset, and the headline
result from the four-policy table with the actual numbers from your final run.

## 2. Method

### 2.1 Two answering methods
- Method A (no retrieval): closed-book answering over a simulated parametric
  memory built from HotpotQA *train*-split gold paragraphs (disjoint from the
  evaluation questions).
- Method B (with retrieval): BM25 over the local corpus (eval-question
  distractor paragraphs, HotpotQA distractor setting) + rank-fusion extraction.

### 2.2 Adaptive decision engine
Pre-retrieval policy: features from the question text alone (question type,
wh-word prefix, length, token entropy, `either/or` option-flag) plus a cheap
lexical familiarity probe against the simulated parametric memory
(`familiarity_probe`: best coverage, titled hits, title IDF). Two policies:
1. Heuristic interpretable baseline (thresholds tuned on validation).
2. Learned: scikit-learn `LogisticRegression` on `(train rows, train labels)`,
   threshold tuned on validation only, evaluated once on test.

_No reference answer / gold titles / outcome labels / retrieved evidence reach
the decision engine at inference time._ Enforced by `tests/test_leakage.py`.

### 2.3 Two workflows
- `run_adaptive`: executes decision + selected path only (the deployed system).
- `run_cross_verify`: executes both paths per question (evaluation harness).
  All latency claims for the deployed system come from the adaptive path-only
  timings; cross-verification totals are reported separately.

### 2.4 Oracle labeling (offline only)
`quality = 0.5·EM + 0.5·F1` vs reference answer. Gap threshold 0.10 →
`retrieval_better` / `no_retrieval_better` / `similar`;
abstention or missing reference → `ambiguous`. Ambiguous rows are excluded
from binary decision metrics and reported as `n_excluded_ambiguous`.

## 3. Dataset
Reproduce: `python scripts/build_dataset.py --force`.
- Source: HotpotQA `distractor` via HF datasets-server, seeded offset
  (`eval_sample_offset_seed=101`), `n=200`, `easy`-level filtered
  (level-ordered split; offset bounded to the easy-dense region — documented).
- Splits: question-level 40/30/30 train/validation/test, seeded shuffle,
  no question/answer/paragraph appears in two splits.
- Corpus: distractor paragraphs of eval questions (~2k docs).
- Parametric memory: train-split gold paragraphs → ~11k snippets.

## 4. Four-policy results (test split)
| policy | composite quality | EM | F1 | mean latency ms | median latency ms | retrieval rate | cost/question |
|---|---|---|---|---|---|---|---|
| never_retrieve | | | | | | 0 | |
| always_retrieve | | | | | | 1.0 | |
| adaptive | | | | | | | |
| oracle (upper bound) | | | | | | | |

(from `summary.json` policies — copy from the produced run)

### 4.1 Decision-engine quality vs oracle labels
- evaluable rows / excluded (ambiguous): _ / _
- decision accuracy vs evaluable oracle labels: _
- RETRIEVE-class precision / recall / F1: _ / _ / _
- beneficial retrievals identified / missed: _ / _
- unnecessary retrievals: _
- retrieval rate: _
- outcome distribution (incl. ambiguous): retrieval_better _ ·
  no_retrieval_better _ · similar _ · ambiguous _

### 4.2 Timing breakdown (validated monotonic clock, per stage)
- decision engine latency: mean _ ms · median _ ms
- retrieval latency: mean _ ms · median _ ms
- generation Method A: mean _ ms
- generation Method B: mean _ ms
- deployed adaptive end-to-end: mean _ ms (avg of selected paths)
- cross-verification (both paths): mean _ ms — evaluation-only
- evaluation time (oracle + metrics): mean _ ms — never mixed with deployed latency

### 4.3 Cost
- retrieval cost model: `adaptive_qa/cost.py` (retrievals × cost-per-retrieval + local compute time).
- estimated cost per question for the four policies + cost saved by adaptive against always-retrieve: _

## 5. Analysis
- When does the engine choose correctly, conditionally constant across
  question types (fluffy descriptions, option-questions, numeric/where/party)?
- Where does it fail? Diagnostic: analyzed missed-beneficial-retrieval rows
  (outcome=retrieval_better, decision=DO_NOT_RETRIEVE): _ most-missed pattern.
- Trade-off statements (charts\*.png): quality-vs-latency curve, precision
  points, cost saved vs quality sacrificed.

## 6. Limitations
- Closed-book "LLM" is a deterministic extraction engine + snippet store,
  not a neural language model. Absolute-quality numbers do not transfer; the
  design does.
- Compression to `easy` level makes the question mix answerable; `hard`
  rows would favor `always_retrieve` more.
- EM/F1 cannot reward paraphrase; ambiguous rows treated separately; LLM
  judge (optional, fixed rubric) would be weak labels — not used by default.
- Latency distributions are conditional on the local CPU (BM25 extractive);
  network LLM timing variance absent; repeats not needed beyond documented
  confidence CIs from bootstrap (optional — see `metrics.stdev`).

## 7. Contributions
1. Question-level pre-retrieval adaptive pipeline with leakage isolation at
   import/AST/test level.
2. Four-policy comparison with structurally distinct deployed-vs-evaluation
   workflows.
3. Reusable "simulated parametric memory" methodology for closed-book
   emulation on a single machine.
4. All artifact per-question JSONL + summary + charts written by
   `scripts/run_experiment.py`, ending point of reproducibility.
