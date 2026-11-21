"""FastAPI backend for the research dashboard.

LD-1 boundaries (requirement §10):
- ``/ask`` answers with the ADAPTIVE workflow only (decision engine ->
  selected path). Reference answers and oracle labels are never exposed to
  the decision engine and never included in the payload of the live mode.
- ``/ask?mode=cross`` performs cross-verification: both paths run, retrieved
  passages and scores are shown, and - only when offline evaluation data is
  available for the question - the decision-correctness flag is attached and
  clearly marked as evaluation-only.
- ``/evaluate/run`` executes a batch of the four-policy experiment.
- ``/evaluate/summary|export`` expose aggregates and per-question CSV/JSON.

Tests call the modules directly; this HTTP layer is a thin, optional shell
(the CLI pipeline works without it)."""

from fastapi import FastAPI

from api import routes_ask, routes_documents, routes_results

app = FastAPI(title="Adaptive QA Research Dashboard", version="0.2.0")
for _m in (routes_ask, routes_documents, routes_results):
    app.include_router(_m.router)
