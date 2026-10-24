"""Retry tests for transient provider failures (timeouts, connection drops, 429/5xx)."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import adaptive_qa.answering.llm_backend as L
from adaptive_qa.answering.llm_backend import AnswerBackendError, LLMBackend


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
              "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(L, "PROJECT_ROOT", Path("/nonexistent-env-root"))
    yield


def test_retries_503_then_succeeds(monkeypatch):
    """Gemini-style 503 'high demand' is transient; next attempt must succeed."""
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            return FakeResponse(
                status_code=503,
                text='[{"error": {"code": 503, "message": "This model is currently experiencing high demand.","status": "UNAVAILABLE"}}]',
            )
        return FakeResponse(payload={"choices": [{"message": {"content": "The answer."}}],
                                     "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})

    monkeypatch.setattr("requests.post", fake_post)
    imported_sleeps = []
    monkeypatch.setattr(L.time, "sleep", lambda s: imported_sleeps.append(s))
    backend = LLMBackend({}, grounded=False)
    ans = backend.answer("q?")
    assert ans.text == "The answer."
    assert len(calls) == 3
    # exponential backoff between attempts: 2s, 4s
    assert imported_sleeps == [2.0, 4.0]


def test_retries_persistent_503_raises_with_attempts(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(status_code=503, text='high demand')

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    backend = LLMBackend({"max_retries": 2}, grounded=False)
    with pytest.raises(AnswerBackendError) as ei:
        backend.answer("q?")
    assert "retried 2x" in str(ei.value)
    assert "high demand" in str(ei.value)
    # initial attempt + 2 retries
    assert len(calls) == 3


def test_retries_timeout_then_succeeds(monkeypatch):
    import requests as _r

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise _r.Timeout("read timed out")
        return FakeResponse(payload={"choices": [{"message": {"content": "late but fine"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    ans = LLMBackend({}, grounded=False).answer("q?")
    assert ans.text == "late but fine"
    assert len(calls) == 2


def test_non_transient_error_not_retried(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(status_code=401, text="bad key")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    with pytest.raises(AnswerBackendError) as ei:
        LLMBackend({}, grounded=False).answer("q?")
    assert len(calls) == 1 and "401" in str(ei.value)


def test_retry_counts_configurable_and_zero_disables(monkeypatch):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(status_code=503, text="down")

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    with pytest.raises(AnswerBackendError):
        LLMBackend({"max_retries": 0}, grounded=False).answer("q?")
    assert len(calls) == 1
