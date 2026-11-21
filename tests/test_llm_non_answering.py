"""Regression tests for non-answering provider replies (all HTTP mocked).

Reproduces the live bug: an OpenAI-compatible routing alias serving a
specialist model whose ``content`` is a classification (observed:
``nvidia/nemotron-3.5-content-safety:free`` returning the literal string
``"User Safety: safe"``) instead of a natural-language answer, and a reasoning
model that spends the whole token budget on chain-of-thought (``content``
null). The backend must treat both as a failed call with a clear error, never
as the grounded answer.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import adaptive_qa.answering.llm_backend as L
from adaptive_qa.answering.llm_backend import AnswerBackendError, LLMBackend
from adaptive_qa.common import RetrievalHit


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
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    monkeypatch.setattr(L, "PROJECT_ROOT", Path("/nonexistent-env-root"))
    # keep retries fast and silent in tests
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    yield


def _hits():
    return [RetrievalHit(doc_id="c1", title="ch6.pdf (page 19)",
                         text="Representing complex attributes in an ER diagram.",
                         score=7.5)]


# ---------------------------------------------------------- classification
def test_safety_classification_not_shown_as_answer(monkeypatch):
    """LIVE BUG: a content-safety model returned 'User Safety: safe' as the
    grounded answer. It must raise a clear error, not be displayed."""
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        # every attempt returns the same classification
        return FakeResponse(payload={
            "model": "nvidia/nemotron-3.5-content-safety:free",
            "choices": [{"message": {"content": "User Safety: safe"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        })

    monkeypatch.setattr("requests.post", fake_post)
    backend = LLMBackend({"max_retries": 1}, grounded=True)
    with pytest.raises(AnswerBackendError) as ei:
        backend.answer("explain about complex attributes in dbms", _hits())
    msg = str(ei.value)
    # clear, diagnostic error; the classification is never the answer
    assert "usable answer" in msg
    assert "nvidia/nemotron-3.5-content-safety:free" in msg
    assert "User Safety" in msg
    # retried once (initial + 1 retry)
    assert len(calls) == 2


def test_classification_rejected_by_helper():
    # whole-string, conservative: only genuine markers are rejected
    assert LLMBackend.is_non_answering_text("User Safety: safe") is True
    assert LLMBackend.is_non_answering_text("Safety: unsafe") is True
    assert LLMBackend.is_non_answering_text("safe") is True
    assert LLMBackend.is_non_answering_text("  ") is True
    # a real answer that merely mentions safety is NOT rejected
    assert LLMBackend.is_non_answering_text(
        "A complex attribute is split into component attributes for clarity [1]."
    ) is False
    assert LLMBackend.is_non_answering_text(
        "This is safe because the ER diagram models entities and their "
        "relationships, which is a safe and standard design practice."
    ) is False


def test_classification_then_success_retries(monkeypatch):
    """First attempt is a classification; the retry returns a real answer."""
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return FakeResponse(payload={"model": "safety-model",
                                         "choices": [{"message": {"content": "User Safety: safe"}}]})
        return FakeResponse(payload={"model": "good-model",
                                     "choices": [{"message": {"content": "Complex attributes have components."}}],
                                     "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14}})

    monkeypatch.setattr("requests.post", fake_post)
    ans = LLMBackend({"max_retries": 2}, grounded=True).answer("q", _hits())
    assert ans.text == "Complex attributes have components."
    assert ans.meta["tokens"]["source"] == "provider-reported"
    assert len(calls) == 2


# ----------------------------------------------------------- reasoning-only
def test_reasoning_only_empty_content_raises(monkeypatch):
    """LIVE BUG: reasoning model burned the whole budget on chain-of-thought,
    leaving content null (finish_reason=length). Must raise, not show
    chain-of-thought as the answer."""
    payload = {
        "model": "nvidia/nemotron-3.5-lightning:free",
        "choices": [{"finish_reason": "length",
                     "message": {"content": None,
                                 "reasoning": "Here's a thinking process: ..."}}],
        "usage": {"prompt_tokens": 1047, "completion_tokens": 1024, "total_tokens": 2071},
    }

    def fake_post(*a, **k):
        return FakeResponse(payload=payload)

    monkeypatch.setattr("requests.post", fake_post)
    with pytest.raises(AnswerBackendError) as ei:
        LLMBackend({"max_retries": 0}, grounded=True).answer("q", _hits())
    assert "usable answer" in str(ei.value)


def test_content_preferred_over_reasoning_when_both_present():
    """When both content and reasoning exist, content (the answer) wins."""
    data = {"choices": [{"message": {"content": "The answer text.",
                                     "reasoning": "internal chain of thought"}}]}
    assert LLMBackend._extract_text(data) == "The answer text."


# ------------------------------------------------------- request/grounding
def test_max_tokens_sent_for_reasoning_headroom(monkeypatch):
    """The request must carry a generous max_tokens so reasoning models have
    budget left to emit an answer after their chain-of-thought."""
    captured = {}

    def fake_post(url, headers=None, timeout=None, data=None):
        captured["body"] = json.loads(data)
        return FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    LLMBackend({"max_tokens": 2048}, grounded=True).answer("q", _hits())
    assert captured["body"]["max_tokens"] == 2048


def test_grounded_still_passes_passages_and_page_titles(monkeypatch):
    """Option B must still receive retrieved passages with page attribution."""
    captured = {}

    def fake_post(url, headers=None, timeout=None, data=None):
        captured["body"] = json.loads(data)
        return FakeResponse(payload={"choices": [{"message": {"content": "answer [1]"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    backend = LLMBackend({}, grounded=True)
    backend.answer("explain complex attributes", _hits())
    user = captured["body"]["messages"][-1]["content"]
    assert "Passages:" in user
    assert "ch6.pdf (page 19)" in user
    assert "Representing complex attributes" in user
    assert "explain complex attributes" in user


def test_closed_book_still_excludes_passages(monkeypatch):
    """Option A must never receive document passages."""
    captured = {}

    def fake_post(url, headers=None, timeout=None, data=None):
        captured["body"] = json.loads(data)
        return FakeResponse(payload={"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr("requests.post", fake_post)
    LLMBackend({}, grounded=False).answer("explain complex attributes", _hits())
    user = captured["body"]["messages"][-1]["content"]
    assert "Passages" not in user
    assert "page 19" not in user
    assert "explain complex attributes" in user
