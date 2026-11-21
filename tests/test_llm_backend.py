"""Tests for the real LLM answering backend — all HTTP is mocked."""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import adaptive_qa.answering.llm_backend as L
from adaptive_qa.answering.llm_backend import (
    AnswerBackendError,
    LLMBackend,
    ModelNotConfiguredError,
)
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
    # always point resolve_chat_kwargs away from any real project .env
    monkeypatch.setattr(L, "PROJECT_ROOT", Path("/nonexistent-env-root"))
    yield


def test_unconfigured_raises_clear_error():
    backend = LLMBackend({}, grounded=False)
    with pytest.raises(ModelNotConfiguredError) as ei:
        backend.answer("question?")
    assert ".env" in str(ei.value)
    assert "LLM_API_KEY" in str(ei.value)


def test_partial_config_raises(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    with pytest.raises(ModelNotConfiguredError):
        LLMBackend({}, grounded=False).answer("q")
    # specifically names the still-missing pieces
    with pytest.raises(ModelNotConfiguredError) as ei:
        LLMBackend({}, grounded=False).answer("q")
    assert "model" in str(ei.value)


def test_usage_provider_reported():
    backend = LLMBackend({}, grounded=False)
    usage = backend.usage_dict("in text", "out", {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
    assert usage == {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15,
                     "source": "provider-reported"}


def test_usage_estimated_when_missing():
    backend = LLMBackend({}, grounded=False)
    usage = backend.usage_dict("five tokens here approximately now", "answer text", None)
    assert usage["source"] == "estimated"
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["total_tokens"] > 0


def test_successful_call_tokens_and_question_flow(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    captured = {}

    def fake_post(url, headers=None, timeout=None, data=None):
        captured["url"] = url
        captured["body"] = json.loads(data)
        captured["auth"] = headers["Authorization"]
        return FakeResponse(payload={
            "choices": [{"message": {"content": "Paris is the capital of France."}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
        })

    monkeypatch.setattr("requests.post", fake_post)
    backend = LLMBackend({}, grounded=False)
    ans = backend.answer("What is the capital of France?")
    assert ans.text == "Paris is the capital of France."
    assert ans.abstained is False
    assert ans.meta["tokens"]["source"] == "provider-reported"
    assert ans.meta["model"] == "test-model"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    # the question reached the model; no passage context present in messages
    body = captured["body"]
    assert "What is the capital of France?" in body["messages"][-1]["content"]
    assert "Passages" not in body["messages"][-1]["content"]


def test_grounded_call_includes_passages_once(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    captured = {}

    def fake_post(url, headers=None, timeout=None, data=None):
        captured["body"] = json.loads(data)
        return FakeResponse(payload={"choices": [{"message": {"content": "The doc states 42."}}]}, status_code=200)

    monkeypatch.setattr("requests.post", fake_post)
    hits = [RetrievalHit(doc_id="c1", title="file.txt (page 2)", text="Passage content", score=3.2)]
    backend = LLMBackend({}, grounded=True)
    ans = backend.answer("What does the doc state?", hits)
    body = captured["body"]
    assert "[1]" in body["messages"][-1]["content"]
    assert "Passage content" in body["messages"][-1]["content"]
    assert ans.meta["retrieved_passage_count"] == 1
    assert ans.meta["retrieved_passages_included_in_input_tokens"] is True
    # retrieved chunks are counted inside B tokens exactly once (as part of input)
    assert ans.meta["tokens"]["source"] in ("provider-reported", "estimated")


def test_http_error_raises_not_fabricates(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResponse(status_code=500, text="boom"))
    with pytest.raises(AnswerBackendError):
        LLMBackend({}, grounded=False).answer("q")


def test_timeout_raises_clean_message(monkeypatch):
    import requests as _requests

    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://fake/v1")
    monkeypatch.setenv("LLM_MODEL", "m")

    def slow(*a, **k):
        raise _requests.Timeout("timed out")

    monkeypatch.setattr("requests.post", slow)
    with pytest.raises(AnswerBackendError) as ei:
        LLMBackend({}, grounded=False).answer("q")
    assert "timed out" in str(ei.value)


def test_env_file_loading_without_overriding(monkeypatch, tmp_path):
    # a real .env provides values; existing exported env wins
    env = tmp_path / ".env"
    env.write_text('LLM_API_KEY="key-from-file"\nLLM_MODEL=m-file\n', encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "m-exported")
    args = L.resolve_chat_kwargs(env)
    assert args["api_key"] == "key-from-file"     # from file
    assert args["model"] == "m-exported"          # exported env takes precedence
