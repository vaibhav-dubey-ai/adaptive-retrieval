"""Real model-backed answering backend (OpenAI-compatible chat API).

Replaces the deterministic extractors when configured: answers are generated
from a general-purpose LLM either closed-book (Option A, no retrieval) or
grounded in retrieved passages (Option B). Configuration comes from
environment variables ONLY at call time (never hardcoded in code or
committed config):

- ``LLM_API_KEY``   (required; falls back to ``OPENAI_API_KEY``)
- ``LLM_BASE_URL``  (falls back to ``OPENAI_BASE_URL``, then config value)
- ``LLM_MODEL``     (falls back to ``OPENAI_MODEL``, then config value)

Transparency contract:
- If the provider returns ``usage`` those token counts are recorded as
  provider-reported. Otherwise they are ESTIMATED from text length and
  clearly labeled ``source: "estimated"``.
- A missing key/timeout/HTTP error raises ``AnswerBackendError`` with an
  actionable message. There is NO silent fallback to the deterministic
  span extractor.
- The deterministic backends remain available and clearly labeled as
  reproducible baselines.

The uploaded/active document collection is NOT accessible to the closed-book
backend: it receives only the question.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Sequence

import requests

from ..common import PROJECT_ROOT
from ..common import GeneratedAnswer, RetrievalHit
from . import AnsweringMethod

# Ordered env fallbacks: generic names first, then legacy OpenAI-style names.
ENV_KEYS = {
    "api_key": ["LLM_API_KEY", "OPENAI_API_KEY"],
    "base_url": ["LLM_BASE_URL", "OPENAI_BASE_URL"],
    "model": ["LLM_MODEL", "OPENAI_MODEL"],
}


class AnswerBackendError(RuntimeError):
    """A model-backed answer call failed (configuration, network, or API)."""


class ModelNotConfiguredError(AnswerBackendError):
    """The answering model is not configured (missing key/dev base URL)."""


def resolve_chat_kwargs(path: Optional[Path] = None) -> Dict:
    """Load current model configuration from real environment variables.

    Reads the project ``.env`` into ``os.environ`` first (without overriding
    values the user already exported), so that setting keys in ``.env`` is
    sufficient without extra tooling. Values are read from the environment
    only; they are never returned, logged, or written anywhere.
    """
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    out = {}
    for name, candidates in ENV_KEYS.items():
        for candidate in candidates:
            val = os.environ.get(candidate)
            if val:
                out[name] = val
                break
        else:
            out[name] = None
    return out


def configured(uri: Optional[Path] = None) -> bool:
    args = resolve_chat_kwargs(uri)
    return bool(args.get("api_key") and args.get("model") and args.get("base_url"))


class LLMBackend(AnsweringMethod):
    """General-purpose LLM answer generation, closed-book or grounded.

    ``grounded=True`` requires the caller to pass retrieved passages; the
    backend counts their tokens as part of Option B's model input (once).
    """

    grounded: bool = False

    def __init__(self, cfg: dict, grounded: bool, name: Optional[str] = None) -> None:
        self.cfg = cfg or {}
        self.grounded = bool(grounded)
        self.model_config = self.cfg.get("model_config", {}) or {}
        self.temperature = float(self.cfg.get("temperature", 0.2))
        self.max_tokens = int(self.cfg.get("max_tokens", 512))
        self.timeout_s = float(self.cfg.get("timeout_s", 90))
        # Retries for TRANSIENT provider failures only (timeouts, connection
        # drops, 429/5xx e.g. "model currently experiencing high demand").
        # Backoff is exponential; after the last attempt the accumulated
        # provider error is surfaced verbatim so the user sees the cause.
        self.max_retries = max(0, int(self.cfg.get("max_retries", 3)))
        self.retry_backoff_s = float(self.cfg.get("retry_backoff_s", 2.0))
        # Local-model model reuse (verified on Ollama 0.40.10, qwen3:4b-instruct):
        # a chat-completion with ``keep_alive`` keeps the model resident after
        # the call, so the NEXT call skips the one-off model load (measured:
        # cold 23.8 s vs warm 7.2 s for the same request). ``None``/empty
        # disables the field entirely (cloud providers reject unknown fields).
        self.keep_alive = self.cfg.get("keep_alive", "30m")
        # Per-instance cold/warm marker: the FIRST call an instance makes may
        # pay provider-side model loading; later calls are warmed. Recorded in
        # the answer meta so latency comparisons can label the bias.
        self._warmed = False
        self.retry_statuses = {429, 500, 502, 503, 504}
        self.system_prompt = self.cfg.get(
            "grounded_system_prompt" if grounded else "closedbook_system_prompt", ""
        )
        self.user_prefix = self.cfg.get("user_prefix_" + ("grounded" if grounded else "closedbook"), "")
        self.name = name or ("llm_grounded" if grounded else "llm_closedbook")

    # -------------------------------------------------------------- prompts
    def _system(self) -> str:
        return (
            self.system_prompt
            or (
                "Answer the user's question using your own knowledge. "
                "You have NOT been given any documents; do not pretend to have read any source. "
                "Answer substantively and briefly (2-4 sentences). If genuinely unknown, say so plainly. "
                "Output ONLY the final answer text. Do NOT include any internal reasoning, "
                "planning, restatement of the question, or phrases like 'The user is asking' "
                "or 'I need to'. Start directly with the answer itself."
            )
            if not self.grounded
            else (
                "Answer the user's question PREFERABLY using the provided document passages. "
                "Prioritize evidence from them; cite which passage(s) [n] support the answer. "
                "If passages are insufficient, supplement with general knowledge and CLEARLY MARK "
                "that part as not document-supported. Do not fabricate document content. "
                "Answer substantively and briefly (2-5 sentences). "
                "Output ONLY the final answer text. Do NOT include any internal reasoning, "
                "planning, restatement of the question, or phrases like 'The user is asking' "
                "or 'I need to'. Start directly with the answer itself."
            )
        )

    # ----------------------------------------------------------- tokenizing
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (≈ lexical tokens + punctuation); labeled estimated."""
        return len(re.findall(r"\S+", text or ""))

    # ------------------------------------------------------------ response parsing
    @staticmethod
    def _extract_text(data: dict) -> str:
        """Pull the visible assistant answer out of a chat-completion payload.

        Handles the documented OpenAI shape (``choices[0].message.content``)
        plus the variations actually seen on OpenAI-compatible routers
        (OpenRouter) and reasoning models:

        - ``message.content`` may be ``null``/empty while the answer sits in
          ``message.reasoning`` (reasoning models that expose only the
          chain-of-thought) or in ``message.reasoning_content``.
        - ``choices[0].text`` (legacy completions-style) is accepted as a
          last resort.

        Returns a stripped string, possibly empty; callers decide whether an
        empty result is fatal. Never returns None.
        """
        choices = data.get("choices") or []
        if not choices:
            return ""
        ch = choices[0] or {}
        msg = ch.get("message") or {}
        # Prefer the visible answer field. Only fall back to reasoning when
        # content is genuinely absent — reasoning is chain-of-thought, not the
        # answer, so it is a last resort (callers may still reject it).
        for key in ("content", "reasoning", "reasoning_content"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # legacy text-completion style
        txt = ch.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        return ""

    @staticmethod
    def is_non_answering_text(text: str) -> bool:
        """True when the model's reply is a classification/marker, not an answer.

        Some OpenAI-compatible routers route to specialist models that return a
        label instead of prose (observed live: ``nvidia/nemotron-3.5-content-
        safety:free`` returns the literal string ``"User Safety: safe"``). Such
        output is never a usable answer, so it is treated like an empty reply:
        retried, then surfaced as a clear provider error rather than shown as
        the grounded answer. Deliberately conservative (whole-string, short)
        so real answers that merely mention safety are not rejected.
        """
        t = (text or "").strip()
        if not t:
            return True
        if len(t) > 120:
            return False
        lowered = t.lower()
        return (
            lowered.startswith("user safety")
            or lowered.startswith("safety:")
            or "safety classification" in lowered
            or lowered in {"safe", "unsafe", "unsafe content", "no answer", "n/a"}
        )

    @staticmethod
    def _finish_reason(data: dict) -> Optional[str]:
        choices = data.get("choices") or []
        if not choices:
            return None
        return (choices[0] or {}).get("finish_reason")

    @staticmethod
    def _routed_model(data: dict) -> Optional[str]:
        """The concrete model that served the call, when the provider reports it.

        OpenRouter routes aliases like ``openrouter/free`` to a concrete model
        and returns it here; useful for reproducibility/transparency.
        """
        val = data.get("model")
        return val if isinstance(val, str) and val else None

    def usage_dict(
        self,
        input_text: str,
        output_text: str,
        provider_usage: Optional[dict],
    ) -> Dict:
        if provider_usage and provider_usage.get("total_tokens") is not None:
            return dict(
                input_tokens=int(provider_usage.get("prompt_tokens") or 0),
                output_tokens=int(provider_usage.get("completion_tokens") or 0),
                total_tokens=int(provider_usage.get("total_tokens") or 0),
                source="provider-reported",
            )
        i = self.estimate_tokens(input_text)
        o = self.estimate_tokens(output_text)
        return {
            "input_tokens": i,
            "output_tokens": o,
            "total_tokens": i + o,
            "source": "estimated",
        }

    # ----------------------------------------------------------------- call
    def _post_chat(self, model_id: str, base_url: str, api_key: str, user_msg: str) -> tuple:
        """One chat-completion request WITH transient-error retries.

        Retried: timeouts, connection errors, HTTP 429/5xx (e.g. Gemini's
        503 "model currently experiencing high demand"), and EMPTY responses
        (some OpenAI-compatible routers / reasoning models intermittently
        return HTTP 200 with no usable answer text). Backoff grows
        exponentially; the final provider error is surfaced verbatim.

        Returns ``(usage, text, routed_model, finish_reason)``.
        """
        url = base_url.rstrip("/") + "/chat/completions"
        attempts = 1 + self.max_retries
        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout_s,
                    data=json.dumps(
                        {
                            "model": model_id,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "messages": [
                                {"role": "system", "content": self._system()},
                                {"role": "user", "content": user_msg},
                            ],
                            **({"keep_alive": self.keep_alive} if self.keep_alive else {}),
                        }
                    ),
                )
            except requests.Timeout as exc:
                last_error = f"timed out after {self.timeout_s}s"
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise AnswerBackendError(
                    f"Answering model timed out after {self.timeout_s}s "
                    f"(retried {self.max_retries}x): {exc}"
                ) from exc
            except requests.ConnectionError as exc:
                last_error = "connection error"
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise AnswerBackendError(
                    f"Cannot reach the answering model at {base_url} "
                    f"(retried {self.max_retries}x): {exc}"
                ) from exc

            if resp.status_code in self.retry_statuses:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise AnswerBackendError(
                    f"Provider unavailable after {attempts} attempts "
                    f"(retried {self.max_retries}x with backoff): {last_error}"
                )
            if resp.status_code >= 400:
                raise AnswerBackendError(
                    f"Provider returned HTTP {resp.status_code}: {resp.text[:200]}"
                )

            # --- parse a 2xx response into an answer -------------------------
            try:
                data = resp.json()
            except ValueError as exc:
                # Non-JSON body on a 2xx — treat as transient malformed reply.
                last_error = f"non-JSON response body: {resp.text[:160]}"
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise AnswerBackendError(
                    f"Provider returned a non-JSON body after {attempts} attempts: {last_error}"
                ) from exc

            text = self._extract_text(data)
            routed = self._routed_model(data)
            finish = self._finish_reason(data)

            # Reject empty, reasoning-only, AND classification/marker replies
            # (e.g. a content-safety model returning "User Safety: safe"):
            # retry, then surface a clear error. Never fabricate a fallback
            # answer. A reasoning fallback is accepted only when it is a
            # substantive answer (a reasoning model whose chain-of-thought
            # never reached a final answer must not be shown as one).
            reasoning_only = (
                not (data.get("choices") or [{}])[0].get("message", {}).get("content")
            )
            if (
                not text
                or self.is_non_answering_text(text)
                or (reasoning_only and len(text.split()) < 10)
            ):
                detail = f"model={routed or model_id!r}, finish_reason={finish!r}"
                if reasoning_only and text:
                    detail += ", reply=chain-of-thought only (no final answer)"
                elif text:
                    detail += f", reply={text[:60]!r}"
                last_error = f"non-answering reply on HTTP 200 ({detail})"
                if attempt + 1 < attempts:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                raise AnswerBackendError(
                    f"Provider returned no usable answer after {attempts} attempts "
                    f"(retried {self.max_retries}x). The model either produced no answer "
                    f"text, only internal reasoning, or a non-answer classification. "
                    f"Details: {detail}. This is typical when a routing alias (e.g. "
                    "openrouter/free) picks a specialist model that is not a general "
                    "assistant — pin a concrete, general-purpose model id in .env "
                    "(LLM_MODEL)."
                )

            return (data.get("usage") or None), text, routed, finish
        # defensive: loop should always return or raise
        raise AnswerBackendError(f"Answering model call failed: {last_error}")

    def answer(self, question: str, passages: Optional[Sequence[RetrievalHit]] = None) -> GeneratedAnswer:
        args = resolve_chat_kwargs()
        missing = [k for k in ("api_key", "base_url", "model") if not args.get(k)]
        if missing:
            raise ModelNotConfiguredError(
                "Answering model is not configured. Missing: " + ", ".join(missing)
                + ". Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in the project .env "
                "(see .env.example). No fallback extractor is used when the model is required."
            )
        if self.grounded:
            if not passages:
                return GeneratedAnswer(
                    text="No passages were retrieved, so a grounded answer cannot be produced.",
                    abstained=True,
                    backend=self.name,
                    meta={"tokens": None, "retrieved_passage_count": 0},
                )
            payload = "\n\n".join(
                f"[{i + 1}] {h.title}\n{h.text}" for i, h in enumerate(passages)
            )
            user_msg = f"{self.user_prefix or ''}Passages:\n{payload}\n\nQuestion: {question}".strip()
        else:
            user_msg = f"{self.user_prefix or ''}Question: {question}".strip()

        try:
            usage, text, routed, finish = self._post_chat(
                args["model"], args["base_url"], args["api_key"], user_msg
            )
        except requests.Timeout as exc:
            raise AnswerBackendError(
                f"Answering model timed out after {self.timeout_s}s: {exc}"
            ) from exc
        except requests.ConnectionError as exc:
            raise AnswerBackendError(
                f"Cannot reach the answering model at {args['base_url']}: {exc}"
            ) from exc

        meta = {
            "model": args["model"],
            "temperature": self.temperature,
            "tokens": self.usage_dict(user_msg, text, usage),
            "grounded": self.grounded,
            # Cold/warm: the first call of this backend instance may include the
            # local model's one-off load or a cloud route's warm-up. Latency
            # reads must not treat that as typical generation time.
            "generation_cold": not self._warmed,
            "keep_alive": self.keep_alive or None,
        }
        self._warmed = True
        if routed and routed != args["model"]:
            # Transparency: when the provider routes an alias (e.g.
            # openrouter/free) to a concrete model, record which one actually
            # answered so runs are reproducible and the user can pin it.
            meta["routed_model"] = routed
        if finish:
            meta["finish_reason"] = finish
            if finish == "length":
                meta["truncated"] = True
                meta["truncation_note"] = (
                    "Answer hit the max_tokens limit and was truncated; raise "
                    "answering.llm.max_tokens for longer answers."
                )
        if self.grounded:
            meta["retrieved_passage_count"] = len(passages)
            meta["retrieved_passages_included_in_input_tokens"] = True
        return GeneratedAnswer(
            text=text,
            abstained=False,
            evidence_doc_ids=tuple(h.doc_id for h in (passages or ())),
            backend=self.name,
            meta=meta,
        )


def build_llm_backends(cfg: dict) -> tuple:
    """Return (closedbook_backend, grounded_backend) from config."""
    sub = (cfg or {}).get("answering", {}).get("llm", {}) or {}
    return LLMBackend(sub, grounded=False), LLMBackend(sub, grounded=True)


# ======================================================================
# Optional hosted backend for the HotpotQA experiment (config: openai_compatible).
# Reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment; off by default.
# ======================================================================

ABSTAIN_TEXT = "ABSTAIN"


class OpenAICompatBackend(AnsweringMethod):
    def __init__(self, grounded: bool, cfg: dict, name_prefix: str = "") -> None:
        self.grounded = bool(grounded)
        self.base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = cfg.get("model") or os.environ.get("OPENAI_MODEL") or ""
        self.system_prompt = cfg.get(
            "method_b_system_prompt" if grounded else "method_a_system_prompt",
            "",
        )
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 120))
        self.timeout_s = float(cfg.get("timeout_s", 60))
        self.name = (name_prefix or "openai_") + ("grounded" if grounded else "closedbook")

    def answer(self, question: str, passages: Optional[Sequence[RetrievalHit]] = None) -> GeneratedAnswer:
        if self.grounded:
            assert passages is not None and len(passages) > 0, "grounded backend requires passages"
            context = "\n\n".join(
                f"[{h.title}] {h.text}"[:1200] for h in passages
            )
            user = f"Passages:\n{context}\n\nQuestion: {question}"
        else:
            user = f"Question: {question}"
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set; hosted backend cannot run. Configure "
                "env or switch answering backends to the deterministic ones."
            )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        abstained = text == ABSTAIN_TEXT
        return GeneratedAnswer(
            text=("I don't know." if abstained else text),
            abstained=abstained,
            backend=self.name,
            meta={"usage": data.get("usage", {})},
        )
