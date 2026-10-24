"""Optional hosted backend (OpenAI-compatible chat completions).

Reads key + base URL from environment (never from code/config snapshots):
- ``OPENAI_API_KEY`` (required)
- ``OPENAI_BASE_URL`` (optional; config value used as fallback)

Used when ``answering.method_a_backend`` / ``method_b_backend`` is set to
``openai_compatible``. OFF by default so the basic demonstration needs no
paid API.
"""

from __future__ import annotations

import os
import json
import urllib.request
from typing import Optional, Sequence

import requests

from ..types import GeneratedAnswer, RetrievalHit
from . import AnsweringMethod

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
