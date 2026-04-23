"""Simple unified LLM client with safe fallback behavior."""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency during bootstrap
    requests = None  # type: ignore[assignment]

from project.config import get_config
from project.agent.nodes.runtime_metrics import record_llm_usage

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_config(cls) -> "LLMClient":
        """Create a client from environment-backed config."""

        config = get_config()
        return cls(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            timeout=config.llm_timeout,
        )

    @classmethod
    def from_memory_config(cls) -> "LLMClient":
        """Create a client for memory extraction.

        MEMORY_LLM_* settings can fully override the default LLM settings. If
        they are absent, config falls back to LLM_API_KEY/LLM_BASE_URL/LLM_MODEL.
        """

        config = get_config()
        return cls(
            api_key=config.memory_llm_api_key,
            base_url=config.memory_llm_base_url,
            model=config.memory_llm_model,
            timeout=config.memory_llm_timeout,
        )

    @property
    def is_configured(self) -> bool:
        """Whether the client has enough configuration for real calls."""

        return bool(self.api_key and self.base_url and self.model and requests is not None)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str | None:
        """Call a chat-completions style API and return text content.

        Returns None on any configuration, request, or parsing failure so callers
        can safely fall back to deterministic logic.
        """

        if not self.is_configured:
            return None

        url = self._build_chat_url()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        started_at = time.monotonic()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            logger.warning("LLM request failed, fallback enabled: %s", exc)
            return None

        text = self._extract_text(data)
        self._record_usage(
            data=data,
            model=model or self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=text or "",
            latency_ms=int((time.monotonic() - started_at) * 1000),
        )
        return text

    def _build_chat_url(self) -> str:
        """Build a flexible OpenAI-compatible chat-completions URL."""

        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str | None:
        """Extract content from common chat completion response shapes."""

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                text = first_choice.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        return None

    @staticmethod
    def _record_usage(
        *,
        data: dict[str, Any],
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_text: str,
        latency_ms: int,
    ) -> None:
        usage = data.get("usage") if isinstance(data, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or _estimate_tokens(f"{system_prompt}\n{user_prompt}")
        )
        completion_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or _estimate_tokens(response_text)
        )
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        record_llm_usage(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
            }
        )


def _estimate_tokens(text: str) -> int:
    """Cheap multilingual token estimate used when provider usage is absent."""

    if not text:
        return 0
    return max(1, len(text) // 4)
