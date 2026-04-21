"""Simple unified LLM client with safe fallback behavior."""

from __future__ import annotations

import logging
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency during bootstrap
    requests = None  # type: ignore[assignment]

from project.config import get_config

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

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            logger.warning("LLM request failed, fallback enabled: %s", exc)
            return None

        return self._extract_text(data)

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
