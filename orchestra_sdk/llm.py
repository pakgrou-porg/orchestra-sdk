"""
orchestra_sdk.llm
==================
Async LLM client with structured output (Pydantic) and retry logic.
Supports OpenRouter, Anthropic, OpenAI, and local LMStudio/Ollama endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMError(Exception):
    pass


class StructuredOutputError(LLMError):
    """Raised when structured output fails after all retries."""
    pass


class LLMRateLimitError(LLMError):
    pass


# ---------------------------------------------------------------------------
# Message type
# ---------------------------------------------------------------------------


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Async LLM client using httpx.
    Supports all OpenAI-compatible endpoints plus Anthropic.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.get_api_key()
        self._base_url = config.get_base_url()

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.provider == "anthropic":
            headers["x-api-key"] = self._api_key or ""
            headers["anthropic-version"] = "2023-06-01"
        elif self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self.config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://orchestra.ai"
            headers["X-Title"] = "Orchestra Conductor"
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _get_endpoint(self) -> str:
        if self.config.provider == "anthropic":
            return f"{self._base_url}/messages"
        return f"{self._base_url}/chat/completions"

    def _parse_response(self, data: dict) -> str:
        """Extract the assistant message content from the API response."""
        if self.config.provider == "anthropic":
            # Anthropic format: {"content": [{"type": "text", "text": "..."}]}
            content = data.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return ""
        else:
            # OpenAI format
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    async def chat(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a chat completion request. Returns the assistant message content.
        Raises LLMError on failure.
        """
        payload = self._build_payload(messages, max_tokens, temperature)
        headers = self._build_headers()
        endpoint = self._get_endpoint()

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

                if response.status_code == 429:
                    raise LLMRateLimitError(
                        f"Rate limited by {self.config.provider}. "
                        "Wait before retrying."
                    )

                if response.status_code >= 400:
                    raise LLMError(
                        f"LLM API error {response.status_code}: {response.text[:500]}"
                    )

                data = response.json()
                content = self._parse_response(data)
                elapsed = time.time() - start
                logger.debug(
                    f"[LLMClient] {self.config.model} responded in {elapsed:.2f}s "
                    f"({len(content)} chars)"
                )
                return content

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise LLMError(
                f"Cannot connect to {self.config.provider} at {self._base_url}: {e}"
            ) from e

    async def structured_output(
        self,
        messages: list[Message],
        schema: Type[T],
        max_retries: int = 2,
        temperature: Optional[float] = None,
    ) -> T:
        """
        Chat completion with Pydantic schema enforcement.
        Instructs the model to output JSON, then validates against the schema.
        Retries up to max_retries times on parse/validation failure.
        Raises StructuredOutputError after all retries are exhausted.
        """
        # Add JSON instruction to the last user message
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        json_instruction = Message.user(
            f"Respond with ONLY valid JSON matching this schema. No prose, no markdown, no code blocks:\n{schema_json}"
        )

        augmented_messages = messages + [json_instruction]

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                raw = await self.chat(
                    augmented_messages,
                    temperature=temperature if temperature is not None else 0.1,
                )

                # Strip markdown code blocks if present
                raw = raw.strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

                parsed = json.loads(raw)
                return schema.model_validate(parsed)

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"[LLMClient] structured_output attempt {attempt + 1}/{max_retries + 1}: "
                    f"JSON parse error: {e}. Raw: {raw[:200]!r}"
                )
                if attempt < max_retries:
                    # Add correction message
                    augmented_messages = augmented_messages + [
                        Message.assistant(raw),
                        Message.user(
                            f"That response was not valid JSON. Error: {e}. "
                            "Try again with ONLY valid JSON, no other text."
                        ),
                    ]

            except ValidationError as e:
                last_error = e
                logger.warning(
                    f"[LLMClient] structured_output attempt {attempt + 1}/{max_retries + 1}: "
                    f"Schema validation error: {e}"
                )
                if attempt < max_retries:
                    augmented_messages = augmented_messages + [
                        Message.assistant(raw),
                        Message.user(
                            f"The JSON did not match the required schema. "
                            f"Validation errors: {e}. "
                            "Try again with corrected JSON."
                        ),
                    ]

        raise StructuredOutputError(
            f"Failed to get valid structured output after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
