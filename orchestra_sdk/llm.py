"""
orchestra_sdk.llm
==================
Async LLM client with structured output (Pydantic) and retry logic.
Supports OpenRouter, Anthropic, OpenAI, and local LMStudio/Ollama endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

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
# Base URL probe helper
# ---------------------------------------------------------------------------


def _probe_base_url(raw_url: str, timeout: float = 5.0) -> str:
    """
    Detect whether a local OpenAI-compatible server uses /v1 or /api/v1.

    Strategy:
      1. Strip any trailing /v1 or /api/v1 from the configured URL to get the
         server root (e.g. "http://10.116.2.145:1234").
      2. Try GET <root>/v1/models  — standard OpenAI path.
      3. If that fails (non-2xx or connection error), try GET <root>/api/v1/models
         — LM Studio ≥ 0.4.x path.
      4. Return the working base URL (including the /v1 or /api/v1 suffix).
      5. If both fail, return the original configured URL unchanged and log a
         warning so the caller can surface the error at request time.

    Only runs for local/custom providers (URLs that are not openrouter.ai,
    anthropic.com, or openai.com). Cloud providers are returned as-is.
    """
    # Don't probe cloud providers
    cloud_hosts = ("openrouter.ai", "anthropic.com", "openai.com")
    if any(h in raw_url for h in cloud_hosts):
        return raw_url

    # Strip known suffixes to get the server root
    root = raw_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break

    candidates = [
        f"{root}/v1",
        f"{root}/api/v1",
    ]

    for candidate in candidates:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{candidate}/models")
                if resp.status_code < 400:
                    if candidate != raw_url.rstrip("/"):
                        logger.info(
                            f"[LLMClient] base_url probe: '{raw_url}' resolved to "
                            f"'{candidate}' (tried {candidates})"
                        )
                    return candidate
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            continue

    logger.warning(
        f"[LLMClient] base_url probe: could not reach server at '{raw_url}'. "
        f"Tried: {candidates}. Proceeding with configured URL — errors will "
        "surface at request time."
    )
    return raw_url.rstrip("/")


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Async LLM client using httpx.
    Supports all OpenAI-compatible endpoints plus Anthropic.

    For local providers (lmstudio, custom, openai_compat), the constructor
    automatically probes the server to resolve whether it uses /v1 or /api/v1,
    so the conductor_config.yaml does not need to specify the exact path.
    """

    def __init__(self, config: LLMConfig, probe: bool = True):
        self.config = config
        self._api_key = config.get_api_key()
        self._raw_url = config.get_base_url()
        self._probe = probe and config.provider in ("lmstudio", "custom", "openai_compat")
        # _base_url is set eagerly for non-probe providers; probe providers defer
        # resolution to the first async call via _ensure_base_url() to avoid
        # blocking the event loop during construction.
        self._base_url: Optional[str] = None if self._probe else self._raw_url
        self._probe_lock = asyncio.Lock()

    async def _ensure_base_url(self) -> str:
        """Resolve base URL, running the synchronous probe in a thread executor
        so the event loop is never blocked during construction or first call."""
        if self._base_url is not None:
            return self._base_url
        async with self._probe_lock:
            # Double-check after acquiring lock
            if self._base_url is not None:
                return self._base_url
            from rich.console import Console as _Console
            _console = _Console()
            _console.print(
                f"  [dim]→ Probing local LLM server at {self._raw_url!r}...[/dim]"
            )
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                self._base_url = await loop.run_in_executor(
                    pool, _probe_base_url, self._raw_url
                )
            _console.print(
                f"  [dim]→ LLM endpoint resolved: {self._base_url!r}[/dim]"
            )
        return self._base_url

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

    async def _get_endpoint(self) -> str:
        base = await self._ensure_base_url()
        if self.config.provider == "anthropic":
            return f"{base}/messages"
        return f"{base}/chat/completions"

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
        endpoint = await self._get_endpoint()

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
                f"Cannot connect to {self.config.provider} at {self._raw_url}: {e}"
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

                # Strip markdown code fences if present.
                # Handles: ```json, ```python, ``` (bare), any language tag.
                raw = raw.strip()
                raw = re.sub(r'^```[a-zA-Z0-9_+\-]*\n?', '', raw)
                if raw.endswith('```'):
                    raw = raw[:-3]
                raw = raw.strip()

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
