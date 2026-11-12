"""Resilient LLM client layer: Anthropic, OpenAI-compatible servers (OpenAI, Ollama, vLLM), and Mock."""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Callable, Protocol

import httpx

logger = logging.getLogger("whytrace.llm")

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


def _post_with_retry(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: float = 120.0,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
) -> dict:
    """Execute an HTTP POST with exponential backoff and jitter for rate-limits and transient 5xx errors."""
    backoff = initial_backoff
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 429 or 500 <= resp.status_code <= 504:
                    if attempt < max_retries:
                        sleep_time = backoff + random.uniform(0.1, 0.5)
                        logger.warning(
                            f"HTTP {resp.status_code} from {url}, retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(sleep_time)
                        backoff *= 2.0
                        continue
                resp.raise_for_status()
                return resp.json()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            if attempt < max_retries:
                sleep_time = backoff + random.uniform(0.1, 0.5)
                logger.warning(
                    f"Network error ({type(e).__name__}) connecting to {url}, retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
                backoff *= 2.0
                continue
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} from {url}: {e.response.text[:200]}")
            raise

    if last_err:
        raise last_err
    raise RuntimeError(f"Failed to complete request to {url} after {max_retries} retries")


class LLM(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 120.0, max_retries: int = 3):
        self.name = f"anthropic:{model}"
        self.model = model
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_tokens: int = 0
        self.total_calls: int = 0

    def complete(self, system: str, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        data = _post_with_retry(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            payload={
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.total_calls += 1
        if "usage" in data:
            self.total_tokens += data["usage"].get("input_tokens", 0) + data["usage"].get("output_tokens", 0)
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAICompatLLM:
    """Works with OpenAI, Ollama (http://localhost:11434/v1), vLLM, LM Studio."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.name = f"openai:{model}@{base_url}"
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.max_retries = max_retries
        is_local = "localhost" in base_url or "127.0.0.1" in base_url
        self.key = api_key or os.environ.get("OPENAI_API_KEY") or ("ollama" if is_local else "")
        if not self.key and not is_local:
            logger.warning(f"No API key provided for OpenAI-compatible endpoint at {base_url}")
        self.total_tokens: int = 0
        self.total_calls: int = 0

    def complete(self, system: str, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post_with_retry(
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload={
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages],
            },
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.total_calls += 1
        if "usage" in data and isinstance(data["usage"], dict):
            self.total_tokens += data["usage"].get("total_tokens", 0)
        return data["choices"][0]["message"]["content"]


class MockLLM:
    """Deterministic LLM for tests and offline runs. `fn(system, messages) -> str`."""

    def __init__(self, fn: Callable[[str, list[Message]], str], name: str = "mock"):
        self.fn, self.name = fn, name
        self.calls = 0

    def complete(self, system: str, messages: list[Message], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.fn(system, messages)


def make_llm(spec: str) -> LLM:
    """`anthropic:<model>` | `ollama:<model>` | `openai:<model>` | `openai:<model>@<base_url>`."""
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(rest)
    if kind == "ollama":
        return OpenAICompatLLM(rest, base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"))
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatLLM(model, base_url=url or "https://api.openai.com/v1")
    raise ValueError(f"unknown LLM spec {spec!r}")
