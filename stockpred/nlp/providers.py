"""LLM backends behind one protocol: Ollama (local), OpenAI-compatible, Anthropic."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import aiohttp

from stockpred.config.settings import Settings

log = logging.getLogger(__name__)
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    async def generate(self, prompt: str, *, temperature: float) -> str: ...
    async def health(self) -> tuple[bool, str]: ...
    async def close(self) -> None: ...


class _Base:
    name = "base"

    def __init__(self, model: str, timeout: float, max_retries: int):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._sess: aiohttp.ClientSession | None = None

    async def _session(self) -> aiohttp.ClientSession:
        if self._sess is None or self._sess.closed:
            self._sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._sess

    async def _post_json(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        """POST with exponential backoff on 429/5xx/timeouts."""
        sess = await self._session()
        last = "unknown"
        for attempt in range(self.max_retries + 1):
            try:
                async with sess.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json(content_type=None)
                    body = (await resp.text())[:300]
                    last = f"HTTP {resp.status}: {body}"
                    if resp.status not in _RETRY_STATUS:
                        raise LLMError(last)
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                delay = 2**attempt
            if attempt < self.max_retries:
                await asyncio.sleep(min(delay, 30))
        raise LLMError(last)

    async def close(self) -> None:
        if self._sess and not self._sess.closed:
            await self._sess.close()


class OllamaProvider(_Base):
    name = "ollama"

    def __init__(self, model: str, base_url: str, timeout: float, max_retries: int):
        super().__init__(model, timeout, max_retries)
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, *, temperature: float) -> str:
        data = await self._post_json(
            f"{self.base_url}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False, "format": "json",
             "options": {"temperature": temperature}},
        )
        return data.get("response", "")

    async def health(self) -> tuple[bool, str]:
        try:
            sess = await self._session()
            async with sess.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    return False, f"HTTP {r.status}"
                names = [m["name"] for m in (await r.json()).get("models", [])]
        except Exception as exc:
            return False, f"cannot reach {self.base_url}: {exc}"
        want = self.model if ":" in self.model else f"{self.model}:latest"
        if want in names or self.model in names:
            return True, "ok"
        return False, f"model {self.model!r} not pulled; have {names}"


class OpenAIProvider(_Base):
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float, max_retries: int):
        super().__init__(model, timeout, max_retries)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def generate(self, prompt: str, *, temperature: float) -> str:
        data = await self._post_json(
            f"{self.base_url}/chat/completions",
            {"model": self.model, "temperature": temperature,
             "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    async def health(self) -> tuple[bool, str]:
        return (True, "ok") if self.api_key else (False, "STOCKPRED_OPENAI_API_KEY not set")


class AnthropicProvider(_Base):
    name = "anthropic"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float, max_retries: int):
        super().__init__(model, timeout, max_retries)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def generate(self, prompt: str, *, temperature: float) -> str:
        data = await self._post_json(
            f"{self.base_url}/v1/messages",
            {"model": self.model, "max_tokens": 600, "temperature": min(temperature, 1.0),
             "messages": [{"role": "user", "content": prompt}]},
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc

    async def health(self) -> tuple[bool, str]:
        return (True, "ok") if self.api_key else (False, "STOCKPRED_ANTHROPIC_API_KEY not set")


def make_provider(cfg: Settings) -> LLMProvider:
    if cfg.llm_provider == "ollama":
        return OllamaProvider(cfg.llm_model, cfg.ollama_base_url, cfg.llm_timeout_s, cfg.llm_max_retries)
    if cfg.llm_provider == "openai":
        return OpenAIProvider(cfg.llm_model, cfg.openai_base_url, cfg.openai_api_key,
                              cfg.llm_timeout_s, cfg.llm_max_retries)
    if cfg.llm_provider == "anthropic":
        return AnthropicProvider(cfg.llm_model, cfg.anthropic_base_url, cfg.anthropic_api_key,
                                 cfg.llm_timeout_s, cfg.llm_max_retries)
    raise ValueError(f"unknown provider {cfg.llm_provider!r}")
