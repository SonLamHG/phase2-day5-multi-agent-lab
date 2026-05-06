"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)


# Approximate USD per 1M tokens (input, output). Update when the pricing page changes.
# Source: OpenAI pricing for the gpt-5 family. Treat these as best-effort estimates.
_COST_PER_MTOKENS: dict[str, tuple[float, float]] = {
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def _retryable_errors() -> tuple[type[BaseException], ...]:
    """Only retry transient OpenAI errors — network, rate limit, server-side."""

    try:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    except ImportError:
        return (TimeoutError,)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _COST_PER_MTOKENS.get(model)
    if rates is None:
        for prefix, value in _COST_PER_MTOKENS.items():
            if model.startswith(prefix):
                rates = value
                break
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    finish_reason: str | None = None


class LLMClient(Protocol):
    """Provider-agnostic LLM client interface."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class OpenAIClient:
    """OpenAI-backed LLM client with retry, timeout, and cost accounting."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env to use OpenAIClient."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            timeout=self._settings.openai_request_timeout,
            max_retries=0,
        )
        self._model = self._settings.openai_model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self._complete_with_retry(system_prompt, user_prompt, temperature, max_tokens)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(_retryable_errors()),
    )
    def _complete_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        from openai import APIError

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, object] = {"model": self._model, "messages": messages}
        # gpt-5 family uses max_completion_tokens, ignores temperature, and
        # spends part of the output budget on internal reasoning. We force
        # `reasoning_effort="minimal"` for cheap factual tasks and inflate the
        # budget so reasoning + answer both fit.
        if self._is_gpt5():
            budget = max(2048, (max_tokens or 1024) * 3)
            kwargs["max_completion_tokens"] = budget
            kwargs["reasoning_effort"] = "minimal"
        else:
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        try:
            # The kwargs shape is built dynamically per model family;
            # the OpenAI SDK overloads are too strict for that pattern.
            response = self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        except APIError as exc:
            logger.warning("OpenAI API error (will retry if attempts remain): %s", exc)
            raise

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        reasoning_tokens: int | None = None
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)
        cost = (
            estimate_cost_usd(self._model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        logger.info(
            "llm_call model=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s "
            "finish_reason=%s cost_usd=%s",
            self._model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            finish_reason,
            cost,
        )
        if not content:
            raise AgentExecutionError(
                f"LLM returned empty content (model={self._model}, "
                f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}). "
                "Increase max_tokens or lower reasoning_effort."
            )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            model=self._model,
            finish_reason=finish_reason,
        )

    def _is_gpt5(self) -> bool:
        return self._model.startswith("gpt-5")


class MockLLMClient:
    """Deterministic stub useful for unit tests and offline development."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "[mock] generated text",
        model: str = "mock-llm",
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self._model = model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        for needle, payload in self._responses.items():
            if needle in user_prompt or needle in system_prompt:
                return self._build(payload)
        return self._build(self._default)

    def _build(self, payload: str) -> LLMResponse:
        input_tokens = max(1, len(payload) // 4)
        output_tokens = max(1, len(payload) // 4)
        return LLMResponse(
            content=payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            model=self._model,
        )


def build_default_llm_client() -> LLMClient:
    """Factory returning a real OpenAI client; raises if API key missing."""

    return OpenAIClient()
