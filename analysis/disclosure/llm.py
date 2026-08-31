"""Judge clients + cost bookkeeping, shared by analysis/disclosure/ (and later analysis/target_visibility/, analysis/continuations/).

Two backends:
  openai      — the project's prepaid OpenAI credits. The default.
  openrouter  — fallback (claude-haiku-4.5), uses the repo's existing client.

Both are OpenAI-shaped, so one call path serves both.
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, AsyncOpenAI,
                    InternalServerError, RateLimitError)
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

load_dotenv()

# USD per million tokens (input, output). List prices; update if they move.
PRICES = {
    # OpenAI
    "gpt-5.6-luna-pro": (0.20, 1.20),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    # OpenRouter fallback
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
}
UNKNOWN_PRICE = (0.50, 2.00)   # deliberately pessimistic


def price_of(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    for known, p in PRICES.items():          # tolerate dated suffixes
        if model.startswith(known):
            return p
    return UNKNOWN_PRICE


def cost_of(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = price_of(model)
    return (prompt_tokens / 1e6) * p_in + (completion_tokens / 1e6) * p_out


def get_client(backend: str) -> AsyncOpenAI:
    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is missing or empty in value-leakage/.env — "
                "add it, or pass --backend openrouter to use the fallback judge.")
        return AsyncOpenAI(api_key=key)
    if backend == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is missing or empty in "
                             "value-leakage/.env")
        return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    raise ValueError(f"backend must be 'openai' or 'openrouter', got {backend!r}")


async def list_models(backend: str, contains: str = "") -> list[str]:
    client = get_client(backend)
    page = await client.models.list()
    ids = sorted(m.id for m in page.data)
    return [i for i in ids if contains.lower() in i.lower()]


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError,
                                   APITimeoutError, InternalServerError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
)
async def _one(client, model, messages, max_tokens, reasoning_effort, extra_body):
    kwargs = dict(model=model, messages=messages)
    # gpt-5-family reasoning models reject max_tokens/temperature and bill any
    # thinking they do, so keep effort at the floor and cap completion tokens.
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
    if extra_body:
        kwargs["extra_body"] = extra_body
    return await client.chat.completions.create(**kwargs)


async def batch(client, model: str, prompts: list[str], max_concurrent: int = 8,
                max_tokens: int = 24, reasoning_effort: str | None = None,
                extra_body: dict | None = None) -> list:
    """One user message per prompt. Exceptions are returned, not raised."""
    sem = asyncio.Semaphore(max_concurrent)

    async def guarded(p):
        async with sem:
            try:
                return await _one(client, model,
                                  [{"role": "user", "content": p}],
                                  max_tokens, reasoning_effort, extra_body)
            except Exception as e:            # noqa: BLE001 — reported per row
                return e

    return await asyncio.gather(*[guarded(p) for p in prompts])
