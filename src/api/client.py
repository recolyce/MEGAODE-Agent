"""OpenAI-compatible chat client. Key and URL come from the environment."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://api.gpugeek.com/v1"
DEFAULT_MODEL = "DeepSeek-V4-Pro"


def llm_config() -> dict[str, str]:
    key = os.environ.get("TMO_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    return {
        "api_key": key,
        "base_url": os.environ.get("TMO_LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("TMO_LLM_MODEL", DEFAULT_MODEL),
    }


def has_llm_key() -> bool:
    return bool(llm_config()["api_key"])


def invoke_text(system: str, user: str, temperature: float = 0.0) -> str:
    llm = get_chat_model(temperature=temperature)
    reply = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return str(reply.content)


def invoke_json(system: str, user: str, temperature: float = 0.0, attempts: int = 3) -> dict:
    """Ask the chat model for a JSON object. Retries; does not fall back to defaults."""
    import json

    last = ""
    for i in range(attempts):
        last = invoke_text(system, user if i == 0 else user + f"\nPrevious reply was not valid JSON: {last[:800]}", temperature)
        start, end = last.find("{"), last.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(last[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"LLM reply had no JSON object after {attempts} attempts: {last[:500]}")


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    cfg = llm_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "Set TMO_LLM_API_KEY or OPENAI_API_KEY. "
            "Optional: TMO_LLM_BASE_URL, TMO_LLM_MODEL."
        )
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=temperature,
    )
