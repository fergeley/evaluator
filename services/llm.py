"""One place that talks to the LLM provider (OpenAI or Gemini)."""
import logging
import re
from typing import List, Dict, Optional

import openai

from config import LLMConfig

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```lang ... ``` block if the model added one."""
    if text is None:
        return ""
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


class LLMClient:
    """Thin wrapper over the OpenAI SDK client, pointed at the configured provider."""

    def __init__(self, config: LLMConfig, max_retries: int = 5, timeout: float = 60.0):
        self.config = config
        self.model = config.model
        # The SDK retries 429/5xx with exponential backoff, which covers
        # free-tier per-minute rate limits.
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        logger.info(f"LLM client ready: {config!r}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = dict(model=self.model, messages=messages, temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            finish = response.choices[0].finish_reason
            raise RuntimeError(
                f"{self.config.provider} model {self.model} returned no text "
                f"(finish_reason={finish})"
            )
        return content
