from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Gemini exposes an OpenAI-compatible API, so the same OpenAI SDK client
# talks to either provider; only the key, base URL and model name change.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash",
}

Provider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class LLMConfig:
    """Everything needed to build an LLM client, resolved from Settings."""
    provider: Provider
    api_key: str
    model: str
    base_url: Optional[str] = None  # None = the OpenAI SDK default

    def __repr__(self) -> str:  # never log the key
        return (f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
                f"base_url={self.base_url!r}, api_key=***)")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider. Leave LLM_PROVIDER unset to pick it from whichever key
    # is present (Gemini if only GEMINI_API_KEY is set, otherwise OpenAI).
    llm_provider: Optional[Provider] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    # Optional overrides; the defaults depend on the provider.
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    @field_validator("llm_provider", "openai_api_key", "gemini_api_key",
                     "llm_model", "llm_base_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        # `GEMINI_API_KEY=` in .env, or an empty deploy variable, means "not set".
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def _check_llm_key(self) -> "Settings":
        # Fail at startup with a clear message instead of on the first query.
        self.llm_config()
        return self

    @property
    def resolved_provider(self) -> Provider:
        if self.llm_provider:
            return self.llm_provider
        if self.gemini_api_key and not self.openai_api_key:
            return "gemini"
        return "openai"

    def llm_config(self) -> LLMConfig:
        provider = self.resolved_provider
        if provider == "gemini":
            api_key = self.gemini_api_key
            base_url = self.llm_base_url or GEMINI_OPENAI_BASE_URL
        else:
            api_key = self.openai_api_key
            base_url = self.llm_base_url
        if not api_key:
            raise ValueError(
                f"LLM provider is '{provider}' but {provider.upper()}_API_KEY is not set"
            )
        return LLMConfig(
            provider=provider,
            api_key=api_key,
            model=self.llm_model or DEFAULT_MODELS[provider],
            base_url=base_url,
        )
