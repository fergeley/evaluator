"""Unit tests for LLM provider resolution in config.Settings (no network)."""
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_MODELS, GEMINI_OPENAI_BASE_URL, Settings

LLM_ENV = ["LLM_PROVIDER", "OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_MODEL",
           "LLM_BASE_URL", "OPENAI_BASE_URL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


def make(**kwargs) -> Settings:
    # _env_file=None so a developer's .env does not leak into the tests
    return Settings(_env_file=None, **kwargs)


def test_openai_defaults():
    cfg = make(openai_api_key="sk-test").llm_config()
    assert cfg.provider == "openai"
    assert cfg.model == DEFAULT_MODELS["openai"]
    assert cfg.base_url is None


def test_gemini_picked_when_only_gemini_key_set():
    cfg = make(gemini_api_key="g-test").llm_config()
    assert cfg.provider == "gemini"
    assert cfg.model == DEFAULT_MODELS["gemini"]
    assert cfg.base_url == GEMINI_OPENAI_BASE_URL
    assert cfg.api_key == "g-test"


def test_explicit_provider_wins_over_detection():
    cfg = make(llm_provider="gemini", openai_api_key="sk-test",
               gemini_api_key="g-test").llm_config()
    assert cfg.provider == "gemini"
    assert cfg.api_key == "g-test"


def test_model_and_base_url_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9999/v1")
    cfg = make().llm_config()
    assert cfg.model == "gemini-2.0-flash"
    assert cfg.base_url == "http://localhost:9999/v1"


def test_openai_base_url_env_is_still_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11435/v1")
    assert make().llm_config().base_url == "http://localhost:11435/v1"


def test_missing_key_fails_at_startup():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make(llm_provider="gemini", openai_api_key="sk-test")


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert make().llm_config().provider == "gemini"


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    make(openai_api_key="sk-test")


def test_repr_hides_key():
    cfg = make(gemini_api_key="g-secret-value").llm_config()
    assert "g-secret-value" not in repr(cfg)
