"""Unit tests for services.llm (no network)."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LLMConfig
from services.llm import LLMClient, strip_code_fences


@pytest.mark.parametrize("raw, expected", [
    ("SELECT * FROM users", "SELECT * FROM users"),
    ("```sql\nSELECT * FROM users\n```", "SELECT * FROM users"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("  ```SQL\nSELECT 1;\n```  ", "SELECT 1;"),
    ("```json\n[{\"a\": 1}]\n```", "[{\"a\": 1}]"),
    ("", ""),
    (None, ""),
])
def test_strip_code_fences(raw, expected):
    assert strip_code_fences(raw) == expected


class FakeCompletions:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


def make_client(content, finish_reason="stop"):
    client = LLMClient(LLMConfig(provider="gemini", api_key="g-test",
                                 model="gemini-2.5-flash",
                                 base_url="http://localhost:1/v1"))
    fake = FakeCompletions(content, finish_reason)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


def test_chat_sends_configured_model():
    client, fake = make_client("SELECT 1")
    out = client.chat([{"role": "user", "content": "q"}], temperature=0.1, max_tokens=50)
    assert out == "SELECT 1"
    assert fake.calls[0]["model"] == "gemini-2.5-flash"
    assert fake.calls[0]["temperature"] == 0.1
    assert fake.calls[0]["max_tokens"] == 50


def test_chat_omits_max_tokens_when_not_given():
    client, fake = make_client("x")
    client.chat([{"role": "user", "content": "q"}], temperature=0.7)
    assert "max_tokens" not in fake.calls[0]


def test_empty_reply_raises_with_finish_reason():
    client, _ = make_client(None, finish_reason="length")
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        client.chat([{"role": "user", "content": "q"}], temperature=0.1)
