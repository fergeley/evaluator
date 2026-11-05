# Malay2SQL Backend

A FastAPI-based backend service that converts Malay language queries into SQL using AI. The service translates Malay queries to English, understands the database schema context, and generates appropriate SQL queries.


## Configuration

Copy `.env.example` to `.env`. SQL generation (and the mock results from `/malay2sql/execute`) goes through one LLM client (`services/llm.py`) that works with either provider:

| Variable | Meaning |
| --- | --- |
| `LLM_PROVIDER` | `openai` or `gemini`. If unset, Gemini is used when only `GEMINI_API_KEY` is set, otherwise OpenAI. |
| `OPENAI_API_KEY` | OpenAI key. Default model `gpt-4o-mini`. |
| `GEMINI_API_KEY` | Google AI Studio key, called through Gemini's OpenAI-compatible endpoint. Default model `gemini-3.5-flash`. |
| `LLM_MODEL` | Optional model override, e.g. `gemini-3.5-flash-lite`. |
| `LLM_BASE_URL` | Optional OpenAI-compatible endpoint override (`OPENAI_BASE_URL` is also read). |

The service refuses to start if the selected provider has no key. Rate-limit (429) and server errors are retried with backoff by the SDK.

## Tests

```bash
pytest tests/test_config.py tests/test_llm.py   # unit tests, no network or models
pytest tests                                     # also needs the models, PostgreSQL and an LLM key
```
