"""
Thin wrapper around the local Ollama LLM.
All agents call generate_json() so the backend and JSON parsing live in one place.
"""
from __future__ import annotations
import json

from nous.config import DEFAULT_MODEL

_VALID_ESCAPES = set('"\\bfnrt/')


def sanitize_json(s: str) -> str:
    """Remove invalid JSON escape sequences produced by the LLM."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s) and s[i+1] not in _VALID_ESCAPES and s[i+1] != 'u':
            result.append(' ')
            i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def extract_json(raw: str) -> dict:
    """Parse the outermost {...} block of an LLM response."""
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s < 0 or e <= s:
        raise ValueError("no JSON object in response")
    return json.loads(sanitize_json(raw[s:e]))


def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    import ollama
    return ollama.generate(model=model, prompt=prompt)["response"]


def generate_json(prompt: str, model: str = DEFAULT_MODEL, retries: int = 2) -> dict:
    """
    Generate and parse a JSON object. Small models occasionally emit malformed JSON,
    so the call is retried up to `retries` more times before the error is raised.
    """
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            return extract_json(generate(prompt, model=model))
        except ValueError as ex:        # json.JSONDecodeError is a ValueError
            last_error = ex
    raise last_error
