"""
Thin wrapper around the local Ollama LLM.
All agents call generate_json() so the backend and JSON parsing live in one place.
"""
from __future__ import annotations
import json

from nous.config import DEFAULT_MODEL

_HEX = set("0123456789abcdefABCDEF")
_LATEX_LEADERS = set("bfrt")     # \beta \frac \rightarrow \text ... start like JSON escapes


def sanitize_json(s: str) -> str:
    r"""
    Repair backslashes in LLM output so json.loads accepts it without corrupting LaTeX.

    - `\\`, `\"`, `\/`, `\uXXXX`, and `\n` stay as valid JSON escapes.
    - `\b \f \r \t` followed by a letter are almost always LaTeX (\beta, \frac,
      \rightarrow, \text), not control characters -> keep the backslash as a literal.
    - any other invalid escape (\alpha, \(, \{ ...) keeps its backslash as a literal.
    Known trade-off: \n followed by a letter is read as a newline, so \nu / \neq lose
    their backslash.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt in ('\\', '"', "/"):
            out.append(c + nxt)
            i += 2
        elif nxt == "u" and n - i >= 6 and all(ch in _HEX for ch in s[i + 2:i + 6]):
            out.append(s[i:i + 6])
            i += 6
        elif nxt in _LATEX_LEADERS and i + 2 < n and s[i + 2].isalpha():
            out.append("\\\\")           # literal backslash; the letters follow normally
            i += 1
        elif nxt in ("b", "f", "n", "r", "t"):
            out.append(c + nxt)
            i += 2
        else:                                # invalid escape (or trailing backslash)
            out.append("\\\\")
            i += 1
    return "".join(out)


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
