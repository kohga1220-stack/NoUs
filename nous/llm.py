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


# --- providers -----------------------------------------------------------
# `model` selects the backend by an optional prefix; no prefix means local Ollama,
# so every existing model name (e.g. "gemma4:e2b") keeps working unchanged.
#   ollama:<name>    local Ollama (default)
#   opencode:<id>    OpenCode Console, OpenAI-compatible API  (env OPENCODE_API_KEY)
#   gemini:<name>    Gemini REST API                          (env GEMINI_API_KEY)
OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT = 300
_PROVIDERS = ("ollama", "opencode", "gemini")


def split_model(model: str) -> tuple[str, str]:
    """Return (provider, model_name). An unknown prefix is part of an Ollama model name."""
    head, sep, rest = model.partition(":")
    if sep and head in _PROVIDERS and rest:
        return head, rest
    return "ollama", model


def _api_key(env: str) -> str:
    import os
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"環境変数 {env} が未設定です（Colab ならシークレットに登録し、os.environ に入れる）")
    return key


RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_RETRIES = 4
NETWORK_ERROR = 599     # 接続・タイムアウトなど、HTTP 応答が得られなかったとき


def _post_once(url: str, headers: dict, body: dict) -> tuple[int, dict | str]:
    """1 回 POST して (HTTPステータス, JSON または本文) を返す。テストでは差し替える。"""
    import requests
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        return NETWORK_ERROR, type(e).__name__          # 例外の文面にはURLが入るため、型名だけ残す
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text


def _post_json(url: str, headers: dict, body: dict) -> dict:
    """429 / 5xx / 接続エラーは指数バックオフ（5, 10, 20, 40 秒）で再試行する。"""
    import time
    delay = 5.0
    for attempt in range(MAX_RETRIES + 1):
        status, data = _post_once(url, headers, body)
        if (status in RETRY_STATUSES or status == NETWORK_ERROR) and attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2
            continue
        break
    if status >= 400:
        # ステータスと本文の先頭だけを出す。リクエストヘッダ（キー）は出さない。
        msg = f"LLM API error {status}: {str(data)[:300]}"
        if status == 403 and "FreeTierError" in str(data):
            msg += "（OpenCode の -free モデルは OpenCode アプリ内でのみ使える。API では -free なしの有料モデルを使う）"
        raise RuntimeError(msg)
    return data


def _generate_opencode(prompt: str, name: str) -> str:
    data = _post_json(
        f"{OPENCODE_BASE_URL}/chat/completions",
        {"Authorization": f"Bearer {_api_key('OPENCODE_API_KEY')}"},
        {"model": name, "messages": [{"role": "user", "content": prompt}]},
    )
    return data["choices"][0]["message"]["content"] or ""


class EmptyResponseError(RuntimeError):
    """The API answered but returned no text (blocked, filtered, or empty). Not retried."""


def _generate_gemini(prompt: str, name: str) -> str:
    data = _post_json(
        f"{GEMINI_BASE_URL}/models/{name}:generateContent",
        {"x-goog-api-key": _api_key("GEMINI_API_KEY")},
        {"contents": [{"parts": [{"text": prompt}]}]},
    )
    candidates = data.get("candidates") or []
    if not candidates:
        # no candidate at all: typically the prompt was blocked (promptFeedback.blockReason)
        reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
        raise EmptyResponseError(f"Gemini returned no candidates (blockReason: {reason})")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(p.get("text", "") for p in parts)
    if not text:
        reason = candidates[0].get("finishReason", "unknown")
        raise EmptyResponseError(f"Gemini returned an empty answer (finishReason: {reason})")
    return text


def generate(prompt: str, model: str = DEFAULT_MODEL, num_ctx: int | None = None) -> str:
    """`num_ctx` (Ollama only): context length in tokens. Ollama's default is short and a longer
    prompt is cut from the START without any error, so long prompts must ask for what they need."""
    provider, name = split_model(model)
    if provider == "opencode":
        return _generate_opencode(prompt, name)
    if provider == "gemini":
        return _generate_gemini(prompt, name)
    import ollama
    if num_ctx:
        return ollama.generate(model=name, prompt=prompt, options={"num_ctx": num_ctx})["response"]
    return ollama.generate(model=name, prompt=prompt)["response"]


def generate_json(prompt: str, model: str = DEFAULT_MODEL, retries: int = 2,
                  num_ctx: int | None = None) -> dict:
    """
    Generate and parse a JSON object. Small models occasionally emit malformed JSON,
    so the call is retried up to `retries` more times before the error is raised.
    """
    last_error: Exception | None = None
    extra = {"num_ctx": num_ctx} if num_ctx else {}
    for _ in range(retries + 1):
        try:
            return extract_json(generate(prompt, model=model, **extra))
        except ValueError as ex:        # json.JSONDecodeError is a ValueError
            last_error = ex
    raise last_error


CHECK_PROMPT = "Reply with the single word OK."


def check_models(models: list[str], gen=None) -> list[dict]:
    """
    Send one tiny prompt to each model and report whether it answered, and how fast.
    Used to verify keys and model IDs before a long run. A failure is reported, not raised.
    An empty (or whitespace-only, or None) answer is a failure. Any API key / token value that
    appears in an error text is replaced with ***, since the output is often pasted into chats.
    """
    import os
    import time
    gen = gen or generate
    secrets = [v for k in ("GEMINI_API_KEY", "OPENCODE_API_KEY", "GH_TOKEN", "OPENALEX_API_KEY")
               if len(v := os.environ.get(k, "")) >= 8]

    def mask(text: str) -> str:
        for v in secrets:
            text = text.replace(v, "***")
        return text

    out = []
    for m in models:
        t0 = time.time()
        try:
            answer = (gen(CHECK_PROMPT, model=m) or "").strip()
            out.append({"model": m, "ok": bool(answer), "seconds": time.time() - t0,
                        "detail": mask(answer[:60]) or "empty answer"})
        except Exception as ex:                     # network, key, model ID, quota ...
            out.append({"model": m, "ok": False, "seconds": time.time() - t0,
                        "detail": mask(f"{type(ex).__name__}: {str(ex)[:200]}")})
    return out
