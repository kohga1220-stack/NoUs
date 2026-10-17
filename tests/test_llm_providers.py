import pytest

from nous import llm


def test_split_model_default_and_unknown_prefix_are_ollama():
    assert llm.split_model("gemma4:e2b") == ("ollama", "gemma4:e2b")
    assert llm.split_model("llama3.2:3b") == ("ollama", "llama3.2:3b")
    assert llm.split_model("plain") == ("ollama", "plain")


def test_split_model_known_prefixes():
    assert llm.split_model("opencode:deepseek-v4.1-flash-free") == ("opencode", "deepseek-v4.1-flash-free")
    assert llm.split_model("gemini:gemini-3.6-flash") == ("gemini", "gemini-3.6-flash")
    assert llm.split_model("ollama:gemma4:e2b") == ("ollama", "gemma4:e2b")
    assert llm.split_model("opencode:") == ("ollama", "opencode:")   # empty name is not a provider call


def test_ollama_path_unchanged(monkeypatch):
    import sys, types
    calls = {}
    fake = types.SimpleNamespace(generate=lambda model, prompt: calls.update(m=model, p=prompt) or {"response": "ok"})
    monkeypatch.setitem(sys.modules, "ollama", fake)
    assert llm.generate("hi", model="gemma4:e2b") == "ok"
    assert calls == {"m": "gemma4:e2b", "p": "hi"}


def test_opencode_request_and_parse(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    seen = {}

    def fake_post(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    assert llm.generate("q", model="opencode:glm-5.3-flash-free") == "answer"
    assert seen["url"] == "https://opencode.ai/inference/openai/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer k"}
    assert seen["body"] == {"model": "glm-5.3-flash-free", "messages": [{"role": "user", "content": "q"}]}


def test_gemini_request_and_parse(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    seen = {}

    def fake_post(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}

    monkeypatch.setattr(llm, "_post_json", fake_post)
    assert llm.generate("q", model="gemini:gemini-3.6-flash") == "ab"
    assert seen["url"].endswith("/models/gemini-3.6-flash:generateContent")
    assert seen["headers"] == {"x-goog-api-key": "g"}


def test_missing_key_raises_without_leaking(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENCODE_API_KEY"):
        llm.generate("q", model="opencode:x")


def test_generate_json_works_through_provider(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setattr(llm, "_post_json", lambda u, h, b: {"choices": [{"message": {"content": 'x {"a": 1} y'}}]})
    assert llm.generate_json("p", model="opencode:m") == {"a": 1}


# --- retry -------------------------------------------------------------

def _scripted(monkeypatch, replies):
    it = iter(replies)
    calls = []
    monkeypatch.setattr(llm, "_post_once", lambda url, headers, body: calls.append(1) or next(it))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return calls


def test_retries_429_and_5xx_then_succeeds(monkeypatch):
    calls = _scripted(monkeypatch, [(429, "slow"), (503, "down"), (200, {"ok": 1})])
    assert llm._post_json("u", {}, {}) == {"ok": 1}
    assert len(calls) == 3


def test_retries_network_error(monkeypatch):
    calls = _scripted(monkeypatch, [(llm.NETWORK_ERROR, "ConnectTimeout"), (200, {"ok": 2})])
    assert llm._post_json("u", {}, {}) == {"ok": 2}
    assert len(calls) == 2


def test_gives_up_after_max_retries(monkeypatch):
    calls = _scripted(monkeypatch, [(429, "slow")] * (llm.MAX_RETRIES + 1))
    with pytest.raises(RuntimeError, match="429"):
        llm._post_json("u", {}, {})
    assert len(calls) == llm.MAX_RETRIES + 1


def test_client_errors_are_not_retried(monkeypatch):
    calls = _scripted(monkeypatch, [(401, "bad key"), (200, {"ok": 3})])
    with pytest.raises(RuntimeError, match="401"):
        llm._post_json("u", {}, {})
    assert len(calls) == 1


def test_network_error_message_has_no_url(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("https://opencode.ai/secret-path?key=SECRET")

    monkeypatch.setattr(requests, "post", boom)
    status, data = llm._post_once("https://opencode.ai/x", {"Authorization": "Bearer SECRET"}, {})
    assert status == llm.NETWORK_ERROR and "SECRET" not in str(data)


# --- Colab 準備セル -----------------------------------------------------

def test_colab_prep_cell_compiles_and_handles_api_models():
    import json
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "nous_colab.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][2]["source"])
    compile(src.replace("get_ipython().system", "print"), "cell2", "exec")
    assert "OPENCODE_API_KEY" in src and "GEMINI_API_KEY" in src
    assert "_ollama_name" in src
