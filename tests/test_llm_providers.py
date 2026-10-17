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
