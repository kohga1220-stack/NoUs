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


@pytest.mark.parametrize("reply, expected", [
    ({"promptFeedback": {"blockReason": "SAFETY"}}, "blockReason: SAFETY"),         # no candidates
    ({"candidates": []}, "blockReason: unknown"),
    ({"candidates": [{"finishReason": "SAFETY"}]}, "finishReason: SAFETY"),         # no content
    ({"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}, "finishReason: MAX_TOKENS"),
    ({"candidates": [{"content": {"parts": [{"text": ""}]}}]}, "finishReason: unknown"),
])
def test_gemini_empty_response_is_a_clear_error_and_is_not_retried(monkeypatch, reply, expected):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    calls = _scripted(monkeypatch, [(200, reply), (200, {"candidates": [{"content": {"parts": [{"text": "late"}]}}]})])
    with pytest.raises(llm.EmptyResponseError, match=expected):
        llm.generate("q", model="gemini:m")
    assert len(calls) == 1                                    # one HTTP call, no retry
    calls = _scripted(monkeypatch, [(200, reply)] * 3)
    with pytest.raises(llm.EmptyResponseError):
        llm.generate_json("q", model="gemini:m")
    assert len(calls) == 1                                    # generate_json retries only ValueError


def test_gemini_error_message_does_not_contain_the_prompt_or_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRETKEY")
    _scripted(monkeypatch, [(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(llm.EmptyResponseError) as ei:
        llm.generate("PRIVATE PROMPT", model="gemini:m")
    assert "SECRETKEY" not in str(ei.value) and "PRIVATE PROMPT" not in str(ei.value)


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


def test_free_tier_403_gets_a_hint_and_no_retry(monkeypatch):
    body = {"type": "error", "error": {"type": "FreeTierError", "message": "free tier can only be used from within OpenCode"}}
    calls = _scripted(monkeypatch, [(403, body), (200, {"ok": 1})])
    with pytest.raises(RuntimeError, match="403") as ei:
        llm._post_json("u", {}, {})
    assert "有料モデル" in str(ei.value) and len(calls) == 1


def test_check_models_reports_each_model_and_never_raises():
    def gen(prompt, model):
        if model == "bad":
            raise RuntimeError("LLM API error 404: model not found")
        if model == "empty":
            return "  "
        return "OK\n"

    r = {x["model"]: x for x in llm.check_models(["good", "bad", "empty"], gen=gen)}
    assert r["good"]["ok"] and r["good"]["detail"] == "OK"
    assert not r["bad"]["ok"] and "404" in r["bad"]["detail"] and r["bad"]["detail"].startswith("RuntimeError")
    assert not r["empty"]["ok"] and r["empty"]["detail"] == "empty answer"


def test_check_llm_command_exit_status(monkeypatch, capsys):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("nous_cli", pathlib.Path(__file__).parent.parent / "nous.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(llm, "check_models", lambda models, gen=None: [
        {"model": models[0], "ok": True, "seconds": 0.5, "detail": "OK"}])
    assert cli.cmd_check_llm(["gemini:x"]) is True
    assert "OK  gemini:x" in capsys.readouterr().out
    monkeypatch.setattr(llm, "check_models", lambda models, gen=None: [
        {"model": models[0], "ok": False, "seconds": 0.5, "detail": "RuntimeError: key unset"}])
    assert cli.cmd_check_llm(["gemini:x"]) is False


def test_check_models_treats_empty_whitespace_and_none_as_failure():
    for answer in ("", "   \n", None):
        r = llm.check_models(["m"], gen=lambda prompt, model, a=answer: a)[0]
        assert not r["ok"] and r["detail"] == "empty answer"


def test_check_models_masks_keys_in_error_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRETKEY_12345678")
    def gen(prompt, model):
        raise RuntimeError("LLM API error 400: bad key SECRETKEY_12345678 for model")
    r = llm.check_models(["gemini:x"], gen=gen)[0]
    assert not r["ok"] and "SECRETKEY" not in r["detail"] and "***" in r["detail"]


def test_num_ctx_reaches_ollama_only_when_asked(monkeypatch):
    import sys, types
    seen = []
    fake = types.SimpleNamespace(generate=lambda **kw: seen.append(kw) or {"response": "{\"a\": 1}"})
    monkeypatch.setitem(sys.modules, "ollama", fake)
    llm.generate("hi", model="gemma4:e2b")
    llm.generate("hi", model="gemma4:e2b", num_ctx=8192)
    assert "options" not in seen[0] and seen[1]["options"] == {"num_ctx": 8192}
    assert llm.generate_json("hi", model="gemma4:e2b", num_ctx=4096) == {"a": 1}
    assert seen[2]["options"] == {"num_ctx": 4096}
    assert llm.generate_json("hi", model="gemma4:e2b") == {"a": 1} and "options" not in seen[3]


def test_known_check_asks_ollama_for_a_long_enough_context(monkeypatch):
    import sys, types
    from nous.evaluation import knowncheck as kc
    seen = []

    def gen(**kw):
        seen.append(kw)
        return {"response": '{"verdict": "not_found", "why": "x"}'}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(generate=gen))
    works = [{"work_id": "W", "title": "T", "year": 1, "text": "T. abstract"}]
    r = kc.check_known("claim", "gemma4:e2b", search=lambda t, n: works, rewrite=False, confirm=False)
    assert r["verdict"] == "not_found" and seen[0]["options"] == {"num_ctx": kc.NUM_CTX}


def test_missing_local_models_lists_only_unpulled_ollama_models(monkeypatch):
    from nous import llm
    monkeypatch.setattr(llm, "ollama_models", lambda: {"gemma4:e2b", "gemma4:12b", "llama3:latest"})
    assert llm.missing_local_models(["gemma4:e2b", "gemma4:31b", None, "gemma4:31b"]) == ["gemma4:31b"]
    assert llm.missing_local_models(["ollama:gemma4:12b", "gemini:gemini-3.6-flash", "opencode:x"]) == []
    assert llm.missing_local_models(["llama3"]) == []                  # bare name resolves to :latest
    monkeypatch.setattr(llm, "ollama_models", lambda: None)            # Ollama not running: no guard
    assert llm.missing_local_models(["gemma4:31b"]) == []
