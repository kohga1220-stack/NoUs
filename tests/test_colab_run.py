import sys

import pytest

from nous import colab_run as cr


def O(**kw):
    return cr.Options(**kw)


def test_every_mode_builds_commands_and_the_default_mode_is_cheap():
    for mode in cr.MODES:
        o = O(refs="hyp:31 hyp:38", note="memo")
        assert cr.build_commands(mode, o), mode
    assert cr.build_commands("状態確認", O()) == [["check-llm", "gemma4:e2b", "gemma4:12b"], ["autoloop", "--report"]]


def test_autoloop_options_become_flags():
    assert cr.build_commands("自律ループ", O(cycles=3, fast=True, gen_model="(既定)")) == [["autoloop", "3", "--fast"]]
    assert cr.build_commands("自律ループ", O(cycles=2, fast=False, gen_model="gemma4:31b")) == \
        [["autoloop", "2", "--gen-model", "gemma4:31b"]]
    assert cr.build_commands("自律ループ", O(cycles=0))[0][1] == "1"            # at least one cycle


def test_refs_note_and_missing_inputs():
    assert cr.build_commands("採用の再点検", O()) == [["knowncheck", "--accepted", "--show"]]
    assert cr.build_commands("採用の再点検", O(refs="hyp:31, hyp:38")) == [["knowncheck", "hyp:31", "hyp:38", "--show"]]
    assert cr.build_commands("善見天へ昇格", O(refs="hyp:31", note="あとで")) == [["vault", "--promote", "hyp:31", "あとで"]]
    assert cr.build_commands("採用を取り消す", O(refs="hyp:38")) == [["autoloop", "--retract", "hyp:38", "取り消し"]]
    for mode in ("善見天へ昇格", "採用を取り消す"):
        with pytest.raises(ValueError):
            cr.build_commands(mode, O())
    with pytest.raises(ValueError):
        cr.build_commands("モデル比較", O(compare_models=" "))
    with pytest.raises(ValueError):
        cr.build_commands("???", O())


def test_model_comparison_and_judges_and_bootstrap():
    cmds = cr.build_commands("モデル比較", O(compare_models="gemma4:12b,gemma4:26b"))
    assert cmds[0] == ["knowncheck", "--controls", "--model", "gemma4:12b"] and len(cmds) == 4
    assert cr.build_commands("評価の一致度", O(judges="gemma4:12b gemma4:31b"))[0] == ["evaluate", "--judges", "gemma4:12b,gemma4:31b"]
    boot = cr.build_commands("初回データ構築", O())
    assert boot[0] == ["openalex", "5"] and ["combocheck"] in boot and boot is not cr.BOOTSTRAP


def test_models_needed_and_gpu_guard():
    assert cr.models_needed("状態確認", O()) == ["gemma4:e2b", "gemma4:12b"]
    assert cr.models_needed("自律ループ", O(gen_model="gemma4:31b")) == ["gemma4:e2b", "gemma4:12b", "gemma4:31b"]
    assert cr.models_needed("浄土へ書き出す", O()) == []
    assert cr.models_needed("モデル比較", O(compare_models="gemma4:26b,gemma4:26b")) == ["gemma4:26b"]
    t4 = 15360
    assert cr.gpu_problems(["gemma4:12b", "gemma4:31b"], t4) == ["gemma4:31b: GPU 15,360 MiB < 目安 22,000 MiB"]
    assert cr.gpu_problems(["gemma4:31b"], 97887) == [] and cr.gpu_problems(["gemma4:31b"], None) == []


def test_secrets_are_masked_and_progress_noise_is_dropped():
    env = {"GEMINI_API_KEY": "AIzaSECRETVALUE123", "GH_TOKEN": "short"}
    assert cr.mask("key=AIzaSECRETVALUE123 ok", env) == "key=*** ok"
    assert cr.mask("short stays", env) == "short stays"                    # too short to be a key: not masked
    assert cr.is_noise("Batches: 100% 1/1 [00:00<00:00, 4.48it/s]") and cr.is_noise("Loading weights: 100% 103/103")
    assert not cr.is_noise("  → REJECTED  hyp:41  composite(excl. novelty)=n/a")


def test_run_command_streams_masks_and_logs(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSECRETVALUE123")
    log = []
    code = cr.run_command([sys.executable, "-c",
                           "import os;print('hello', os.environ['GEMINI_API_KEY']);print('Batches: 100% 1/1 [00:00<00:00, 9.9it/s]');print('done')"], log)
    assert code == 0
    text = "\n".join(log[1:])                                               # without the "$ command" line
    assert "hello ***" in text and "AIzaSECRET" not in text and "Batches" not in text and "done" in text
    assert log[0].startswith("$ ") and log[-1].startswith("[exit 0")
    assert "hello ***" in capsys.readouterr().out
    assert cr.run_command([sys.executable, "-c", "import sys;sys.exit(3)"], []) == 3


def test_write_log_is_markdown_and_capped(tmp_path):
    p = cr.write_log(tmp_path / "実行ログ" / "a.md", "自律ループ", ["開始: now"], ["line"] * 5)
    assert p.read_text(encoding="utf-8").startswith("# 実行ログ: 自律ループ") and "- 開始: now" in p.read_text(encoding="utf-8")
    big = cr.write_log(tmp_path / "b.md", "x", [], ["x" * 1000] * 600)
    assert "途中省略" in big.read_text(encoding="utf-8") and big.stat().st_size < 450_000


def test_run_from_form_without_setup_says_so(monkeypatch, capsys):
    monkeypatch.setattr(cr, "ROOT", cr.Path("/nonexistent"))
    assert cr.run_from_form({"MODE": "状態確認"})["ok"] is False
    assert "ワンセット準備" in capsys.readouterr().out


def test_run_from_form_runs_commands_saves_exports_and_logs(monkeypatch, tmp_path, capsys):
    calls, saved = [], []
    monkeypatch.setattr(cr, "ROOT", tmp_path)
    monkeypatch.setattr(cr, "ensure_ollama", lambda wait=30: True)
    monkeypatch.setattr(cr, "gpu_mib", lambda: 97887)
    monkeypatch.setattr(cr, "gpu_name", lambda: "TestGPU, 97887 MiB")
    monkeypatch.setattr(cr, "log_path", lambda mode, now=None: tmp_path / "logs" / f"{mode}.md")
    monkeypatch.setattr("nous.llm.missing_local_models", lambda models: ["gemma4:31b"])

    def fake_run(argv, log, cwd=None, echo=True):
        calls.append(argv[1:] if argv[0] == sys.executable else argv)
        log.append("$ " + " ".join(argv))
        return 0

    monkeypatch.setattr(cr, "run_command", fake_run)
    monkeypatch.setattr(cr.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "abc123 commit"})())
    res = cr.run_from_form({"MODE": "自律ループ", "CYCLES": 2, "FAST": True, "GEN_MODEL": "gemma4:31b", "save": lambda: saved.append(1)})
    assert res["ok"] and saved == [1]
    assert calls[0] == ["ollama", "pull", "gemma4:31b"]                      # a missing model is pulled first
    assert calls[1] == ["nous.py", "autoloop", "2", "--fast", "--gen-model", "gemma4:31b"]
    assert calls[-1] == ["nous.py", "vault"]                                 # then the vault is rewritten
    text = (tmp_path / "logs" / "自律ループ.md").read_text(encoding="utf-8")
    assert "コード: abc123 commit" in text and "TestGPU" in text and "結果: 成功" in text


def test_run_from_form_refuses_a_model_the_gpu_cannot_hold(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cr, "ROOT", tmp_path)
    monkeypatch.setattr(cr, "gpu_mib", lambda: 15360)
    ran = []
    monkeypatch.setattr(cr, "run_command", lambda *a, **k: ran.append(a) or 0)
    res = cr.run_from_form({"MODE": "自律ループ", "GEN_MODEL": "gemma4:31b", "save": lambda: None})
    assert res == {"ok": False, "why": "gpu"} and not ran
    assert "載らない" in capsys.readouterr().out


def test_a_failing_command_stops_the_plan_but_still_saves_and_logs(monkeypatch, tmp_path):
    saved, calls = [], []
    monkeypatch.setattr(cr, "ROOT", tmp_path)
    monkeypatch.setattr(cr, "ensure_ollama", lambda wait=30: True)
    monkeypatch.setattr(cr, "gpu_mib", lambda: None)
    monkeypatch.setattr(cr, "log_path", lambda mode, now=None: tmp_path / "x.md")
    monkeypatch.setattr("nous.llm.missing_local_models", lambda models: [])
    monkeypatch.setattr(cr.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())

    def fake_run(argv, log, cwd=None, echo=True):
        calls.append(argv)
        return 2 if "evaluate" in argv else 0

    monkeypatch.setattr(cr, "run_command", fake_run)
    res = cr.run_from_form({"MODE": "評価の一致度", "save": lambda: saved.append(1)})
    assert not res["ok"] and saved == [1] and len(calls) == 1                # stopped at evaluate; reliability not run
    assert "結果: 失敗" in (tmp_path / "x.md").read_text(encoding="utf-8")


def _form_cell():
    import json
    import pathlib
    nb = json.loads((pathlib.Path(__file__).parent.parent / "notebooks" / "nous_colab.ipynb").read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "#@title ▶ 実行" in "".join(c["source"]))


def test_notebook_form_matches_the_runner():
    import re
    src = _form_cell()
    compile(src, "form", "exec")
    modes = re.search(r'MODE = "[^"]*"\s+#@param (\[.*?\])', src).group(1)
    import json as _j
    assert _j.loads(modes) == cr.MODES                                           # the dropdown is the runner's mode list
    ns = {}
    exec(src.split("# ---- 以下は触らない")[0].replace("#@param", "#"), ns)       # the form variables alone
    o = cr.Options.from_globals(ns)
    assert o.cycles == 3 and o.fast is True and o.gen_model == "(既定)" and o.export_vault is True
    assert ns["MODE"] == "状態確認"                                                # the default (run-all) mode is the cheap one
    for mode in cr.MODES:
        cr.build_commands(mode, cr.Options(refs="hyp:1"))


def test_notebook_form_pulls_before_importing_the_runner_and_needs_the_setup_cell():
    src = _form_cell()
    assert src.index('git_auto("pull"') < src.index("import nous.colab_run")
    assert "先に、上の「1. ワンセット準備」" in src and '"save" not in globals()' in src
