import json
import pathlib
import subprocess
import textwrap

import pytest

NB_DIR = pathlib.Path(__file__).parent.parent / "notebooks"
NBS = [NB_DIR / "nous_colab.ipynb", NB_DIR / "nous_colab_manual.ipynb"]      # the one-cell copy, then the step-4 copy


def _cells():
    for nb in NBS:
        yield from json.loads(nb.read_text(encoding="utf-8"))["cells"]
TOKEN = "ghp_TESTTOKEN1234567890"


def _helper_sources():
    """The git()/git_auto() helper as written in each notebook cell that has one (dedented)."""
    out = []
    for c in _cells():
        s = "".join(c["source"])
        if c["cell_type"] == "code" and "def git(" in s:
            start = s.index("def git(")
            start = s.rfind("\n", 0, start) + 1
            end = s.index("git_auto(", s.index("def git_auto"))          # up to the first real call
            end = s.rfind("\n", 0, s.index("\n", s.index("def git_auto(*args")) + 1)
            body = s[start:]
            # keep only the two function definitions
            lines, keep = body.split("\n"), []
            indent = len(lines[0]) - len(lines[0].lstrip())
            for i, l in enumerate(lines):
                if i and l.strip() and (len(l) - len(l.lstrip())) <= indent and not l.lstrip().startswith(("def ", '"""')):
                    break
                keep.append(l)
            out.append(textwrap.dedent("\n".join(keep)))
    return out


class _Result:
    def __init__(self, code, text=""):
        self.returncode, self.stdout, self.stderr = code, "", text


def _run(src, monkeypatch, behaviour):
    """Execute the helper with a fake subprocess.run; `behaviour(env_has_token)` -> _Result."""
    calls = []

    def fake_run(cmd, **kw):
        has = "Authorization" in kw["env"].get("GIT_CONFIG_VALUE_0", "")
        calls.append(has)
        return behaviour(has, kw["env"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    ns = {"os": __import__("os"), "subprocess": subprocess, "base64": __import__("base64")}
    exec(src, ns)
    return ns, calls


def test_notebook_has_the_helper_in_both_places_and_imports_what_it_uses():
    assert len(_helper_sources()) == 2
    for c in _cells():
        s = "".join(c["source"])
        if c["cell_type"] == "code" and "def git(" in s:
            assert s.index("import base64") < s.index("def git(")


@pytest.mark.parametrize("src", _helper_sources(), ids=["one-cell", "step-4"])
def test_public_repo_is_tried_without_the_token_first(src, monkeypatch, capsys):
    ns, calls = _run(src, monkeypatch, lambda has, env: _Result(0))
    ns["git_auto"]("clone", "-q", "url", "dir", token=TOKEN)
    assert calls == [False]                      # the token was never sent


@pytest.mark.parametrize("src", _helper_sources(), ids=["one-cell", "step-4"])
def test_falls_back_to_the_token_when_anonymous_access_fails_and_is_quiet_about_it(src, monkeypatch, capsys):
    ns, calls = _run(src, monkeypatch, lambda has, env: _Result(0) if has else
                     _Result(128, "fatal: could not read Username"))
    ns["git_auto"]("clone", "-q", "url", "dir", token=TOKEN)
    assert calls == [False, True]
    assert "could not read Username" not in capsys.readouterr().out      # first failure is not shown


@pytest.mark.parametrize("src", _helper_sources(), ids=["one-cell", "step-4"])
def test_token_never_appears_in_output_or_error(src, monkeypatch, capsys):
    def behaviour(has, env):
        if has:
            return _Result(128, f"fatal: bad credentials {TOKEN} {env['GIT_CONFIG_VALUE_0']}")
        return _Result(128, "fatal: could not read Username")

    ns, calls = _run(src, monkeypatch, behaviour)
    with pytest.raises(RuntimeError) as e:
        ns["git_auto"]("clone", "-q", "url", "dir", token=TOKEN)
    shown = capsys.readouterr().out + str(e.value)
    assert TOKEN not in shown and "Basic" not in shown.replace("Authorization: Basic ***", "")
    assert "***" in shown and calls == [False, True]


@pytest.mark.parametrize("src", _helper_sources(), ids=["one-cell", "step-4"])
def test_without_a_token_the_error_is_shown_once(src, monkeypatch, capsys):
    ns, calls = _run(src, monkeypatch, lambda has, env: _Result(128, "fatal: repository not found"))
    with pytest.raises(RuntimeError):
        ns["git_auto"]("clone", "-q", "url", "dir")
    assert calls == [False] and "repository not found" in capsys.readouterr().out
