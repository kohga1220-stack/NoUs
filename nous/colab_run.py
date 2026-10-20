"""
Colab: one form for everything. The notebook's second cell calls `run_from_form(globals())`.

It (1) checks that the one-cell setup was run, (2) pulls the models the chosen job needs and refuses a
model the GPU cannot hold, (3) runs the commands with live output, (4) saves the data to Drive and
rewrites the 浄土 vault, and (5) keeps the whole output as a Markdown log in `浄土/実行ログ/`, so
results never have to be copied by hand.

Everything that can be tested without Colab is a plain function here.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/content/NoUs")
SECRET_ENV = ("OPENALEX_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "GEMINI_API_KEY", "OPENCODE_API_KEY", "HF_TOKEN")
DEFAULT_MODELS = ("gemma4:e2b", "gemma4:12b")           # generation/judging and the known-work check
MAX_LOG_CHARS = 400_000
# GPU memory (MiB) a model needs: a rough guide from the model file sizes (31b / 26b ≈ 17-20 GB), not a measurement
GPU_NEED_MIB = {"gemma4:31b": 22000, "gemma4:26b": 19000}

MODES = ["状態確認", "自律ループ", "採用の再点検", "浄土へ書き出す", "善見天へ昇格", "採用を取り消す",
         "モデル比較", "評価の一致度", "初回データ構築"]
SAVES = {"自律ループ", "採用の再点検", "善見天へ昇格", "採用を取り消す", "評価の一致度", "初回データ構築"}

# the first-time data build, in the order of the old notebook (C, D, E, G2, G3, G3b, G3c)
BOOTSTRAP = [["openalex", "5"], ["embed"], ["abstract"], ["voids", "--refresh"], ["fillvoids", "--openalex"],
             ["embed"], ["abstract"], ["trends"], ["migrate-domains"], ["litcheck"], ["calibrate"], ["combocheck"]]


@dataclass
class Options:
    cycles: int = 3
    fast: bool = True
    gen_model: str = "(既定)"
    compare_models: str = "gemma4:12b,gemma4:26b"
    judges: str = "gemma4:12b"
    refs: str = ""
    note: str = ""
    save_after: bool = True
    export_vault: bool = True
    ignore_gpu_limit: bool = False

    @classmethod
    def from_globals(cls, g: dict) -> "Options":
        m = {"CYCLES": "cycles", "FAST": "fast", "GEN_MODEL": "gen_model", "COMPARE_MODELS": "compare_models",
             "JUDGES": "judges", "REFS": "refs", "NOTE": "note", "SAVE_AFTER": "save_after",
             "EXPORT_VAULT": "export_vault", "IGNORE_GPU_LIMIT": "ignore_gpu_limit"}
        return cls(**{attr: g[key] for key, attr in m.items() if key in g})


def split_list(text: str) -> list[str]:
    return [x for x in re.split(r"[,\s、]+", text or "") if x]


def gen_model_of(o: Options) -> str | None:
    return None if o.gen_model in ("", "(既定)") else o.gen_model


def build_commands(mode: str, o: Options) -> list[list[str]]:
    """The `nous.py` commands of a mode, as argument lists. Raises ValueError for a missing input."""
    refs = split_list(o.refs)
    if mode == "状態確認":
        return [["check-llm", *DEFAULT_MODELS], ["autoloop", "--report"]]
    if mode == "自律ループ":
        cmd = ["autoloop", str(max(1, int(o.cycles)))]
        if o.fast:
            cmd.append("--fast")
        if gen_model_of(o):
            cmd += ["--gen-model", gen_model_of(o)]
        return [cmd]
    if mode == "採用の再点検":
        return [["knowncheck", *(refs or ["--accepted"]), "--show"]]
    if mode == "浄土へ書き出す":
        return [["vault"]]
    if mode == "善見天へ昇格":
        if not refs:
            raise ValueError("REFS に昇格する仮説を書いてください（例: hyp:31）")
        return [["vault", "--promote", r, *([o.note] if o.note else [])] for r in refs]
    if mode == "採用を取り消す":
        if not refs:
            raise ValueError("REFS に取り消す仮説を書いてください（例: hyp:38）")
        return [["autoloop", "--retract", r, o.note or "取り消し"] for r in refs]
    if mode == "モデル比較":
        models = split_list(o.compare_models)
        if not models:
            raise ValueError("COMPARE_MODELS にモデル名を書いてください")
        return [["knowncheck", flag, "--model", m] for m in models for flag in ("--controls", "--negatives")]
    if mode == "評価の一致度":
        judges = ",".join(split_list(o.judges)) or "gemma4:12b"
        return [["evaluate", "--judges", judges], ["reliability"]]
    if mode == "初回データ構築":
        return [list(c) for c in BOOTSTRAP]
    raise ValueError(f"unknown mode: {mode}")


def models_needed(mode: str, o: Options) -> list[str]:
    """Models (Ollama or API) the mode will use."""
    out: list[str] = []
    if mode in ("状態確認", "自律ループ", "初回データ構築"):
        out += list(DEFAULT_MODELS)
    if mode == "自律ループ" and gen_model_of(o):
        out.append(gen_model_of(o))
    if mode == "採用の再点検":
        out.append("gemma4:12b")
    if mode == "モデル比較":
        out += split_list(o.compare_models)
    if mode == "評価の一致度":
        out += ["gemma4:e2b", *split_list(o.judges)]
    return list(dict.fromkeys(out))


def gpu_problems(models: list[str], gpu_mib: int | None) -> list[str]:
    """Models that do not fit the GPU (empty when the GPU memory is unknown)."""
    if not gpu_mib:
        return []
    return [f"{m}: GPU {gpu_mib:,} MiB < 目安 {need:,} MiB" for m in models
            if (need := GPU_NEED_MIB.get(m)) and gpu_mib < need]


def mask(text: str, env=None) -> str:
    env = os.environ if env is None else env
    for k in SECRET_ENV:
        v = env.get(k)
        if v and len(v) >= 8:
            text = text.replace(v, "***")
    return text


_NOISE = re.compile(r"(\d+%\||it/s\]|s/it\]|^model\.safetensors|Loading weights:|^Batches:)")


def is_noise(line: str) -> bool:
    return bool(_NOISE.search(line))


def gpu_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or "GPU なし"
    except Exception:
        return "GPU なし"


def run_command(argv: list[str], log: list[str], cwd=None, echo: bool = True) -> int:
    """Run with live output; the cleaned, secret-masked text is appended to `log`."""
    t0 = time.time()
    log.append(f"$ {' '.join(argv)}")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=env, cwd=cwd)
    for raw in p.stdout:
        line = mask(raw.rstrip("\n"), env)
        if is_noise(line):
            continue
        if echo:
            print(line, flush=True)
        log.append(line)
    code = p.wait()
    log.append(f"[exit {code}, {time.time() - t0:.0f}s]")
    return code


def write_log(path: Path, mode: str, header: list[str], log: list[str]) -> Path:
    body = "\n".join(log)
    if len(body) > MAX_LOG_CHARS:
        body = body[:50_000] + "\n…（途中省略）…\n" + body[-(MAX_LOG_CHARS - 50_000):]
    text = f"# 実行ログ: {mode}\n\n" + "\n".join(f"- {h}" for h in header) + "\n\n## 出力\n\n```text\n" + body + "\n```\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def log_path(mode: str, now: float | None = None) -> Path:
    from nous.vault import default_vault_dir
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return default_vault_dir() / "実行ログ" / f"{stamp}-{mode}.md"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama(wait: int = 30) -> bool:
    if ollama_up():
        return True
    subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(wait):
        time.sleep(1)
        if ollama_up():
            return True
    return False


def run_from_form(g: dict) -> dict:
    """Entry point called by the notebook with its globals()."""
    mode = g.get("MODE", "状態確認")
    o = Options.from_globals(g)
    if not ROOT.exists() or not callable(g.get("save")):
        print("先に、上の「1. ワンセット準備」のセルを実行してください。")
        return {"ok": False, "why": "setup not run"}
    try:
        cmds = build_commands(mode, o)
    except ValueError as ex:
        print(f"入力が足りません: {ex}")
        return {"ok": False, "why": str(ex)}
    needed = models_needed(mode, o)
    mem = gpu_mib()
    bad = [] if o.ignore_gpu_limit else gpu_problems(needed, mem)
    if bad:
        print("この GPU では載らないモデルがあります（何も実行していません）:")
        print("\n".join("  " + b for b in bad))
        print("→ L4（24GB）以上に変えるか、IGNORE_GPU_LIMIT を ON にしてください。")
        return {"ok": False, "why": "gpu"}
    log: list[str] = []
    if needed:
        if not ensure_ollama():
            print("Ollama が起動していません。上の「1. ワンセット準備」を実行し直してください。")
            return {"ok": False, "why": "ollama"}
        from nous.llm import missing_local_models
        for m in missing_local_models(needed):
            print(f"取得: ollama pull {m}")
            run_command(["ollama", "pull", m], log)
    commit = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    header = [f"開始: {time.strftime('%Y-%m-%d %H:%M:%S')}", f"コード: {commit}", f"GPU: {gpu_name()}",
              f"設定: {o}"]
    ok = True
    t0 = time.time()
    for c in cmds:
        code = run_command([sys.executable, "nous.py", *c], log, cwd=ROOT)
        if code:
            ok = False
            print(f"\n[終了コード {code}] ここで止めます: {' '.join(c)}")
            break
    if o.save_after and mode in SAVES:
        print("\nsave():")
        g["save"]()
    if o.export_vault and mode in SAVES and ok:      # (mode 浄土へ書き出す already is the export)
        print("\n浄土へ書き出し:")
        run_command([sys.executable, "nous.py", "vault"], log, cwd=ROOT)
    header.append(f"所要: {time.time() - t0:.0f}秒 / 結果: {'成功' if ok else '失敗'}")
    path = None
    try:
        path = write_log(log_path(mode), mode, header, log)
        print(f"\nログを保存しました: {path}")
    except Exception as ex:                                   # a failed log must not hide the result
        print(f"\n（ログの保存に失敗: {type(ex).__name__}）")
    return {"ok": ok, "log": str(path) if path else None}
