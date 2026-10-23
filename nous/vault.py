"""
浄土 (Jōdo): the memory archive. An Obsidian vault written from the NoUs database.

Everything NoUs has looked at is kept here, whether or not a hypothesis passed:

  善見天  the centre. `目次.md` (front page) and the hypotheses a human promoted
  浄土    kept notes: accepted candidates and not-yet-judged hypotheses, papers, field pairs
  灰      hypotheses the 焼却人 sorted out (rejected / retracted). Kept, with the reason.

The vault is a *derived export*: SQLite (`data/nous.db`, the 記憶域) stays the source of truth and
ChromaDB (the 憶質) the semantic index. Running `nous.py vault` again rewrites the machine-written
notes; anything that is not in the manifest (e.g. `人のメモ/`) is never touched.

焼却人 never deletes or edits a memory: it only decides the tier and says why. A tier is
  灰      validated < 0 (retracted), or the loop rejected it
  善見天  a human promoted it (`nous.py vault --promote hyp:31`), unless it was retracted
  浄土    everything else (accepted by the loop, or never judged)

Notes are named by id (`hyp-31.md`, `W2741809807.md`) so that they never collide and links survive
renames; the readable title is in the note and in `aliases`.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from nous.config import DB_PATH, ROOT_DIR
from nous.domains import parent_of

# names of the stores, shown on the front page
NAMES = {"記憶域": "SQLite（data/nous.db）作業用の正本", "憶質": "ChromaDB（意味の索引）"}

TIERS = ("善見天", "浄土", "灰")
DIR_CENTER, DIR_HYP, DIR_ASH = "善見天", "仮説", "灰"
DIR_SCEPTER, DIR_PAPER, DIR_PAIR, DIR_HUMAN = "討論の仮説", "論文", "分野ペア", "人のメモ"
FRONT_PAGE = f"{DIR_CENTER}/目次.md"
RUN_HISTORY = "実行履歴.md"
MANIFEST = ".nous-vault.json"
AUTO_ITEMS = ("novelty_embedding", "novelty_literature", "combo_novelty", "combo_checked")
MARK_PROMOTE = "promote"


def default_vault_dir() -> Path:
    """$NOUS_VAULT, else the 浄土 folder next to NoUs_data on Google Drive (Colab), else ./浄土."""
    if os.environ.get("NOUS_VAULT"):
        return Path(os.environ["NOUS_VAULT"])
    drive = Path("/content/drive/MyDrive")
    return (drive / "AI" / "projects" / "浄土") if drive.exists() else ROOT_DIR / "浄土"


# ------------------------------------------------------------------ #
#  Small text helpers (pure)                                          #
# ------------------------------------------------------------------ #

def yaml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(round(v, 4)) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(yaml_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def frontmatter(props: dict) -> str:
    lines = [f"{k}: {yaml_value(v)}" for k, v in props.items() if v is not None and v != ""]
    return "---\n" + "\n".join(lines) + "\n---\n"


def link_text(title: str | None) -> str:
    """A title that is safe inside [[target|title]]."""
    return re.sub(r"[\[\]|#^]", " ", title or "").strip() or "(no title)"


def wlink(target: str, title: str | None = None) -> str:
    return f"[[{target}|{link_text(title)}]]" if title else f"[[{target}]]"


def cell(x) -> str:
    return "" if x is None else str(x).replace("|", "¦").replace("\n", " ")


def quote(text: str) -> str:
    """A Markdown blockquote that survives line breaks."""
    return "\n".join("> " + line for line in (text or "").strip().splitlines()) or "> "


def short(text: str, n: int = 60) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------ #
#  焼却人: the tier of a hypothesis (pure)                            #
# ------------------------------------------------------------------ #

def classify(validated, run: dict | None, mark: str | None) -> tuple[str, str]:
    """(tier, reason). Never deletes anything: it only says where the memory is kept and why."""
    if validated is not None and validated < 0:
        return "灰", "取り消し（retract）"
    if mark == MARK_PROMOTE:
        return "善見天", "人が昇格"
    status = (run or {}).get("status")
    if status in ("rejected", "retracted"):
        return "灰", f"自律ループが棄却: {(run or {}).get('reason') or ''}".strip()
    if status == "accepted":
        return "浄土", "自律ループが採用（これらのチェックでは既存文献に見つからなかった。新発見の主張ではない）"
    return "浄土", "未判定"


# ------------------------------------------------------------------ #
#  Reading the database                                               #
# ------------------------------------------------------------------ #

def _has(conn, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _all(conn, table: str, sql: str, args=()) -> list[dict]:
    if not _has(conn, table):
        return []
    try:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    except sqlite3.OperationalError:
        return []


def collect(db_path) -> dict:
    """Read everything the vault needs (read-only). Missing tables simply give empty lists."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        d: dict = {}
        d["hyps"] = _all(conn, "hypotheses", """
            SELECT h.id, h.hypothesis_text AS text, h.confidence, h.source_ids, h.validated,
                   h.timestamp, e.query, e.domain_a, e.domain_b
            FROM hypotheses h LEFT JOIN explorations e ON e.id = h.exploration_id ORDER BY h.id""")
        d["runs"] = _all(conn, "loop_runs", "SELECT * FROM loop_runs ORDER BY id")
        d["evals"] = _all(conn, "evaluations", "SELECT target_ref, rater, item, score FROM evaluations")
        d["lit"] = _all(conn, "literature_checks",
                        "SELECT target_ref, rank, work_id, title, year, cited_by, sim FROM literature_checks ORDER BY target_ref, rank")
        d["pairs"] = _all(conn, "combo_checks", "SELECT * FROM combo_checks")
        d["pair_titles"] = _all(conn, "combo_titles", "SELECT concept_a, concept_b, title, year, work_id FROM combo_titles")
        d["groups"] = _all(conn, "combo_groups", "SELECT * FROM combo_groups")
        d["known"] = _all(conn, "known_checks", "SELECT * FROM known_checks ORDER BY id")
        d["marks"] = _all(conn, "vault_marks", "SELECT * FROM vault_marks")
        d["articles"] = _all(conn, "articles", "SELECT title, domain, summary, url FROM articles")
        d["links"] = _all(conn, "field_links", "SELECT field_a, field_b, observed, lift FROM field_links")
        d["scepters"] = _all(conn, "scepter_hypotheses", """
            SELECT s.id, s.scepter, s.domain, s.hypothesis AS text, s.confidence, s.structural_pattern,
                   s.validation_question, d.query FROM scepter_hypotheses s LEFT JOIN debates d ON d.id = s.debate_id ORDER BY s.id""")
        d["db_bytes"] = os.path.getsize(db_path)
        return d
    finally:
        conn.close()


def mark_hypothesis(ref: str, mark: str | None, note: str = "", db_path=None) -> None:
    """Record (or, with mark=None, clear) a human decision in the database."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS vault_marks (
        ref TEXT PRIMARY KEY, mark TEXT, note TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    if mark is None:
        conn.execute("DELETE FROM vault_marks WHERE ref = ?", (ref,))
    else:
        conn.execute("INSERT OR REPLACE INTO vault_marks (ref, mark, note) VALUES (?, ?, ?)", (ref, mark, note))
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ #
#  Building the notes (pure: data -> {path: text})                    #
# ------------------------------------------------------------------ #

def work_key(work_id: str | None, title: str | None) -> str:
    """Note name of a paper: its OpenAlex id (`W123`), else a stable key from the title."""
    if work_id:
        return re.sub(r"[^\w-]", "", work_id.rsplit("/", 1)[-1])
    return "t-" + format(abs(hash_title(title or "")), "x")


def hash_title(t: str) -> int:
    """Stable (not salted like hash()) small hash for titles without an id."""
    h = 1469598103934665603
    for ch in t.strip().lower():
        h = ((h ^ ord(ch)) * 1099511628211) % (1 << 64)
    return h


def build(d: dict, now: str) -> dict[str, str]:
    arts_by_title = {(a["title"] or "").strip().lower(): a for a in d["articles"]}
    runs_by_ref: dict[str, dict] = {}
    for r in d["runs"]:
        if r.get("hyp_ref"):
            runs_by_ref[r["hyp_ref"]] = r                       # rows are in id order: the last wins
    marks = {m["ref"]: m for m in d["marks"]}
    known = {}
    for k in d["known"]:
        known[k["target_ref"]] = k                               # latest check wins
    lit_by_ref: dict[str, list[dict]] = {}
    for l in d["lit"]:
        lit_by_ref.setdefault(l["target_ref"], []).append(l)
    pairs_by_ref: dict[str, list[dict]] = {}
    for p in d["pairs"]:
        pairs_by_ref.setdefault(p["target_ref"], []).append(p)
    titles_by_pair: dict[tuple, list[dict]] = {}
    for t in d["pair_titles"]:
        titles_by_pair.setdefault((t["concept_a"], t["concept_b"]), []).append(t)
    group_by_ref = {g["target_ref"]: g for g in d["groups"]}
    scores = _score_table(d["evals"])
    links = {frozenset((l["field_a"], l["field_b"])): l for l in d["links"]}

    papers: dict[str, dict] = {}          # note name -> {title, year, cited_by, url, summary, roles, refs}

    def paper(work_id, title, year=None, cited_by=None, url=None, summary=None) -> str:
        key = work_key(work_id, title)
        p = papers.setdefault(key, {"title": title, "year": year, "cited_by": cited_by,
                                    "url": url, "summary": summary, "refs": []})
        for field, val in (("title", title), ("year", year), ("cited_by", cited_by),
                           ("url", url), ("summary", summary)):
            if val and not p.get(field):
                p[field] = val
        return key

    files: dict[str, str] = {}
    hyp_rows = []                         # for the front page / indexes

    for h in d["hyps"]:
        ref = f"hyp:{h['id']}"
        run = runs_by_ref.get(ref)
        mark = (marks.get(ref) or {}).get("mark")
        tier, why = classify(h.get("validated"), run, mark)
        sc = scores.get(ref, {})
        # papers this hypothesis touched
        closest = []
        for l in lit_by_ref.get(ref, [])[:5]:
            key = paper(l["work_id"], l["title"], l["year"], l["cited_by"])
            papers[key]["refs"].append((ref, "最も近い既存研究"))
            closest.append((key, l))
        bridge = []
        try:
            sources = json.loads(h.get("source_ids") or "[]")
        except json.JSONDecodeError:
            sources = []
        for t in sources:
            a = arts_by_title.get((t or "").strip().lower())
            key = paper(a["url"] if a else None, t, url=(a or {}).get("url"), summary=(a or {}).get("summary"))
            papers[key]["refs"].append((ref, "橋渡し（生成の材料）"))
            bridge.append((key, t))
        pair_rows = pairs_by_ref.get(ref, [])
        evidence = []
        for p in pair_rows:
            for t in titles_by_pair.get((p["concept_a"], p["concept_b"]), [])[:3]:
                key = paper(t["work_id"], t["title"], t["year"])
                papers[key]["refs"].append((ref, f"概念の共起: {p['concept_a']} × {p['concept_b']}"))
                evidence.append((p, key, t))
        folder = {"善見天": DIR_CENTER, "浄土": DIR_HYP, "灰": DIR_ASH}[tier]
        path = f"{folder}/hyp-{h['id']}.md"
        files[path] = _render_hypothesis(h, ref, tier, why, run, (marks.get(ref) or {}), known.get(ref),
                                         sc, closest, bridge, pair_rows, evidence,
                                         group_by_ref.get(ref), links)
        hyp_rows.append({"ref": ref, "id": h["id"], "tier": tier, "path": path, "text": h.get("text") or "",
                         "status": (run or {}).get("status") or "unjudged", "created": h.get("timestamp"),
                         "composite": sc.get("composite_excl_novelty")})

    for s in d["scepters"]:
        ref = f"scepter:{s['id']}"
        sc = scores.get(ref, {})
        files[f"{DIR_SCEPTER}/scepter-{s['id']}.md"] = (
            frontmatter({"type": "討論の仮説", "ref": ref, "tier": "浄土", "scepter": s.get("scepter"),
                         "parent": parent_of(s.get("scepter") or ""),
                         "domain": s.get("domain"), "confidence": s.get("confidence"),
                         "composite": sc.get("composite_excl_novelty"), "tags": ["討論の仮説", "浄土"],
                         "aliases": [ref]})
            + f"\n# {ref}（{s.get('scepter') or ''}）\n\n" + quote(s.get("text")) + "\n\n"
            + f"- 討論のテーマ: {s.get('query') or ''}\n- 構造パターン: {s.get('structural_pattern') or ''}\n"
            + f"- 検証のための問い: {s.get('validation_question') or ''}\n\n"
            + _score_section(sc))

    for key, p in sorted(papers.items()):
        files[f"{DIR_PAPER}/{key}.md"] = _render_paper(key, p)

    pair_runs: dict[frozenset, list[dict]] = {}
    for r in d["runs"]:
        pair_runs.setdefault(frozenset((r["field_a"], r["field_b"])), []).append(r)
    for k, runs in pair_runs.items():
        files[f"{DIR_PAIR}/{_pair_name(runs[0])}.md"] = _render_pair(runs, links.get(k))

    files[RUN_HISTORY] = _render_runs(d["runs"], runs_by_ref)
    files[FRONT_PAGE] = _render_front(hyp_rows, d, now)
    return files


def _pair_name(run: dict) -> str:
    a, b = sorted((str(run["field_a"]).rsplit("/", 1)[-1], str(run["field_b"]).rsplit("/", 1)[-1]))
    return f"pair-{a}-{b}"


def _score_table(evals: list[dict]) -> dict[str, dict]:
    by: dict[str, dict] = {}
    for e in evals:
        if e["score"] is None:
            continue
        t = by.setdefault(e["target_ref"], {"items": {}, "auto": {}, "raters": set()})
        if e["item"] in AUTO_ITEMS:
            t["auto"][e["item"]] = e["score"]
        else:
            t["items"].setdefault(e["item"], []).append(e["score"])
            t["raters"].add(e["rater"])
    out = {}
    for ref, t in by.items():
        means = {i: mean(v) for i, v in t["items"].items()}
        rest = [v for i, v in means.items() if i != "novelty"]
        out[ref] = {"item_means": means, "n_raters": len(t["raters"]), "auto": t["auto"],
                    "human": sorted(r for r in t["raters"] if str(r).startswith("human")),
                    "composite_excl_novelty": mean(rest)}
    return out


def _score_section(sc: dict) -> str:
    if not sc:
        return "## 評価\n\n（評価なし）\n"
    lines = ["## 評価", "", f"評価者 {sc['n_raters']} 人の平均（novelty は LLM 間で一致しないため合成値から除外）。"
             "評価者の組が仮説ごとに違うことがあるので、行どうしの比較には注意。", "",
             "| 項目 | 平均 |", "|---|---|"]
    lines += [f"| {i} | {fmt(v)} |" for i, v in sc["item_means"].items()]
    lines.append(f"| 合成（novelty 除く） | {fmt(sc['composite_excl_novelty'])} |")
    if sc["human"]:
        lines.append(f"\n人の評価あり: {', '.join(sc['human'])}")
    return "\n".join(lines) + "\n"


def _render_hypothesis(h, ref, tier, why, run, mark, known, sc, closest, bridge, pair_rows, evidence, group, links) -> str:
    run = run or {}
    props = {
        "type": "仮説", "ref": ref, "tier": tier, "status": run.get("status") or "unjudged",
        "created": h.get("timestamp"), "field_a": run.get("name_a") or h.get("domain_a"),
        "field_b": run.get("name_b") or h.get("domain_b"), "confidence": h.get("confidence"),
        "composite": sc.get("composite_excl_novelty") if sc else run.get("composite"),
        "combo": run.get("combo_label"), "lit_novelty": run.get("lit_novelty"),
        "known": (known or {}).get("verdict"), "reward": run.get("reward"),
        "tags": ["仮説", tier], "aliases": [ref],
    }
    out = [frontmatter(props), f"# {ref}  {short(h.get('text') or '', 50)}\n", quote(h.get("text")) + "\n"]
    out += ["## 焼却人の分類", "", f"- 層: **{tier}**", f"- 理由: {why}"]
    if mark.get("note"):
        out.append(f"- 人のメモ（昇格時）: {mark['note']}")
    out += ["", "## 実行（自律ループ）", ""]
    if run:
        pair_title = f"{run.get('name_a')} × {run.get('name_b')}"
        out += [f"- 結果: {run.get('status')} — {run.get('reason') or ''}",
                f"- 分野ペア: {wlink(_pair_name(run), pair_title)}",
                f"- クエリ: {run.get('query') or ''}",
                f"- 組合せ判定: {run.get('combo_label')} / 文献の新規性: {fmt(run.get('lit_novelty'))} / 報酬: {fmt(run.get('reward'))}"]
    else:
        out.append("- 自律ループの記録なし（手動または旧い仮説）")
    out += ["", "## 既知研究チェック", ""]
    if known:
        out += [f"- 判定: **{known.get('verdict')}**（{known.get('model')}）", f"- 理由: {known.get('why') or ''}"]
        if known.get("title"):
            out.append(f"- 既出と判定された論文: {known['title']} ({known.get('year')}) — 「{known.get('evidence') or ''}」")
        try:
            works = json.loads(known.get("works") or "[]")
        except json.JSONDecodeError:
            works = []
        out += [f"  - {w}" for w in works]
    else:
        out.append("（記録なし）")
    out += ["", "## 概念の組合せ", ""]
    if pair_rows:
        out += ["| 概念A | 概念B | 共起 | 緩い共起 | 隣接性 |", "|---|---|---|---|---|"]
        out += [f"| {cell(p['concept_a'])} | {cell(p['concept_b'])} | {cell(p['joint'])} | {cell(p.get('loose'))} | {fmt(p.get('adjacency'))} |"
                for p in pair_rows]
        if group:
            out.append(f"\n全体の組合せ（参考）: {group.get('concepts')} — 共起 {group.get('joint')}、緩い共起 {group.get('loose')}")
    else:
        out.append("（記録なし）")
    if evidence:
        out += ["", "共起した論文の例:"]
        seen = set()
        for p, key, t in evidence:
            if key not in seen:
                seen.add(key)
                out.append(f"- {p['concept_a']} × {p['concept_b']}: {wlink(key, t['title'])} ({t.get('year')})")
    out += ["", "## 最も近い既存研究", ""]
    out += [f"{i}. {wlink(k, l['title'])} ({l.get('year')}、類似度 {fmt(l.get('sim'))}、被引用 {l.get('cited_by')})"
            for i, (k, l) in enumerate(closest, 1)] or ["（記録なし）"]
    out += ["", "## 生成の材料にした論文（橋渡し）", ""]
    out += [f"- {wlink(k, t)}" for k, t in bridge] or ["（記録なし）"]
    out += ["", _score_section(sc)]
    return "\n".join(out)


def _render_paper(key: str, p: dict) -> str:
    props = {"type": "論文", "openalex": key if key.startswith("W") else None, "title": p.get("title"),
             "year": p.get("year"), "cited_by": p.get("cited_by"), "url": p.get("url"),
             "tags": ["論文"], "aliases": [p.get("title")] if p.get("title") else None}
    out = [frontmatter(props), f"# {p.get('title') or key}\n"]
    if p.get("summary"):
        out += ["> " + p["summary"].strip().replace("\n", " "), "（要旨の先頭 500 字まで）\n"]
    out.append("## 参照元\n")
    seen = set()
    for ref, role in p["refs"]:
        if (ref, role) not in seen:
            seen.add((ref, role))
            out.append(f"- {wlink('hyp-' + ref.split(':')[1])} — {role}")
    return "\n".join(out) + "\n"


def _render_pair(runs: list[dict], link: dict | None) -> str:
    r0 = runs[0]
    props = {"type": "分野ペア", "field_a": r0.get("name_a"), "field_b": r0.get("name_b"),
             "shared_works": (link or {}).get("observed"), "lift": (link or {}).get("lift"),
             "tags": ["分野ペア"], "aliases": [f"{r0.get('name_a')} × {r0.get('name_b')}"]}
    out = [frontmatter(props), f"# {r0.get('name_a')} × {r0.get('name_b')}\n"]
    if link and link.get("observed") is not None:
        out.append(f"共有論文 {link['observed']:,} 件、lift {fmt(link.get('lift'), 3)}（低いほど分野どうしの接点が少ない）\n")
    out += ["## 実行", ""]
    for r in sorted(runs, key=lambda x: x["id"]):
        who = wlink("hyp-" + r["hyp_ref"].split(":")[1]) if r.get("hyp_ref") else "（仮説なし）"
        out.append(f"- run {r['id']} ({r.get('created_at')}): {r.get('status')} {who} — {r.get('reason') or ''}")
    return "\n".join(out) + "\n"


def _render_runs(runs: list[dict], runs_by_ref: dict) -> str:
    out = [frontmatter({"type": "実行履歴", "tags": ["実行履歴"]}), "# 実行履歴（自律ループ）\n",
           "| run | 日時 | 分野ペア | 結果 | 仮説 | composite | 組合せ | 文献新規性 | 理由 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda x: -x["id"]):
        h = wlink("hyp-" + r["hyp_ref"].split(":")[1], r["hyp_ref"]) if r.get("hyp_ref") else ""
        out.append(f"| {r['id']} | {cell(r.get('created_at'))} | {cell(r.get('name_a'))} × {cell(r.get('name_b'))} | "
                   f"{cell(r.get('status'))} | {h} | {fmt(r.get('composite'))} | {cell(r.get('combo_label'))} | "
                   f"{fmt(r.get('lit_novelty'))} | {cell(short(r.get('reason') or '', 80))} |")
    return "\n".join(out) + "\n"


def _render_front(hyp_rows: list[dict], d: dict, now: str) -> str:
    by_tier = {t: [h for h in hyp_rows if h["tier"] == t] for t in TIERS}
    out = [frontmatter({"type": "目次", "tags": ["目次"]}), "# 善見天（浄土の目次）\n",
           f"最終書き出し: {now}\n",
           "NoUs が見たものを、仮説が通ったかどうかに関係なく残す場所です。この目次と各ノートは "
           "`nous.py vault` が書き出します（手で書き換えても、次の書き出しで上書きされます）。"
           "自分のメモは `人のメモ/` に書いてください（上書きされません）。\n",
           "## 構成", ""]
    out += [f"- **{k}**: {v}" for k, v in NAMES.items()]
    out += ["- **メモキーパー**: 収集（OpenAlex などから、判断せずに書き込む）",
            "- **焼却人**: 選別（層を決めて理由を残す。記憶そのものは削除も書き換えもしない）", "",
            "## 層", "", "| 層 | 件数 | 意味 |", "|---|---|---|",
            f"| 善見天 | {len(by_tier['善見天'])} | 人が昇格させた仮説 |",
            f"| 浄土 | {len(by_tier['浄土'])} | 採用（候補）または未判定 |",
            f"| 灰 | {len(by_tier['灰'])} | 棄却・取り消し（理由つきで保存） |", ""]
    out.append("## 善見天（昇格した仮説）\n")
    out += [f"- {wlink('hyp-' + str(h['id']), h['ref'])} {short(h['text'])}" for h in by_tier["善見天"]] or ["（まだありません。`nous.py vault --promote hyp:N` で昇格）"]
    out.append("\n## 浄土（採用の候補）\n")
    acc = [h for h in by_tier["浄土"] if h["status"] == "accepted"]
    out += [f"- {wlink('hyp-' + str(h['id']), h['ref'])} {short(h['text'])}" for h in acc] or ["（なし）"]
    out.append("\n## 最近の仮説\n")
    for h in sorted(hyp_rows, key=lambda x: -x["id"])[:15]:
        out.append(f"- {wlink('hyp-' + str(h['id']), h['ref'])}［{h['tier']}・{h['status']}］{short(h['text'])}")
    out += ["", "## 他", "", f"- {wlink(RUN_HISTORY[:-3], '実行履歴')}",
            f"- 仮説 {len(hyp_rows)} 件、討論の仮説 {len(d['scepters'])} 件、実行 {len(d['runs'])} 回"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ #
#  Writing (the only part that touches the vault folder)              #
# ------------------------------------------------------------------ #

def write_vault(out: Path, files: dict[str, str], now: str, force: bool = False) -> dict:
    """Write changed notes, keep the rest. Removes only notes that an earlier export wrote and
    that are no longer produced; a much smaller export removes nothing unless `force`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST
    previous: list[str] = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
        except (json.JSONDecodeError, OSError):
            previous = []
    stats = {"written": 0, "unchanged": 0, "removed": 0, "kept_stale": 0}
    for rel, text in files.items():
        p = out / rel
        if p.exists() and p.read_text(encoding="utf-8") == text:
            stats["unchanged"] += 1
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, p)
        stats["written"] += 1
    stale = [r for r in previous if r not in files]
    tracked = set(files)
    if stale and not force and len(files) < 0.5 * len(previous):
        stats["kept_stale"] = len(stale)                     # looks like a wrong/empty database: remove nothing
        tracked.update(stale)                                # ...but keep tracking them for a later --force
        stale = []
    for rel in stale:
        p = out / rel
        if p.exists():
            p.unlink()
            stats["removed"] += 1
            try:
                p.parent.rmdir()
            except OSError:
                pass
    human = out / DIR_HUMAN
    human.mkdir(exist_ok=True)
    readme = human / "README.md"
    if not readme.exists():
        readme.write_text("# 人のメモ\n\nここは NoUs が書き換えない場所です。自分のメモを置いてください。\n"
                          "他のノートへは `[[hyp-31]]` のようにリンクできます。\n", encoding="utf-8")
    manifest_path.write_text(json.dumps({"generator": "nous.vault", "version": 1, "generated_at": now,
                                         "files": sorted(tracked)}, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    return stats


def export(out=None, db_path=None, force: bool = False, now: str | None = None) -> dict:
    now = now or time.strftime("%Y-%m-%d %H:%M")
    out = Path(out) if out else default_vault_dir()
    d = collect(db_path or DB_PATH)
    files = build(d, now)
    stats = write_vault(out, files, now, force=force)
    folders: dict[str, int] = {}
    for rel in files:
        top = rel.split("/", 1)[0] if "/" in rel else rel
        folders[top] = folders.get(top, 0) + 1
    # check from this side that the notes are on the (possibly Drive-mounted) disk
    missing = [rel for rel in files if not (out / rel).exists()]
    stats.update(out=str(out), notes=len(files), hypotheses=len(d["hyps"]), db_bytes=d["db_bytes"],
                 folders=folders, missing=missing, front_page=files.get(FRONT_PAGE, ""))
    return stats
