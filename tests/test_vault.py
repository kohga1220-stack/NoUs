import json
import sqlite3

import pytest

from nous import loop, vault
from nous.collector import openalex, wikipedia
from nous.evaluation import combination, literature, store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "nous.db"
    for mod in (wikipedia, loop, openalex, store, combination, literature):
        monkeypatch.setattr(mod, "DB_PATH", path)
    monkeypatch.setattr(wikipedia, "DATA_DIR", tmp_path)
    wikipedia.init_db()
    conn = sqlite3.connect(path)
    loop._ensure_table(conn)
    literature._ensure_table(conn)
    combination._ensure_table(conn)
    openalex._ensure_tables(conn)
    store.ensure_tables(path)
    conn.execute("INSERT INTO articles (title, domain, summary, url) VALUES (?,?,?,?)",
                 ("Bridge paper", "physics", "An abstract about bridges.", "https://openalex.org/W111"))
    conn.execute("INSERT INTO explorations (query, domain_a, domain_b) VALUES ('q', 'A', 'B')")
    for text, src in (("Hypothesis one: X scales with Y.", ["Bridge paper"]), ("Hypothesis two.", []),
                      ("Hypothesis three, retracted.", [])):
        conn.execute("INSERT INTO hypotheses (exploration_id, hypothesis_text, confidence, source_ids) VALUES (1,?,0.5,?)",
                     (text, json.dumps(src)))
    conn.execute("UPDATE hypotheses SET validated = -1 WHERE id = 3")
    conn.execute("INSERT INTO field_links (field_a, field_b, observed, expected, lift) VALUES ('1','2',5000,10,0.1)")
    for ref, status, reason in (("hyp:1", "accepted", "not found"), ("hyp:2", "rejected", "combination already studied")):
        conn.execute("INSERT INTO loop_runs (field_a, field_b, name_a, name_b, query, hyp_ref, status, reason, composite, combo_label, lit_novelty, reward)"
                     " VALUES ('1','2','Physics','Biology','q',?,?,?,3.5,'unexplored',0.4,0.7)", (ref, status, reason))
    conn.execute("INSERT INTO loop_runs (field_a, field_b, name_a, name_b, status, reason) VALUES ('3','4','C','D','error','boom')")
    conn.execute("INSERT INTO literature_checks (target_ref, rank, work_id, title, year, cited_by, sim) VALUES ('hyp:1',1,'W222','Close work | with bar',2001,9,0.81)")
    conn.execute("INSERT INTO combo_checks (target_ref, concept_a, concept_b, n_a, n_b, joint, expected, lift, method, adjacency, loose)"
                 " VALUES ('hyp:1','X','Y',1000,1000,2,1.0,2.0,'m',0.5,30)")
    conn.execute("INSERT INTO combo_titles (concept_a, concept_b, title, year, work_id) VALUES ('X','Y','Joint paper',2010,'W333')")
    conn.commit()
    conn.close()
    store.save_scores("hyp:1", "llm:a", {"novelty": 2, "testability": 4, "specificity": 4, "coherence": 5, "structural_depth": 3})
    store.save_scores("hyp:1", "llm:b", {"novelty": 3, "testability": 5, "specificity": 4, "coherence": 4, "structural_depth": 3})
    from nous.evaluation.knowncheck import save_result
    save_result("hyp:1", "gemma4:12b", {"verdict": "related", "why": "close but different", "works": ["Close work (2001)"]}, db_path=path)
    return path


def _export(db, tmp_path, **kw):
    out = tmp_path / "浄土"
    return out, vault.export(out=out, db_path=db, now="2026-10-20 10:00", **kw)


def test_export_writes_tiers_papers_and_history(db, tmp_path):
    out, st = _export(db, tmp_path)
    assert (out / "仮説" / "hyp-1.md").exists()                    # accepted -> 浄土 tier
    assert (out / "灰" / "hyp-2.md").exists()                      # rejected -> 灰, kept with the reason
    assert (out / "灰" / "hyp-3.md").exists()                      # retracted -> 灰
    assert (out / "実行履歴.md").exists() and (out / "善見天" / "目次.md").exists()
    assert (out / "論文" / "W111.md").exists() and (out / "論文" / "W222.md").exists() and (out / "論文" / "W333.md").exists()
    assert (out / "分野ペア" / "pair-1-2.md").exists()
    assert (out / "人のメモ" / "README.md").exists()
    assert st["hypotheses"] == 3 and st["written"] == st["notes"] and st["unchanged"] == 0


def test_hypothesis_note_has_everything_that_was_judged(db, tmp_path):
    out, _ = _export(db, tmp_path)
    note = (out / "仮説" / "hyp-1.md").read_text(encoding="utf-8")
    assert note.startswith("---\n") and 'tier: "浄土"' in note and 'known: "related"' in note
    assert "close but different" in note and "Close work (2001)" in note          # known-check why and works
    assert "[[W222|Close work   with bar]]" in note                               # '|' in a title cannot break the link
    assert "[[W111|Bridge paper]]" in note and "[[W333|Joint paper]]" in note
    assert "| X | Y | 2 | 30 | 0.50 |" in note
    assert "評価者 2 人" in note and "| testability | 4.50 |" in note
    assert "新発見の主張ではない" in note
    ash = (out / "灰" / "hyp-2.md").read_text(encoding="utf-8")
    assert "combination already studied" in ash and 'tier: "灰"' in ash
    assert "取り消し" in (out / "灰" / "hyp-3.md").read_text(encoding="utf-8")


def test_paper_note_lists_what_refers_to_it(db, tmp_path):
    out, _ = _export(db, tmp_path)
    p = (out / "論文" / "W222.md").read_text(encoding="utf-8")
    assert 'openalex: "W222"' in p and "[[hyp-1]]" in p and "最も近い既存研究" in p
    assert "An abstract about bridges." in (out / "論文" / "W111.md").read_text(encoding="utf-8")


def test_second_export_changes_nothing_and_promotion_moves_the_note(db, tmp_path):
    out, first = _export(db, tmp_path)
    _, second = _export(db, tmp_path)
    assert second["written"] == 0 and second["unchanged"] == first["notes"]      # nothing to sync to Drive
    vault.mark_hypothesis("hyp:2", vault.MARK_PROMOTE, "worth a look", db_path=db)
    _export(db, tmp_path)
    assert (out / "善見天" / "hyp-2.md").exists() and not (out / "灰" / "hyp-2.md").exists()   # moved, not duplicated
    assert "worth a look" in (out / "善見天" / "hyp-2.md").read_text(encoding="utf-8")
    assert "hyp-2" in (out / "善見天" / "目次.md").read_text(encoding="utf-8")
    vault.mark_hypothesis("hyp:2", None, db_path=db)
    _export(db, tmp_path)
    assert (out / "灰" / "hyp-2.md").exists()


def test_human_notes_and_foreign_files_are_never_touched(db, tmp_path):
    out, _ = _export(db, tmp_path)
    mine = out / "人のメモ" / "my.md"
    mine.write_text("mine", encoding="utf-8")
    other = out / "仮説" / "hyp-99.md"                      # not written by the exporter
    other.write_text("not in the manifest", encoding="utf-8")
    (out / ".obsidian").mkdir()
    _export(db, tmp_path)
    assert mine.read_text(encoding="utf-8") == "mine" and other.exists() and (out / ".obsidian").exists()


def test_a_much_smaller_export_removes_nothing_unless_forced(db, tmp_path):
    out, _ = _export(db, tmp_path)
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    st = vault.export(out=out, db_path=empty, now="t")
    assert st["kept_stale"] > 0 and st["removed"] == 0 and (out / "仮説" / "hyp-1.md").exists()
    st = vault.export(out=out, db_path=empty, now="t", force=True)
    assert st["removed"] > 0 and not (out / "仮説" / "hyp-1.md").exists()
    assert (out / "人のメモ" / "README.md").exists()


def test_classify_tiers():
    assert vault.classify(-1, {"status": "accepted"}, vault.MARK_PROMOTE)[0] == "灰"      # retracted wins
    assert vault.classify(0, {"status": "rejected"}, vault.MARK_PROMOTE)[0] == "善見天"   # a human decision wins over the loop
    assert vault.classify(0, {"status": "rejected", "reason": "x"}, None)[0] == "灰"
    assert vault.classify(0, {"status": "accepted"}, None)[0] == "浄土"
    assert vault.classify(0, None, None) == ("浄土", "未判定")


def test_yaml_and_links_are_safe():
    assert vault.frontmatter({"a": 'say "hi"', "b": None, "c": 0, "d": ["x", "y"], "e": True}) == \
        '---\na: "say \\"hi\\""\nc: 0\nd: ["x", "y"]\ne: true\n---\n'
    assert vault.wlink("W1", "a|b]c") == "[[W1|a b c]]"
    assert vault.work_key("https://openalex.org/W123", "t") == "W123"
    assert vault.work_key(None, "Same Title") == vault.work_key(None, " same title ")   # stable, id-free key


def test_multiline_hypothesis_stays_inside_the_blockquote():
    assert vault.quote("a\nb") == "> a\n> b"


def test_export_reports_what_is_on_disk(db, tmp_path):
    out, st = _export(db, tmp_path)
    assert st["missing"] == [] and st["folders"]["論文"] == 3 and st["folders"]["灰"] == 2
    assert st["folders"]["仮説"] == 1 and st["folders"]["実行履歴.md"] == 1 and "善見天（浄土の目次）" in st["front_page"]
