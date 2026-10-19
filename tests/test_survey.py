import sqlite3

from nous.evaluation import combination as cb, literature, survey


def _db(tmp_path, monkeypatch):
    db = tmp_path / "n.db"
    for mod in (cb, literature, survey, __import__("nous.evaluation.store", fromlist=["x"])):
        monkeypatch.setattr(mod, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    literature._ensure_table(con)
    literature._ensure_controls_table(con)
    con.executescript("""
        CREATE TABLE explorations (id INTEGER PRIMARY KEY, query TEXT);
        CREATE TABLE hypotheses (id INTEGER PRIMARY KEY, exploration_id INTEGER, hypothesis_text TEXT);
        INSERT INTO explorations VALUES (1, 'q');
        INSERT INTO hypotheses VALUES (1, 1, 'rare pair only'), (2, 1, 'one studied pair'), (3, 1, 'close paper');
    """)

    def P(a, b, joint):
        return {"a": a, "b": b, "n_a": 1000, "n_b": 1000, "joint": joint, "expected": 0, "lift": 0,
                "adjacency": 0.5, "loose": None}

    cb._save_pairs(con, "hyp:1", [P("A", "B", 0), P("A", "C", 3)], "filter")
    cb._save_pairs(con, "hyp:2", [P("A", "B", 0), P("A", "C", 147)], "filter")      # a rare pair AND a studied pair
    cb._save_pairs(con, "hyp:3", [P("A", "B", 0)], "filter")
    for ref, sim in (("hyp:1", 0.40), ("hyp:2", 0.50), ("hyp:3", 0.73)):
        con.execute("INSERT INTO literature_checks (target_ref, rank, work_id, title, year, cited_by, sim) "
                    "VALUES (?, 1, 'W', 'T', 2000, 0, ?)", (ref, sim))
    con.execute("INSERT INTO literature_controls (label, novelty, top_title, top_year, top_sim) "
                "VALUES ('famous', 0.2, 'x', 1990, 0.8), ('famous2', 0.4, 'y', 1990, 0.6)")
    con.commit()
    con.close()
    return db


def test_rules_catch_what_they_are_meant_to(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    dec = survey.decisions(survey.load(db), threshold=0.70)
    assert dec["hyp:1"] == {"R0": False, "R1": False, "R2": False, "R3": False, "R4": False}
    assert dec["hyp:2"]["R0"] is False and dec["hyp:2"]["R1"] is True      # the rarest pair decides in R0
    assert dec["hyp:3"]["R2"] is True and dec["hyp:3"]["R0"] is False
    assert dec["hyp:3"]["R3"] and dec["hyp:2"]["R4"]


def test_control_recall_and_report(tmp_path, monkeypatch, capsys):
    db = _db(tmp_path, monkeypatch)
    assert survey.control_recall([("a", 0.8), ("b", 0.6)], thresholds=(0.7,)) == [(0.7, 1, 2)]
    survey.report(db, threshold=0.70)
    out = capsys.readouterr().out
    assert "stored hypotheses with pair data: 3" in out and "R1" in out
    assert "hyp:2" in out and "hyp:3" in out                     # listed as 'newly rejected'


def test_old_database_without_the_tables_does_not_crash(tmp_path):
    db = tmp_path / "old.db"
    sqlite3.connect(db).close()
    data = survey.load(db)
    assert data["pairs"] == {} and survey.decisions(data) == {}
