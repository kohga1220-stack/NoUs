"""
Evaluation storage — every score given to every hypothesis by every rater.

Targets are referenced by a string `ref`:
    hyp:<id>      a row in `hypotheses`          (nous.py hyp)
    scepter:<id>  a row in `scepter_hypotheses`  (one Scepter's hypothesis in a debate)
    verdict:<id>  the NOUS verdict of debate <id>
"""
from __future__ import annotations
import json
import sqlite3

from nous.config import DB_PATH


def ensure_tables(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            target_ref     TEXT NOT NULL,
            rater          TEXT NOT NULL,     -- e.g. "llm:gemma4:e2b:Scepter-N", "human:kohga"
            item           TEXT NOT NULL,     -- rubric item or automatic metric name
            score          REAL,
            rationale      TEXT,
            prompt_version TEXT,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (target_ref, rater, item)
        );
    """)
    conn.commit()
    conn.close()


def save_scores(target_ref: str, rater: str, scores: dict[str, float],
                rationales: dict[str, str] | None = None,
                prompt_version: str = "", db_path=None):
    ensure_tables(db_path)
    rationales = rationales or {}
    conn = sqlite3.connect(db_path or DB_PATH)
    for item, score in scores.items():
        conn.execute("""
            INSERT OR REPLACE INTO evaluations
            (target_ref, rater, item, score, rationale, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (target_ref, rater, item, score, rationales.get(item, ""), prompt_version))
    conn.commit()
    conn.close()


def load_scores(item: str | None = None, db_path=None) -> list[dict]:
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path or DB_PATH)
    sql = "SELECT target_ref, rater, item, score FROM evaluations"
    args: tuple = ()
    if item:
        sql += " WHERE item = ?"
        args = (item,)
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [{"target_ref": r[0], "rater": r[1], "item": r[2], "score": r[3]} for r in rows]


def rated_by(rater: str, db_path=None) -> set[str]:
    ensure_tables(db_path)
    conn = sqlite3.connect(db_path or DB_PATH)
    rows = conn.execute("SELECT DISTINCT target_ref FROM evaluations WHERE rater = ?",
                        (rater,)).fetchall()
    conn.close()
    return {r[0] for r in rows}


def _table_exists(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def load_targets(db_path=None) -> list[dict]:
    """All evaluable hypotheses as {ref, query, text, domain}."""
    conn = sqlite3.connect(db_path or DB_PATH)
    targets: list[dict] = []

    if _table_exists(conn, "hypotheses") and _table_exists(conn, "explorations"):
        for hid, query, text in conn.execute("""
            SELECT h.id, e.query, h.hypothesis_text
            FROM hypotheses h JOIN explorations e ON h.exploration_id = e.id
        """):
            targets.append({"ref": f"hyp:{hid}", "query": query, "text": text or "",
                            "domain": "nous"})

    if _table_exists(conn, "scepter_hypotheses") and _table_exists(conn, "debates"):
        for sid, query, scepter, text in conn.execute("""
            SELECT s.id, d.query, s.scepter, s.hypothesis
            FROM scepter_hypotheses s JOIN debates d ON s.debate_id = d.id
        """):
            targets.append({"ref": f"scepter:{sid}", "query": query, "text": text or "",
                            "domain": scepter})

        for did, query, verdict in conn.execute("SELECT id, query, verdict FROM debates"):
            try:
                v = json.loads(verdict or "{}")
            except json.JSONDecodeError:
                v = {}
            text = v.get("verdict", "")
            targets.append({"ref": f"verdict:{did}", "query": query, "text": text,
                            "domain": "NOUS"})

    conn.close()
    # parse-error placeholders are not real hypotheses
    return [t for t in targets if t["text"] and not t["text"].startswith("[")]
