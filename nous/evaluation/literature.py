"""
Literature check — is a hypothesis already in the published record?

For each hypothesis, OpenAlex semantic search (GTE-Large embeddings over ~300M
titles + abstracts) returns the closest existing works. Nous re-embeds those works
with its own model and records:

  novelty_literature = 1 - max cosine(hypothesis, closest works)

together with the closest works themselves, so a human can see *what* prior work
the hypothesis resembles. Cost: one semantic search per hypothesis ($0.001 each).
"""
from __future__ import annotations
import sqlite3
import time

from nous.config import DB_PATH
from nous.evaluation import store

RATER = "auto:openalex"
ITEM  = "novelty_literature"


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS literature_checks (
            target_ref  TEXT,
            rank        INTEGER,
            work_id     TEXT,
            title       TEXT,
            year        INTEGER,
            cited_by    INTEGER,
            sim         REAL,
            checked_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (target_ref, rank)
        )
    """)
    conn.commit()


def rank_prior_work(hyp_emb: list[float], works: list[dict],
                    work_embs: list[list[float]]) -> list[dict]:
    """Attach cosine similarity to each work and sort, most similar first (pure)."""
    from nous.engine.connector import cosine_sim
    ranked = [{**w, "sim": round(cosine_sim(hyp_emb, e), 4)} for w, e in zip(works, work_embs)]
    ranked.sort(key=lambda w: w["sim"], reverse=True)
    return ranked


def search_prior_work(text: str, n: int = 10) -> list[dict]:
    from nous.collector.openalex import get, reconstruct_abstract, short_id
    data = get("works", {
        "search.semantic": text[:2000],
        "per_page": n,
        "select": "id,display_name,publication_year,abstract_inverted_index,cited_by_count",
    })
    works = []
    for w in data.get("results", []):
        title = w.get("display_name") or ""
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        works.append({
            "work_id":  short_id(w["id"]),
            "title":    title,
            "year":     w.get("publication_year"),
            "cited_by": w.get("cited_by_count", 0),
            "text":     f"{title}. {abstract}".strip(),
        })
    return works


def check_literature(refs: list[str] | None = None, n: int = 10,
                     verbose: bool = True) -> int:
    """Run the literature check for every (or the given) hypothesis not yet checked."""
    from nous.engine.embedder import get_model

    targets = store.load_targets()
    if refs:
        targets = [t for t in targets if t["ref"] in set(refs)]
    done = store.rated_by(RATER)
    todo = [t for t in targets if t["ref"] not in done]
    if not todo:
        if verbose:
            print("All hypotheses already checked against the literature.")
        return 0

    model = get_model()
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    checked = 0
    for t in todo:
        try:
            works = search_prior_work(t["text"], n=n)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {t['ref']}: {ex}")
            continue
        time.sleep(1.1)   # semantic search is limited to 1 request/second
        if not works:
            continue

        hyp_emb = model.encode(t["text"]).tolist()
        work_embs = model.encode([w["text"] for w in works]).tolist()
        ranked = rank_prior_work(hyp_emb, works, work_embs)
        novelty = round(1.0 - ranked[0]["sim"], 4)

        conn.execute("DELETE FROM literature_checks WHERE target_ref = ?", (t["ref"],))
        conn.executemany("""
            INSERT INTO literature_checks (target_ref, rank, work_id, title, year, cited_by, sim)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [(t["ref"], i + 1, w["work_id"], w["title"], w["year"], w["cited_by"], w["sim"])
              for i, w in enumerate(ranked)])
        conn.commit()
        store.save_scores(t["ref"], RATER, {ITEM: novelty})
        checked += 1

        if verbose:
            top = ranked[0]
            print(f"  {t['ref']:<12} novelty={novelty:.2f}  closest: "
                  f"{top['title'][:70]} ({top['year']}, sim={top['sim']:.2f})")
    conn.close()
    if verbose:
        print(f"\nDone. {checked} hypotheses checked.")
    return checked


def closest_prior_work(ref: str, k: int = 1) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    rows = conn.execute("""
        SELECT title, year, sim, work_id FROM literature_checks
        WHERE target_ref = ? ORDER BY rank LIMIT ?
    """, (ref, k)).fetchall()
    conn.close()
    return [{"title": r[0], "year": r[1], "sim": r[2], "work_id": r[3]} for r in rows]
