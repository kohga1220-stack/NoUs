"""
Retrieve past explorations and hypotheses from SQLite.
Past hypotheses are also fed back into ChromaDB as new knowledge.
"""
import sqlite3
import json

from nous.config import DB_PATH
from nous.domains import HYPOTHESIS_DOMAIN


def get_recent_hypotheses(limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT h.id, e.query, h.hypothesis_text, h.confidence, h.source_ids, h.timestamp
        FROM hypotheses h
        JOIN explorations e ON h.exploration_id = e.id
        ORDER BY h.timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id":         r[0],
            "query":      r[1],
            "hypothesis": r[2],
            "confidence": r[3],
            "sources":    json.loads(r[4]) if r[4] else [],
            "timestamp":  r[5],
        }
        for r in rows
    ]


def sync_hypotheses_to_chroma(only_ids: set[int] | None = None):
    """
    Feed past hypotheses back into ChromaDB
    so they become part of the searchable knowledge base (self-growth).
    """
    from nous.engine.embedder import get_model, get_collection

    hyps = get_recent_hypotheses(limit=100)
    if only_ids is not None:
        hyps = [h for h in hyps if h["id"] in only_ids]
    if not hyps:
        return

    model      = get_model()
    collection = get_collection()
    existing   = set(collection.get(include=[])["ids"])

    new = [h for h in hyps if f"hyp_{h['id']}" not in existing]
    if not new:
        print("No new hypotheses to sync.")
        return

    texts      = [h["hypothesis"] for h in new]
    embeddings = model.encode(texts).tolist()
    collection.add(
        ids        = [f"hyp_{h['id']}" for h in new],
        embeddings = embeddings,
        documents  = texts,
        metadatas  = [{"title": f"Hypothesis: {h['query']}", "domain": HYPOTHESIS_DOMAIN} for h in new],
    )
    print(f"Synced {len(new)} hypotheses to ChromaDB.")


if __name__ == "__main__":
    hyps = get_recent_hypotheses(5)
    for h in hyps:
        print(f"[{h['timestamp']}] {h['query']}")
        print(f"  {h['hypothesis'][:120]}...")
