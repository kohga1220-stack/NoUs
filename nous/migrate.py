"""
One-off data migrations.

migrate_domains(): rewrite legacy domain names (physics, history, ...) to OpenAlex
field slugs in SQLite and in both ChromaDB collections. Idempotent.
"""
from __future__ import annotations
import sqlite3

from nous.config import DB_PATH
from nous.domains import LEGACY_TO_OPENALEX, canonical_domain


def migrate_sqlite(db_path=None) -> int:
    conn = sqlite3.connect(db_path or DB_PATH)
    changed = 0
    for old, new in LEGACY_TO_OPENALEX.items():
        if old == new:
            continue
        # a canonical title may already exist (UNIQUE title) — only the domain changes here
        changed += conn.execute("UPDATE articles SET domain = ? WHERE domain = ?",
                                (new, old)).rowcount
    conn.commit()
    conn.close()
    return changed


def migrate_collection(collection) -> int:
    res = collection.get(include=["metadatas"])
    ids, metas = [], []
    for doc_id, meta in zip(res["ids"], res["metadatas"]):
        new = canonical_domain(meta.get("domain", ""))
        if new != meta.get("domain"):
            ids.append(doc_id)
            metas.append({**meta, "domain": new})
    for i in range(0, len(ids), 500):
        collection.update(ids=ids[i:i + 500], metadatas=metas[i:i + 500])
    return len(ids)


def migrate_domains(verbose: bool = True) -> dict[str, int]:
    from nous.engine.embedder import get_collection
    from nous.engine.structure import get_struct_collection

    out = {
        "sqlite_articles": migrate_sqlite(),
        "chroma_knowledge": migrate_collection(get_collection()),
        "chroma_structures": migrate_collection(get_struct_collection()),
    }
    if verbose:
        for k, v in out.items():
            print(f"  {k:<18} {v} updated")
    return out
