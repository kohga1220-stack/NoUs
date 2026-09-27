"""
Embed articles from SQLite into ChromaDB for semantic search.
"""
from __future__ import annotations
import sqlite3

from nous.config import DB_PATH, CHROMA_PATH

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION = "nous_knowledge"

_model = None


def get_model():
    """Load the SentenceTransformer once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION)


def load_all_articles() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, domain, summary, full_text FROM articles")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "domain": r[2], "summary": r[3], "text": r[4]} for r in rows]


def already_indexed(collection) -> set[str]:
    results = collection.get(include=[])
    return set(results["ids"])


def embed_all(batch_size: int = 32):
    print("Loading model...")
    model = get_model()
    collection = get_collection()
    indexed = already_indexed(collection)

    articles = load_all_articles()
    new_articles = [a for a in articles if str(a["id"]) not in indexed]

    if not new_articles:
        print("All articles already indexed.")
        return

    print(f"Embedding {len(new_articles)} articles...")

    for i in range(0, len(new_articles), batch_size):
        batch = new_articles[i:i + batch_size]
        texts = [
            f"{a['title']}. {a['summary']}" for a in batch
        ]
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
        collection.add(
            ids        = [str(a["id"]) for a in batch],
            embeddings = embeddings,
            documents  = texts,
            metadatas  = [{"title": a["title"], "domain": a["domain"]} for a in batch],
        )
        print(f"  Indexed {min(i + batch_size, len(new_articles))} / {len(new_articles)}")

    print("Done.")


if __name__ == "__main__":
    embed_all()
