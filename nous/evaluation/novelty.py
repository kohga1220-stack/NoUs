"""
Automatic (LLM-free) metrics computed from embeddings.

novelty_embedding = 1 - (max cosine similarity between the hypothesis and any
                         article in the knowledge base)
A hypothesis that merely paraphrases an existing article scores near 0.
"""
from __future__ import annotations

from nous.domains import HYPOTHESIS_DOMAIN


def novelty_from_sims(sims: list[float]) -> float | None:
    if not sims:
        return None
    return round(1.0 - max(sims), 4)


def embedding_novelty(text: str, k: int = 5) -> float | None:
    """Novelty of `text` against the article index (hypotheses themselves excluded)."""
    from nous.engine.connector import cosine_sim
    from nous.engine.embedder import get_collection, get_model

    collection = get_collection()
    if collection.count() == 0:
        return None
    emb = get_model().encode(text).tolist()
    raw = collection.query(
        query_embeddings=[emb],
        n_results=min(k, collection.count()),
        where={"domain": {"$ne": HYPOTHESIS_DOMAIN}},
        include=["embeddings"],
    )
    sims = [cosine_sim(emb, e) for e in raw["embeddings"][0]]
    return novelty_from_sims(sims)
