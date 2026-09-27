"""
Connection Engine — the core of Nous.
Finds structurally similar concepts across different domains.
"""
from __future__ import annotations
import numpy as np

from nous.domains import HYPOTHESIS_DOMAIN
from nous.engine.embedder import get_model, get_collection

SAME_DOMAIN_PENALTY = 0.3
STRUCTURAL_WEIGHT   = 0.6


def nous_score(
    semantic_sim: float,
    same_domain: bool,
    structural_sim: float | None = None,
) -> float:
    """
    Cross-domain resonance score.
    Suppresses same-domain matches. When a structural similarity is supplied it is
    blended in (0.4 semantic / 0.6 structural); otherwise the score is purely semantic.
    """
    base = semantic_sim - (SAME_DOMAIN_PENALTY if same_domain else 0.0)
    if structural_sim is None:
        return base
    return base * (1 - STRUCTURAL_WEIGHT) + structural_sim * STRUCTURAL_WEIGHT


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def infer_source_domain(candidates: list[dict]) -> str | None:
    """The query's home domain = domain of its semantically closest article."""
    ranked = sorted(
        (c for c in candidates if c["domain"] != HYPOTHESIS_DOMAIN),
        key=lambda c: c["semantic_sim"], reverse=True,
    )
    return ranked[0]["domain"] if ranked else None


def rank_cross_domain(
    query_embedding: list[float],
    documents: list[str],
    metadatas: list[dict],
    embeddings: list[list[float]],
    source_domain: str | None = None,
    n_results: int = 10,
) -> list[dict]:
    """
    Re-rank retrieved candidates by nous_score.
    Candidates from `source_domain` (inferred when not given) are penalized so that
    results from *other* domains rise to the top.
    """
    candidates = [
        {
            "title":        meta["title"],
            "domain":       meta["domain"],
            "semantic_sim": cosine_sim(query_embedding, emb),
            "summary":      doc[:300],
        }
        for doc, meta, emb in zip(documents, metadatas, embeddings)
    ]
    if source_domain is None:
        source_domain = infer_source_domain(candidates)

    for c in candidates:
        c["same_domain"] = source_domain is not None and c["domain"] == source_domain
        c["nous_score"]  = nous_score(c["semantic_sim"], same_domain=c["same_domain"])
        c["source_domain"] = source_domain

    candidates.sort(key=lambda x: x["nous_score"], reverse=True)
    return candidates[:n_results]


def find_cross_domain_connections(
    query: str,
    source_domain: str | None = None,
    target_domain: str | None = None,
    n_results: int = 10,
) -> list[dict]:
    """
    Given a concept (query), find the most structurally resonant concepts
    from other domains.

    Args:
        query:         Natural language concept or question.
        source_domain: The query's own domain; its articles are penalized.
                       Inferred from the nearest article when omitted.
        target_domain: If set, only return results from this domain.
        n_results:     Number of results to return (3x candidates are retrieved
                       before reranking).

    Returns:
        List of dicts with title, domain, nous_score, semantic_sim, summary.
    """
    model = get_model()
    collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    query_embedding = model.encode(query).tolist()

    kwargs = {}
    if target_domain is not None:
        kwargs["where"] = {"domain": target_domain}

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results * 3, total),
        include=["documents", "metadatas", "embeddings"],
        **kwargs,
    )

    return rank_cross_domain(
        query_embedding,
        raw["documents"][0],
        raw["metadatas"][0],
        raw["embeddings"][0],
        source_domain=source_domain,
        n_results=n_results,
    )


def find_bridge(concept_a: str, concept_b: str, n: int = 5) -> list[dict]:
    """
    Find concepts that structurally bridge two given concepts across domains.
    These are candidates for novel analogical hypotheses.
    """
    model = get_model()
    collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    emb_a = model.encode(concept_a)
    emb_b = model.encode(concept_b)
    midpoint = ((emb_a + emb_b) / 2).tolist()

    raw = collection.query(
        query_embeddings=[midpoint],
        n_results=min(n * 2, total),
        include=["documents", "metadatas", "embeddings"],
    )

    bridges = []
    for doc, meta, emb in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["embeddings"][0],
    ):
        sim_a = cosine_sim(emb_a.tolist(), emb)
        sim_b = cosine_sim(emb_b.tolist(), emb)
        bridge_score = (sim_a + sim_b) / 2
        bridges.append({
            "title":        meta["title"],
            "domain":       meta["domain"],
            "bridge_score": bridge_score,
            "summary":      doc[:300],
        })

    bridges.sort(key=lambda x: x["bridge_score"], reverse=True)
    return bridges[:n]


if __name__ == "__main__":
    print("=== Cross-domain search: 'phase transition' ===")
    results = find_cross_domain_connections("phase transition", n_results=5)
    for r in results:
        print(f"  [{r['domain']}] {r['title']}  score={r['nous_score']:.3f}")

    print("\n=== Bridge: 'quantum entanglement' ↔ 'social network' ===")
    bridges = find_bridge("quantum entanglement", "social network", n=5)
    for b in bridges:
        print(f"  [{b['domain']}] {b['title']}  bridge={b['bridge_score']:.3f}")
