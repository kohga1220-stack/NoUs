"""
Build a full knowledge graph from all indexed articles.
Nodes = articles, edges = cross-domain semantic similarity (nous_score).
Identifies bridge nodes (concepts connecting distant domains) and void zones.
"""
from __future__ import annotations
from itertools import combinations

import networkx as nx

from nous.engine.connector import cosine_sim, nous_score
from nous.engine.structure import structural_sim


def build_graph_from(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    threshold: float = 0.25,
    max_edges_per_node: int = 6,
    structures: dict[str, dict] | None = None,
) -> nx.Graph:
    """
    Pure graph construction (no I/O).

    `structures` ({id: {"embedding", "motifs"}}) blends structural similarity into the
    edge score for pairs where both articles have a structural abstraction.

    Node attributes: title, domain, centrality
    Edge attributes: weight   (nous_score — higher = more similar)
                     distance (1 - weight — used for shortest-path metrics)
    """
    G = nx.Graph()

    for doc_id, meta in zip(ids, metadatas):
        G.add_node(doc_id, title=meta["title"], domain=meta["domain"])

    edge_candidates: dict[str, list[tuple]] = {i: [] for i in ids}

    for (i_idx, i_id), (j_idx, j_id) in combinations(enumerate(ids), 2):
        if metadatas[i_idx]["domain"] == metadatas[j_idx]["domain"]:
            continue

        sim = cosine_sim(embeddings[i_idx], embeddings[j_idx])
        st_sim = None
        if structures and i_id in structures and j_id in structures:
            a, b = structures[i_id], structures[j_id]
            st_sim = structural_sim(a["embedding"], a["motifs"], b["embedding"], b["motifs"])
        score = nous_score(sim, same_domain=False, structural_sim=st_sim)

        if score >= threshold:
            edge_candidates[i_id].append((score, j_id))
            edge_candidates[j_id].append((score, i_id))

    for node_id, candidates in edge_candidates.items():
        candidates.sort(reverse=True)
        for score, neighbor_id in candidates[:max_edges_per_node]:
            if not G.has_edge(node_id, neighbor_id):
                G.add_edge(
                    node_id, neighbor_id,
                    weight=round(score, 4),
                    distance=round(max(1.0 - score, 1e-6), 4),
                )

    centrality = nx.degree_centrality(G) if G.number_of_nodes() > 1 else {n: 0.0 for n in G}
    for node_id in G.nodes:
        G.nodes[node_id]["centrality"] = round(centrality[node_id], 4)

    return G


def build_graph(threshold: float = 0.25, max_edges_per_node: int = 6) -> nx.Graph:
    """
    Build the full Nous knowledge graph from ChromaDB.
    Only cross-domain edges above threshold are kept.

    Args:
        threshold:           Minimum nous_score to create an edge.
        max_edges_per_node:  Limit edges per node to keep graph readable.
    """
    from nous.engine.embedder import get_collection
    from nous.engine.structure import load_structures

    collection = get_collection()
    result = collection.get(include=["embeddings", "metadatas"])

    ids = result["ids"]
    if not ids:
        raise RuntimeError("ChromaDB is empty — run 'python nous.py embed' first.")

    structures = load_structures()
    print(f"Computing edges for {len(ids)} nodes "
          f"({len(structures)} with structural abstractions)...")
    G = build_graph_from(ids, result["embeddings"], result["metadatas"],
                         threshold=threshold, max_edges_per_node=max_edges_per_node,
                         structures=structures)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def find_bridge_nodes(G: nx.Graph, top_n: int = 10) -> list[dict]:
    """
    Bridge nodes connect multiple domains — candidates for analogical insight.
    Ranked by betweenness centrality × (number of distinct domains connected)^1.5.
    Betweenness uses `distance` (1 - similarity) so strong links count as short paths.
    """
    if G.number_of_edges() == 0:
        return []

    betweenness = nx.betweenness_centrality(G, weight="distance")

    bridges = []
    for node_id, data in G.nodes(data=True):
        neighbor_domains = {G.nodes[n]["domain"] for n in G.neighbors(node_id)}
        domain_span = len(neighbor_domains)
        score = betweenness[node_id] * (domain_span ** 1.5)
        bridges.append({
            "id":             node_id,
            "title":          data["title"],
            "domain":         data["domain"],
            "bridge_score":   round(score, 6),
            "domains_linked": sorted(neighbor_domains),
            "degree":         G.degree(node_id),
        })

    bridges.sort(key=lambda x: x["bridge_score"], reverse=True)
    return bridges[:top_n]


def find_void_zones(G: nx.Graph) -> list[dict]:
    """
    Void zones = pairs of domains with few or no connecting edges.
    These are unexplored territory — where Nous should look next.
    """
    domain_pairs: dict[tuple, int] = {}
    all_domains = sorted({data["domain"] for _, data in G.nodes(data=True)})

    for d1, d2 in combinations(all_domains, 2):
        domain_pairs[(d1, d2)] = 0

    for u, v in G.edges():
        d1 = G.nodes[u]["domain"]
        d2 = G.nodes[v]["domain"]
        if d1 != d2:
            key = tuple(sorted([d1, d2]))
            domain_pairs[key] = domain_pairs.get(key, 0) + 1

    voids = [
        {"domain_a": k[0], "domain_b": k[1], "connections": v}
        for k, v in domain_pairs.items()
    ]
    voids.sort(key=lambda x: x["connections"])
    return voids
