"""
Unit tests for the pure (no LLM / no ChromaDB) parts of Nous.
"""
import sqlite3

import networkx as nx
import pytest

from nous import config
from nous.collector.arxiv import parse_arxiv_id
from nous.domains import DOMAIN_TO_SCEPTER, domains_for, scepter_for
from nous.engine.connector import nous_score, rank_cross_domain
from nous.engine.knowledge_graph import build_graph_from, find_bridge_nodes, find_void_zones
from nous.llm import extract_json, sanitize_json
from nous.scepter.registry import SCEPTER_MAP


# ---------------- nous_score / cross-domain ranking ---------------- #

def test_nous_score_penalizes_same_domain():
    assert nous_score(0.8, same_domain=True) == pytest.approx(0.5)
    assert nous_score(0.8, same_domain=False) == pytest.approx(0.8)


def test_nous_score_structural_blend_only_when_given():
    assert nous_score(0.5, False, structural_sim=1.0) == pytest.approx(0.5 * 0.4 + 0.6)


def test_rank_cross_domain_infers_source_and_demotes_it():
    q = [1.0, 0.0]
    docs  = ["a", "b", "c"]
    metas = [
        {"title": "Phase transition", "domain": "physics"},     # nearest -> home domain
        {"title": "Critical point",   "domain": "physics"},
        {"title": "Tipping point",    "domain": "sociology"},
    ]
    embs = [[1.0, 0.0], [0.95, 0.31], [0.8, 0.6]]
    res = rank_cross_domain(q, docs, metas, embs, n_results=3)
    assert res[0]["source_domain"] == "physics"
    assert res[0]["domain"] == "sociology"          # cross-domain result wins
    assert all(r["same_domain"] == (r["domain"] == "physics") for r in res)


def test_rank_cross_domain_explicit_source():
    q = [1.0, 0.0]
    metas = [{"title": "x", "domain": "physics"}, {"title": "y", "domain": "biology"}]
    embs = [[1.0, 0.0], [0.9, 0.44]]
    res = rank_cross_domain(q, ["", ""], metas, embs, source_domain="biology")
    assert res[0]["domain"] == "physics"


# ---------------- knowledge graph ---------------- #

def _toy_graph():
    ids = ["p1", "b1", "s1", "e1"]
    metas = [
        {"title": "P", "domain": "physics"},
        {"title": "B", "domain": "biology"},
        {"title": "S", "domain": "sociology"},
        {"title": "E", "domain": "economics"},
    ]
    embs = [[1, 0, 0], [0.9, 0.43, 0], [0, 1, 0], [0, 0.9, 0.43]]
    return build_graph_from(ids, embs, metas, threshold=0.25)


def test_graph_only_cross_domain_edges_with_distance():
    G = _toy_graph()
    for u, v, d in G.edges(data=True):
        assert G.nodes[u]["domain"] != G.nodes[v]["domain"]
        assert d["distance"] == pytest.approx(max(1 - d["weight"], 1e-6), abs=1e-4)


def test_betweenness_prefers_strong_links():
    # a–b strong, b–c strong, a–c weak: the shortest path a→c should go through b
    G = nx.Graph()
    for n, dom in [("a", "x"), ("b", "y"), ("c", "z")]:
        G.add_node(n, title=n, domain=dom)
    G.add_edge("a", "b", weight=0.9, distance=0.1)
    G.add_edge("b", "c", weight=0.9, distance=0.1)
    G.add_edge("a", "c", weight=0.3, distance=0.7)
    top = find_bridge_nodes(G, top_n=1)[0]
    assert top["id"] == "b" and top["bridge_score"] > 0


def test_void_zones_sorted_ascending():
    voids = find_void_zones(_toy_graph())
    counts = [v["connections"] for v in voids]
    assert counts == sorted(counts)
    assert len(voids) == 6  # C(4,2)


# ---------------- domains ---------------- #

def test_every_collector_domain_maps_to_a_real_scepter():
    for d, s in DOMAIN_TO_SCEPTER.items():
        assert s in SCEPTER_MAP, d
    assert scepter_for("psychology") == "Scepter-H"
    assert "physics" in domains_for("Scepter-N")
    assert {"economics", "sociology", "social_sciences"} <= set(SCEPTER_MAP["Scepter-S"].collector_domains)


# ---------------- LLM JSON parsing ---------------- #

def test_sanitize_and_extract_json():
    raw = 'Sure! {"hypothesis": "a \\x b", "confidence": 0.7} done'
    assert "\\x" not in sanitize_json('"\\x"')
    out = extract_json(raw)
    assert out["confidence"] == 0.7


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        extract_json("no json here")


# ---------------- arXiv IDs ---------------- #

def test_parse_arxiv_id_keeps_archive_prefix():
    assert parse_arxiv_id("http://arxiv.org/abs/2401.01735v1") == "2401.01735v1"
    assert parse_arxiv_id("http://arxiv.org/abs/cond-mat/0607151v1") == "cond-mat/0607151v1"


# ---------------- hypothesis links ---------------- #

def test_linker_dedup_is_orientation_independent_and_saves_strength(tmp_path, monkeypatch):
    from nous.memory import linker

    db = tmp_path / "nous.db"
    conn = sqlite3.connect(db)
    # old schema without rationale / strength
    conn.execute("CREATE TABLE hypothesis_links (id INTEGER PRIMARY KEY, hyp_id_a INTEGER, "
                 "hyp_id_b INTEGER, link_type TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(linker, "DB_PATH", db)

    linker.save_link(2, 1, "supports", "because", 0.8)   # stored as B->A
    existing = linker.get_existing_links()
    assert linker.is_linked(existing, 1, 2)
    assert linker.is_linked(existing, 2, 1)

    row = sqlite3.connect(db).execute(
        "SELECT rationale, strength FROM hypothesis_links").fetchone()
    assert row == ("because", 0.8)


def test_config_paths_are_under_repo():
    assert config.DB_PATH.parent == config.DATA_DIR
    assert config.DATA_DIR.parent == config.ROOT_DIR
