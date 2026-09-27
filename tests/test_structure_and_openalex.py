import pytest

from nous.collector import openalex as oa
from nous.engine import connector
from nous.engine.connector import rank_cross_domain
from nous.engine.knowledge_graph import build_graph_from
from nous.engine.structure import motif_jaccard, normalize_motifs, structural_sim


# ---------------- structure ---------------- #

def test_normalize_motifs_filters_to_vocabulary():
    assert normalize_motifs(["Positive Feedback", "threshold", "made-up", "threshold"]) == \
        ["positive_feedback", "threshold"]
    assert normalize_motifs("nope") == []


def test_structural_sim_combines_embedding_and_motifs():
    assert motif_jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert structural_sim([1, 0], ["threshold"], [1, 0], ["threshold"]) == pytest.approx(1.0)
    assert structural_sim([1, 0], ["threshold"], [0, 1], ["delay"]) == pytest.approx(0.0, abs=1e-6)


def test_structural_similarity_can_outrank_semantic_similarity():
    # "b" is semantically closer, but "c" shares the structure
    q = [1.0, 0.0]
    metas = [{"title": "home", "domain": "physics"},
             {"title": "b", "domain": "biology"},
             {"title": "c", "domain": "sociology"}]
    embs = [[1.0, 0.0], [0.9, 0.44], [0.5, 0.87]]
    plain = rank_cross_domain(q, ["", "", ""], metas, embs, ids=["h", "b", "c"])
    assert plain[0]["title"] == "b"
    blended = rank_cross_domain(q, ["", "", ""], metas, embs, ids=["h", "b", "c"],
                                structural_sims={"b": 0.1, "c": 0.9})
    assert blended[0]["title"] == "c"
    assert blended[0]["structural_sim"] == 0.9


def test_graph_uses_structures_when_both_ends_have_them():
    ids = ["a", "b"]
    metas = [{"title": "A", "domain": "x"}, {"title": "B", "domain": "y"}]
    embs = [[1, 0], [0, 1]]                                 # semantically unrelated
    assert build_graph_from(ids, embs, metas).number_of_edges() == 0
    st = {"a": {"embedding": [1, 0], "motifs": ["threshold"]},
          "b": {"embedding": [1, 0], "motifs": ["threshold"]}}
    G = build_graph_from(ids, embs, metas, structures=st)
    assert G.number_of_edges() == 1                          # linked by structure


class _FakeModel:
    def encode(self, text):
        import numpy as np
        return np.array([1.0, 0.0]) if "query" in text else np.array([0.0, 1.0])


class _FakeCollection:
    def __init__(self, rows):
        self.rows = rows  # id -> (doc, meta, emb)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, include, where=None):
        ids = [i for i in self.rows if where is None or self.rows[i][1]["domain"] == where["domain"]]
        ids = ids[:1]                                   # semantic search only finds "p1"
        return {"ids": [ids],
                "documents": [[self.rows[i][0] for i in ids]],
                "metadatas": [[self.rows[i][1] for i in ids]],
                "embeddings": [[self.rows[i][2] for i in ids]]}

    def get(self, ids, include):
        return {"ids": ids,
                "documents": [self.rows[i][0] for i in ids],
                "metadatas": [self.rows[i][1] for i in ids],
                "embeddings": [self.rows[i][2] for i in ids]}


def test_find_connections_merges_structural_only_hits(monkeypatch):
    rows = {"p1": ("phase", {"title": "Phase transition", "domain": "physics"}, [1.0, 0.0]),
            "s1": ("riot", {"title": "Riot threshold model", "domain": "sociology"}, [0.0, 1.0])}
    monkeypatch.setattr(connector, "get_model", lambda: _FakeModel())
    monkeypatch.setattr(connector, "get_collection", lambda: _FakeCollection(rows))
    monkeypatch.setattr(connector, "_structural_candidates",
                        lambda q, m, n, llm: (([0.0, 1.0], ["threshold"]), ["s1"]))
    from nous.engine import structure
    monkeypatch.setattr(structure, "load_structures", lambda ids=None: {
        "s1": {"embedding": [0.0, 1.0], "motifs": ["threshold"], "structure": ""}})

    res = connector.find_cross_domain_connections("query", n_results=5)
    titles = [r["title"] for r in res]
    assert "Riot threshold model" in titles            # found only via structure
    riot = next(r for r in res if r["title"] == "Riot threshold model")
    assert riot["structural_sim"] == pytest.approx(1.0)


# ---------------- OpenAlex ---------------- #

def test_reconstruct_abstract_and_parse_work():
    inv = {"Despite": [0], "growing": [1], "interest": [2], "in": [3, 5], "work": [4], "x": [6]}
    assert oa.reconstruct_abstract(inv) == "Despite growing interest in work in x"
    work = {"id": "https://openalex.org/W1", "display_name": "T", "publication_year": 2020,
            "abstract_inverted_index": inv, "cited_by_count": 3,
            "primary_topic": {"display_name": "Topic",
                              "field": {"id": "https://openalex.org/fields/32", "display_name": "Psychology"}}}
    w = oa.parse_work(work)
    assert w["domain"] == "psychology" and w["openalex_id"] == "W1" and w["text"].startswith("T\n\n")
    assert oa.parse_work({**work, "abstract_inverted_index": None}) is None


def test_slug_and_short_id_and_scepter_mapping():
    from nous.domains import scepter_for
    assert oa.slugify("Economics, Econometrics and Finance") == "economics_econometrics_and_finance"
    assert oa.short_id("https://openalex.org/fields/27") == "27"
    assert scepter_for(oa.slugify("Arts and Humanities")) == "Scepter-H"
    assert scepter_for(oa.slugify("Pharmacology, Toxicology and Pharmaceutics")) == "Scepter-A"


def test_lift_matrix_flags_voids():
    totals = {"A": 100, "B": 100, "C": 100}
    cooc = {("A", "B"): 40, ("B", "A"): 40, ("A", "C"): 1}
    table = oa.lift_matrix(cooc, totals, 300)
    assert table[0]["lift"] == 0.0 and {table[0]["field_a"], table[0]["field_b"]} == {"B", "C"}
    ab = next(t for t in table if {t["field_a"], t["field_b"]} == {"A", "B"})
    assert ab["observed"] == 80 and ab["lift"] == pytest.approx(80 / (2 * 100 * 100 / 300))


def test_growth_stats_detects_acceleration():
    counts = {2010 + i: int(100 * (1.05 ** min(i, 5)) * (1.20 ** max(i - 5, 0))) for i in range(11)}
    g = oa.growth_stats(counts, window=5)
    assert g["previous_cagr"] == pytest.approx(0.05, abs=0.01)
    assert g["recent_cagr"] == pytest.approx(0.20, abs=0.01)
    assert g["acceleration"] > 0.1
    assert oa.growth_stats({2020: 1}, window=5)["acceleration"] is None


def test_openalex_get_sends_key_and_retries(monkeypatch):
    calls = []

    class R:
        def __init__(self, code): self.status_code = code
        def json(self): return {"ok": True}
        def raise_for_status(self): raise AssertionError("should not be called")

    seq = iter([R(429), R(200)])

    def fake_get(url, params, timeout):
        calls.append(params)
        return next(seq)

    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    monkeypatch.setattr(oa.requests, "get", fake_get)
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)
    assert oa.get("works", {"filter": "x"}) == {"ok": True}
    assert len(calls) == 2 and calls[0]["api_key"] == "k"
