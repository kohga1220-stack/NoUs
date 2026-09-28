import sqlite3

import pytest

from nous import llm
from nous.domains import canonical_domain, scepter_for
from nous.evaluation import literature
from nous.evaluation.runner import _judges, human_vs_llm, reliability_report
from nous.evaluation.store import clean_hypothesis_text
from nous.migrate import migrate_collection, migrate_sqlite


# ① retry on malformed JSON
def test_generate_json_retries_then_succeeds(monkeypatch):
    replies = iter(["not json", '{"a": 1'  , '{"a": 2}'])
    monkeypatch.setattr(llm, "generate", lambda prompt, model: next(replies))
    assert llm.generate_json("p", retries=2) == {"a": 2}


def test_generate_json_gives_up(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda prompt, model: "never json")
    with pytest.raises(ValueError):
        llm.generate_json("p", retries=1)


# ② reliability split
def _rows():
    rows = []
    human = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}
    for t, h in human.items():
        rows.append({"target_ref": t, "rater": "human:me", "item": "novelty", "score": h})
        rows.append({"target_ref": t, "rater": "llm:m:A", "item": "novelty", "score": h + 1})
        rows.append({"target_ref": t, "rater": "llm:m:B", "item": "novelty", "score": h + 1})
    return rows


def test_human_vs_llm_detects_leniency():
    e = next(x for x in human_vs_llm(_rows()) if x["item"] == "novelty")
    assert e["n_targets"] == 4 and e["small_n"]
    assert e["bias"] == pytest.approx(1.0)
    assert e["r"] == pytest.approx(1.0)
    assert e["icc"] < 1.0          # absolute agreement penalises the constant offset


def test_llm_only_reliability_excludes_humans():
    e = next(x for x in reliability_report(_rows(), "llm:") if x["item"] == "novelty")
    assert e["n_raters"] == 2 and e["small_n"]


# ④ judges
def test_judges_single_model_uses_personas_multi_model_neutral():
    single = _judges(["g"])
    assert len(single) == 5 and all(r.startswith("llm:g:Scepter-") for r, _, _ in single)
    multi = _judges(["g", "l", "q"])
    assert [r for r, _, _ in multi] == ["llm:g:neutral", "llm:l:neutral", "llm:q:neutral"]
    assert all(p == "" for _, p, _ in multi)


# ⑤ cleaning stored hypothesis text
def test_clean_hypothesis_text():
    assert clean_hypothesis_text("[parse error: x]") is None
    assert clean_hypothesis_text('{"structural_pattern": "s", "hypothesis": "H here"}') == "H here"
    assert clean_hypothesis_text('{"structural_pattern": "s"}') is None
    assert clean_hypothesis_text("{ broken") is None
    assert clean_hypothesis_text("  plain  ") == "plain"


# ③ literature check
def test_search_and_rank_prior_work(monkeypatch):
    from nous.collector import openalex
    captured = {}

    def fake_get(path, params):
        captured.update(params)
        return {"results": [
            {"id": "https://openalex.org/W1", "display_name": "Far", "publication_year": 2001,
             "abstract_inverted_index": {"x": [0]}, "cited_by_count": 5},
            {"id": "https://openalex.org/W2", "display_name": "Near", "publication_year": 2020,
             "abstract_inverted_index": None, "cited_by_count": 9},
        ]}
    monkeypatch.setattr(openalex, "get", fake_get)
    works = literature.search_prior_work("h" * 3000, n=2)
    assert len(captured["search.semantic"]) == 2000
    assert works[0]["work_id"] == "W1" and works[1]["text"] == "Near."
    ranked = literature.rank_prior_work([1.0, 0.0], works, [[0.0, 1.0], [1.0, 0.0]])
    assert ranked[0]["title"] == "Near" and ranked[0]["sim"] == pytest.approx(1.0)


# ⑥ domain unification
def test_canonical_domain_and_scepter():
    assert canonical_domain("physics") == "physics_and_astronomy"
    assert canonical_domain("history") == "arts_and_humanities"
    assert canonical_domain("neuroscience") == "neuroscience"
    for legacy in ("physics", "history", "sociology", "biology", "economics"):
        assert scepter_for(legacy) == scepter_for(canonical_domain(legacy))


def test_migrate_sqlite(tmp_path):
    db = tmp_path / "n.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, domain TEXT)")
    con.executemany("INSERT INTO articles (title, domain) VALUES (?, ?)",
                    [("a", "physics"), ("b", "psychology"), ("c", "history"), ("d", "medicine")])
    con.commit(); con.close()
    assert migrate_sqlite(db) == 2
    assert migrate_sqlite(db) == 0          # idempotent
    doms = [r[0] for r in sqlite3.connect(db).execute("SELECT domain FROM articles ORDER BY id")]
    assert doms == ["physics_and_astronomy", "psychology", "arts_and_humanities", "medicine"]


def test_migrate_collection_updates_only_legacy():
    class Col:
        def __init__(self):
            self.metas = {"1": {"title": "A", "domain": "physics"},
                          "2": {"title": "B", "domain": "medicine"}}
            self.updated = []

        def get(self, include):
            return {"ids": list(self.metas), "metadatas": list(self.metas.values())}

        def update(self, ids, metadatas):
            self.updated += list(zip(ids, metadatas))

    c = Col()
    assert migrate_collection(c) == 1
    assert c.updated == [("1", {"title": "A", "domain": "physics_and_astronomy"})]
