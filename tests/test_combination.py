import sqlite3

import pytest
import requests

from nous.collector import openalex
from nous.evaluation import combination as cb


def test_sanitize_and_build_query():
    assert cb.sanitize_phrase('phase-transition, "Ising": model|x') == "phase-transition Ising model x"
    assert cb.build_query(["phase transition", "opinion, dynamics"]) == \
        '"phase transition" AND "opinion dynamics"'


def test_pair_stats_and_labels():
    st = cb.pair_stats(1000, 2000, 20, 2_000_000)      # expected = 1.0
    assert st["expected"] == pytest.approx(1.0) and st["lift"] == pytest.approx(20.0)
    assert cb.pair_stats(1, 1, 0, 0)["lift"] == 0.0
    assert cb.label_for(0.05) == "unexplored"
    assert cb.label_for(0.5) == "sparse"
    assert cb.label_for(3.0) == "well_trodden"


def test_summarize_uses_most_novel_pair_and_flags_rare_terms():
    pairs = [{"a": "A", "b": "B", "lift": 5.0, "joint": 500},
             {"a": "A", "b": "C", "lift": 0.02, "joint": 1}]
    res = cb.summarize({"A": 900, "B": 800, "C": 700}, pairs)
    assert res["label"] == "unexplored" and res["pair"]["b"] == "C"
    assert res["combo_novelty"] == pytest.approx(0.98)
    # a coined / rarely used term -> no verdict
    res = cb.summarize({"A": 900, "B": 3}, pairs[:1])
    assert res["label"] == "unrecognized_terms" and res["combo_novelty"] is None
    assert cb.summarize({}, [])["label"] == "unrecognized_terms"
    # commonplace pair -> novelty 0
    assert cb.summarize({"A": 100, "B": 100}, [{"a": "A", "b": "B", "lift": 12.0, "joint": 9}]
                        )["combo_novelty"] == 0.0


def test_parse_concepts_dedupes_limits_and_cleans():
    data = {"concepts": ["Phase transition", "phase transition", "opinion dynamics",
                         "a very long phrase that is not a standard term at all", "Ising, model",
                         "network science", "extra"]}
    out = cb.parse_concepts(data)
    assert out == ["Phase transition", "opinion dynamics", "Ising model", "network science"]
    assert cb.parse_concepts("nope") == []


class _Resp:
    def __init__(self, code): self.status_code = code


def test_counter_counts_caches_and_falls_back_to_search(monkeypatch):
    calls = []

    def fake_get(path, params=None):
        calls.append(dict(params))
        if "filter" in params:
            raise requests.HTTPError(response=_Resp(400))
        return {"meta": {"count": 1234}}

    monkeypatch.setattr(openalex, "get", fake_get)
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)
    c = cb.Counter()
    assert c.count('"x"') == 1234 and c.mode == "search"
    assert c.count('"x"') == 1234                       # cached
    assert sum(1 for p in calls if "search" in p) == 1
    assert c.total == 1234


def test_counter_reraises_non_400(monkeypatch):
    def fake_get(path, params=None):
        raise requests.HTTPError(response=_Resp(500))
    monkeypatch.setattr(openalex, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        cb.Counter().count('"x"')


def test_measure_and_end_to_end_storage(tmp_path, monkeypatch):
    table = {'"A"': 1000, '"B"': 2000, '"C"': 10, '"A" AND "B"': 400, '"A" AND "C"': 0,
             '"B" AND "C"': 0}

    class FakeCounter:
        mode = "filter"
        total = 2_000_000
        def count(self, q): return table[q]

    counts, pairs = cb.measure(["A", "B", "C"], FakeCounter())
    assert counts == {"A": 1000, "B": 2000, "C": 10}
    ab = next(p for p in pairs if (p["a"], p["b"]) == ("A", "B"))
    assert ab["lift"] == pytest.approx(400 / 1.0)
    assert cb.summarize(counts, pairs)["label"] == "unrecognized_terms"   # C has only 10 works

    db = tmp_path / "n.db"
    monkeypatch.setattr(cb, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    cb._save_pairs(con, "hyp:1", pairs, "filter")
    cb._save_pairs(con, "hyp:1", pairs, "filter")       # re-saving replaces, not duplicates
    assert con.execute("SELECT COUNT(*) FROM combo_checks").fetchone()[0] == 3
    con.close()
    assert cb.combo_summary("hyp:1")["label"] == "unrecognized_terms"
    assert cb.combo_summary("hyp:404") is None
    assert cb.label_counts() == {"unrecognized_terms": 1}


def test_runner_aggregate_carries_combo_novelty():
    from nous.evaluation.runner import aggregate
    rows = [{"target_ref": "hyp:1", "rater": "llm:m:A", "item": "novelty", "score": 4.0},
            {"target_ref": "hyp:1", "rater": "auto:openalex-combo", "item": "combo_novelty", "score": 0.9}]
    a = aggregate(rows)[0]
    assert a["combo_novelty"] == 0.9 and a["composite"] == 4.0
