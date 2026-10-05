import sqlite3

import pytest
import requests

from nous.collector import openalex
from nous.evaluation import combination as cb


def P(a="A", b="B", n_a=1000, n_b=1000, joint=0, adjacency=None):
    return {"a": a, "b": b, "n_a": n_a, "n_b": n_b, "joint": joint, "expected": 0.0,
            "lift": 0.0, "adjacency": adjacency}


def test_sanitize_and_build_query():
    assert cb.sanitize_phrase('phase-transition, "Ising": model|x') == "phase-transition Ising model x"
    assert cb.build_query(["phase transition", "opinion, dynamics"]) == \
        '"phase transition" AND "opinion dynamics"'


def test_pair_stats_is_informational():
    st = cb.pair_stats(1000, 2000, 20, 2_000_000)
    assert st["expected"] == pytest.approx(1.0) and st["lift"] == pytest.approx(20.0)
    assert cb.pair_stats(1, 1, 0, 0)["lift"] == 0.0


def test_pair_status_uses_absolute_joint_counts():
    assert cb.pair_status(P(joint=0)) == "none"
    assert cb.pair_status(P(joint=1)) == "few_papers"
    assert cb.pair_status(P(joint=9)) == "few_papers"
    assert cb.pair_status(P(joint=10)) == "studied"


def test_informative_absence_needs_both_concepts_common():
    assert cb.is_informative(P(n_a=800, n_b=900, joint=0))
    assert not cb.is_informative(P(n_a=800, n_b=120, joint=0))     # narrow phrase: can't tell
    assert cb.is_informative(P(n_a=20, n_b=20, joint=3))           # any joint work is informative


def test_verdicts_reproduce_the_observed_failure_modes():
    counts = {"A": 1000, "B": 1000}
    # tiny joint count is NOT "well trodden" (the old lift rule said so)
    r = cb.summarize(counts, [P(joint=3, n_a=5000, n_b=10000)])
    assert r["label"] == "few_papers" and r["combo_novelty"] == pytest.approx(0.7)
    # zero joint with a narrow concept is inconclusive, not "unexplored"
    r = cb.summarize({"A": 1000, "B": 80}, [P(joint=0, n_a=1000, n_b=80)])
    assert r["label"] == "inconclusive" and r["combo_novelty"] is None
    # zero joint between two common concepts is a real gap
    r = cb.summarize(counts, [P(joint=0)])
    assert r["label"] == "unexplored" and r["combo_novelty"] == 1.0
    # an established literature
    r = cb.summarize(counts, [P(joint=390)])
    assert r["label"] == "studied" and r["combo_novelty"] == 0.0
    # coined / rare term
    assert cb.summarize({"A": 900, "B": 3}, [P()])["label"] == "unrecognized_terms"
    assert cb.summarize({}, [])["label"] == "unrecognized_terms"


def test_most_novel_informative_pair_decides_and_uninformative_pairs_are_ignored():
    pairs = [P("A", "B", joint=500),
             P("A", "C", n_b=60, joint=0),          # uninformative zero, must not win
             P("B", "C", n_b=60, joint=0),
             P("A", "D", joint=4)]
    counts = {"A": 1000, "B": 1000, "C": 60, "D": 1000}
    r = cb.summarize(counts, pairs)
    assert r["label"] == "few_papers" and r["pair"]["b"] == "D"


def test_bridge_candidate_needs_adjacent_literatures():
    counts = {"A": 1000, "B": 1000}
    near = cb.summarize(counts, [P(joint=0, adjacency=0.8)], adj_threshold=0.5)
    far = cb.summarize(counts, [P(joint=0, adjacency=0.1)], adj_threshold=0.5)
    assert near["bridge_candidate"] and not far["bridge_candidate"]
    assert not cb.summarize(counts, [P(joint=0, adjacency=0.8)], adj_threshold=None)["bridge_candidate"]
    assert not cb.summarize(counts, [P(joint=50, adjacency=0.9)], adj_threshold=0.5)["bridge_candidate"]


def test_bhattacharyya_and_threshold():
    assert cb.bhattacharyya({"x": 5, "y": 5}, {"x": 1, "y": 1}) == pytest.approx(1.0)
    assert cb.bhattacharyya({"x": 5}, {"y": 5}) == pytest.approx(0.0)
    assert cb.bhattacharyya({}, {"x": 1}) is None
    assert cb.adjacency_threshold([0.6, 0.8], [0.1, 0.3]) == pytest.approx(0.45)
    assert cb.adjacency_threshold([0.3, 0.8], [0.4]) is None       # not separable
    assert cb.adjacency_threshold([], [0.1]) is None


def test_parse_concepts_dedupes_limits_and_cleans():
    data = {"concepts": ["Phase transition", "phase transition", "opinion dynamics",
                         "a very long phrase that is not a standard term at all", "Ising, model",
                         "network science", "extra"]}
    assert cb.parse_concepts(data) == ["Phase transition", "opinion dynamics", "Ising model",
                                       "network science"]
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
    assert c.count('"x"') == 1234
    assert sum(1 for p in calls if "search" in p) == 1
    assert c.total == 1234


def test_counter_reraises_non_400(monkeypatch):
    def fake_get(path, params=None):
        raise requests.HTTPError(response=_Resp(500))
    monkeypatch.setattr(openalex, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        cb.Counter().count('"x"')


def test_distribution_tries_coarser_grouping_when_a_grouping_is_rejected(monkeypatch):
    seen = []

    def fake_get(path, params=None):
        if "group_by" in params:
            seen.append(params["group_by"])
            if params["group_by"] != "primary_topic.field.id":
                raise requests.HTTPError(response=_Resp(400))
            return {"group_by": [{"key": "https://openalex.org/fields/17", "count": 30},
                                 {"key": "unknown", "count": 5}]}
        return {"meta": {"count": 10}}          # the text filter itself is accepted

    monkeypatch.setattr(openalex, "get", fake_get)
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)
    c = cb.Counter()
    assert c.distribution('"x"') == {"17": 30}
    assert c.mode == "filter"
    assert seen[-1] == "primary_topic.field.id" and len(seen) == 3
    c.distribution('"y"')
    assert seen.count("primary_topic.subfield.id") == 1        # remembered the working attribute


class _FakeCounter:
    mode = "filter"
    total = 2_000_000
    table = {'"A"': 1000, '"B"': 2000, '"C"': 60, '"A" AND "B"': 400, '"A" AND "C"': 0,
             '"B" AND "C"': 0}
    dists = {'"A"': {"1": 5, "2": 5}, '"B"': {"1": 4, "2": 6}, '"C"': {"9": 3}}

    def count(self, q): return self.table[q]
    def try_count(self, q): return self.table.get(q, 2)
    def distribution(self, q): return self.dists[q]


def test_measure_and_storage_roundtrip(tmp_path, monkeypatch):
    counts, pairs = cb.measure(["A", "B", "C"], _FakeCounter())
    assert counts == {"A": 1000, "B": 2000, "C": 60}
    ab = next(p for p in pairs if (p["a"], p["b"]) == ("A", "B"))
    assert ab["joint"] == 400 and ab["adjacency"] > 0.9
    ac = next(p for p in pairs if (p["a"], p["b"]) == ("A", "C"))
    assert ac["adjacency"] == pytest.approx(0.0)

    db = tmp_path / "n.db"
    monkeypatch.setattr(cb, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    cb._save_pairs(con, "hyp:1", pairs, "filter")
    cb._save_pairs(con, "hyp:1", pairs, "filter")             # replaces, no duplicates
    assert con.execute("SELECT COUNT(*) FROM combo_checks").fetchone()[0] == 3
    cb._set_meta(con, "adj_threshold", 0.5)
    con.close()
    assert cb.get_adj_threshold() == 0.5
    s = cb.combo_summary("hyp:1")
    assert s["label"] == "studied"       # A x B (400 works); the zero pairs involve the narrow C
    assert cb.combo_summary("hyp:404") is None
    assert cb.stored_refs() == ["hyp:1"]


def test_old_database_gets_the_adjacency_column(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE combo_checks (target_ref TEXT, concept_a TEXT, concept_b TEXT,
                   n_a INTEGER, n_b INTEGER, joint INTEGER, expected REAL, lift REAL,
                   method TEXT, checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                   PRIMARY KEY (target_ref, concept_a, concept_b))""")
    con.execute("INSERT INTO combo_checks (target_ref, concept_a, concept_b, n_a, n_b, joint, expected, lift) "
                "VALUES ('hyp:1', 'A', 'B', 900, 900, 0, 0.0, 0.0)")
    con.commit(); con.close()
    monkeypatch.setattr(cb, "DB_PATH", db)
    # relabelled from stored counts without any API call: zero joint, both concepts common
    assert cb.combo_summary("hyp:1")["label"] == "unexplored"

    counter = _FakeCounter()
    counter.dists = {'"A"': {"1": 1}, '"B"': {"1": 1}}
    assert cb.fill_adjacency(counter, verbose=False) == 1
    assert cb.combo_summary("hyp:1")["pair"]["adjacency"] == pytest.approx(1.0)


def test_rescore_replaces_old_scores(tmp_path, monkeypatch):
    from nous.evaluation import store
    db = tmp_path / "n.db"
    monkeypatch.setattr(cb, "DB_PATH", db)
    monkeypatch.setattr(store, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    cb._save_pairs(con, "hyp:1", [P(n_a=900, n_b=900, joint=0)], "filter")     # unexplored -> 1.0
    cb._save_pairs(con, "hyp:2", [P(n_a=900, n_b=60, joint=0)], "filter")      # inconclusive
    con.close()
    store.save_scores("hyp:2", cb.RATER, {"combo_novelty": 0.95})              # stale, old rule
    assert cb.rescore_all() == 2
    rows = {(r["target_ref"], r["item"]): r["score"] for r in store.load_scores()
            if r["rater"] == cb.RATER}
    assert rows == {("hyp:1", "combo_novelty"): 1.0, ("hyp:2", "combo_checked"): 1.0}


def test_runner_aggregate_carries_combo_novelty():
    from nous.evaluation.runner import aggregate
    rows = [{"target_ref": "hyp:1", "rater": "llm:m:A", "item": "novelty", "score": 4.0},
            {"target_ref": "hyp:1", "rater": "auto:openalex-combo", "item": "combo_novelty", "score": 0.9}]
    a = aggregate(rows)[0]
    assert a["combo_novelty"] == 0.9 and a["composite"] == 4.0


def test_print_report_groups_by_label_and_marks_bridge_candidates(tmp_path, monkeypatch, capsys):
    from nous.evaluation import store, literature
    db = tmp_path / "n.db"
    monkeypatch.setattr(cb, "DB_PATH", db)
    monkeypatch.setattr(store, "DB_PATH", db)
    monkeypatch.setattr(literature, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    cb._save_pairs(con, "hyp:1", [P("A", "B", joint=0, adjacency=0.8)], "filter")
    cb._save_pairs(con, "hyp:2", [P("C", "D", joint=0, adjacency=0.1)], "filter")
    cb._save_pairs(con, "hyp:3", [P("E", "F", joint=300, adjacency=0.9)], "filter")
    cb._set_meta(con, "adj_threshold", 0.5)
    con.close()
    cb.print_report()
    out = capsys.readouterr().out
    assert "unexplored: 2" in out and "studied: 1" in out
    assert "hyp:1       ★" in out and "hyp:2       ★" not in out
    assert "Bridge candidates: 1" in out


def test_loose_query_drops_stopwords_and_groups_words():
    assert cb.build_loose_query(["phase transition", "theory of mind"]) == \
        "(phase AND transition) AND (theory AND mind)"


def test_robust_means_gap_survives_rewording():
    counts = {"A": 1000, "B": 1000}
    r = cb.summarize(counts, [P(joint=0, adjacency=0.8) | {"loose": 3}], adj_threshold=0.5)
    assert r["robust"] and r["bridge_candidate"]
    r = cb.summarize(counts, [P(joint=0, adjacency=0.8) | {"loose": 400}], adj_threshold=0.5)
    assert not r["robust"] and r["bridge_candidate"]          # phrase-only gap
    assert not cb.summarize(counts, [P(joint=0)], None)["robust"]          # loose not measured
    assert not cb.summarize(counts, [P(joint=50) | {"loose": None}], None)["robust"]


def test_rank_candidates_robust_first_then_adjacency():
    def item(ref, adj, robust):
        return ref, {"bridge_candidate": True, "robust": robust, "pair": {"adjacency": adj}}
    ranked = cb.rank_candidates([item("a", 0.9, False), item("b", 0.5, True), item("c", 0.7, True),
                                 ("d", {"bridge_candidate": False, "robust": True,
                                        "pair": {"adjacency": 1.0}})])
    assert [r for r, _ in ranked] == ["c", "b", "a"]


def test_counter_titles_and_try_count(monkeypatch):
    def fake_get(path, params=None):
        if params.get("per_page") == 1:
            if "(" in params["filter"]:
                raise requests.HTTPError(response=_Resp(400))
            return {"meta": {"count": 7}}
        return {"results": [{"id": "https://openalex.org/W1", "display_name": "T", "publication_year": 2001}]}
    monkeypatch.setattr(openalex, "get", fake_get)
    monkeypatch.setattr(cb.time, "sleep", lambda s: None)
    c = cb.Counter()
    assert c.try_count('"x"') == 7
    assert c.try_count("(a AND b)") is None and c.mode == "filter"      # no fallback triggered
    assert c.titles('"x"') == [{"title": "T", "year": 2001, "work_id": "W1"}]


def test_fill_loose_and_titles_for_stored_pairs(tmp_path, monkeypatch):
    db = tmp_path / "n.db"
    monkeypatch.setattr(cb, "DB_PATH", db)
    con = sqlite3.connect(db)
    cb._ensure_table(con)
    cb._save_pairs(con, "hyp:1", [P("A", "B", joint=3), P("A", "C", joint=0), P("B", "C", joint=40)],
                   "filter")
    con.close()

    class C(_FakeCounter):
        def try_count(self, q): return 5
        def titles(self, q, n=5): return [{"title": "Paper", "year": 2010, "work_id": "W9"}]

    assert cb.fill_loose(C(), verbose=False) == 2            # only the pairs with joint < 10
    assert cb.fill_titles(C(), verbose=False) == 1           # only 1 <= joint < 10
    assert cb.load_titles("A", "B")[0]["title"] == "Paper" and cb.load_titles("A", "C") == []
    assert cb.combo_summary("hyp:1")["pair"]["loose"] == 5


def test_claim_text_drops_the_structure_framing():
    text = ("Concept A from [chemistry] (starch) and Concept B from [medicine] (tendon) share "
            "structure Z (thermodynamic constraints governing polymer packing). This suggests "
            "Hypothesis H: The localized density of proteoglycans in tendon matrix is governed "
            "by liquid-liquid phase separation principles.")
    assert cb.claim_text(text).startswith("The localized density of proteoglycans")
    assert "polymer packing" not in cb.claim_text(text)
    assert cb.claim_text("Plain hypothesis without a marker, but long enough to keep.") == \
        "Plain hypothesis without a marker, but long enough to keep."
    assert cb.claim_text("Hypothesis H: too short") == "Hypothesis H: too short"   # nothing useful after it


def test_aggregate_excludes_novelty_from_loop_composite():
    from nous.evaluation.runner import aggregate
    rows = [{"target_ref": "hyp:1", "rater": "llm:m:A", "item": i, "score": s}
            for i, s in (("novelty", 5.0), ("testability", 3.0), ("specificity", 3.0))]
    a = aggregate(rows)[0]
    assert a["composite_excl_novelty"] == pytest.approx(3.0)
    assert a["composite"] == pytest.approx(11 / 3)
