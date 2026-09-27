import sqlite3

import numpy as np
import pytest

from nous.evaluation import rubric, store
from nous.evaluation.novelty import novelty_from_sims
from nous.evaluation.reliability import icc, interpret
from nous.evaluation.runner import aggregate, rating_matrix, reliability_report

# Shrout & Fleiss (1979), Table 2: 6 targets × 4 judges
SF = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]


def test_icc_matches_shrout_fleiss_published_values():
    r = icc(SF)
    expected = {"ICC1": .17, "ICC2": .29, "ICC3": .71, "ICC1k": .44, "ICC2k": .62, "ICC3k": .91}
    for k, v in expected.items():
        assert r[k] == pytest.approx(v, abs=0.005), k


def test_icc_perfect_agreement_and_errors():
    assert icc([[1, 1], [3, 3], [5, 5]])["ICC2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        icc([[1, np.nan], [2, 2]])
    with pytest.raises(ValueError):
        icc([[1, 2]])
    assert interpret(0.8) == "good" and interpret(0.3) == "poor"


def _rows():
    rows = []
    for t, scores in enumerate(SF):
        for j, s in enumerate(scores):
            rows.append({"target_ref": f"hyp:{t}", "rater": f"llm:m:J{j}", "item": "novelty", "score": s})
    rows.append({"target_ref": "hyp:0", "rater": "auto:embedding", "item": "novelty_embedding", "score": 0.4})
    rows.append({"target_ref": "hyp:99", "rater": "llm:m:J0", "item": "novelty", "score": 3})  # incomplete
    return rows


def test_rating_matrix_keeps_complete_targets_only():
    m, targets, raters = rating_matrix(_rows(), "novelty")
    assert m.shape == (6, 4) and "hyp:99" not in targets and len(raters) == 4
    rep = {e["item"]: e for e in reliability_report(_rows())}
    assert rep["novelty"]["ICC2"] == pytest.approx(.29, abs=0.005)
    assert "ICC2" not in rep["testability"]


def test_aggregate_ranks_by_composite_and_keeps_auto_metric_separate():
    agg = aggregate(_rows())
    assert agg[0]["ref"] == "hyp:4"               # row [10,5,6,9] has the highest mean
    top0 = next(a for a in agg if a["ref"] == "hyp:0")
    assert top0["novelty_embedding"] == 0.4
    assert top0["item_means"]["novelty"] == pytest.approx(6.0)


def test_novelty_from_sims():
    assert novelty_from_sims([0.2, 0.9]) == pytest.approx(0.1)
    assert novelty_from_sims([]) is None


def test_llm_judge_clamps_and_drops_bad_items(monkeypatch):
    monkeypatch.setattr(rubric, "generate_json", lambda prompt, model: {
        "scores": {"novelty": 7, "testability": "2", "specificity": "n/a"},
        "rationale": {"novelty": "new", "bogus": "x"},
    })
    scores, rat = rubric.llm_judge("H", "Q", persona="physics")
    assert scores == {"novelty": 5.0, "testability": 2.0}
    assert rat == {"novelty": "new"}


def test_store_roundtrip_and_targets(tmp_path):
    db = tmp_path / "nous.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE explorations (id INTEGER PRIMARY KEY, query TEXT);
        CREATE TABLE hypotheses (id INTEGER PRIMARY KEY, exploration_id INTEGER, hypothesis_text TEXT);
        CREATE TABLE debates (id INTEGER PRIMARY KEY, query TEXT, verdict TEXT);
        CREATE TABLE scepter_hypotheses (id INTEGER PRIMARY KEY, debate_id INTEGER, scepter TEXT, hypothesis TEXT);
        INSERT INTO explorations VALUES (1, 'q1');
        INSERT INTO hypotheses VALUES (1, 1, 'H one'), (2, 1, '[parse error: x]');
        INSERT INTO debates VALUES (1, 'q2', '{"verdict": "V one"}');
        INSERT INTO scepter_hypotheses VALUES (1, 1, 'Scepter-N', 'S one');
    """)
    conn.commit(); conn.close()

    refs = {t["ref"] for t in store.load_targets(db)}
    assert refs == {"hyp:1", "scepter:1", "verdict:1"}

    store.save_scores("hyp:1", "human:me", {"novelty": 3}, db_path=db)
    store.save_scores("hyp:1", "human:me", {"novelty": 4}, db_path=db)   # replace, not duplicate
    rows = store.load_scores(db_path=db)
    assert rows == [{"target_ref": "hyp:1", "rater": "human:me", "item": "novelty", "score": 4.0}]
    assert store.rated_by("human:me", db_path=db) == {"hyp:1"}
