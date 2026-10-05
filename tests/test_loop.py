import sqlite3

import pytest

from nous import loop
from nous.collector import openalex


def L(a, b, lift, observed=100):
    return {"field_a": a, "field_b": b, "name_a": a, "name_b": b,
            "observed": observed, "lift": lift}


LINKS = [L("1", "2", 0.5), L("3", "4", 0.1), L("5", "6", 0.3), L("7", "8", 0.05, observed=0)]


def test_choose_pair_takes_lowest_lift_and_skips_pairs_without_shared_works():
    assert loop.choose_pair(LINKS, [])["field_a"] == "3"           # 0.1; the 0.05 pair has no works
    assert loop.choose_pair([L("7", "8", 0.01, observed=0)], []) is None
    assert loop.choose_pair([], []) is None


def test_choose_pair_skips_tried_pairs_in_either_order():
    hist = [{"field_a": "4", "field_b": "3", "reward": 0.0}]
    assert loop.choose_pair(LINKS, hist)["field_a"] == "5"


def test_rewarded_fields_raise_priority_of_their_other_pairs():
    links = [L("1", "2", 0.1), L("1", "3", 0.2), L("4", "5", 0.15)]
    hist = [{"field_a": "1", "field_b": "9", "reward": 0.9}]
    # without feedback the lowest lift wins; field 1 earned a reward so a pair with it overtakes
    assert loop.choose_pair(links, [])["field_b"] == "2"
    best = loop.choose_pair(links, hist, exploit_weight=1.0)
    assert best["field_a"] == "1"
    zero = [{"field_a": "1", "field_b": "9", "reward": 0.0}]
    assert loop.choose_pair(links, zero, exploit_weight=1.0)["field_b"] == "2"


def test_decide_rejects_only_on_evidence_of_prior_work():
    assert loop.decide("studied", 0.9, 0.27)[0] is False
    assert loop.decide("unexplored", 0.20, 0.27)[0] is False         # lit-novelty at/below cut-off
    ok, why = loop.decide("few_papers", 0.40, 0.27)
    assert ok and "not found" in why
    ok, why = loop.decide("unrecognized_terms", 0.40, 0.27)          # unverifiable: accepted, flagged
    assert ok and "unverified" in why
    ok, why = loop.decide("unexplored", None, None)
    assert ok and "not compared" in why


def test_reward_and_query():
    assert loop.reward_for(True, 4.0) == pytest.approx(0.8)
    assert loop.reward_for(False, 4.0) == 0.0 and loop.reward_for(True, None) == 0.0
    assert loop.make_query(L("1", "2", 0.1), [{"title": "Bridge paper"}]) == "1 and 2: Bridge paper"
    assert loop.make_query(L("1", "2", 0.1), []) == "1 and 2"


def _steps(label="unexplored", lit=0.4, composite=4.0, fail=None, log=None):
    log = log if log is not None else []

    def generate(q):
        if fail == "generate":
            raise RuntimeError("ollama down")
        return {"id": 7}

    return loop.Steps(
        collect=lambda link, n: [{"title": "T"}],
        index=lambda: log.append("index"),
        generate=generate,
        judge=lambda ref: composite,
        check=lambda ref: (label, lit),
        cutoff=lambda: 0.27,
        sync=lambda hid: log.append(f"sync:{hid}"),
    ), log


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "n.db"
    monkeypatch.setattr(loop, "DB_PATH", db)
    monkeypatch.setattr(openalex, "DB_PATH", db)
    monkeypatch.setattr(openalex, "load_field_links", lambda: LINKS)
    return db


def test_cycle_accepts_and_syncs_only_accepted(env):
    steps, log = _steps()
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "accepted" and rec["hyp_ref"] == "hyp:7" and rec["reward"] == pytest.approx(0.8)
    assert log == ["index", "sync:7"]

    steps, log = _steps(label="studied")
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "rejected" and rec["reward"] == 0.0
    assert "sync:7" not in log
    assert len(loop.load_history()) == 2


def test_cycle_logs_errors_and_does_not_retry_same_pair(env):
    steps, _ = _steps(fail="generate")
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "error" and "ollama down" in rec["reason"]
    assert loop.load_history()[0]["field_a"] == "3"
    steps, _ = _steps()
    assert loop.run_cycle(steps, verbose=False)["field_a"] == "5"        # moved on to the next pair


def test_cycle_stops_when_no_pair_is_left(env):
    for _ in range(3):
        assert loop.run_cycle(_steps()[0], verbose=False)
    assert loop.run_cycle(_steps()[0], verbose=False) is None


def test_report_and_plan_print(env, capsys):
    loop.run_cycle(_steps()[0], verbose=False)
    loop.print_report()
    loop.print_plan(top=2)
    out = capsys.readouterr().out
    assert "accepted" in out and "Next pairs" in out
