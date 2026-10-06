import pytest

from nous import loop
from nous.collector import openalex


def _system(extra_gap=False):
    """
    Two clusters of neighbouring fields: x1..x3 and y1..y3. Within a cluster fields meet often
    (high lift); across clusters rarely. x1-x2 is a *gap between neighbours*: same links to
    everything else, but a low direct lift.
    """
    xs, ys = ["x1", "x2", "x3"], ["y1", "y2", "y3"]
    links = []

    def add(a, b, lift, obs):
        links.append({"field_a": a, "field_b": b, "name_a": a, "name_b": b,
                      "lift": lift, "observed": obs})

    for grp in (xs, ys):
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                add(a, b, 5.0, 50_000)
    for a in xs:
        for b in ys:
            add(a, b, 0.1, 5_000)
    for l in links:
        if (l["field_a"], l["field_b"]) == ("x1", "x2"):
            l["lift"], l["observed"] = 0.05, 3_000
        if extra_gap and (l["field_a"], l["field_b"]) == ("y1", "y2"):
            l["lift"], l["observed"] = 0.06, 3_000
    return links


def test_choose_pair_prefers_a_gap_between_neighbours_over_unrelated_fields():
    best = loop.choose_pair(_system(), [])
    assert {best["field_a"], best["field_b"]} == {"x1", "x2"}
    c = best["components"]
    assert c["neighbour"] == pytest.approx(1.0) and c["void"] == pytest.approx(1.0)
    # an unrelated cross-cluster pair has an opposite profile -> no priority
    scores = loop.score_links(_system())
    assert scores[frozenset(("x3", "y3"))]["neighbour"] < 0


def test_pairs_sharing_too_few_works_are_skipped():
    assert loop.choose_pair(_system(), [], min_shared=4_000)["observed"] >= 4_000
    assert loop.choose_pair(_system(), [], min_shared=10 ** 9) is None
    assert loop.choose_pair([], []) is None


def test_choose_pair_skips_tried_pairs_in_either_order():
    hist = [{"field_a": "x2", "field_b": "x1", "reward": 0.0}]
    best = loop.choose_pair(_system(), hist)
    assert {best["field_a"], best["field_b"]} != {"x1", "x2"}


def test_rewarded_fields_raise_priority_of_their_other_pairs():
    links = _system(extra_gap=True)
    assert {loop.choose_pair(links, [])["field_a"]} == {"x1"}               # lowest lift first
    hist = [{"field_a": "y1", "field_b": "y9", "reward": 0.9}]
    best = loop.choose_pair(links, hist, exploit_weight=1.0)
    assert {best["field_a"], best["field_b"]} == {"y1", "y2"}
    zero = [{"field_a": "y1", "field_b": "y9", "reward": 0.0}]
    best = loop.choose_pair(links, zero, exploit_weight=1.0)
    assert {best["field_a"], best["field_b"]} == {"x1", "x2"}


def test_decide_rejects_only_on_evidence_of_prior_work():
    assert loop.decide("studied", 0.9, 0.27)[0] is False
    assert loop.decide("unexplored", 0.20, 0.27)[0] is False         # lit-novelty at/below cut-off
    ok, why = loop.decide("few_papers", 0.40, 0.27, composite=3.5, floor=3.0)
    assert ok and "not found" in why
    ok, why = loop.decide("unrecognized_terms", 0.40, 0.27)          # unverifiable: accepted, flagged
    assert ok and "unverified" in why
    ok, why = loop.decide("unexplored", None, None)
    assert ok and "not compared" in why and "no quality floor" in why


def test_reward_and_query():
    assert loop.reward_for(True, 4.0) == pytest.approx(0.8)
    assert loop.reward_for(False, 4.0) == 0.0 and loop.reward_for(True, None) == 0.0
    link = {"name_a": "1", "name_b": "2"}
    assert loop.make_query(link, [{"title": "Bridge paper"}]) == "1 and 2: Bridge paper"
    assert loop.make_query(link, []) == "1 and 2"


def _steps(label="unexplored", lit=0.4, composite=4.0, fail=None, log=None):
    log = log if log is not None else []

    def generate(q, works):
        if fail == "generate":
            raise RuntimeError("ollama down")
        log.append(f"works:{len(works)}")
        return {"id": 7}

    return loop.Steps(
        collect=lambda link, n: [{"title": "T"}],
        index=lambda: log.append("index"),
        generate=generate,
        judge=lambda ref: composite,
        check=lambda ref: (label, lit),
        cutoff=lambda: 0.27,
        sync=lambda hid: log.append(f"sync:{hid}"),
        describe=lambda ref: f"  hypothesis of {ref}",
    ), log


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "n.db"
    monkeypatch.setattr(loop, "DB_PATH", db)
    monkeypatch.setattr(openalex, "DB_PATH", db)
    monkeypatch.setattr(openalex, "load_field_links", lambda: _system())
    return db


def test_cycle_accepts_and_syncs_only_accepted(env, capsys):
    steps, log = _steps()
    rec = loop.run_cycle(steps, verbose=True)
    assert rec["status"] == "accepted" and rec["hyp_ref"] == "hyp:7" and rec["reward"] == pytest.approx(0.8)
    assert log == ["index", "works:1", "sync:7"]                    # the bridge works reach generation
    assert "hypothesis of hyp:7" in capsys.readouterr().out

    steps, log = _steps(label="studied")
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "rejected" and rec["reward"] == 0.0
    assert "sync:7" not in log
    assert len(loop.load_history()) == 2


def test_cycle_logs_errors_and_does_not_retry_same_pair(env):
    steps, _ = _steps(fail="generate")
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "error" and "ollama down" in rec["reason"]
    first = {loop.load_history()[0]["field_a"], loop.load_history()[0]["field_b"]}
    steps, _ = _steps()
    rec = loop.run_cycle(steps, verbose=False)
    assert {rec["field_a"], rec["field_b"]} != first                # moved on to another pair


def test_cycle_stops_when_no_pair_is_left(env):
    n = 0
    while loop.run_cycle(_steps()[0], verbose=False, min_shared=1000):
        n += 1
        assert n < 100
    assert n >= 1


def test_report_and_plan_print(env, capsys):
    loop.run_cycle(_steps()[0], verbose=False)
    loop.print_report()
    loop.print_plan(top=2)
    out = capsys.readouterr().out
    assert "accepted" in out and "Next pairs" in out and "neighbour=" in out


def test_pairs_that_meet_as_often_as_chance_are_not_gaps():
    links = _system()
    scores = loop.score_links(links)
    hist = [{"field_a": l["field_a"], "field_b": l["field_b"], "reward": 0.0}
            for l in links if scores[frozenset((l["field_a"], l["field_b"]))]["void"] >= loop.MIN_VOID]
    assert loop.choose_pair(links, hist) is None           # only high-lift pairs are left
    assert loop.choose_pair(links, hist, min_void=0.0) is not None


def test_a_poor_first_run_no_longer_dominates_the_next_choice():
    links = _system(extra_gap=True)
    hist = [{"field_a": "y1", "field_b": "y9", "reward": 0.64}]
    best = loop.choose_pair(links, hist)                     # default exploit weight is small
    assert {best["field_a"], best["field_b"]} == {"x1", "x2"}


def test_quality_floor_and_rejection_below_it():
    assert loop.quality_floor([3.0] * 9) is None
    assert loop.quality_floor([None, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 4.0]) == 3.0
    ok, why = loop.decide("unexplored", 0.4, 0.27, composite=2.75, floor=3.4)
    assert not ok and "below the median" in why
    ok, _ = loop.decide("unexplored", 0.4, 0.27, composite=3.4, floor=3.4)
    assert ok


def test_cycle_rejects_a_hypothesis_below_the_quality_floor(env):
    steps, log = _steps(composite=2.0)
    steps.floor = lambda: 3.4
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == "rejected" and "below the median" in rec["reason"]
    assert "sync:7" not in log


def test_bridge_prompt_forbids_the_template_and_lists_the_works():
    from nous.engine.hypothesis import build_bridge_prompt
    p = build_bridge_prompt("A and B", [{"domain": "X", "title": "Paper one", "summary": "abstract"}])
    assert "Paper one" in p and "Do NOT write" in p and '"hypothesis"' in p
