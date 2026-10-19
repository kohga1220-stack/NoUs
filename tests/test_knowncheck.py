import pytest

from nous import loop
from nous.evaluation import knowncheck as kc
from tests.test_loop import env  # noqa: F401  (fixture)

ABSTRACT = ("We show that the static dielectric constant of aqueous electrolyte solutions decreases "
            "with the ion concentration because of the structure of ionic hydration shells.")
WORKS = [
    {"work_id": "W1", "title": "Unrelated paper", "year": 2001, "text": "Unrelated paper. Something else entirely."},
    {"work_id": "W2", "title": "Static dielectric constant of electrolytes", "year": 2000,
     "text": f"Static dielectric constant of electrolytes. {ABSTRACT}"},
]
GOOD_QUOTE = "the static dielectric constant of aqueous electrolyte solutions decreases with the ion concentration"


def test_quote_supported_accepts_the_text_and_light_rewording_but_not_inventions():
    assert kc.quote_supported(GOOD_QUOTE, ABSTRACT)
    assert kc.quote_supported("static dielectric constant of aqueous electrolyte solutions decrease with ion concentration",
                              ABSTRACT)                                           # small wording change
    assert not kc.quote_supported("magnetic monopoles condense in a quantum spin liquid at low temperature", ABSTRACT)
    assert not kc.quote_supported("too short", "too short text")                  # shorter than 20 characters


def test_parse_verdict_established_needs_a_verifiable_quote():
    ok = kc.parse_verdict({"verdict": "established", "work": 2, "quote": GOOD_QUOTE}, WORKS)
    assert ok["verdict"] == "established" and ok["title"].startswith("Static dielectric") and ok["year"] == 2000
    fake = kc.parse_verdict({"verdict": "established", "work": 2,
                             "quote": "magnetic monopoles condense in a quantum spin liquid at low temperature"}, WORKS)
    assert fake["verdict"] == "related" and "not found" in fake["evidence"]
    wrong_work = kc.parse_verdict({"verdict": "established", "work": 1, "quote": GOOD_QUOTE}, WORKS)
    assert wrong_work["verdict"] == "related"                                      # quote is from another work
    assert kc.parse_verdict({"verdict": "established", "work": 9, "quote": GOOD_QUOTE}, WORKS)["verdict"] == "related"
    assert kc.parse_verdict({"verdict": "established", "work": None, "quote": ""}, WORKS)["verdict"] == "related"


def test_parse_verdict_other_answers_and_garbage():
    assert kc.parse_verdict({"verdict": "related"}, WORKS)["verdict"] == "related"
    assert kc.parse_verdict({"verdict": "not_found"}, WORKS)["verdict"] == "not_found"
    for bad in (None, [], "established", {"verdict": "maybe"}, {}):
        assert kc.parse_verdict(bad, WORKS)["verdict"] == "unknown"


def test_prompt_has_the_claim_the_works_and_the_narrow_question():
    p = kc.build_prompt("Hypothesis claim X", WORKS)
    assert "Hypothesis claim X" in p and "[2] Static dielectric constant of electrolytes (2000)" in p
    assert "Being on the same topic is NOT enough" in p and '"verdict"' in p


def test_check_known_uses_only_the_claim_and_never_raises():
    seen = {}

    def search(text, n):
        seen["q"] = text
        return WORKS

    def ask(prompt, model):
        seen["prompt"], seen["model"] = prompt, model
        return {"verdict": "established", "work": 2, "quote": GOOD_QUOTE}

    long_text = "Concept A from [X] and B from [Y] share structure Z. Hypothesis H: " + "dielectric constant scales with ion radii. " * 3
    r = kc.check_known(long_text, "m", search=search, ask=ask)
    assert r["verdict"] == "established" and seen["model"] == "m"
    assert seen["q"].startswith("dielectric constant scales") and "share structure Z" not in seen["prompt"]

    def boom(*a, **k):
        raise RuntimeError("down")

    assert kc.check_known("t", "m", search=boom, ask=ask)["verdict"] == "unknown"
    assert kc.check_known("t", "m", search=lambda t, n: WORKS, ask=boom)["verdict"] == "unknown"
    assert kc.check_known("t", "m", search=lambda t, n: [], ask=ask)["verdict"] == "not_found"


def test_apply_known_only_a_verified_established_rejects():
    est = {"verdict": "established", "title": "T", "year": 2000, "evidence": "says it"}
    ok, why = loop.apply_known(est, "fine")
    assert not ok and "already established: T (2000)" in why
    for v in ("related", "not_found", "unknown"):
        ok, why = loop.apply_known({"verdict": v}, "fine")
        assert ok and "known-work check: " + v in why


def _steps_with_known(verdict):
    from tests.test_loop import _steps
    steps, log = _steps()
    steps.known = lambda ref: {"verdict": verdict, "title": "T", "year": 2000, "evidence": "quote"}
    return steps, log


@pytest.mark.parametrize("verdict, status", [("established", "rejected"), ("related", "accepted"),
                                             ("not_found", "accepted"), ("unknown", "accepted")])
def test_cycle_applies_the_known_work_check_after_the_other_checks(env, verdict, status):
    steps, log = _steps_with_known(verdict)
    rec = loop.run_cycle(steps, verbose=False)
    assert rec["status"] == status
    assert ("sync:7" in log) == (status == "accepted")
    if status == "rejected":
        assert rec["reward"] == 0.0 and "already established" in rec["reason"]


def test_known_check_is_not_called_for_hypotheses_already_rejected(env):
    from tests.test_loop import _steps
    steps, _ = _steps(label="studied")
    steps.known = lambda ref: (_ for _ in ()).throw(AssertionError("called"))
    assert loop.run_cycle(steps, verbose=False)["status"] == "rejected"


def test_cli_controls_and_accepted_retract(monkeypatch, capsys):
    from nous.evaluation import store
    calls = []
    monkeypatch.setattr(kc, "check_known", lambda text, model: {
        "verdict": "established" if "MARKER" in text else "related", "title": "T", "year": 2001, "evidence": "q"})
    monkeypatch.setattr(kc, "_accepted_refs", lambda: ["hyp:1", "hyp:2"])
    monkeypatch.setattr(store, "load_targets", lambda *a, **k: [{"ref": "hyp:1", "text": "MARKER claim"},
                                                                {"ref": "hyp:2", "text": "new claim"}])
    monkeypatch.setattr(loop, "retract", lambda ref, reason, **k: calls.append((ref, reason)))
    kc.run(["--accepted", "--retract", "--model", "m"])
    assert [c[0] for c in calls] == ["hyp:1"] and calls[0][1].startswith("retracted: established: T")
    kc.run(["--controls"])
    assert "flagged as established: 0/10" in capsys.readouterr().out
