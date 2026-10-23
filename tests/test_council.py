"""The Scepter council: 10 field Scepters + 10 auditors, sparse (top-k) review."""
import sqlite3

import pytest

from nous import core
from nous.domains import DOMAIN_TO_SCEPTER, FIELD_GROUPS, field_scepter_for, parent_of, scepter_for
from nous.output.graph import scepter_color
from nous.scepter.registry import ALL_SCEPTERS, AUDITORS, COUNCIL_MAP, FIELD_SCEPTERS

OPENALEX_FIELDS = [d for d in DOMAIN_TO_SCEPTER
                   if d not in ("philosophy", "history", "linguistics", "sociology", "economics",
                                "physics", "biology")]


def test_sizes_and_judges_stay_at_five():
    assert len(ALL_SCEPTERS) == 5            # `evaluate` judges are not expanded
    assert len(FIELD_SCEPTERS) == 10
    assert len(AUDITORS) == 10
    assert len(COUNCIL_MAP) == 25            # unique names


def test_every_openalex_field_is_in_exactly_one_group_under_its_parent():
    assert len(OPENALEX_FIELDS) == 26
    seen = {}
    for name, parent, fields in FIELD_GROUPS:
        for f in fields:
            assert f not in seen, f
            seen[f] = name
            assert DOMAIN_TO_SCEPTER[f] == parent, f
    assert set(seen) == set(OPENALEX_FIELDS)


def test_field_lookup_and_parent():
    assert field_scepter_for("neuroscience") == "Scepter-N3"
    assert field_scepter_for("physics") == "Scepter-N1"     # legacy name goes through canonical_domain
    assert field_scepter_for("nonexistent") is None
    assert parent_of("Scepter-N3") == "Scepter-N" == parent_of("Scepter-N")
    assert scepter_for("neuroscience") == "Scepter-N"        # parent mapping unchanged


def test_field_scepter_owns_only_its_fields():
    n3 = next(s for s in FIELD_SCEPTERS if s.name == "Scepter-N3")
    assert "neuroscience" in n3.collector_domains
    assert "mathematics" not in n3.collector_domains
    assert n3._owns("neuroscience") and not n3._owns("mathematics")
    assert ALL_SCEPTERS[2]._owns("mathematics")              # parents unchanged


def test_colours_are_distinct_hex():
    names = [m for m in COUNCIL_MAP]
    cols = [scepter_color(n) for n in names]
    assert all(c.startswith("#") and len(c) == 7 for c in cols)
    assert len(set(cols)) == len(cols)


def test_select_for_audit_takes_top_k_by_confidence_and_skips_errors():
    hs = [{"scepter": "a", "hypothesis": "x", "confidence": 0.2},
          {"scepter": "b", "hypothesis": "[parse error: boom]", "confidence": 0.99},
          {"scepter": "c", "hypothesis": "y", "confidence": "0.9"},
          {"scepter": "d", "hypothesis": "z", "confidence": 0.5},
          {"scepter": "e", "hypothesis": "w", "confidence": None}]
    assert [h["scepter"] for h in core.select_for_audit(hs, 2)] == ["c", "d"]
    assert core.select_for_audit(hs, 0) == []


@pytest.fixture
def fake_llm(tmp_path, monkeypatch):
    db = tmp_path / "nous.db"
    monkeypatch.setattr(core, "DB_PATH", db)
    monkeypatch.setattr(core, "find_cross_domain_connections", lambda *a, **k: [])
    calls = []

    def fake(prompt, model=None, **kw):
        calls.append(prompt)
        if "NOUS — the supreme" in prompt:
            return {"verdict": "v", "confidence": 0.5}
        if "auditor of NOUS" in prompt:
            return {"link_type": "contradicts", "strength": 0.8, "rationale": "r", "challenge": "c"}
        if "OTHER SCEPTER'S HYPOTHESIS" in prompt:
            return {"link_type": "supports", "strength": 0.5, "rationale": "r"}
        return {"hypothesis": "h", "confidence": 0.5, "structural_pattern": "p",
                "validation_question": "q", "domains_connected": []}

    monkeypatch.setattr("nous.scepter.base.generate_json", fake)
    monkeypatch.setattr(core, "generate_json", fake)
    return db, calls


def test_council_debate_uses_about_41_calls(fake_llm):
    db, calls = fake_llm
    res = core.debate("q", verbose=False)
    assert len(res["hypotheses"]) == 10
    assert len(res["critiques"]) == 10 * 3
    assert len(calls) == 10 + 30 + 1
    assert {c["from_scepter"] for c in res["critiques"]} == {a.name for a in AUDITORS}
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from scepter_hypotheses").fetchone()[0] == 10
    assert conn.execute("select count(*) from scepter_critiques").fetchone()[0] == 30


def test_classic_debate_is_unchanged_all_pairs(fake_llm):
    _db, calls = fake_llm
    res = core.debate("q2", verbose=False, classic=True)
    assert len(res["hypotheses"]) == 5
    assert len(res["critiques"]) == 5 * 4
    assert len(calls) == 5 + 20 + 1


def test_explicit_scepters_without_auditors_critique_each_other(fake_llm):
    _db, calls = fake_llm
    res = core.debate("q3", scepters=FIELD_SCEPTERS[:3], verbose=False)
    assert len(res["critiques"]) == 3 * 2
