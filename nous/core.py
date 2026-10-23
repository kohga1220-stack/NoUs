"""
NOUS Core — the central orchestrator.

Workflow for a 'debate' session:
  1. Retrieve cross-domain context from ChromaDB
  2. Activate the council: 10 field Scepters (generators) + 10 role auditors
  3. Each field Scepter generates its domain-specific hypothesis (sequentially)
  4. Each auditor checks the top-k hypotheses (by confidence) on its one question.
     (`classic`: the 5 parent Scepters critique every other one, all pairs.)
  5. NOUS Core synthesizes: weighs evidence, resolves conflicts, issues verdict
  6. Save everything to SQLite memory
"""
from __future__ import annotations
import json
import sqlite3

from nous.config import DB_PATH, DEFAULT_MODEL
from nous.engine.connector import find_cross_domain_connections
from nous.llm import generate_json
from nous.scepter.registry import ALL_SCEPTERS, AUDITORS, FIELD_SCEPTERS, Scepter

LINK_SYMBOLS = {"supports": "✓", "contradicts": "✗", "extends": "+"}
TOP_K = 3   # hypotheses each auditor reviews


# ------------------------------------------------------------------ #
#  Database helpers                                                   #
# ------------------------------------------------------------------ #

def _ensure_debate_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS debates (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT,
            verdict   TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS scepter_hypotheses (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id INTEGER,
            scepter   TEXT,
            domain    TEXT,
            hypothesis TEXT,
            confidence REAL,
            structural_pattern TEXT,
            validation_question TEXT
        );
        CREATE TABLE IF NOT EXISTS scepter_critiques (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id     INTEGER,
            from_scepter  TEXT,
            to_scepter    TEXT,
            link_type     TEXT,
            strength      REAL,
            rationale     TEXT,
            challenge     TEXT
        );
    """)
    conn.commit()
    conn.close()


def _save_debate(query: str, hypotheses: list[dict],
                 critiques: list[dict], verdict: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO debates (query, verdict) VALUES (?, ?)",
              (query, json.dumps(verdict, ensure_ascii=False)))
    debate_id = c.lastrowid
    for h in hypotheses:
        c.execute("""
            INSERT INTO scepter_hypotheses
            (debate_id, scepter, domain, hypothesis, confidence,
             structural_pattern, validation_question)
            VALUES (?,?,?,?,?,?,?)
        """, (debate_id, h.get("scepter"), h.get("domain"),
              h.get("hypothesis"), h.get("confidence", 0.5),
              h.get("structural_pattern", ""), h.get("validation_question", "")))
    for cr in critiques:
        c.execute("""
            INSERT INTO scepter_critiques
            (debate_id, from_scepter, to_scepter, link_type, strength, rationale, challenge)
            VALUES (?,?,?,?,?,?,?)
        """, (debate_id, cr.get("from_scepter"), cr.get("to_scepter"),
              cr.get("link_type"), cr.get("strength", 0.5),
              cr.get("rationale", ""), cr.get("challenge", "")))
    conn.commit()
    conn.close()
    return debate_id


# ------------------------------------------------------------------ #
#  NOUS Core synthesis                                                #
# ------------------------------------------------------------------ #

def _synthesize(query: str, hypotheses: list[dict],
                critiques: list[dict], model: str) -> dict:
    hyp_text = "\n\n".join(
        f"[{h['scepter']} — {h['domain']}]\n"
        f"Hypothesis: {h.get('hypothesis','')}\n"
        f"Confidence: {h.get('confidence',0)}\n"
        f"Validation: {h.get('validation_question','')}"
        for h in hypotheses
    )
    crit_text = "\n".join(
        f"  {cr['from_scepter']} {LINK_SYMBOLS.get(cr['link_type'],'?')} "
        f"{cr['to_scepter']}: {cr.get('rationale','')}"
        + (f" [challenge: {cr['challenge']}]" if cr.get("challenge") and cr.get("link_type") == "contradicts" else "")
        for cr in critiques
    )

    prompt = f"""You are NOUS — the supreme integrating intelligence that oversees all Scepters.

QUERY: {query}

SCEPTER HYPOTHESES:
{hyp_text}

REVIEWS (✓ supports / passes, ✗ contradicts / fails, + extends / passes with a change):
{crit_text}

Your task as NOUS:
1. Identify the deepest structural insight that emerges from the convergence of ALL Scepters
2. Resolve any contradictions by proposing a higher-order principle
3. Identify which validation question is most experimentally tractable RIGHT NOW
4. Predict: what human limitation does this query expose, and what breakthrough direction does the synthesis suggest?

Respond ONLY with JSON:
{{
  "unified_structural_pattern": "The meta-pattern visible only from integrating all domains",
  "verdict": "NOUS's final unified hypothesis — bolder and more general than any single Scepter's",
  "resolved_contradictions": ["If any Scepters contradicted each other, how NOUS resolves it"],
  "priority_validation": "The single most tractable experiment across all Scepter proposals",
  "human_limitation_exposed": "What cognitive or institutional limit does this query reveal?",
  "breakthrough_direction": "The most promising direction for transcending that limitation",
  "confidence": 0.0-1.0
}}"""

    try:
        return generate_json(prompt, model=model)
    except Exception as ex:
        return {"verdict": f"[synthesis error: {ex}]", "confidence": 0.0}


def select_for_audit(hypotheses: list[dict], k: int = TOP_K) -> list[dict]:
    """The k most confident usable hypotheses (ties keep generation order)."""
    def conf(h: dict) -> float:
        try:
            return float(h.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    usable = [h for h in hypotheses if not str(h.get("hypothesis", "")).startswith("[parse error")]
    return sorted(usable, key=lambda h: -conf(h))[:max(k, 0)]


def previous_debates(query: str) -> list[int]:
    """Ids of earlier debates with exactly the same question (case/space-insensitive)."""
    _ensure_debate_tables()
    key = " ".join(query.lower().split())
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT id, query FROM debates").fetchall()
    finally:
        conn.close()
    return [i for i, q in rows if " ".join((q or "").lower().split()) == key]


# ------------------------------------------------------------------ #
#  Main debate orchestration                                          #
# ------------------------------------------------------------------ #

def debate(
    query: str,
    scepters: list[Scepter] | None = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    auditors: list | None = None,
    top_k: int = TOP_K,
    classic: bool = False,
) -> dict:
    """
    Run a full NOUS debate session on a query.

    Args:
        query:    The concept or question to investigate.
        scepters: Generators to activate (default: the 10 field Scepters).
        auditors: Role auditors (default: all 10 when `scepters` is not given, else none).
                  With no auditors, the generators critique each other (all pairs).
        top_k:    Hypotheses each auditor reviews (the most confident ones).
        classic:  The original debate: the 5 parent Scepters, all-pairs critique.
        model:    Ollama model to use for all agents (Scepters and NOUS).
        verbose:  Print progress to stdout.

    Returns:
        Full debate result dict.
    """
    _ensure_debate_tables()
    if classic:
        scepters, auditors = ALL_SCEPTERS, []
    elif scepters is None:
        scepters = FIELD_SCEPTERS
        if auditors is None:
            auditors = AUDITORS
    auditors = auditors or []
    # Re-instantiate so every agent in this session uses the requested model
    scepters = [type(s)(model=model) for s in scepters]
    auditors = [type(a)(model=model) for a in auditors]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  NOUS DEBATE: {query}")
        print(f"{'='*60}\n")

    # Step 1: retrieve cross-domain context
    if verbose:
        print("[NOUS] Retrieving cross-domain context...")
    context = find_cross_domain_connections(query, n_results=12, llm_model=model)

    # Step 2: each Scepter generates its hypothesis
    hypotheses: list[dict] = []
    for scepter in scepters:
        if verbose:
            print(f"[{scepter.name}] Generating {scepter.domain} hypothesis...")
        h = scepter.generate(query, context)
        hypotheses.append(h)
        if verbose:
            conf = h.get("confidence", 0)
            hyp_short = h.get("hypothesis", "")[:100]
            print(f"         → {hyp_short}... (conf={conf})")

    # Step 3: review. Auditors check the top-k hypotheses; without auditors, all-pairs critique.
    critiques: list[dict] = []
    if auditors:
        targets = select_for_audit(hypotheses, top_k)
        if verbose:
            print(f"\n[NOUS] {len(auditors)} auditors review {len(targets)} hypotheses "
                  f"({len(auditors) * len(targets)} reviews)...")
        for auditor in auditors:
            for target in targets:
                cr = auditor.review(target)
                critiques.append(cr)
                if verbose:
                    sym = LINK_SYMBOLS.get(cr.get("link_type", ""), "?")
                    print(f"  {cr['from_scepter']} {sym} {cr['to_scepter']}: "
                          f"{cr.get('rationale','')[:80]}")
    else:
        if verbose:
            print(f"\n[NOUS] Running inter-Scepter critique ({len(scepters)*(len(scepters)-1)} pairs)...")
        for i, scepter in enumerate(scepters):
            for j, other_hyp in enumerate(hypotheses):
                if i == j:
                    continue
                cr = scepter.critique(hypotheses[i], other_hyp)
                critiques.append(cr)
                if verbose:
                    sym = LINK_SYMBOLS.get(cr.get("link_type", ""), "?")
                    print(f"  {cr['from_scepter']} {sym} {cr['to_scepter']}: "
                          f"{cr.get('rationale','')[:80]}")

    # Step 4: NOUS synthesizes
    if verbose:
        print("\n[NOUS] Synthesizing final verdict...")
    verdict = _synthesize(query, hypotheses, critiques, model)

    # Step 5: save to DB
    debate_id = _save_debate(query, hypotheses, critiques, verdict)

    result = {
        "debate_id":  debate_id,
        "query":      query,
        "hypotheses": hypotheses,
        "critiques":  critiques,
        "verdict":    verdict,
    }

    if verbose:
        _print_verdict(verdict)

    return result


def _print_verdict(verdict: dict):
    print(f"\n{'='*60}")
    print("  NOUS VERDICT")
    print(f"{'='*60}")
    print(f"\n[Unified Pattern]\n{verdict.get('unified_structural_pattern','')}")
    print(f"\n[Verdict]\n{verdict.get('verdict','')}")
    if verdict.get("resolved_contradictions"):
        print("\n[Resolved Contradictions]")
        for r in verdict["resolved_contradictions"]:
            print(f"  • {r}")
    print(f"\n[Priority Validation]\n{verdict.get('priority_validation','')}")
    print(f"\n[Human Limitation Exposed]\n{verdict.get('human_limitation_exposed','')}")
    print(f"\n[Breakthrough Direction]\n{verdict.get('breakthrough_direction','')}")
    print(f"\nConfidence: {verdict.get('confidence',0)}")
    print(f"{'='*60}\n")


def show_debates(limit: int = 5):
    _ensure_debate_tables()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT d.id, d.query, d.timestamp, COUNT(h.id) as n_hyp
        FROM debates d
        LEFT JOIN scepter_hypotheses h ON h.debate_id = d.id
        GROUP BY d.id ORDER BY d.id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    print("\n=== Recent NOUS Debates ===\n")
    for r in rows:
        print(f"  #{r[0]} [{r[2]}] {r[1][:60]}  ({r[3]} Scepters)")
