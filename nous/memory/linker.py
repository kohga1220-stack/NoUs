"""
Automatically analyze relationships between stored hypotheses using Ollama,
then store links in hypothesis_links table.
Link types: 'supports', 'contradicts', 'extends'
"""
from __future__ import annotations
import sqlite3
from itertools import combinations

from nous.config import DB_PATH, DEFAULT_MODEL
from nous.llm import generate_json


def _ensure_link_columns(conn: sqlite3.Connection):
    """Migrate older DBs: add rationale / strength columns if missing."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(hypothesis_links)").fetchall()]
    if "rationale" not in cols:
        conn.execute("ALTER TABLE hypothesis_links ADD COLUMN rationale TEXT")
    if "strength" not in cols:
        conn.execute("ALTER TABLE hypothesis_links ADD COLUMN strength REAL")
    conn.commit()


def get_all_hypotheses() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT h.id, e.query, h.hypothesis_text, h.confidence, h.timestamp
        FROM hypotheses h
        JOIN explorations e ON h.exploration_id = e.id
        ORDER BY h.id
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "query": r[1], "hypothesis": r[2],
         "confidence": r[3], "timestamp": r[4]}
        for r in rows
    ]


def get_existing_links() -> set[frozenset[int]]:
    """Linked pairs, orientation-independent (A->B and B->A are the same pair)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT hyp_id_a, hyp_id_b FROM hypothesis_links").fetchall()
    conn.close()
    return {frozenset((r[0], r[1])) for r in rows}


def is_linked(existing: set[frozenset[int]], id_a: int, id_b: int) -> bool:
    return frozenset((id_a, id_b)) in existing


def save_link(hyp_id_a: int, hyp_id_b: int, link_type: str,
              rationale: str = "", strength: float | None = None):
    conn = sqlite3.connect(DB_PATH)
    _ensure_link_columns(conn)
    conn.execute("""
        INSERT INTO hypothesis_links (hyp_id_a, hyp_id_b, link_type, rationale, strength)
        VALUES (?, ?, ?, ?, ?)
    """, (hyp_id_a, hyp_id_b, link_type, rationale, strength))
    conn.commit()
    conn.close()


def analyze_pair(hyp_a: dict, hyp_b: dict, model: str) -> dict | None:
    prompt = f"""You are Nous, analyzing relationships between two scientific hypotheses.

HYPOTHESIS A (id={hyp_a['id']}, query="{hyp_a['query']}"):
{hyp_a['hypothesis']}

HYPOTHESIS B (id={hyp_b['id']}, query="{hyp_b['query']}"):
{hyp_b['hypothesis']}

Analyze the logical and structural relationship between these two hypotheses.
Choose exactly one relationship type:
- "supports": A and B reinforce each other; if one is true, it increases the probability of the other
- "contradicts": A and B are in tension; if one is true, it challenges the other
- "extends": One hypothesis is a special case, elaboration, or logical consequence of the other

Respond ONLY with JSON:
{{
  "link_type": "supports" | "contradicts" | "extends",
  "direction": "A->B" | "B->A" | "bidirectional",
  "rationale": "One sentence explaining why.",
  "strength": 0.0-1.0
}}"""

    try:
        return generate_json(prompt, model=model)
    except Exception as e:
        print(f"    [WARN] parse error: {e}")
        return None


def link_all(model: str = DEFAULT_MODEL):
    hypotheses = get_all_hypotheses()
    if len(hypotheses) < 2:
        print("Need at least 2 hypotheses. Run 'nous.py hyp' first.")
        return

    existing = get_existing_links()
    pairs = list(combinations(hypotheses, 2))
    new_links = 0

    print(f"Analyzing {len(pairs)} hypothesis pair(s) with {model}...\n")

    for hyp_a, hyp_b in pairs:
        if is_linked(existing, hyp_a["id"], hyp_b["id"]):
            print(f"  [SKIP] #{hyp_a['id']} ↔ #{hyp_b['id']} already linked")
            continue

        print(f"  Pair #{hyp_a['id']} ({hyp_a['query']}) ↔ #{hyp_b['id']} ({hyp_b['query']})")
        result = analyze_pair(hyp_a, hyp_b, model)
        if not result:
            continue

        link_type  = result.get("link_type", "extends")
        direction  = result.get("direction", "bidirectional")
        rationale  = result.get("rationale", "")
        try:
            strength = float(result.get("strength", 0.5))
        except (TypeError, ValueError):
            strength = 0.5

        # store canonical direction
        if direction == "B->A":
            id_a, id_b = hyp_b["id"], hyp_a["id"]
        else:
            id_a, id_b = hyp_a["id"], hyp_b["id"]

        save_link(id_a, id_b, link_type, rationale, strength)
        existing.add(frozenset((id_a, id_b)))
        new_links += 1

        symbol = {"supports": "⟶✓", "contradicts": "⟶✗", "extends": "⟶+"}.get(link_type, "⟶?")
        print(f"    {symbol} [{link_type}] (strength={strength:.2f})")
        print(f"    {rationale}\n")

    print(f"Done. {new_links} new link(s) stored.")


def show_network():
    conn = sqlite3.connect(DB_PATH)
    _ensure_link_columns(conn)
    c = conn.cursor()

    c.execute("""
        SELECT l.hyp_id_a, l.hyp_id_b, l.link_type, l.rationale, l.strength,
               ea.query, eb.query
        FROM hypothesis_links l
        JOIN hypotheses ha ON l.hyp_id_a = ha.id
        JOIN hypotheses hb ON l.hyp_id_b = hb.id
        JOIN explorations ea ON ha.exploration_id = ea.id
        JOIN explorations eb ON hb.exploration_id = eb.id
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        print("No hypothesis links yet.")
        return

    SYMBOLS = {"supports": "✓ supports", "contradicts": "✗ contradicts", "extends": "+ extends"}
    print("\n=== Hypothesis Network ===\n")
    for r in rows:
        sym = SYMBOLS.get(r[2], r[2])
        strength = f" ({r[4]:.2f})" if r[4] is not None else ""
        print(f"  #{r[0]} [{r[5]}]  {sym}{strength}  #{r[1]} [{r[6]}]")
        if r[3]:
            print(f"     └─ {r[3]}")
    print()
