"""
Autonomous loop (phase 4): explore literature voids, propose a hypothesis, and reject it
when the literature already knows it.

One cycle
  1. choose   a pair of OpenAlex fields that rarely meet (low lift) and has not been tried;
              pairs whose fields produced well-rated hypotheses before are preferred
  2. collect  the most-cited works tagged with BOTH fields (humanity's existing dockings)
  3. index    embed the new works (and abstract their structures, if requested)
  4. propose  one hypothesis seeded by that pair and its top bridge work
  5. judge    LLM rubric (5 judges) + embedding novelty
  6. check    litcheck (closest prior work) and combocheck (is the pairing already studied?)
  7. decide   REJECT when combocheck says `studied` or the lit-novelty is at/below the
              calibrated "known" cut-off; otherwise ACCEPT and feed it back into the
              knowledge base (`sync`). Every cycle is logged in `loop_runs`.

reward = composite / 5 for an accepted hypothesis, 0 for a rejected one. Rewards steer the
choice of later pairs (fields that yielded accepted hypotheses are tried more).
Nothing here claims a discovery: accepted means "not found in the literature by these checks".
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import Callable

from nous.config import DB_PATH, DEFAULT_MODEL

EXPLOIT_WEIGHT = 0.5      # how strongly past rewards of a pair's fields raise its priority


# ------------------------------------------------------------------ #
#  Pure decision logic                                                #
# ------------------------------------------------------------------ #

def choose_pair(links: list[dict], history: list[dict], tried_limit: int = 1,
                exploit_weight: float = EXPLOIT_WEIGHT) -> dict | None:
    """
    links:   field_links rows ({field_a, field_b, observed, lift, ...}); lowest lift = biggest void.
    history: past runs ({field_a, field_b, reward}).
    A pair already tried `tried_limit` times is skipped. Priority = void score in [0, 1]
    (1 = lowest lift among candidates) + exploit_weight * mean reward of past runs that
    touched either field. Pairs with no shared works (observed == 0) are ignored: there
    is nothing to dock.
    """
    cands = [l for l in links if l["observed"] > 0]
    if not cands:
        return None
    cands = sorted(cands, key=lambda l: (l["lift"], l["field_a"], l["field_b"]))
    n = len(cands)
    tried: dict[frozenset, int] = {}
    field_rewards: dict[str, list[float]] = {}
    for h in history:
        k = frozenset((h["field_a"], h["field_b"]))
        tried[k] = tried.get(k, 0) + 1
        for f in (h["field_a"], h["field_b"]):
            field_rewards.setdefault(f, []).append(h["reward"] or 0.0)

    best, best_score = None, None
    for rank, l in enumerate(cands):
        if tried.get(frozenset((l["field_a"], l["field_b"])), 0) >= tried_limit:
            continue
        void = 1.0 - rank / max(n - 1, 1)
        rs = field_rewards.get(l["field_a"], []) + field_rewards.get(l["field_b"], [])
        bonus = exploit_weight * (sum(rs) / len(rs)) if rs else 0.0
        score = void + bonus
        if best_score is None or score > best_score:
            best, best_score = l, score
    return best


def decide(combo_label: str | None, lit_novelty: float | None,
           lit_cutoff: float | None) -> tuple[bool, str]:
    """Accept or reject from the literature checks only. Returns (accepted, reason)."""
    if combo_label == "studied":
        return False, "combination already studied (>= 10 joint papers)"
    if lit_novelty is not None and lit_cutoff is not None and lit_novelty <= lit_cutoff:
        return False, f"lit-novelty {lit_novelty:.2f} <= known cut-off {lit_cutoff:.2f}"
    notes = []
    if combo_label in (None, "unrecognized_terms", "inconclusive"):
        notes.append(f"combination unverified ({combo_label or 'not checked'})")
    if lit_novelty is None or lit_cutoff is None:
        notes.append("lit-novelty not compared with a cut-off")
    return True, "; ".join(notes) if notes else "not found in the literature by these checks"


def reward_for(accepted: bool, composite: float | None) -> float:
    return round(composite / 5.0, 4) if accepted and composite is not None else 0.0


def make_query(link: dict, works: list[dict]) -> str:
    """Seed text for retrieval/generation: the two fields and the most-cited bridging work."""
    q = f"{link['name_a']} and {link['name_b']}"
    if works and works[0].get("title"):
        q += f": {works[0]['title']}"
    return q


# ------------------------------------------------------------------ #
#  Storage                                                            #
# ------------------------------------------------------------------ #

def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS loop_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_a TEXT, field_b TEXT, name_a TEXT, name_b TEXT,
            query TEXT, hyp_ref TEXT,
            status TEXT,            -- accepted / rejected / error
            reason TEXT,
            composite REAL, combo_label TEXT, lit_novelty REAL, reward REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.commit()


def load_history() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        rows = conn.execute("SELECT field_a, field_b, reward FROM loop_runs").fetchall()
    finally:
        conn.close()
    return [{"field_a": a, "field_b": b, "reward": r} for a, b, r in rows]


def _log_run(rec: dict):
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    cols = ("field_a", "field_b", "name_a", "name_b", "query", "hyp_ref", "status", "reason",
            "composite", "combo_label", "lit_novelty", "reward")
    conn.execute(f"INSERT INTO loop_runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 tuple(rec.get(c) for c in cols))
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ #
#  Steps (injectable so the loop can be tested without network/LLM)   #
# ------------------------------------------------------------------ #

@dataclass
class Steps:
    collect: Callable[[dict, int], list[dict]]      # (link, per_pair) -> works
    index: Callable[[], None]
    generate: Callable[[str], dict]                 # query -> {"id": hypothesis id, ...}
    judge: Callable[[str], float | None]            # ref -> rubric composite
    check: Callable[[str], tuple[str | None, float | None]]   # ref -> (combo label, lit-novelty)
    cutoff: Callable[[], float | None]
    sync: Callable[[int], None]                     # accepted hypothesis id -> knowledge base


def default_steps(model: str = DEFAULT_MODEL, abstract: bool = False) -> Steps:
    def collect(link, per_pair):
        from nous.collector.openalex import collect_pair_bridges
        from nous.collector.wikipedia import init_db
        init_db()
        return collect_pair_bridges(link, per_pair)

    def index():
        from nous.engine.embedder import embed_all
        embed_all()
        if abstract:
            from nous.engine.structure import build_structure_index
            build_structure_index(model=model)

    def generate(query):
        from nous.engine.hypothesis import generate_hypothesis
        return generate_hypothesis(query, model=model)

    def judge(ref):
        from nous.evaluation import store
        from nous.evaluation.runner import aggregate, evaluate
        evaluate(refs=[ref], model=model, verbose=False)
        for a in aggregate(store.load_scores()):
            if a["ref"] == ref:
                return a["composite"]
        return None

    def check(ref):
        from nous.evaluation import store
        from nous.evaluation.combination import check_combinations, combo_summary
        from nous.evaluation.literature import RATER, check_literature
        check_literature(refs=[ref], verbose=False)
        check_combinations(refs=[ref], model=model, verbose=False, with_references=False)
        c = combo_summary(ref)
        lit = next((r["score"] for r in store.load_scores("novelty_literature")
                    if r["target_ref"] == ref and r["rater"] == RATER), None)
        return (c["label"] if c else None), lit

    def cutoff():
        from nous.evaluation.literature import known_threshold
        return known_threshold()

    def sync(hid):
        from nous.memory.store import sync_hypotheses_to_chroma
        sync_hypotheses_to_chroma(only_ids={hid})

    return Steps(collect, index, generate, judge, check, cutoff, sync)


# ------------------------------------------------------------------ #
#  Cycle and driver                                                   #
# ------------------------------------------------------------------ #

def run_cycle(steps: Steps, per_pair: int = 10, verbose: bool = True) -> dict | None:
    from nous.collector.openalex import load_field_links
    link = choose_pair(load_field_links(), load_history())
    if link is None:
        if verbose:
            print("No untried field pair left (or no field links yet: run 'nous.py voids --refresh').")
        return None
    rec = {"field_a": link["field_a"], "field_b": link["field_b"],
           "name_a": link["name_a"], "name_b": link["name_b"]}
    if verbose:
        print(f"\n--- pair: {link['name_a']} × {link['name_b']} (lift={link['lift']:.3f}, "
              f"shared works={link['observed']:,})")
    try:
        works = steps.collect(link, per_pair)
        rec["query"] = make_query(link, works)
        if verbose:
            print(f"  collected {len(works)} bridge works; query: {rec['query'][:100]}")
        steps.index()
        result = steps.generate(rec["query"])
        hid = result.get("id")
        if hid is None:
            raise RuntimeError("hypothesis was not saved")
        rec["hyp_ref"] = f"hyp:{hid}"
        rec["composite"] = steps.judge(rec["hyp_ref"])
        rec["combo_label"], rec["lit_novelty"] = steps.check(rec["hyp_ref"])
        accepted, reason = decide(rec["combo_label"], rec["lit_novelty"], steps.cutoff())
        rec["status"], rec["reason"] = ("accepted" if accepted else "rejected"), reason
        rec["reward"] = reward_for(accepted, rec["composite"])
        if accepted:
            steps.sync(hid)
    except Exception as ex:
        rec.update(status="error", reason=f"{type(ex).__name__}: {ex}", reward=0.0)
    _log_run(rec)
    if verbose:
        comp = f"{rec['composite']:.2f}" if rec.get("composite") is not None else "n/a"
        print(f"  → {rec['status'].upper()}  {rec.get('hyp_ref', '')}  composite={comp}  "
              f"combination={rec.get('combo_label')}  reason: {rec['reason']}")
    return rec


def autoloop(cycles: int = 1, per_pair: int = 10, model: str = DEFAULT_MODEL,
             abstract: bool = False, verbose: bool = True) -> list[dict]:
    steps = default_steps(model=model, abstract=abstract)
    out = []
    for _ in range(cycles):
        rec = run_cycle(steps, per_pair=per_pair, verbose=verbose)
        if rec is None:
            break
        out.append(rec)
    if verbose and out:
        acc = sum(1 for r in out if r["status"] == "accepted")
        print(f"\nDone. {len(out)} cycles: {acc} accepted, "
              f"{sum(1 for r in out if r['status'] == 'rejected')} rejected, "
              f"{sum(1 for r in out if r['status'] == 'error')} errors.")
    return out


def print_plan(top: int = 10):
    """Which pairs would be explored next (database only, no API calls)."""
    from nous.collector.openalex import load_field_links
    links, hist = load_field_links(), load_history()
    if not links:
        print("No field links yet: run 'nous.py voids --refresh' first.")
        return
    remaining = list(links)
    print(f"\n=== Next pairs the loop would explore ({len(hist)} runs so far) ===")
    for i in range(top):
        l = choose_pair(remaining, hist)
        if not l:
            break
        print(f"  {i + 1:>2}. {l['name_a']} × {l['name_b']}  lift={l['lift']:.3f}  "
              f"shared={l['observed']:,}")
        hist = hist + [{"field_a": l["field_a"], "field_b": l["field_b"], "reward": 0.0}]


def print_report(limit: int = 30):
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    rows = conn.execute("""SELECT id, name_a, name_b, hyp_ref, status, composite, combo_label,
                                  lit_novelty, reward, reason, created_at
                           FROM loop_runs ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    if not rows:
        print("No loop runs yet — run 'nous.py autoloop'.")
        return
    print("\n=== Autonomous loop runs (latest first) ===")
    for r in rows:
        comp = f"{r[5]:.2f}" if r[5] is not None else " n/a"
        lit = f"{r[7]:.2f}" if r[7] is not None else "n/a"
        print(f"  #{r[0]:<3} {r[4]:<8} {r[3] or '-':<9} composite={comp} combo={r[6]} lit={lit} "
              f"reward={r[8]:.2f}  {r[1]} × {r[2]}")
        print(f"        {r[9]}")
