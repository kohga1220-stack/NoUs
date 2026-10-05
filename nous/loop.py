"""
Autonomous loop (phase 4): explore literature voids, propose a hypothesis, and reject it
when the literature already knows it.

One cycle
  1. choose   a pair of OpenAlex fields that rarely meet directly (low lift) although their
              links to all other fields look alike (similar co-occurrence profile): a gap
              between neighbours, not just two unrelated fields. Pairs sharing fewer than
              MIN_SHARED works are skipped; pairs whose fields produced well-rated
              hypotheses before are preferred.
  2. collect  the most-cited works tagged with BOTH fields (humanity's existing dockings)
  3. index    embed the new works (and abstract their structures, if requested)
  4. propose  one hypothesis built from those bridging works (not from the whole knowledge base)
  5. judge    LLM rubric (5 judges) + embedding novelty
  6. check    litcheck (closest prior work) and combocheck on the concepts of the hypothesis's
              own claim (is that pairing already studied?)
  7. decide   REJECT when combocheck says `studied` or the lit-novelty is at/below the
              calibrated "known" cut-off; otherwise ACCEPT and feed it back into the
              knowledge base (`sync`). Every cycle is logged in `loop_runs`.

reward = (rubric composite WITHOUT the novelty item) / 5 for an accepted hypothesis, 0 for a
rejected one. The LLM novelty rating is left out because, against careful human ratings (n=10),
it was 1.3 points more lenient and uncorrelated; novelty is judged by the literature checks. Rewards steer the
choice of later pairs (fields that yielded accepted hypotheses are tried more).
Nothing here claims a discovery: accepted means "not found in the literature by these checks".
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import Callable

from nous.config import DB_PATH, DEFAULT_MODEL

EXPLOIT_WEIGHT = 0.5      # how strongly past rewards of a pair's fields raise its priority
MIN_SHARED = 1000         # heuristic: pairs sharing fewer works are mostly noise / mis-tagged


# ------------------------------------------------------------------ #
#  Pure decision logic                                                #
# ------------------------------------------------------------------ #

def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sxx * syy) ** 0.5


def score_links(links: list[dict]) -> dict[frozenset, dict]:
    """
    Per field pair: `void` in [0, 1] (1 = lowest direct lift of all pairs with shared works) and
    `neighbour` in [-1, 1] = Pearson correlation of the two fields' log-lift profiles over all
    OTHER fields (do they connect to the rest of science in the same way?).
    A high neighbour score with a high void score = a gap between similar fields.
    """
    import math
    lift: dict[frozenset, float] = {}
    for l in links:
        lift[frozenset((l["field_a"], l["field_b"]))] = max(l["lift"], 1e-3)
    fields = sorted({f for k in lift for f in k})
    cands = sorted((l for l in links if l["observed"] > 0), key=lambda l: l["lift"])
    n = len(cands)
    out: dict[frozenset, dict] = {}
    for rank, l in enumerate(cands):
        a, b = l["field_a"], l["field_b"]
        xs, ys = [], []
        for g in fields:
            if g in (a, b):
                continue
            la, lb = lift.get(frozenset((a, g))), lift.get(frozenset((b, g)))
            if la is not None and lb is not None:
                xs.append(math.log(la))
                ys.append(math.log(lb))
        out[frozenset((a, b))] = {"void": 1.0 - rank / max(n - 1, 1),
                                  "neighbour": _pearson(xs, ys)}
    return out


def choose_pair(links: list[dict], history: list[dict], tried_limit: int = 1,
                exploit_weight: float = EXPLOIT_WEIGHT, min_shared: int = MIN_SHARED) -> dict | None:
    """
    Pick the next field pair. links: field_links rows ({field_a, field_b, observed, lift, ...}).
    history: past runs ({field_a, field_b, reward}).
    Skipped: pairs tried `tried_limit` times, pairs sharing fewer than `min_shared` works.
    priority = void * max(neighbour, 0) + exploit_weight * mean reward of past runs that
    touched either field. The chosen link is returned with its components under "components".
    """
    scores = score_links(links)
    tried: dict[frozenset, int] = {}
    field_rewards: dict[str, list[float]] = {}
    for h in history:
        k = frozenset((h["field_a"], h["field_b"]))
        tried[k] = tried.get(k, 0) + 1
        for f in (h["field_a"], h["field_b"]):
            field_rewards.setdefault(f, []).append(h["reward"] or 0.0)

    best, best_p = None, None
    for l in sorted(links, key=lambda l: (l["lift"], l["field_a"], l["field_b"])):
        k = frozenset((l["field_a"], l["field_b"]))
        if l["observed"] < max(min_shared, 1) or k not in scores or tried.get(k, 0) >= tried_limit:
            continue
        comp = scores[k]
        rs = field_rewards.get(l["field_a"], []) + field_rewards.get(l["field_b"], [])
        bonus = exploit_weight * (sum(rs) / len(rs)) if rs else 0.0
        priority = comp["void"] * max(comp["neighbour"], 0.0) + bonus
        if best_p is None or priority > best_p:
            best, best_p = {**l, "components": {**comp, "bonus": bonus, "priority": priority}}, priority
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
    generate: Callable[[str, list[dict]], dict]     # (query, bridge works) -> {"id": hypothesis id, ...}
    judge: Callable[[str], float | None]            # ref -> rubric composite
    check: Callable[[str], tuple[str | None, float | None]]   # ref -> (combo label, lit-novelty)
    cutoff: Callable[[], float | None]
    sync: Callable[[int], None]                     # accepted hypothesis id -> knowledge base
    describe: Callable[[str], str] | None = None    # ref -> text shown after the cycle


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

    def generate(query, works):
        from nous.engine.hypothesis import generate_hypothesis
        context = [{"domain": w["field"], "title": w["title"], "summary": w["summary"]}
                   for w in works[:8]]
        return generate_hypothesis(query, model=model, context=context or None)

    def judge(ref):
        from nous.evaluation import store
        from nous.evaluation.runner import aggregate, evaluate
        evaluate(refs=[ref], model=model, verbose=False)
        for a in aggregate(store.load_scores()):
            if a["ref"] == ref:
                return a["composite_excl_novelty"]
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

    def describe(ref):
        from nous.evaluation import store
        from nous.evaluation.combination import _ensure_table as _ct, _load_pairs, load_titles
        text = next((t["text"] for t in store.load_targets() if t["ref"] == ref), "")
        conn = sqlite3.connect(DB_PATH)
        try:
            _ct(conn)
            pairs = _load_pairs(conn, ref)
        finally:
            conn.close()
        lines = [f"  hypothesis: {text[:400]}"]
        for p in pairs:
            lines.append(f"  concepts: {p['a']} × {p['b']}  joint={p['joint']} loose={p['loose']} "
                         f"adjacency={p['adjacency']}")
            for w in load_titles(p["a"], p["b"])[:3]:
                lines.append(f"    - {w['title'][:90]} ({w['year']})")
        return "\n".join(lines)

    return Steps(collect, index, generate, judge, check, cutoff, sync, describe)


# ------------------------------------------------------------------ #
#  Cycle and driver                                                   #
# ------------------------------------------------------------------ #

def run_cycle(steps: Steps, per_pair: int = 10, verbose: bool = True,
              min_shared: int = MIN_SHARED) -> dict | None:
    from nous.collector.openalex import load_field_links
    link = choose_pair(load_field_links(), load_history(), min_shared=min_shared)
    if link is None:
        if verbose:
            print("No untried field pair left (or no field links yet: run 'nous.py voids --refresh').")
        return None
    rec = {"field_a": link["field_a"], "field_b": link["field_b"],
           "name_a": link["name_a"], "name_b": link["name_b"]}
    if verbose:
        c = link.get("components", {})
        print(f"\n--- pair: {link['name_a']} × {link['name_b']} (lift={link['lift']:.3f}, "
              f"shared works={link['observed']:,}, neighbour={c.get('neighbour', 0):.2f})")
    try:
        works = steps.collect(link, per_pair)
        rec["query"] = make_query(link, works)
        if verbose:
            print(f"  collected {len(works)} bridge works; query: {rec['query'][:100]}")
        steps.index()
        result = steps.generate(rec["query"], works)
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
        print(f"  → {rec['status'].upper()}  {rec.get('hyp_ref', '')}  composite(excl. novelty)={comp}  "
              f"combination={rec.get('combo_label')}  reason: {rec['reason']}")
        if rec.get("hyp_ref") and steps.describe:
            try:
                print(steps.describe(rec["hyp_ref"]))
            except Exception as ex:
                print(f"  (could not describe: {ex})")
    return rec


def autoloop(cycles: int = 1, per_pair: int = 10, model: str = DEFAULT_MODEL,
             abstract: bool = False, verbose: bool = True, min_shared: int = MIN_SHARED) -> list[dict]:
    steps = default_steps(model=model, abstract=abstract)
    out = []
    for _ in range(cycles):
        rec = run_cycle(steps, per_pair=per_pair, verbose=verbose, min_shared=min_shared)
        if rec is None:
            break
        out.append(rec)
    if verbose and out:
        acc = sum(1 for r in out if r["status"] == "accepted")
        print(f"\nDone. {len(out)} cycles: {acc} accepted, "
              f"{sum(1 for r in out if r['status'] == 'rejected')} rejected, "
              f"{sum(1 for r in out if r['status'] == 'error')} errors.")
    return out


def print_plan(top: int = 10, min_shared: int = MIN_SHARED):
    """Which pairs would be explored next, with their score components (database only)."""
    from nous.collector.openalex import load_field_links
    links, hist = load_field_links(), load_history()
    if not links:
        print("No field links yet: run 'nous.py voids --refresh' first.")
        return
    print(f"\n=== Next pairs the loop would explore ({len(hist)} runs so far, "
          f"min shared works {min_shared:,}) ===")
    print("  void = 1 for the lowest direct lift; neighbour = similarity of the two fields' links "
          "to all other fields (-1..1)")
    for i in range(top):
        l = choose_pair(links, hist, min_shared=min_shared)
        if not l:
            break
        c = l["components"]
        print(f"  {i + 1:>2}. {l['name_a']} × {l['name_b']}  lift={l['lift']:.3f}  "
              f"shared={l['observed']:,}  void={c['void']:.2f}  neighbour={c['neighbour']:.2f}  "
              f"priority={c['priority']:.2f}")
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
