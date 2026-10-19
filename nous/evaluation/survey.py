"""
Which rejection rule would catch known work without throwing away new ideas?

Reads the stored checks (no API calls) and compares candidate rules on the stored hypotheses:

  R0  current       reject when the most novel informative pair has >= STUDIED_MIN joint works
  R1  any pair      reject when ANY pair has >= STUDIED_MIN joint works
  R2  close paper   reject when the closest published work has similarity >= T (litcheck)
  R3  R0 or R2
  R4  R1 or R2

and reports how many hypotheses each would reject and which ones only that rule rejects.
Ground truth: the famous ideas of `calibrate` (litcheck controls) are known work, so a good
similarity threshold T should catch most of them; the stored hypotheses have no labels, so the
rows only 'newly rejected' by a rule are listed for a human to read.

Usage:  python -m nous.evaluation.survey [T]        (default T = 0.70)
"""
from __future__ import annotations
import sqlite3
import sys

from nous.config import DB_PATH
from nous.evaluation import store
from nous.evaluation.combination import STUDIED_MIN, is_informative

DEFAULT_T = 0.70


def _rows(conn, sql, args=()):
    try:
        return conn.execute(sql, args).fetchall()
    except sqlite3.OperationalError:          # table missing in an older database
        return []


def load(db_path=None) -> dict:
    conn = sqlite3.connect(db_path or DB_PATH)
    pairs: dict[str, list[dict]] = {}
    for ref, a, b, n_a, n_b, joint in _rows(
            conn, "SELECT target_ref, concept_a, concept_b, n_a, n_b, joint FROM combo_checks"):
        pairs.setdefault(ref, []).append({"a": a, "b": b, "n_a": n_a, "n_b": n_b, "joint": joint})
    closest = {ref: (sim, title, year) for ref, sim, title, year in _rows(
        conn, "SELECT target_ref, sim, title, year FROM literature_checks WHERE rank = 1")}
    controls = [(label, sim) for label, sim in _rows(
        conn, "SELECT label, top_sim FROM literature_controls")]
    status = {ref: st for ref, st in _rows(
        conn, "SELECT hyp_ref, status FROM loop_runs WHERE hyp_ref IS NOT NULL ORDER BY id")}
    conn.close()
    try:
        text = {t["ref"]: t["text"] for t in store.load_targets(db_path or DB_PATH)}
    except sqlite3.OperationalError:
        text = {}
    return {"pairs": pairs, "closest": closest, "controls": controls, "status": status, "text": text}


def rule_r0(pairs: list[dict]) -> bool:
    informative = [p for p in pairs if is_informative(p)]
    return bool(informative) and min(p["joint"] for p in informative) >= STUDIED_MIN


def rule_r1(pairs: list[dict]) -> bool:
    return any(p["joint"] >= STUDIED_MIN for p in pairs)


def decisions(data: dict, threshold: float = DEFAULT_T) -> dict[str, dict[str, bool]]:
    """ref -> {rule: rejected?} for every hypothesis that has pair data."""
    out = {}
    for ref, pairs in data["pairs"].items():
        sim = data["closest"].get(ref, (None,))[0]
        r0, r1 = rule_r0(pairs), rule_r1(pairs)
        r2 = sim is not None and sim >= threshold
        out[ref] = {"R0": r0, "R1": r1, "R2": r2, "R3": r0 or r2, "R4": r1 or r2}
    return out


def control_recall(controls: list[tuple[str, float]], thresholds=(0.60, 0.65, 0.70, 0.73, 0.75, 0.80)):
    sims = [s for _, s in controls if s is not None]
    return [(t, sum(s >= t for s in sims), len(sims)) for t in thresholds]


def report(db_path=None, threshold: float = DEFAULT_T) -> None:
    data = load(db_path)
    dec = decisions(data, threshold)
    n = len(dec)
    print(f"stored hypotheses with pair data: {n}   with litcheck: "
          f"{sum(1 for r in dec if r in data['closest'])}")
    print(f"\nfamous ideas (known work) caught by 'closest paper similarity >= T'  [{len(data['controls'])} ideas]")
    for t, k, m in control_recall(data["controls"]):
        print(f"  T={t:.2f}: {k}/{m}")
    print(f"\nrules at T={threshold:.2f}: hypotheses that would be REJECTED")
    for rule, label in (("R0", "current: rarest pair >= 10"), ("R1", "any pair >= 10"),
                        ("R2", "closest paper sim >= T"), ("R3", "R0 or R2"), ("R4", "R1 or R2")):
        print(f"  {rule}  {label:<28} {sum(d[rule] for d in dec.values()):3d} / {n}")
    for rule, base in (("R1", "R0"), ("R2", "R0")):
        rows = [r for r, d in dec.items() if d[rule] and not d[base]]
        print(f"\nrejected by {rule} but not by {base}: {len(rows)}")
        for r in sorted(rows):
            sim = data["closest"].get(r, (None, "", ""))
            mx = max(p["joint"] for p in data["pairs"][r])
            snippet = (data["text"].get(r, "") or "").replace("\n", " ")[:70]
            sim_s = f"sim={sim[0]:.2f}" if sim[0] is not None else "sim=n/a"
            print(f"  {r:<11} maxpair={mx:<5} {sim_s}  {data['status'].get(r, '-'):<8} {snippet}")


if __name__ == "__main__":
    report(threshold=float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_T)
