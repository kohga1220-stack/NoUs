"""
Evaluation loop: score hypotheses with multiple LLM judges (+ optional humans),
rank them, and check how much the judges agree (ICC).
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np

from nous.config import DEFAULT_MODEL
from nous.evaluation import store
from nous.evaluation.reliability import icc, interpret
from nous.evaluation.rubric import ITEMS, PROMPT_VERSION, RUBRIC, clamp_score, llm_judge

AUTO_RATER = "auto:embedding"


def _judges(model: str) -> list[tuple[str, str]]:
    """(rater_id, persona) — one judge per Scepter lens."""
    from nous.scepter.registry import ALL_SCEPTERS
    return [(f"llm:{model}:{s.name}", f"{s.domain} — {s.lens}") for s in ALL_SCEPTERS]


def evaluate(refs: list[str] | None = None, model: str = DEFAULT_MODEL,
             with_novelty: bool = True, verbose: bool = True) -> int:
    """Score every (or the given) target with all judges; skips already-rated pairs."""
    targets = store.load_targets()
    if refs:
        targets = [t for t in targets if t["ref"] in set(refs)]
    if not targets:
        print("No hypotheses to evaluate. Run 'nous.py hyp' or 'nous.py debate' first.")
        return 0

    n_saved = 0
    for rater, persona in _judges(model):
        done = store.rated_by(rater)
        for t in targets:
            if t["ref"] in done:
                continue
            try:
                scores, rationales = llm_judge(t["text"], t["query"], persona, model)
            except Exception as ex:
                if verbose:
                    print(f"  [WARN] {rater} on {t['ref']}: {ex}")
                continue
            if scores:
                store.save_scores(t["ref"], rater, scores, rationales, PROMPT_VERSION)
                n_saved += 1
                if verbose:
                    s = " ".join(f"{k[:4]}={v:g}" for k, v in scores.items())
                    print(f"  {t['ref']:<12} {rater.split(':')[-1]:<10} {s}")

    if with_novelty:
        done = store.rated_by(AUTO_RATER)
        try:
            from nous.evaluation.novelty import embedding_novelty
            for t in targets:
                if t["ref"] in done:
                    continue
                nov = embedding_novelty(t["text"])
                if nov is not None:
                    store.save_scores(t["ref"], AUTO_RATER, {"novelty_embedding": nov})
        except Exception as ex:
            if verbose:
                print(f"  [WARN] embedding novelty skipped: {ex}")

    return n_saved


def human_rate(rater_name: str, limit: int = 10):
    """Interactive rating in the terminal with the same rubric the LLM judges use."""
    rater = f"human:{rater_name}"
    done = store.rated_by(rater)
    todo = [t for t in store.load_targets() if t["ref"] not in done][:limit]
    if not todo:
        print("Nothing left to rate.")
        return

    print("\nRUBRIC (1-5):")
    for item, (q, a1, a3, a5) in RUBRIC.items():
        print(f"  {item}: {q}\n     1={a1} / 3={a3} / 5={a5}")

    for t in todo:
        print(f"\n{'-'*60}\n[{t['ref']}] query: {t['query']}\n\n{t['text']}\n")
        scores = {}
        for item in ITEMS:
            while True:
                raw = input(f"  {item} (1-5, s=skip target, q=quit): ").strip().lower()
                if raw == "q":
                    if scores:
                        store.save_scores(t["ref"], rater, scores, prompt_version=PROMPT_VERSION)
                    return
                if raw == "s":
                    scores = {}
                    break
                v = clamp_score(raw)
                if v is not None and raw.isdigit():
                    scores[item] = v
                    break
            if raw == "s":
                break
        if scores:
            store.save_scores(t["ref"], rater, scores, prompt_version=PROMPT_VERSION)


# ------------------------------------------------------------------ #
#  Aggregation (pure)                                                 #
# ------------------------------------------------------------------ #

def aggregate(rows: list[dict]) -> list[dict]:
    """
    Mean score per item per target across raters (rubric items only), plus
    `composite` = mean of the item means and the automatic novelty metric.
    Sorted by composite, best first.
    """
    by_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    auto: dict[str, float] = {}
    raters: dict[str, set] = defaultdict(set)
    for r in rows:
        if r["score"] is None:
            continue
        if r["item"] == "novelty_embedding":
            auto[r["target_ref"]] = r["score"]
            continue
        if r["item"] in ITEMS:
            by_target[r["target_ref"]][r["item"]].append(r["score"])
            raters[r["target_ref"]].add(r["rater"])

    out = []
    for ref, items in by_target.items():
        means = {i: float(np.mean(v)) for i, v in items.items()}
        out.append({
            "ref": ref,
            "item_means": means,
            "composite": float(np.mean(list(means.values()))),
            "n_raters": len(raters[ref]),
            "novelty_embedding": auto.get(ref),
        })
    out.sort(key=lambda x: x["composite"], reverse=True)
    return out


def rating_matrix(rows: list[dict], item: str,
                  rater_prefix: str | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """
    targets × raters matrix for one item, keeping only raters (matching the prefix)
    and targets with complete data.
    """
    cells: dict[tuple[str, str], float] = {}
    for r in rows:
        if r["item"] != item or r["score"] is None:
            continue
        if rater_prefix and not r["rater"].startswith(rater_prefix):
            continue
        cells[(r["target_ref"], r["rater"])] = r["score"]

    raters  = sorted({rt for _, rt in cells})
    targets = sorted({t for t, _ in cells})
    complete = [t for t in targets if all((t, rt) in cells for rt in raters)]
    m = np.array([[cells[(t, rt)] for rt in raters] for t in complete], dtype=float)
    return m, complete, raters


def reliability_report(rows: list[dict], rater_prefix: str | None = None) -> list[dict]:
    report = []
    for item in ITEMS:
        m, targets, raters = rating_matrix(rows, item, rater_prefix)
        entry = {"item": item, "n_targets": len(targets), "n_raters": len(raters)}
        if len(targets) >= 2 and len(raters) >= 2:
            res = icc(m)
            entry.update({k: res[k] for k in ("ICC2", "ICC2k", "ICC3")})
            entry["verdict"] = interpret(res["ICC2k"])
        report.append(entry)
    return report


# ------------------------------------------------------------------ #
#  Printing                                                           #
# ------------------------------------------------------------------ #

def print_scores(top: int = 15):
    agg = aggregate(store.load_scores())
    if not agg:
        print("No evaluations yet. Run 'nous.py evaluate' first.")
        return
    targets = {t["ref"]: t for t in store.load_targets()}
    print("\n=== Hypothesis ranking (rubric composite, 1-5) ===\n")
    for a in agg[:top]:
        t = targets.get(a["ref"], {})
        nov = a["novelty_embedding"]
        nov_s = f"  emb-novelty={nov:.2f}" if nov is not None else ""
        print(f"  {a['composite']:.2f}  {a['ref']:<12} (raters={a['n_raters']}){nov_s}")
        print("        " + " ".join(f"{k}={v:.1f}" for k, v in a["item_means"].items()))
        if t:
            print(f"        [{t['query']}] {t['text'][:110]}...")


def print_reliability(rater_prefix: str | None = None):
    rows = store.load_scores()
    print("\n=== Inter-rater reliability (ICC, Shrout & Fleiss 1979) ===")
    print("ICC2 = single judge, absolute agreement; ICC2k = mean of all judges.")
    print("Interpretation follows Koo & Li (2016) on ICC2k.\n")
    for e in reliability_report(rows, rater_prefix):
        if "ICC2" not in e:
            print(f"  {e['item']:<17} insufficient data "
                  f"(targets={e['n_targets']}, raters={e['n_raters']})")
            continue
        print(f"  {e['item']:<17} ICC2={e['ICC2']:.2f}  ICC2k={e['ICC2k']:.2f}  "
              f"ICC3={e['ICC3']:.2f}  → {e['verdict']}  "
              f"(targets={e['n_targets']}, raters={e['n_raters']})")
