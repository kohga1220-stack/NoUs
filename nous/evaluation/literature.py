"""
Literature check — is a hypothesis already in the published record?

For each hypothesis, OpenAlex semantic search (GTE-Large embeddings over ~300M
titles + abstracts) returns the closest existing works. Nous re-embeds those works
with its own model and records:

  novelty_literature = 1 - max cosine(hypothesis, closest works)

together with the closest works themselves, so a human can see *what* prior work
the hypothesis resembles. Cost: one semantic search per hypothesis ($0.001 each).
"""
from __future__ import annotations
import sqlite3
import time

from nous.config import DB_PATH
from nous.evaluation import store

RATER = "auto:openalex"
ITEM  = "novelty_literature"


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS literature_checks (
            target_ref  TEXT,
            rank        INTEGER,
            work_id     TEXT,
            title       TEXT,
            year        INTEGER,
            cited_by    INTEGER,
            sim         REAL,
            checked_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (target_ref, rank)
        )
    """)
    conn.commit()


def rank_prior_work(hyp_emb: list[float], works: list[dict],
                    work_embs: list[list[float]]) -> list[dict]:
    """Attach cosine similarity to each work and sort, most similar first (pure)."""
    from nous.engine.connector import cosine_sim
    ranked = [{**w, "sim": round(cosine_sim(hyp_emb, e), 4)} for w, e in zip(works, work_embs)]
    ranked.sort(key=lambda w: w["sim"], reverse=True)
    return ranked


def search_prior_work(text: str, n: int = 10) -> list[dict]:
    from nous.collector.openalex import get, reconstruct_abstract, short_id
    data = get("works", {
        "search.semantic": text[:2000],
        "per_page": n,
        "select": "id,display_name,publication_year,abstract_inverted_index,cited_by_count",
    })
    works = []
    for w in data.get("results", []):
        title = w.get("display_name") or ""
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        works.append({
            "work_id":  short_id(w["id"]),
            "title":    title,
            "year":     w.get("publication_year"),
            "cited_by": w.get("cited_by_count", 0),
            "text":     f"{title}. {abstract}".strip(),
        })
    return works


def check_literature(refs: list[str] | None = None, n: int = 10,
                     verbose: bool = True) -> int:
    """Run the literature check for every (or the given) hypothesis not yet checked."""
    from nous.engine.embedder import get_model

    targets = store.load_targets()
    if refs:
        targets = [t for t in targets if t["ref"] in set(refs)]
    done = store.rated_by(RATER)
    todo = [t for t in targets if t["ref"] not in done]
    if not todo:
        if verbose:
            print("All hypotheses already checked against the literature.")
        return 0

    model = get_model()
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    checked = 0
    for t in todo:
        try:
            works = search_prior_work(t["text"], n=n)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {t['ref']}: {ex}")
            continue
        time.sleep(1.1)   # semantic search is limited to 1 request/second
        if not works:
            continue

        hyp_emb = model.encode(t["text"]).tolist()
        work_embs = model.encode([w["text"] for w in works]).tolist()
        ranked = rank_prior_work(hyp_emb, works, work_embs)
        novelty = round(1.0 - ranked[0]["sim"], 4)

        conn.execute("DELETE FROM literature_checks WHERE target_ref = ?", (t["ref"],))
        conn.executemany("""
            INSERT INTO literature_checks (target_ref, rank, work_id, title, year, cited_by, sim)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [(t["ref"], i + 1, w["work_id"], w["title"], w["year"], w["cited_by"], w["sim"])
              for i, w in enumerate(ranked)])
        conn.commit()
        store.save_scores(t["ref"], RATER, {ITEM: novelty})
        checked += 1

        if verbose:
            top = ranked[0]
            print(f"  {t['ref']:<12} novelty={novelty:.2f}  closest: "
                  f"{top['title'][:70]} ({top['year']}, sim={top['sim']:.2f})")
    conn.close()
    if verbose:
        print(f"\nDone. {checked} hypotheses checked.")
    return checked


def closest_prior_work(ref: str, k: int = 1) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    rows = conn.execute("""
        SELECT title, year, sim, work_id FROM literature_checks
        WHERE target_ref = ? ORDER BY rank LIMIT ?
    """, (ref, k)).fetchall()
    conn.close()
    return [{"title": r[0], "year": r[1], "sim": r[2], "work_id": r[3]} for r in rows]


# ------------------------------------------------------------------ #
#  Calibration                                                        #
# ------------------------------------------------------------------ #
#
# lit-novelty = 1 - max cosine is not an absolute quantity: its scale depends on the
# embedding model and on how long the texts are. To learn what value means "this idea
# is already in the literature", we measure the same quantity on well-known, published
# ideas written in the same style as Nous hypotheses (positive controls).
# The label is the classic source; the sentence is a paraphrase written for this test.

KNOWN_IDEAS: list[tuple[str, str]] = [
    ("Bak-Tang-Wiesenfeld 1987",
     "Large systems of many interacting components can organize themselves into a critical "
     "state without external tuning, where a small perturbation triggers avalanches of all "
     "sizes whose distribution follows a power law."),
    ("Watts-Strogatz 1998",
     "Networks can combine high local clustering with short average path lengths when a few "
     "random long-range links are added to a regular lattice, producing small-world structure."),
    ("Barabasi-Albert 1999",
     "Scale-free networks with power-law degree distributions emerge from growth combined "
     "with preferential attachment, in which new nodes link preferentially to well-connected nodes."),
    ("West-Brown-Enquist 1997",
     "Metabolic rate scales with body mass to the three-quarter power because resources are "
     "distributed through space-filling, fractal-like hierarchical branching networks."),
    ("Kahneman-Tversky 1979",
     "People evaluate outcomes as gains and losses relative to a reference point rather than "
     "final wealth, and losses loom larger than equivalent gains, which shapes decisions under risk."),
    ("Tononi 2004",
     "Consciousness corresponds to integrated information: a system is conscious to the extent "
     "that it generates information as a whole that exceeds the information of its parts."),
    ("Friston 2010",
     "Biological agents maintain their organization by minimizing variational free energy, "
     "which amounts to minimizing surprise about sensory states through perception and action."),
    ("Dunbar 1992",
     "Primate neocortex size correlates with social group size, suggesting that the computational "
     "demands of tracking social relationships drove the evolution of large brains."),
    ("Williams-Bargh 2008",
     "Physical experiences of warmth, such as holding a warm cup, prime judgments of interpersonal "
     "warmth, linking bodily sensation to social cognition through embodied metaphor."),
    ("Castellano-Fortunato-Loreto 2009",
     "Collective opinion formation can be modeled with statistical physics, where agents adopt "
     "the states of their neighbors and the population undergoes transitions from disorder to consensus."),
]


def quantiles(values: list[float]) -> dict[str, float]:
    """min / q25 / median / q75 / max of a list (pure)."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "q25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "q75": float(np.percentile(a, 75)),
            "max": float(a.max())}


def summarize_calibration(known: list[float], hypotheses: list[float]) -> dict:
    """
    Suggest a 'probably already known' cut-off and apply it to the hypotheses.

    threshold = 75th percentile of the known ideas' lit-novelty. About three quarters of
    the classic ideas score at or below it, so a hypothesis at or below it looks at least
    as 'already published' as most classics. It is a heuristic for triage, not a proof.
    """
    k = quantiles(known)
    threshold = k["q75"]
    flagged = [h for h in hypotheses if h <= threshold]
    return {
        "known": k,
        "hypotheses": quantiles(hypotheses) if hypotheses else None,
        "threshold": threshold,
        "n_hypotheses": len(hypotheses),
        "n_flagged": len(flagged),
        "separated": bool(hypotheses) and quantiles(hypotheses)["median"] > k["median"],
    }


def _ensure_controls_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS literature_controls (
            label      TEXT PRIMARY KEY,
            novelty    REAL,
            top_title  TEXT,
            top_year   INTEGER,
            top_sim    REAL,
            checked_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def calibrate(n: int = 10, verbose: bool = True) -> dict | None:
    """
    Measure lit-novelty for the known ideas (10 semantic searches, about $0.01), then compare
    with the Nous hypotheses already checked by `litcheck`.
    """
    from nous.engine.embedder import get_model

    model = get_model()
    conn = sqlite3.connect(DB_PATH)
    _ensure_controls_table(conn)

    known: list[float] = []
    for label, text in KNOWN_IDEAS:
        try:
            works = search_prior_work(text, n=n)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {label}: {ex}")
            continue
        time.sleep(1.1)
        if not works:
            continue
        hyp_emb = model.encode(text).tolist()
        ranked = rank_prior_work(hyp_emb, works, model.encode([w["text"] for w in works]).tolist())
        top = ranked[0]
        novelty = round(1.0 - top["sim"], 4)
        known.append(novelty)
        conn.execute("INSERT OR REPLACE INTO literature_controls "
                     "(label, novelty, top_title, top_year, top_sim) VALUES (?,?,?,?,?)",
                     (label, novelty, top["title"], top["year"], top["sim"]))
        conn.commit()
        if verbose:
            print(f"  {label:<34} lit-novelty={novelty:.2f}  closest: "
                  f"{top['title'][:55]} ({top['year']}, sim={top['sim']:.2f})")
    conn.close()

    if len(known) < 3:
        print("Too few control results to calibrate.")
        return None

    hyp = [r["score"] for r in store.load_scores(ITEM) if r["rater"] == RATER
           and r["target_ref"].split(":")[0] in ("hyp", "scepter", "verdict")]
    result = summarize_calibration(known, hyp)

    if verbose:
        k, h = result["known"], result["hypotheses"]
        print("\n=== lit-novelty calibration ===")
        print(f"  known ideas   (n={len(known)}):  min {k['min']:.2f}  q25 {k['q25']:.2f}  "
              f"median {k['median']:.2f}  q75 {k['q75']:.2f}  max {k['max']:.2f}")
        if h:
            print(f"  Nous hypotheses (n={len(hyp)}): min {h['min']:.2f}  q25 {h['q25']:.2f}  "
                  f"median {h['median']:.2f}  q75 {h['q75']:.2f}  max {h['max']:.2f}")
        print(f"\n  suggested 'probably already known' cut-off: lit-novelty <= {result['threshold']:.2f}")
        if hyp:
            print(f"  hypotheses at or below it: {result['n_flagged']} / {result['n_hypotheses']}")
            print("  Nous hypotheses are more novel than the classics (median)."
                  if result["separated"] else
                  "  Nous hypotheses are NOT more novel than well-known classics (median).")
        print("  Note: 10 controls only; the cut-off is a triage heuristic, not proof of prior work.")
    return result


def known_threshold() -> float | None:
    """Cut-off saved by `calibrate` (75th percentile of the known ideas' lit-novelty), if any."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_controls_table(conn)
        vals = [r[0] for r in conn.execute("SELECT novelty FROM literature_controls")]
    finally:
        conn.close()
    return quantiles(vals)["q75"] if len(vals) >= 3 else None
