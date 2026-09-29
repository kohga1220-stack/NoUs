"""
Combination check — is the *pairing* of ideas new, even if each idea is old?

Nous hypotheses combine established ideas from different fields. Comparing a whole
hypothesis with single papers (litcheck) is biased against long multi-concept statements.
Here the hypothesis is decomposed into 2-4 standard academic concepts; OpenAlex then
counts, over title + abstract:

    n_A, n_B   works mentioning concept A / B
    joint      works mentioning A AND B
    lift       joint / (n_A * n_B / N)      (N = all works)

lift >= 1 : the pair co-occurs at least as often as chance   -> well trodden
lift  < 1 : the pair meets less often than chance            -> sparse
lift << 1 : (almost) nobody has combined them                -> unexplored

Reference pairs (well-known combinations and deliberately unrelated ones) are measured
in the same run so the numbers can be read against a scale.
Cost: about 10 count queries per hypothesis at $0.0001 each.
"""
from __future__ import annotations
import itertools
import re
import sqlite3
import time

import requests

from nous.config import DB_PATH, DEFAULT_MODEL
from nous.evaluation import store

RATER = "auto:openalex-combo"
ITEM = "combo_novelty"

MIN_KNOWN = 50            # a concept with fewer works is probably not a standard term
UNEXPLORED_LIFT = 0.1
WELL_TRODDEN_LIFT = 1.0

# (concept A, concept B) — well-known cross-field pairings, and pairs with no reason to meet
KNOWN_PAIRS = [
    ("Ising model", "opinion dynamics"),
    ("self-organized criticality", "neuronal avalanches"),
    ("evolutionary game theory", "cooperation"),
    ("network science", "epidemic spreading"),
]
UNRELATED_PAIRS = [
    ("gravitational waves", "dental caries"),
    ("sourdough fermentation", "quantum field theory"),
    ("Byzantine art", "lithium-ion battery"),
]


# ------------------------------------------------------------------ #
#  Pure helpers                                                       #
# ------------------------------------------------------------------ #

def sanitize_phrase(s: str) -> str:
    """Keep letters, digits, spaces and hyphens (commas, quotes, ':' and '|' would break a filter)."""
    s = re.sub(r"[^\w\s-]", " ", str(s), flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def build_query(phrases: list[str]) -> str:
    """'"phase transition" AND "opinion dynamics"'"""
    return " AND ".join(f'"{sanitize_phrase(p)}"' for p in phrases)


def pair_stats(n_a: int, n_b: int, joint: int, total: int) -> dict:
    expected = n_a * n_b / total if total else 0.0
    return {"expected": round(expected, 3), "lift": (joint / expected) if expected else 0.0}


def label_for(lift: float) -> str:
    if lift < UNEXPLORED_LIFT:
        return "unexplored"
    if lift < WELL_TRODDEN_LIFT:
        return "sparse"
    return "well_trodden"


def summarize(concept_counts: dict[str, int], pairs: list[dict]) -> dict:
    """
    One verdict per hypothesis.
      unrecognized_terms : some concept has < MIN_KNOWN works (not a standard term) -> no score
      otherwise the *most novel* pair (lowest lift) decides: the hypothesis is new in
      combination if at least one of its pairings is rare.
    combo_novelty = 1 - min(1, lift of that pair)   (0 = commonplace, 1 = never combined)
    """
    if not pairs or any(n < MIN_KNOWN for n in concept_counts.values()):
        return {"label": "unrecognized_terms", "combo_novelty": None, "pair": None}
    best = min(pairs, key=lambda p: p["lift"])
    return {"label": label_for(best["lift"]),
            "combo_novelty": round(1.0 - min(1.0, best["lift"]), 4),
            "pair": best}


def parse_concepts(data) -> list[str]:
    """LLM JSON -> 2..4 distinct sanitized concepts (pure)."""
    raw = data.get("concepts", []) if isinstance(data, dict) else []
    out: list[str] = []
    for c in raw:
        s = sanitize_phrase(c)
        if s and len(s.split()) <= 6 and s.lower() not in {x.lower() for x in out}:
            out.append(s)
    return out[:4]


# ------------------------------------------------------------------ #
#  OpenAlex counting                                                  #
# ------------------------------------------------------------------ #

class Counter:
    """Counts works matching a boolean phrase query, caching results."""

    def __init__(self):
        self.mode = "filter"          # title + abstract; falls back to `search` (adds full text)
        self.cache: dict[str, int] = {}
        self._total: int | None = None

    def _request(self, query: str) -> int:
        from nous.collector.openalex import get
        if self.mode == "filter":
            data = get("works", {"filter": f"title_and_abstract.search:{query}",
                                 "per_page": 1, "select": "id"})
        else:
            data = get("works", {"search": query, "per_page": 1, "select": "id"})
        return int(data.get("meta", {}).get("count", 0))

    def count(self, query: str) -> int:
        if query in self.cache:
            return self.cache[query]
        try:
            n = self._request(query)
        except requests.HTTPError as ex:
            if self.mode == "filter" and ex.response is not None and ex.response.status_code == 400:
                print("  [note] title_and_abstract.search not accepted; using `search` "
                      "(title + abstract + full text) instead.")
                self.mode = "search"
                self._total = None
                self.cache.clear()
                n = self._request(query)
            else:
                raise
        time.sleep(0.15)
        self.cache[query] = n
        return n

    @property
    def total(self) -> int:
        if self._total is None:
            from nous.collector.openalex import get
            self._total = int(get("works", {"per_page": 1, "select": "id"})["meta"]["count"])
        return self._total


def measure(concepts: list[str], counter: Counter) -> tuple[dict[str, int], list[dict]]:
    """Counts for each concept and lift for each pair of concepts."""
    counts = {c: counter.count(build_query([c])) for c in concepts}
    pairs = []
    for a, b in itertools.combinations(concepts, 2):
        joint = counter.count(build_query([a, b]))
        st = pair_stats(counts[a], counts[b], joint, counter.total)
        pairs.append({"a": a, "b": b, "n_a": counts[a], "n_b": counts[b], "joint": joint,
                      "expected": st["expected"], "lift": st["lift"]})
    return counts, pairs


# ------------------------------------------------------------------ #
#  LLM decomposition                                                  #
# ------------------------------------------------------------------ #

def decompose(text: str, model: str = DEFAULT_MODEL) -> list[str]:
    from nous.llm import generate_json
    prompt = f"""Extract the 2 to 4 core concepts combined in this research hypothesis.

Rules:
- Each concept is a STANDARD academic term of 1-4 words that researchers actually put in paper
  titles or abstracts (e.g. "phase transition", "opinion dynamics", "embodied cognition").
- Do NOT use labels the hypothesis itself invented. Replace them with the standard term for
  the underlying idea.
- Prefer concepts from different fields.

HYPOTHESIS:
{text[:1500]}

Respond ONLY with JSON: {{"concepts": ["...", "..."]}}"""
    return parse_concepts(generate_json(prompt, model=model))


# ------------------------------------------------------------------ #
#  Storage                                                            #
# ------------------------------------------------------------------ #

def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS combo_checks (
            target_ref TEXT, concept_a TEXT, concept_b TEXT,
            n_a INTEGER, n_b INTEGER, joint INTEGER, expected REAL, lift REAL,
            method TEXT, checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (target_ref, concept_a, concept_b)
        )
    """)
    conn.commit()


def _save_pairs(conn, ref: str, pairs: list[dict], method: str):
    conn.execute("DELETE FROM combo_checks WHERE target_ref = ?", (ref,))
    conn.executemany("""
        INSERT INTO combo_checks (target_ref, concept_a, concept_b, n_a, n_b, joint, expected, lift, method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [(ref, p["a"], p["b"], p["n_a"], p["n_b"], p["joint"], p["expected"], p["lift"], method)
          for p in pairs])
    conn.commit()


def combo_summary(ref: str) -> dict | None:
    """Verdict for one target, rebuilt from the stored pairs (used by `scores`)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        rows = conn.execute("SELECT concept_a, concept_b, n_a, n_b, joint, expected, lift "
                            "FROM combo_checks WHERE target_ref = ?", (ref,)).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    pairs = [dict(zip(("a", "b", "n_a", "n_b", "joint", "expected", "lift"), r)) for r in rows]
    counts = {}
    for p in pairs:
        counts[p["a"]], counts[p["b"]] = p["n_a"], p["n_b"]
    return summarize(counts, pairs)


# ------------------------------------------------------------------ #
#  Commands                                                           #
# ------------------------------------------------------------------ #

def _fmt_pair(p: dict) -> str:
    return (f"{p['a']} × {p['b']}: joint={p['joint']:,} "
            f"(expected {p['expected']:,.1f}, lift={p['lift']:.3f})")


def measure_references(counter: Counter, verbose: bool = True) -> dict[str, list[dict]]:
    out = {"known": [], "unrelated": []}
    for kind, pairs in (("known", KNOWN_PAIRS), ("unrelated", UNRELATED_PAIRS)):
        for a, b in pairs:
            _, ps = measure([a, b], counter)
            out[kind].append(ps[0])
    if verbose:
        print("\n=== Reference pairs (scale for reading the numbers) ===")
        print("  well-known combinations:")
        for p in out["known"]:
            print(f"    {_fmt_pair(p)}")
        print("  deliberately unrelated:")
        for p in out["unrelated"]:
            print(f"    {_fmt_pair(p)}")
    return out


def check_combinations(refs: list[str] | None = None, model: str = DEFAULT_MODEL,
                       verbose: bool = True) -> int:
    """Decompose each hypothesis, count concepts and pairs on OpenAlex, store and print."""
    counter = Counter()
    measure_references(counter, verbose)

    targets = store.load_targets()
    if refs:
        targets = [t for t in targets if t["ref"] in set(refs)]
    done = store.rated_by(RATER)
    todo = [t for t in targets if t["ref"] not in done]

    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    n = 0
    if verbose and todo:
        print("\n=== Hypotheses ===")
    for t in todo:
        try:
            concepts = decompose(t["text"], model=model)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {t['ref']}: decomposition failed ({ex})")
            continue
        if len(concepts) < 2:
            if verbose:
                print(f"  {t['ref']:<12} fewer than 2 concepts extracted — skipped")
            continue
        try:
            counts, pairs = measure(concepts, counter)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {t['ref']}: {ex}")
            continue
        _save_pairs(conn, t["ref"], pairs, counter.mode)
        res = summarize(counts, pairs)
        # always mark the target as checked; only informative verdicts get a score
        store.save_scores(t["ref"], RATER, {ITEM: res["combo_novelty"]} if res["combo_novelty"] is not None
                          else {"combo_checked": 1.0})
        n += 1
        if verbose:
            print(f"  {t['ref']:<12} {res['label']:<18} concepts: {', '.join(concepts)}")
            if res["pair"]:
                print(f"               most novel pair — {_fmt_pair(res['pair'])}")
            else:
                low = [c for c, k in counts.items() if k < MIN_KNOWN]
                if low:
                    print(f"               rarely-used terms (<{MIN_KNOWN} works): {', '.join(low)}")
    conn.close()
    if verbose:
        print(f"\nDone. {n} hypotheses checked ({counter.mode} search).")
    return n


def label_counts() -> dict[str, int]:
    """How many checked hypotheses fall in each verdict class."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        refs = [r[0] for r in conn.execute("SELECT DISTINCT target_ref FROM combo_checks")]
    finally:
        conn.close()
    out: dict[str, int] = {}
    for ref in refs:
        s = combo_summary(ref)
        if s:
            out[s["label"]] = out.get(s["label"], 0) + 1
    return out

