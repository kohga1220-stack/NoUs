"""
Combination check — is the *pairing* of ideas rare in the literature, and is it worth trying?

A Nous hypothesis combines established ideas from different fields. It is decomposed into
2-4 standard academic concepts, and every pair (A, B) is read on two independent axes:

1. HAS ANYONE COMBINED THEM?   (counts of works whose title/abstract mention ...)
       n_A, n_B   A / B            joint   A AND B
   status      joint >= 10          studied      an established literature exists
               1 <= joint < 10      few_papers   only a handful of papers
               joint == 0           none         nobody combined them
   A count of zero only means something when both concepts are common enough
   (min(n_A, n_B) >= MIN_COMMON); otherwise the phrases are too narrow to tell
   (`inconclusive`). The independence expectation n_A*n_B/N is shown, but it is not used
   for the verdict: for niche concepts it is ~0 by construction, so lift explodes.

2. ARE THE TWO LITERATURES NEIGHBOURS?   (adjacency)
   The works about A and the works about B are each spread over OpenAlex subfields.
   adjacency = Bhattacharyya coefficient of the two distributions (0 = disjoint fields,
   1 = identical). High adjacency with few or no joint papers is a real gap: related
   literatures that nobody has connected. Low adjacency with no joint papers is usually
   just an unrelated pair (nobody writes about it because there is no reason to).
   The cut-off between the two is read off the reference pairs of the same run.

Reference pairs (famous combinations and deliberately unrelated ones) are measured every
time so the numbers can be read against a scale.
Cost: about 10 count queries + 1 group_by per concept per hypothesis at $0.0001 each.
"""
from __future__ import annotations
import itertools
import math
import re
import sqlite3
import time

import requests

from nous.config import DB_PATH, DEFAULT_MODEL
from nous.evaluation import store

RATER = "auto:openalex-combo"
ITEM = "combo_novelty"

MIN_KNOWN = 50            # a concept with fewer works is probably not a standard term
STUDIED_MIN = 10          # joint >= this: an established literature
MIN_COMMON = 500          # a zero is informative only if both concepts have >= this many works

GROUP_ATTRS = ["primary_topic.subfield.id", "topics.subfield.id", "primary_topic.field.id"]

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

LABEL_ORDER = ["unexplored", "few_papers", "studied", "inconclusive", "unrecognized_terms"]


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
    """Independence expectation and lift (informational only)."""
    expected = n_a * n_b / total if total else 0.0
    return {"expected": round(expected, 3), "lift": (joint / expected) if expected else 0.0}


def pair_status(pair: dict) -> str:
    """studied / few_papers / none, from the absolute number of joint works."""
    j = pair["joint"]
    if j >= STUDIED_MIN:
        return "studied"
    return "few_papers" if j >= 1 else "none"


def is_informative(pair: dict) -> bool:
    """A pair says something unless it has zero joint works AND a concept is too narrow."""
    return pair["joint"] > 0 or min(pair["n_a"], pair["n_b"]) >= MIN_COMMON


def bhattacharyya(dist_a: dict, dist_b: dict) -> float | None:
    """Overlap of two count distributions over the same keys: sum sqrt(p_a * p_b) in [0, 1]."""
    ta, tb = sum(dist_a.values()), sum(dist_b.values())
    if ta <= 0 or tb <= 0:
        return None
    return sum(math.sqrt((dist_a[k] / ta) * (dist_b[k] / tb)) for k in dist_a.keys() & dist_b.keys())


def adjacency_threshold(known: list[float], unrelated: list[float]) -> float | None:
    """
    Cut-off between 'neighbouring literatures' and 'unrelated' read off the reference pairs:
    the midpoint between the lowest famous pair and the highest unrelated pair, if the two
    groups are separable; None otherwise (no bridge flag is then given).
    """
    known = [x for x in known if x is not None]
    unrelated = [x for x in unrelated if x is not None]
    if not known or not unrelated or min(known) <= max(unrelated):
        return None
    return (min(known) + max(unrelated)) / 2


def summarize(concept_counts: dict[str, int], pairs: list[dict],
              adj_threshold: float | None = None) -> dict:
    """
    One verdict per hypothesis, from its most novel *informative* pair.

      unrecognized_terms  a concept has < MIN_KNOWN works (not a standard term)
      inconclusive        every pair has zero joint works and a too-narrow concept
      unexplored          an informative pair with joint == 0
      few_papers          fewest joint works is 1..9
      studied             every informative pair has >= 10 joint works

    combo_novelty: 1.0 unexplored, 1 - joint/10 for few_papers, 0.0 studied, None otherwise.
    bridge_candidate: unexplored / few_papers AND the pair's literatures are neighbours
    (adjacency >= adj_threshold).
    """
    if not pairs or any(n < MIN_KNOWN for n in concept_counts.values()):
        return {"label": "unrecognized_terms", "combo_novelty": None, "pair": None,
                "bridge_candidate": False}
    informative = [p for p in pairs if is_informative(p)]
    if not informative:
        return {"label": "inconclusive", "combo_novelty": None, "pair": None,
                "bridge_candidate": False}
    best = min(informative, key=lambda p: (p["joint"], -(p.get("adjacency") or 0.0)))
    status = pair_status(best)
    label = {"none": "unexplored", "few_papers": "few_papers", "studied": "studied"}[status]
    novelty = {"unexplored": 1.0, "few_papers": round(1.0 - best["joint"] / STUDIED_MIN, 4),
               "studied": 0.0}[label]
    adj = best.get("adjacency")
    bridge = (label in ("unexplored", "few_papers") and adj is not None
              and adj_threshold is not None and adj >= adj_threshold)
    return {"label": label, "combo_novelty": novelty, "pair": best, "bridge_candidate": bridge}


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
    """Counts works (and their subfield distribution) for a boolean phrase query, with caching."""

    def __init__(self):
        self.mode = "filter"          # title + abstract; falls back to `search` (adds full text)
        self.cache: dict[str, int] = {}
        self.dist_cache: dict[str, dict[str, int]] = {}
        self._total: int | None = None
        self._attr: str | None = None

    def _text_params(self, query: str) -> dict:
        if self.mode == "filter":
            return {"filter": f"title_and_abstract.search:{query}"}
        return {"search": query}

    def _fall_back(self):
        print("  [note] title_and_abstract.search not accepted; using `search` "
              "(title + abstract + full text) instead.")
        self.mode = "search"
        self._total = None
        self.cache.clear()
        self.dist_cache.clear()

    @staticmethod
    def _is_400(ex: requests.HTTPError) -> bool:
        return ex.response is not None and ex.response.status_code == 400

    def count(self, query: str) -> int:
        from nous.collector.openalex import get
        if query in self.cache:
            return self.cache[query]
        try:
            data = get("works", {**self._text_params(query), "per_page": 1, "select": "id"})
        except requests.HTTPError as ex:
            if self.mode == "filter" and self._is_400(ex):
                self._fall_back()
                return self.count(query)
            raise
        n = int(data.get("meta", {}).get("count", 0))
        time.sleep(0.15)
        self.cache[query] = n
        return n

    def distribution(self, query: str) -> dict[str, int]:
        """Works matching `query`, counted per OpenAlex subfield (falls back to coarser groupings)."""
        from nous.collector.openalex import get, short_id
        if query in self.dist_cache:
            return self.dist_cache[query]
        attrs = [self._attr] if self._attr else GROUP_ATTRS
        last: Exception | None = None
        for attr in attrs:
            try:
                data = get("works", {**self._text_params(query), "group_by": attr})
            except requests.HTTPError as ex:
                if not self._is_400(ex):
                    raise
                if self.mode == "filter":
                    # was it the text filter or the grouping? count() falls back to `search`
                    # when the text filter is what OpenAlex rejects
                    self.count(build_query(["probe"]))
                    if self.mode == "search":
                        return self.distribution(query)
                last = ex
                continue
            self._attr = attr
            dist = {short_id(g["key"]): int(g["count"]) for g in data.get("group_by", [])
                    if str(g.get("key")) not in ("unknown", "-111")}
            time.sleep(0.15)
            self.dist_cache[query] = dist
            return dist
        raise last if last else RuntimeError("no grouping attribute accepted")

    @property
    def total(self) -> int:
        if self._total is None:
            from nous.collector.openalex import get
            self._total = int(get("works", {"per_page": 1, "select": "id"})["meta"]["count"])
        return self._total


def measure(concepts: list[str], counter: Counter,
            with_adjacency: bool = True) -> tuple[dict[str, int], list[dict]]:
    """Counts for each concept; joint, expectation and adjacency for each pair of concepts."""
    counts = {c: counter.count(build_query([c])) for c in concepts}
    pairs = []
    for a, b in itertools.combinations(concepts, 2):
        joint = counter.count(build_query([a, b]))
        st = pair_stats(counts[a], counts[b], joint, counter.total)
        adj = None
        if with_adjacency:
            adj = bhattacharyya(counter.distribution(build_query([a])),
                                counter.distribution(build_query([b])))
        pairs.append({"a": a, "b": b, "n_a": counts[a], "n_b": counts[b], "joint": joint,
                      "expected": st["expected"], "lift": st["lift"], "adjacency": adj})
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
  Prefer broad, widely used terms over narrow or compound phrases.
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
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS combo_checks (
            target_ref TEXT, concept_a TEXT, concept_b TEXT,
            n_a INTEGER, n_b INTEGER, joint INTEGER, expected REAL, lift REAL,
            method TEXT, checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (target_ref, concept_a, concept_b)
        );
        CREATE TABLE IF NOT EXISTS combo_meta (key TEXT PRIMARY KEY, value REAL);
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(combo_checks)").fetchall()]
    if "adjacency" not in cols:
        conn.execute("ALTER TABLE combo_checks ADD COLUMN adjacency REAL")
    conn.commit()


def _save_pairs(conn, ref: str, pairs: list[dict], method: str):
    conn.execute("DELETE FROM combo_checks WHERE target_ref = ?", (ref,))
    conn.executemany("""
        INSERT INTO combo_checks
        (target_ref, concept_a, concept_b, n_a, n_b, joint, expected, lift, adjacency, method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [(ref, p["a"], p["b"], p["n_a"], p["n_b"], p["joint"], p["expected"], p["lift"],
           p.get("adjacency"), method) for p in pairs])
    conn.commit()


def _load_pairs(conn, ref: str) -> list[dict]:
    rows = conn.execute("SELECT concept_a, concept_b, n_a, n_b, joint, expected, lift, adjacency "
                        "FROM combo_checks WHERE target_ref = ?", (ref,)).fetchall()
    return [dict(zip(("a", "b", "n_a", "n_b", "joint", "expected", "lift", "adjacency"), r))
            for r in rows]


def _set_meta(conn, key: str, value: float | None):
    if value is None:
        conn.execute("DELETE FROM combo_meta WHERE key = ?", (key,))
    else:
        conn.execute("INSERT OR REPLACE INTO combo_meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_adj_threshold() -> float | None:
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        row = conn.execute("SELECT value FROM combo_meta WHERE key = 'adj_threshold'").fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def combo_summary(ref: str) -> dict | None:
    """Verdict for one target, rebuilt from the stored pairs (used by `scores` and the report)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        pairs = _load_pairs(conn, ref)
    finally:
        conn.close()
    if not pairs:
        return None
    counts = {}
    for p in pairs:
        counts[p["a"]], counts[p["b"]] = p["n_a"], p["n_b"]
    return summarize(counts, pairs, get_adj_threshold())


def stored_refs() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(conn)
        return [r[0] for r in conn.execute("SELECT DISTINCT target_ref FROM combo_checks")]
    finally:
        conn.close()


def label_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for ref in stored_refs():
        s = combo_summary(ref)
        if s:
            out[s["label"]] = out.get(s["label"], 0) + 1
    return out


# ------------------------------------------------------------------ #
#  Commands                                                           #
# ------------------------------------------------------------------ #

def _fmt_pair(p: dict) -> str:
    adj = p.get("adjacency")
    adj_s = f", adjacency={adj:.2f}" if adj is not None else ""
    return (f"{p['a']} × {p['b']}: n_A={p['n_a']:,}, n_B={p['n_b']:,}, "
            f"joint={p['joint']:,}{adj_s}")


def measure_references(counter: Counter, verbose: bool = True) -> dict:
    """Famous vs. unrelated pairs; also stores the adjacency cut-off derived from them."""
    out = {"known": [], "unrelated": []}
    for kind, pairs in (("known", KNOWN_PAIRS), ("unrelated", UNRELATED_PAIRS)):
        for a, b in pairs:
            _, ps = measure([a, b], counter)
            out[kind].append(ps[0])
    thr = adjacency_threshold([p["adjacency"] for p in out["known"]],
                              [p["adjacency"] for p in out["unrelated"]])
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    _set_meta(conn, "adj_threshold", thr)
    conn.close()
    out["adj_threshold"] = thr
    if verbose:
        print("\n=== Reference pairs (scale for reading the numbers) ===")
        print("  well-known combinations:")
        for p in out["known"]:
            print(f"    {_fmt_pair(p)}   [{pair_status(p)}]")
        print("  deliberately unrelated:")
        for p in out["unrelated"]:
            print(f"    {_fmt_pair(p)}   [{pair_status(p)}]")
        if thr is None:
            print("  adjacency cut-off: none (famous and unrelated pairs are not separable) "
                  "— no bridge flag will be given")
        else:
            print(f"  adjacency cut-off (neighbouring literatures): >= {thr:.2f}")
    return out


def fill_adjacency(counter: Counter, verbose: bool = True) -> int:
    """Add the adjacency of every stored pair that lacks it (2 group_by calls per new concept)."""
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    rows = conn.execute("SELECT DISTINCT concept_a, concept_b FROM combo_checks "
                        "WHERE adjacency IS NULL").fetchall()
    n = 0
    for a, b in rows:
        adj = bhattacharyya(counter.distribution(build_query([a])),
                            counter.distribution(build_query([b])))
        if adj is None:
            continue
        conn.execute("UPDATE combo_checks SET adjacency = ? WHERE concept_a = ? AND concept_b = ?",
                     (adj, a, b))
        n += 1
    conn.commit()
    conn.close()
    if verbose and rows:
        print(f"  adjacency added for {n} stored concept pairs")
    return n


def rescore_all() -> int:
    """Rewrite each stored target's score from its stored pairs (verdict rules may have changed)."""
    n = 0
    for ref in stored_refs():
        s = combo_summary(ref)
        if not s:
            continue
        store.delete_scores(ref, RATER)
        store.save_scores(ref, RATER, {ITEM: s["combo_novelty"]} if s["combo_novelty"] is not None
                          else {"combo_checked": 1.0})
        n += 1
    return n


def print_report():
    """All stored verdicts grouped by label, from the database (no OpenAlex calls)."""
    refs = stored_refs()
    if not refs:
        print("No combination checks stored yet — run 'nous.py combocheck'.")
        return
    thr = get_adj_threshold()
    groups: dict[str, list[tuple[str, dict]]] = {}
    for ref in sorted(refs):
        s = combo_summary(ref)
        groups.setdefault(s["label"], []).append((ref, s))

    print("\n=== Combination check — all stored hypotheses ===")
    print("  " + "   ".join(f"{lbl}: {len(groups.get(lbl, []))}" for lbl in LABEL_ORDER))
    print("  (unexplored / few_papers with neighbouring literatures are marked ★ = bridge candidate)")
    if thr is None:
        print("  (no adjacency cut-off available: no ★ flags)")
    for lbl in LABEL_ORDER:
        items = groups.get(lbl, [])
        if not items:
            continue
        print(f"\n[{lbl}]  {len(items)}")
        for ref, s in items:
            star = " ★" if s["bridge_candidate"] else ""
            if s["pair"]:
                print(f"  {ref:<11}{star} {_fmt_pair(s['pair'])}")
            else:
                print(f"  {ref:<11}")
    stars = [(ref, s) for lbl in groups for ref, s in groups[lbl] if s["bridge_candidate"]]
    print(f"\nBridge candidates: {len(stars)}")
    for ref, s in sorted(stars, key=lambda x: -(x[1]["pair"].get("adjacency") or 0)):
        print(f"  {ref:<11} {s['label']:<10} {_fmt_pair(s['pair'])}")


def check_combinations(refs: list[str] | None = None, model: str = DEFAULT_MODEL,
                       verbose: bool = True) -> int:
    """
    1. reference pairs  2. decompose + count new hypotheses  3. adjacency for stored pairs
    4. re-score every stored hypothesis  5. print the report.
    """
    counter = Counter()
    measure_references(counter, verbose)

    targets = store.load_targets()
    if refs:
        targets = [t for t in targets if t["ref"] in set(refs)]
    done = set(stored_refs())
    todo = [t for t in targets if t["ref"] not in done]

    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    n = 0
    if verbose and todo:
        print(f"\nChecking {len(todo)} new hypotheses ...")
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
            _, pairs = measure(concepts, counter)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {t['ref']}: {ex}")
            continue
        _save_pairs(conn, t["ref"], pairs, counter.mode)
        n += 1
    conn.close()

    if verbose:
        print("\nAdjacency for previously stored pairs ...")
    fill_adjacency(counter, verbose)
    rescore_all()
    if verbose:
        print_report()
        print(f"\nDone. {n} new hypotheses checked ({counter.mode} search).")
    return n
