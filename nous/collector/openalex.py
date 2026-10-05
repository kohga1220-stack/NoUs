"""
OpenAlex — the open catalog of the global research system (~300M works).

Used for three things:
  1. collect_fields()      : sample works (with abstracts) from each of the 26 fields
                             into the knowledge base
  2. field_cooccurrence()  : how often works of field A are also tagged with field B,
                             across the WHOLE literature (group_by, no sampling)
                             → literature-grounded Void Zones
  3. field_trends()        : works per publication year for each field
                             → growth / acceleration of each discipline

Access (as of 2026): free API key recommended ($1/day free budget; $0.10/day without a key).
Set it with:  export OPENALEX_API_KEY=...
Cost: a list/group_by call is $0.0001, so a full refresh of all three steps is well under $0.05.
"""
from __future__ import annotations
import os
import re
import sqlite3
import time

import requests

from nous.config import DB_PATH, RAW_PATH

API = "https://api.openalex.org"


# ------------------------------------------------------------------ #
#  HTTP                                                               #
# ------------------------------------------------------------------ #

def _params(extra: dict) -> dict:
    p = dict(extra)
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        p["api_key"] = key
    mailto = os.environ.get("OPENALEX_MAILTO")
    if mailto:
        p["mailto"] = mailto
    return p


def get(path: str, params: dict | None = None, max_retries: int = 5) -> dict:
    """GET with exponential backoff on 429 / 5xx."""
    url = f"{API}/{path.lstrip('/')}"
    for attempt in range(max_retries):
        r = requests.get(url, params=_params(params or {}), timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenAlex: max retries exceeded for {url}")


# ------------------------------------------------------------------ #
#  Pure helpers                                                       #
# ------------------------------------------------------------------ #

def slugify(name: str) -> str:
    """'Economics, Econometrics and Finance' -> 'economics_econometrics_and_finance'"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def short_id(openalex_id: str) -> str:
    """'https://openalex.org/fields/27' -> '27';  'https://openalex.org/W123' -> 'W123'"""
    return str(openalex_id).rstrip("/").rsplit("/", 1)[-1]


def reconstruct_abstract(inverted: dict | None) -> str:
    """Rebuild plaintext from OpenAlex's abstract_inverted_index."""
    if not inverted:
        return ""
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    positions.sort()
    return " ".join(w for _, w in positions)


def parse_work(work: dict) -> dict | None:
    """OpenAlex work JSON -> article dict (None when there is no abstract or field)."""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    pt = work.get("primary_topic") or {}
    field = (pt.get("field") or {}).get("display_name")
    title = work.get("display_name") or work.get("title")
    if not abstract or not field or not title:
        return None
    return {
        "openalex_id": short_id(work["id"]),
        "title":       title,
        "domain":      slugify(field),
        "field":       field,
        "topic":       pt.get("display_name", ""),
        "year":        work.get("publication_year"),
        "cited_by":    work.get("cited_by_count", 0),
        "summary":     abstract[:500],
        "text":        f"{title}\n\n{abstract}",
        "url":         work["id"],
    }


def lift_matrix(cooc: dict[tuple[str, str], int], totals: dict[str, int],
                n_total: int) -> list[dict]:
    """
    Observed vs. expected co-occurrence of two fields.

    cooc[(a, b)]  : works whose primary field is a and that are also tagged with field b
    totals[f]     : works whose primary field is f
    n_total       : all works considered

    lift = P(tagged b | primary a) / P(primary b)
    lift < 1  → the fields meet less often than their sizes predict  (a void)
    lift > 1  → they meet more often than chance                      (a highway)
    Pairs are symmetrised by summing both directions.
    Because a work carries up to 3 topics, lifts are inflated roughly uniformly;
    read them as a ranking (relative), not as absolute probabilities.
    """
    fields = sorted(totals)
    out = []
    for i, a in enumerate(fields):
        for b in fields[i + 1:]:
            obs = cooc.get((a, b), 0) + cooc.get((b, a), 0)
            exp = 2 * totals[a] * totals[b] / n_total if n_total else 0
            out.append({
                "field_a": a, "field_b": b,
                "observed": obs,
                "expected": round(exp, 2),
                "lift": round(obs / exp, 4) if exp else 0.0,
            })
    out.sort(key=lambda x: x["lift"])
    return out


def growth_stats(counts_by_year: dict[int, int], window: int = 5) -> dict:
    """
    Compound annual growth over the last `window` years vs. the `window` before,
    and the change between them (acceleration). Uses complete years only.
    """
    years = sorted(y for y in counts_by_year if counts_by_year[y] > 0)
    if len(years) < 2 * window + 1:
        return {"recent_cagr": None, "previous_cagr": None, "acceleration": None}

    def cagr(y0, y1):
        a, b = counts_by_year.get(y0, 0), counts_by_year.get(y1, 0)
        if a <= 0 or b <= 0:
            return None
        return (b / a) ** (1 / (y1 - y0)) - 1

    last = years[-1]
    recent = cagr(last - window, last)
    prev = cagr(last - 2 * window, last - window)
    acc = None if recent is None or prev is None else recent - prev
    return {"recent_cagr": recent, "previous_cagr": prev, "acceleration": acc}


# ------------------------------------------------------------------ #
#  Database                                                           #
# ------------------------------------------------------------------ #

def _ensure_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS oa_fields (
            field_id     TEXT PRIMARY KEY,
            name         TEXT,
            slug         TEXT,
            domain_name  TEXT,
            works_count  INTEGER
        );
        CREATE TABLE IF NOT EXISTS field_links (
            field_a   TEXT,
            field_b   TEXT,
            observed  INTEGER,
            expected  REAL,
            lift      REAL,
            computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (field_a, field_b)
        );
        CREATE TABLE IF NOT EXISTS field_trends (
            field_id  TEXT,
            year      INTEGER,
            works     INTEGER,
            PRIMARY KEY (field_id, year)
        );
    """)
    conn.commit()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_tables(conn)
    return conn


# ------------------------------------------------------------------ #
#  1. Fields and works                                                #
# ------------------------------------------------------------------ #

def fetch_fields() -> list[dict]:
    data = get("fields", {"per_page": 100})
    fields = [{
        "field_id":    short_id(f["id"]),
        "name":        f["display_name"],
        "slug":        slugify(f["display_name"]),
        "domain_name": (f.get("domain") or {}).get("display_name", ""),
        "works_count": f.get("works_count", 0),
    } for f in data.get("results", [])]

    conn = _connect()
    conn.executemany("""INSERT OR REPLACE INTO oa_fields
                        (field_id, name, slug, domain_name, works_count)
                        VALUES (:field_id, :name, :slug, :domain_name, :works_count)""", fields)
    conn.commit()
    conn.close()
    return fields


def load_fields() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT field_id, name, slug, domain_name, works_count FROM oa_fields").fetchall()
    conn.close()
    return [dict(zip(("field_id", "name", "slug", "domain_name", "works_count"), r)) for r in rows]


def fetch_works(field_id: str, n: int = 50, mode: str = "cited",
                seed: int = 42) -> list[dict]:
    """
    mode='cited'  : most-cited works of the field (its landmark theories)
    mode='recent' : most-cited works of the last 3 years (its current frontier)
    mode='sample' : a reproducible random sample
    """
    select = "id,display_name,publication_year,abstract_inverted_index,primary_topic,cited_by_count"
    flt = f"primary_topic.field.id:{field_id},has_abstract:true"
    params = {"per_page": min(n * 2, 100), "select": select}
    if mode == "recent":
        flt += f",publication_year:>{time.gmtime().tm_year - 3}"
        params["sort"] = "cited_by_count:desc"
    elif mode == "sample":
        params["sample"] = min(n * 2, 100)
        params["seed"] = seed
    else:
        params["sort"] = "cited_by_count:desc"
    params["filter"] = flt

    data = get("works", params)
    works = [w for w in (parse_work(x) for x in data.get("results", [])) if w]
    return works[:n]


def save_work(w: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR IGNORE INTO articles (title, domain, summary, full_text, page_id, url)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (w["title"], w["domain"], w["summary"], w["text"], None, w["url"]))
    conn.commit()
    conn.close()
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    (RAW_PATH / f"openalex_{w['openalex_id']}.txt").write_text(w["text"], encoding="utf-8")


def collect_fields(per_field: int = 20, modes: tuple[str, ...] = ("cited", "recent"),
                   verbose: bool = True) -> int:
    """Sample works from every field into the knowledge base (~2 calls per field per mode)."""
    from nous.collector.wikipedia import init_db
    init_db()
    fields = load_fields() or fetch_fields()
    total = 0
    for f in fields:
        if verbose:
            print(f"\n[{f['slug']}] {f['name']}")
        for mode in modes:
            try:
                works = fetch_works(f["field_id"], n=per_field, mode=mode)
            except Exception as ex:
                print(f"  [WARN] {mode}: {ex}")
                continue
            for w in works:
                save_work(w)
                total += 1
            if verbose:
                print(f"  {mode:<6} {len(works)} works")
            time.sleep(0.2)
    if verbose:
        print(f"\nDone. {total} works collected.")
    return total


# ------------------------------------------------------------------ #
#  2. Field co-occurrence → literature-grounded void zones            #
# ------------------------------------------------------------------ #

def field_cooccurrence(verbose: bool = True) -> list[dict]:
    """
    26 group_by calls: for each primary field A, count works also tagged with field B.
    Stores and returns the lift table (lowest lift = strongest void first).
    """
    fields = load_fields() or fetch_fields()
    totals: dict[str, int] = {}
    cooc: dict[tuple[str, str], int] = {}

    for f in fields:
        data = get("works", {"filter": f"primary_topic.field.id:{f['field_id']}",
                             "group_by": "topics.field.id"})
        totals[f["field_id"]] = data.get("meta", {}).get("count", 0)
        for g in data.get("group_by", []):
            other = short_id(g["key"])
            if other != f["field_id"]:
                cooc[(f["field_id"], other)] = g["count"]
        if verbose:
            print(f"  {f['name']:<45} {totals[f['field_id']]:>12,} works")
        time.sleep(0.2)

    table = lift_matrix(cooc, {k: v for k, v in totals.items() if v > 0}, sum(totals.values()))

    conn = _connect()
    conn.executemany("""INSERT OR REPLACE INTO field_links
                        (field_a, field_b, observed, expected, lift)
                        VALUES (:field_a, :field_b, :observed, :expected, :lift)""", table)
    conn.commit()
    conn.close()
    return table


def load_field_links() -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT l.field_a, fa.name, l.field_b, fb.name, l.observed, l.expected, l.lift
        FROM field_links l
        LEFT JOIN oa_fields fa ON fa.field_id = l.field_a
        LEFT JOIN oa_fields fb ON fb.field_id = l.field_b
        ORDER BY l.lift ASC
    """).fetchall()
    conn.close()
    keys = ("field_a", "name_a", "field_b", "name_b", "observed", "expected", "lift")
    return [dict(zip(keys, r)) for r in rows]


def collect_pair_bridges(link: dict, per_pair: int = 10) -> list[dict]:
    """Fetch and save the most-cited works tagged with both fields of one pair; returns them."""
    data = get("works", {
        "filter": f"topics.field.id:{link['field_a']},topics.field.id:{link['field_b']},has_abstract:true",
        "sort": "cited_by_count:desc",
        "per_page": min(per_pair * 2, 100),
        "select": "id,display_name,publication_year,abstract_inverted_index,primary_topic,cited_by_count",
    })
    works = [w for w in (parse_work(x) for x in data.get("results", [])) if w][:per_pair]
    for w in works:
        save_work(w)
    return works


def collect_void_bridges(top: int = 10, per_pair: int = 10, verbose: bool = True) -> int:
    """
    For the `top` lowest-lift field pairs that still have *some* shared works, fetch the
    most-cited works tagged with BOTH fields — humanity's rare existing "dockings" of
    the two disciplines — into the knowledge base.
    """
    from nous.collector.wikipedia import init_db
    init_db()
    links = [l for l in load_field_links() if l["observed"] > 0][:top]
    if not links:
        print("No field links yet — run 'nous.py voids --refresh' first.")
        return 0
    total = 0
    for l in links:
        works = collect_pair_bridges(l, per_pair)
        total += len(works)
        if verbose:
            print(f"  {l['name_a']} × {l['name_b']}  (lift={l['lift']:.3f}) → {len(works)} works")
        time.sleep(0.2)
    if verbose:
        print(f"\nDone. {total} bridge works collected.")
    return total


# ------------------------------------------------------------------ #
#  3. Field trends                                                    #
# ------------------------------------------------------------------ #

def field_trends(since: int = 1950, verbose: bool = True) -> dict[str, dict]:
    """Works per publication year for every field (26 group_by calls)."""
    fields = load_fields() or fetch_fields()
    this_year = time.gmtime().tm_year
    out = {}
    conn = _connect()
    for f in fields:
        data = get("works", {"filter": f"primary_topic.field.id:{f['field_id']},"
                                       f"publication_year:{since}-{this_year - 1}",
                             "group_by": "publication_year"})
        counts = {int(g["key"]): g["count"] for g in data.get("group_by", [])
                  if str(g["key"]).lstrip("-").isdigit() and int(g["key"]) > 0}
        conn.executemany("INSERT OR REPLACE INTO field_trends (field_id, year, works) VALUES (?,?,?)",
                         [(f["field_id"], y, c) for y, c in counts.items()])
        out[f["field_id"]] = {"name": f["name"], "counts": counts, **growth_stats(counts)}
        if verbose:
            g = out[f["field_id"]]
            rc = f"{g['recent_cagr']*100:+.1f}%/yr" if g["recent_cagr"] is not None else "n/a"
            ac = f"{g['acceleration']*100:+.1f}pt" if g["acceleration"] is not None else "n/a"
            print(f"  {f['name']:<45} growth {rc:>11}  accel {ac:>8}")
        time.sleep(0.2)
    conn.commit()
    conn.close()
    return out
