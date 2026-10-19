"""
Known-work check — does a close published work already STATE this claim?

Counts and embedding similarity cannot tell "same topic" from "same claim" (a long LLM-written
hypothesis scores 0.5-0.7 against a paper that says the same thing). So the closest works are
fetched with their abstracts and an LLM is asked, per hypothesis, one narrow question:

  established  a listed work states or directly tests THIS claim (same variables, same relation)
  related      the same topic or variables, but not this claim
  not_found    nothing listed is about it

An 'established' answer must quote the work. The quote is checked against the abstract text; if
it cannot be found there, the answer is downgraded to 'related' (an invented quote must never
reject a hypothesis). Any failure (search, model, malformed answer) gives 'unknown', which never
rejects. Cost: one OpenAlex semantic search + one LLM call per hypothesis.

Usage:  python nous.py knowncheck [--model M] [--n N] [--rewrite] [--show] [--controls | --accepted [--retract] | hyp:26 ...]
"""
from __future__ import annotations
import re

from nous.config import DEFAULT_MODEL

DEFAULT_KNOWN_MODEL = "gemini:gemini-3.6-flash"
N_WORKS = 5
ABSTRACT_CHARS = 800
QUOTE_MIN_CHARS = 20
QUOTE_MIN_OVERLAP = 0.7           # share of the quote's words (>= 4 letters) found in the abstract
VERDICTS = ("established", "related", "not_found")


def build_prompt(claim: str, works: list[dict]) -> str:
    listed = "\n\n".join(f"[{i + 1}] {w['title']} ({w.get('year')})\n{(w.get('text') or '')[:ABSTRACT_CHARS]}"
                         for i, w in enumerate(works))
    return f"""You check whether a research hypothesis is ALREADY established in the published literature.

HYPOTHESIS:
{claim[:1500]}

PUBLISHED WORKS (title, year, abstract):
{listed}

Decide, using ONLY the works above:
- "established": one of the works states or directly tests THIS specific claim (the same variables
  AND the same kind of relationship). Being on the same topic is NOT enough.
- "related": the works study the same topic or variables, but none states this claim.
- "not_found": none of the works is about this claim.
For "established", give the number of that work and copy ONE short sentence from its abstract
(at most 200 characters, exactly as written) that states the claim.

Respond ONLY with JSON: {{"verdict": "established" | "related" | "not_found", "work": <number or null>, "quote": "<sentence or empty>"}}"""


def build_rewrite_prompt(claim: str) -> str:
    return f"""Rewrite this research hypothesis as a plain search query for a literature database:
one or two short sentences stating only the core claim in the vocabulary used in paper abstracts.
Leave out the method, examples and any hedging.

HYPOTHESIS:
{claim[:1500]}

Respond ONLY with JSON: {{"query": "<one or two sentences>"}}"""


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 4]


def quote_supported(quote: str, text: str) -> bool:
    """The quote is (nearly) in the text: enough of its longer words occur in it (pure)."""
    if len((quote or "").strip()) < QUOTE_MIN_CHARS:
        return False
    qw = _words(quote)
    if not qw:
        return False
    have = set(_words(text))
    return sum(w in have for w in qw) / len(qw) >= QUOTE_MIN_OVERLAP


def parse_verdict(data, works: list[dict]) -> dict:
    """LLM JSON -> {verdict, work, title, year, evidence}; an unverifiable 'established' becomes 'related'."""
    out = {"verdict": "unknown", "work": None, "title": None, "year": None, "evidence": ""}
    if not isinstance(data, dict) or data.get("verdict") not in VERDICTS:
        return out
    out["verdict"] = data["verdict"]
    if out["verdict"] != "established":
        return out
    idx = data.get("work")
    quote = str(data.get("quote") or "").strip()
    work = works[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(works) else None
    if work is None or not quote_supported(quote, work.get("text", "")):
        out["verdict"] = "related"
        out["evidence"] = "claimed established, but the quote was not found in the cited abstract"
        return out
    out.update(work=idx, title=work["title"], year=work.get("year"), evidence=quote[:200])
    return out


def _merge(first: list[dict], second: list[dict], n: int) -> list[dict]:
    """Alternate the two result lists, drop duplicates (by work id, else title), keep n."""
    seen, out = set(), []
    for pair in zip(first + [None] * len(second), second + [None] * len(first)):
        for w in pair:
            if not w:
                continue
            key = w.get("work_id") or w["title"]
            if key not in seen:
                seen.add(key)
                out.append(w)
    return out[:n]


def check_known(text: str, model: str = DEFAULT_KNOWN_MODEL, n: int = N_WORKS,
                search=None, ask=None, rewrite: bool = False) -> dict:
    """Never raises: any failure is reported as verdict 'unknown'.
    rewrite=True also searches with a short plain-language version of the claim (one more
    model call); the two result lists are merged."""
    from nous.evaluation.combination import claim_text
    claim = claim_text(text)
    empty = {"work": None, "title": None, "year": None, "works": []}
    try:
        if search is None:
            from nous.evaluation.literature import search_prior_work as search
        if ask is None:
            from nous.llm import generate_json as ask
        works = search(claim, n=n)
    except Exception as ex:
        return {"verdict": "unknown", "evidence": f"search failed: {type(ex).__name__}", **empty}
    if rewrite:
        try:
            q = str((ask(build_rewrite_prompt(claim), model=model) or {}).get("query") or "").strip()
            if q:
                works = _merge(works, search(q, n=n), n)
        except Exception:
            pass                                   # the plain search results are still usable
    titles = [f"{w['title'][:80]} ({w.get('year')})" for w in works]
    if not works:
        return {"verdict": "not_found", "evidence": "no works returned", **empty}
    try:
        data = ask(build_prompt(claim, works), model=model)
    except Exception as ex:
        return {"verdict": "unknown", "evidence": f"model failed: {type(ex).__name__}",
                **{**empty, "works": titles}}
    res = parse_verdict(data, works)
    res["works"] = titles
    return res


def describe(res: dict) -> str:
    if res["verdict"] == "established":
        return f"established: {res['title']} ({res['year']}) — \"{res['evidence']}\""
    return f"{res['verdict']}" + (f" ({res['evidence']})" if res.get("evidence") else "")


# ------------------------------------------------------------------ #
#  Command line                                                       #
# ------------------------------------------------------------------ #

def _accepted_refs() -> list[str]:
    import sqlite3
    from nous.config import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT DISTINCT hyp_ref FROM loop_runs WHERE status = 'accepted' "
                            "AND hyp_ref IS NOT NULL").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return [r[0] for r in rows]


def run(args: list[str]) -> None:
    from nous.evaluation import store
    from nous.evaluation.literature import KNOWN_IDEAS
    model = DEFAULT_KNOWN_MODEL
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        args = args[:i] + args[i + 2:]
    n, rewrite, show = N_WORKS, "--rewrite" in args, "--show" in args
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    opts = dict(n=n, rewrite=rewrite)
    retract = "--retract" in args
    if "--controls" in args:
        hits = 0
        print(f"famous (known) ideas, judged by {model}:")
        for label, text in KNOWN_IDEAS:
            r = check_known(text, model, **opts)
            hits += r["verdict"] == "established"
            print(f"  {label:<30} {describe(r)[:150]}")
            if show:
                for t in r["works"]:
                    print(f"        - {t}")
        print(f"\nflagged as established: {hits}/{len(KNOWN_IDEAS)}  (all should be)")
        return
    refs = _accepted_refs() if "--accepted" in args else [a for a in args if not a.startswith("--")]
    if not refs:
        print("Usage: python nous.py knowncheck [--model M] [--n N] [--rewrite] [--show] "
              "[--controls | --accepted [--retract] | hyp:26 ...]")
        return
    texts = {t["ref"]: t["text"] for t in store.load_targets()}
    from nous import loop
    for ref in refs:
        if ref not in texts:
            print(f"  {ref:<11} not found")
            continue
        r = check_known(texts[ref], model, **opts)
        print(f"  {ref:<11} {describe(r)[:220]}")
        if show:
            for t in r["works"]:
                print(f"        - {t}")
        if retract and r["verdict"] == "established":
            loop.retract(ref, f"retracted: {describe(r)}"[:300])
