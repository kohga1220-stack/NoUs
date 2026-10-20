"""
Known-work check — does a close published work already STATE this claim?

Counts and embedding similarity cannot tell "same topic" from "same claim" (a long LLM-written
hypothesis scores 0.5-0.7 against a paper that says the same thing). So the closest works are
fetched with their abstracts and an LLM is asked, per hypothesis, one narrow question:

  established  a listed work states or directly tests THIS claim (same variables, same relation)
  related      the same topic or variables, but not this claim
  not_found    nothing listed is about it

An 'established' answer must quote the work. The quote is checked against the abstract text; if
it cannot be found there, or a second, separate question ("does this quote assert the same central
claim?") is answered no, the answer is downgraded to 'related' (an invented or off-target quote
must never reject a hypothesis). Any failure (search, model, malformed answer) gives 'unknown', which never
rejects. Cost: one OpenAlex semantic search + one LLM call per hypothesis.

Usage:  python nous.py knowncheck [--model M] [--n N] [--no-rewrite] [--strict] [--show] [--controls | --negatives | --accepted [--retract] | hyp:26 ...]
"""
from __future__ import annotations
import time
import re

from nous.config import DEFAULT_MODEL

DEFAULT_KNOWN_MODEL = "gemini:gemini-3.6-flash"
LOCAL_KNOWN_MODEL = "gemma4:12b"   # free; live check on the 10 famous ideas: 8/10 found, 0/6 false (gemma4:e2b: 3/10)
NUM_CTX = 8192                    # the prompt is ~3,000 tokens; Ollama's default context would cut it
N_WORKS = 10
ABSTRACT_CHARS = 800
QUOTE_MIN_CHARS = 20
QUOTE_MAX_CHARS = 350             # one or two sentences
QUOTE_MIN_OVERLAP = 0.7           # share of the quote's words (>= 4 letters) found in the abstract
VERDICTS = ("established", "related", "not_found")


_DEFINITION = {
    # live run on 10 famous ideas, strict wording: 5/10 found; the model kept answering 'related'
    # because no work repeated EVERY part of a multi-part claim (e.g. Bak-Tang-Wiesenfeld 1988 itself)
    True: """- "established": one of the works states or directly tests THIS specific claim (the same variables
  AND the same kind of relationship). Being on the same topic is NOT enough.""",
    False: """- "established": one of the works states or directly tests the CENTRAL claim of the hypothesis, i.e.
  its main relationship between the variables. Parts of the hypothesis that the work does not mention
  (extra examples, mechanisms, numbers) do not matter as long as the work does not contradict them.
  Being on the same topic is NOT enough: the work must assert that relationship.
  If the hypothesis applies a known principle to a NEW system, quantity or factor, then that application
  IS its central claim: a work that states only the general principle (for other systems) does NOT
  establish it.""",
}


def default_known_model() -> str | None:
    """The model used when none is given: the local gemma4:12b if Ollama has it (free), else Gemini
    if GEMINI_API_KEY is set, else None (the check is off)."""
    import os
    from nous.llm import ollama_models
    if LOCAL_KNOWN_MODEL in (ollama_models() or set()):
        return LOCAL_KNOWN_MODEL
    return DEFAULT_KNOWN_MODEL if os.environ.get("GEMINI_API_KEY") else None


def build_prompt(claim: str, works: list[dict], strict: bool = False) -> str:
    listed = "\n\n".join(f"[{i + 1}] {w['title']} ({w.get('year')})\n{(w.get('text') or '')[:ABSTRACT_CHARS]}"
                         for i, w in enumerate(works))
    return f"""You check whether a research hypothesis is ALREADY established in the published literature.

HYPOTHESIS:
{claim[:1500]}

PUBLISHED WORKS (title, year, abstract):
{listed}

Decide, using ONLY the works above:
{_DEFINITION[strict]}
- "related": the works study the same topic or variables, but none states that claim.
- "not_found": none of the works is about this claim.
For "established", give the number of that work and copy one or two consecutive sentences from its
abstract (at most {QUOTE_MAX_CHARS} characters, exactly as written) that state the claim.

Respond ONLY with JSON: {{"verdict": "established" | "related" | "not_found", "work": <number or null>, "quote": "<sentences or empty>", "why": "<one short sentence: what the closest work says, and how it differs if not established>"}}"""


def build_rewrite_prompt(claim: str) -> str:
    return f"""Rewrite this research hypothesis as a plain search query for a literature database:
one or two short sentences stating only the core claim in the vocabulary used in paper abstracts.
Leave out the method, examples and any hedging.

HYPOTHESIS:
{claim[:1500]}

Respond ONLY with JSON: {{"query": "<one or two sentences>"}}"""


def build_confirm_prompt(claim: str, title: str, quote: str) -> str:
    return f"""Does this sentence from a published abstract assert the same central claim as the hypothesis?
Be strict: the same topic is not enough; the same variables and the same kind of relationship are needed.
Details the hypothesis adds on top of the central claim do not matter.

HYPOTHESIS:
{claim[:1500]}

SENTENCE (from "{title}"):
{quote}

Respond ONLY with JSON: {{"same": true | false}}"""


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
    out = {"verdict": "unknown", "work": None, "title": None, "year": None, "evidence": "", "why": ""}
    if not isinstance(data, dict) or data.get("verdict") not in VERDICTS:
        return out
    out["verdict"] = data["verdict"]
    out["why"] = str(data.get("why") or "")[:200]
    if out["verdict"] != "established":
        return out
    idx = data.get("work")
    quote = str(data.get("quote") or "").strip()
    work = works[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(works) else None
    if work is None or not quote_supported(quote, work.get("text", "")):
        out["verdict"] = "related"
        out["evidence"] = "claimed established, but the quote was not found in the cited abstract"
        return out
    out.update(work=idx, title=work["title"], year=work.get("year"), evidence=quote[:QUOTE_MAX_CHARS])
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
                search=None, ask=None, rewrite: bool = True, confirm: bool = True,
                strict: bool = False) -> dict:
    """Never raises: any failure is reported as verdict 'unknown'.
    rewrite: also search with a short plain-language version of the claim (one more model
    call); the two result lists are merged (live check on 10 famous ideas: 5/10 -> 8/10 found).
    confirm: an 'established' answer must survive a second, separate yes/no question about its
    quote (a live run accepted a quote about body size for a claim about neocortex size)."""
    from nous.evaluation.combination import claim_text
    claim = claim_text(text)
    empty = {"work": None, "title": None, "year": None, "works": []}
    timing = {"search": 0.0, "llm": 0.0}          # seconds; shown after the cycle's `known` time

    def timed(key, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            timing[key] += time.monotonic() - t0
    try:
        if search is None:
            from nous.evaluation.literature import search_prior_work as search
        if ask is None:
            from nous.llm import generate_json

            def ask(prompt, model):
                return generate_json(prompt, model=model, num_ctx=NUM_CTX)
        raw_search, raw_ask = search, ask
        search = lambda *a, **kw: timed("search", raw_search, *a, **kw)
        ask = lambda *a, **kw: timed("llm", raw_ask, *a, **kw)
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
        data = ask(build_prompt(claim, works, strict), model=model)
    except Exception as ex:
        return {"verdict": "unknown", "evidence": f"model failed: {type(ex).__name__}",
                **{**empty, "works": titles}}
    res = parse_verdict(data, works)
    res["works"] = titles
    res["timing"] = timing
    if confirm and res["verdict"] == "established":
        try:
            same = (ask(build_confirm_prompt(claim, res["title"], res["evidence"]), model=model) or {}).get("same")
        except Exception:
            same = None
        if same is not True:
            res.update(verdict="related", evidence="the quote was not confirmed as the same claim"
                       if same is False else "second check unavailable; not rejecting")
    res["timing"] = {k: round(v, 1) for k, v in timing.items()}
    return res


def describe(res: dict) -> str:
    if res["verdict"] == "established":
        return f"established: {res['title']} ({res['year']}) — \"{res['evidence']}\""
    return f"{res['verdict']}" + (f" ({res['evidence']})" if res.get("evidence") else "")


# Claims that no published work should state (invented cross-domain links, and known principles
# applied to a new system or factor). A good check never calls any of them 'established'.
NEGATIVE_CONTROLS: list[tuple[str, str]] = [
    ("gravitational waves x dental caries",
     "The amplitude of the gravitational wave background determines the incidence of dental caries, "
     "because tidal strain modulates the remineralization of tooth enamel."),
    ("sourdough as a critical system",
     "Sourdough starter microbial populations organize themselves into a critical state, so that small "
     "changes in the feeding schedule trigger avalanches of acidity change whose sizes follow a power law."),
    ("cathode packing x mosaic entropy",
     "The packing density of lithium-ion cathode particles sets the information entropy of Byzantine "
     "mosaic tile patterns, because both are governed by the same space-filling constraint."),
    ("Ising opinions among bakers (hard)",
     "Opinion dynamics among sourdough bakers follow Ising-model dynamics, with the coupling strength "
     "set by oven temperature."),
    ("3/4 scaling of poets (hard)",
     "The number of followers of a poet on social media scales with the three-quarter power of the "
     "poet's body mass, because attention is distributed through fractal branching networks."),
    ("integrated information of sourdough (hard)",
     "A sourdough colony is conscious to the extent that it integrates information as a whole, and this "
     "integration increases with the hydration level of the dough."),
]


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
    model = default_known_model() or DEFAULT_KNOWN_MODEL
    if "--model" in args:
        i = args.index("--model")
        model = args[i + 1]
        args = args[:i] + args[i + 2:]
    n, rewrite, show = N_WORKS, "--no-rewrite" not in args, "--show" in args
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    opts = dict(n=n, rewrite=rewrite, strict="--strict" in args)
    retract = "--retract" in args
    if "--negatives" in args:
        hits = 0
        print(f"claims that should NOT be established, judged by {model}:")
        for label, text in NEGATIVE_CONTROLS:
            r = check_known(text, model, **opts)
            hits += r["verdict"] == "established"
            print(f"  {label:<44} {describe(r)[:150]}")
            if show:
                if r.get("why"):
                    print(f"        why: {r['why']}")
                for t in r["works"]:
                    print(f"        - {t}")
        print(f"\nwrongly flagged as established: {hits}/{len(NEGATIVE_CONTROLS)}  (should be 0)")
        return
    if "--controls" in args:
        hits = 0
        print(f"famous (known) ideas, judged by {model}:")
        for label, text in KNOWN_IDEAS:
            r = check_known(text, model, **opts)
            hits += r["verdict"] == "established"
            print(f"  {label:<30} {describe(r)[:150]}")
            if show:
                if r.get("why"):
                    print(f"        why: {r['why']}")
                for t in r["works"]:
                    print(f"        - {t}")
        print(f"\nflagged as established: {hits}/{len(KNOWN_IDEAS)}  (all should be)")
        return
    refs = _accepted_refs() if "--accepted" in args else [a for a in args if not a.startswith("--")]
    if not refs:
        print("Usage: python nous.py knowncheck [--model M] [--n N] [--no-rewrite] [--strict] [--show] "
              "[--controls | --negatives | --accepted [--retract] | hyp:26 ...]")
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
            if r.get("why"):
                print(f"        why: {r['why']}")
            for t in r["works"]:
                print(f"        - {t}")
        if retract and r["verdict"] == "established":
            loop.retract(ref, f"retracted: {describe(r)}"[:300])
