"""
Rubric for judging hypotheses — used identically by LLM judges and human raters,
so their scores are comparable and inter-rater reliability can be computed.
"""
from __future__ import annotations

from nous.config import DEFAULT_MODEL
from nous.llm import generate_json

PROMPT_VERSION = "rubric-v1"
SCALE = (1, 5)

# item -> (question, anchors for 1 / 3 / 5)
RUBRIC: dict[str, tuple[str, str, str, str]] = {
    "novelty": (
        "How new is the idea relative to established knowledge?",
        "a restatement of a well-known idea",
        "a known idea applied in a somewhat new setting",
        "a connection that, to your knowledge, has not been proposed",
    ),
    "testability": (
        "Could the claim be checked by an experiment, dataset analysis, or computation?",
        "unfalsifiable or purely metaphorical",
        "testable in principle but no clear procedure",
        "names a concrete test whose outcome could refute it",
    ),
    "specificity": (
        "How precise are the variables, mechanisms and predicted effects?",
        "vague buzzwords",
        "names the variables but not the direction or size of effects",
        "specifies variables, mechanism and expected direction of effect",
    ),
    "coherence": (
        "Is the reasoning internally consistent and compatible with basic facts?",
        "self-contradictory or factually wrong",
        "plausible with gaps",
        "consistent and factually sound",
    ),
    "structural_depth": (
        "Does the cross-domain link rest on a shared *structure* (mechanism, dynamics, "
        "mathematical form) rather than a surface analogy or shared words?",
        "surface wordplay or loose metaphor",
        "partial structural correspondence",
        "a precise structural mapping between the domains",
    ),
}

ITEMS = list(RUBRIC)


def rubric_text() -> str:
    lines = []
    for item, (q, a1, a3, a5) in RUBRIC.items():
        lines.append(f"- {item}: {q}\n    1 = {a1}; 3 = {a3}; 5 = {a5}")
    return "\n".join(lines)


def clamp_score(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = SCALE
    return min(max(v, lo), hi)


def llm_judge(hypothesis: str, query: str, persona: str = "",
              model: str = DEFAULT_MODEL) -> tuple[dict[str, float], dict[str, str]]:
    """
    Score one hypothesis on every rubric item with an LLM.
    `persona` gives the judge a disciplinary viewpoint (e.g. a Scepter's lens) so that
    several judges act as distinct raters.
    Returns (scores, rationales); items the model failed to score are omitted.
    """
    persona_text = f"\nYour disciplinary viewpoint: {persona}\n" if persona else ""
    keys = ", ".join(f'"{i}": 1-5' for i in ITEMS)
    prompt = f"""You are a strict scientific reviewer rating a research hypothesis.{persona_text}
Rate it on each item using the 1-5 scale and anchors below. Be critical: most
LLM-generated hypotheses deserve 2-3; reserve 5 for exceptional cases.

RUBRIC:
{rubric_text()}

ORIGINAL QUERY: {query}

HYPOTHESIS:
{hypothesis}

Respond ONLY with JSON:
{{
  "scores": {{{keys}}},
  "rationale": {{"<item>": "one short sentence per item"}}
}}"""

    data = generate_json(prompt, model=model)
    raw_scores = data.get("scores", {}) if isinstance(data, dict) else {}
    raw_rat    = data.get("rationale", {}) if isinstance(data, dict) else {}

    scores = {}
    for item in ITEMS:
        v = clamp_score(raw_scores.get(item))
        if v is not None:
            scores[item] = v
    rationales = {k: str(v) for k, v in raw_rat.items() if k in scores} if isinstance(raw_rat, dict) else {}
    return scores, rationales
