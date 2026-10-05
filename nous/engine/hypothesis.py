"""
Hypothesis generation via local Ollama LLM.
Uses cross-domain connections found by connector.py as context.
"""
import json
import sqlite3

from nous.config import DB_PATH, DEFAULT_MODEL
from nous.engine.connector import find_cross_domain_connections, find_bridge
from nous.llm import extract_json, generate


def build_prompt(query: str, connections: list[dict], bridges: list[dict]) -> str:
    conn_text = "\n".join(
        f"- [{r['domain']}] {r['title']}: {r['summary'][:200]}"
        for r in connections
    )
    bridge_text = "\n".join(
        f"- [{b['domain']}] {b['title']}: {b['summary'][:200]}"
        for b in bridges
    ) or "(none)"
    return f"""You are Nous, an AI that discovers novel connections between distant fields of human knowledge.

QUERY: {query}

CROSS-DOMAIN RESONANCES (concepts from other fields that share structural patterns with the query):
{conn_text}

BRIDGE CONCEPTS (concepts that structurally connect multiple domains):
{bridge_text}

Your task:
1. Identify the deepest structural pattern that the query shares with concepts from OTHER domains.
2. Generate a novel hypothesis: "Concept A from domain X and concept B from domain Y share structure Z, which suggests hypothesis H."
3. Propose one concrete next question or experiment that could validate this hypothesis.

Be bold. Prioritize surprising, non-obvious connections over safe ones.
Format your response as JSON:
{{
  "structural_pattern": "...",
  "hypothesis": "...",
  "confidence": 0.0-1.0,
  "validation_question": "...",
  "domains_connected": ["domain1", "domain2"]
}}"""


def generate_hypothesis(
    query: str,
    model: str = DEFAULT_MODEL,
    n_connections: int = 8,
    save: bool = True,
    context: list[dict] | None = None,
) -> dict:
    """
    `context` (optional): articles ({domain, title, summary}) to build the hypothesis from,
    instead of searching the whole knowledge base. The autonomous loop passes the works
    that bridge the two chosen fields, so the hypothesis is about that pair.
    """
    if context is not None:
        connections, bridges = context, []
    else:
        connections = find_cross_domain_connections(query, n_results=n_connections, llm_model=model)
        bridges     = find_bridge(query, connections[0]["title"] if connections else query, n=4)

    prompt = build_prompt(query, connections, bridges)

    print(f"Generating hypothesis via {model}...")
    raw = generate(prompt, model=model)

    try:
        result = extract_json(raw)
    except Exception:
        result = {"hypothesis": raw, "confidence": 0.5, "domains_connected": []}

    result["query"]   = query
    result["sources"] = [r["title"] for r in connections[:5]]

    if save:
        result["id"] = _save_hypothesis(query, result)

    return result


def _save_hypothesis(query: str, result: dict) -> int:
    domains = result.get("domains_connected") or []
    domain_a = domains[0] if len(domains) > 0 else "unknown"
    domain_b = domains[1] if len(domains) > 1 else "unknown"

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO explorations (query, domain_a, domain_b)
        VALUES (?, ?, ?)
    """, (query, domain_a, domain_b))
    exploration_id = c.lastrowid

    c.execute("""
        INSERT INTO hypotheses (exploration_id, hypothesis_text, confidence, source_ids)
        VALUES (?, ?, ?, ?)
    """, (
        exploration_id,
        result.get("hypothesis", ""),
        result.get("confidence", 0.5),
        json.dumps(result.get("sources", [])),
    ))
    hypothesis_id = c.lastrowid
    conn.commit()
    conn.close()
    return hypothesis_id


if __name__ == "__main__":
    result = generate_hypothesis("phase transition in complex systems")
    print(json.dumps(result, indent=2))
