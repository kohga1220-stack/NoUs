"""
Domain taxonomy — maps the fine-grained collector domains (used in SQLite,
ChromaDB and the knowledge graph) onto the 5 Scepter domains.
"""
from __future__ import annotations

# collector domain -> Scepter name
DOMAIN_TO_SCEPTER: dict[str, str] = {
    "philosophy":       "Scepter-H",
    "history":          "Scepter-H",
    "linguistics":      "Scepter-H",
    "psychology":       "Scepter-H",   # Psychology is listed under Humanities in registry.py
    "economics":        "Scepter-S",
    "sociology":        "Scepter-S",
    "physics":          "Scepter-N",
    "biology":          "Scepter-N",
    "mathematics":      "Scepter-N",
    "computer_science": "Scepter-A",
}

HYPOTHESIS_DOMAIN = "hypothesis"


def scepter_for(domain: str) -> str | None:
    """Return the Scepter responsible for a collector domain (None if unmapped)."""
    return DOMAIN_TO_SCEPTER.get(domain)


def domains_for(scepter_name: str) -> list[str]:
    """Return all collector domains covered by a Scepter."""
    return sorted(d for d, s in DOMAIN_TO_SCEPTER.items() if s == scepter_name)
