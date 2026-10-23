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

# OpenAlex's 26 fields (slugified display names) -> Scepter
DOMAIN_TO_SCEPTER.update({
    "arts_and_humanities":                          "Scepter-H",
    "psychology":                                   "Scepter-H",
    "social_sciences":                              "Scepter-S",
    "economics_econometrics_and_finance":           "Scepter-S",
    "business_management_and_accounting":           "Scepter-S",
    "decision_sciences":                            "Scepter-S",
    "mathematics":                                  "Scepter-N",
    "physics_and_astronomy":                        "Scepter-N",
    "chemistry":                                    "Scepter-N",
    "earth_and_planetary_sciences":                 "Scepter-N",
    "agricultural_and_biological_sciences":         "Scepter-N",
    "biochemistry_genetics_and_molecular_biology":  "Scepter-N",
    "immunology_and_microbiology":                  "Scepter-N",
    "neuroscience":                                 "Scepter-N",
    "computer_science":                             "Scepter-A",
    "engineering":                                  "Scepter-A",
    "chemical_engineering":                         "Scepter-A",
    "materials_science":                            "Scepter-A",
    "energy":                                       "Scepter-A",
    "medicine":                                     "Scepter-A",
    "dentistry":                                    "Scepter-A",
    "nursing":                                      "Scepter-A",
    "pharmacology_toxicology_and_pharmaceutics":    "Scepter-A",
    "health_professions":                           "Scepter-A",
    "veterinary":                                   "Scepter-A",
    "environmental_science":                        "Scepter-I",
})

HYPOTHESIS_DOMAIN = "hypothesis"

# The 5 parent Scepters are split into 10 field Scepters (the generators of a council debate).
# (name, parent Scepter, OpenAlex field slugs). Every OpenAlex field is in exactly one group,
# and the group's parent is the field's DOMAIN_TO_SCEPTER entry (checked by tests).
FIELD_GROUPS: list[tuple[str, str, tuple[str, ...]]] = [
    ("Scepter-H1", "Scepter-H", ("arts_and_humanities",)),
    ("Scepter-H2", "Scepter-H", ("psychology",)),
    ("Scepter-S1", "Scepter-S", ("social_sciences",)),
    ("Scepter-S2", "Scepter-S", ("economics_econometrics_and_finance",
                                 "business_management_and_accounting", "decision_sciences")),
    ("Scepter-N1", "Scepter-N", ("mathematics", "physics_and_astronomy")),
    ("Scepter-N2", "Scepter-N", ("chemistry", "earth_and_planetary_sciences")),
    ("Scepter-N3", "Scepter-N", ("agricultural_and_biological_sciences",
                                 "biochemistry_genetics_and_molecular_biology",
                                 "immunology_and_microbiology", "neuroscience")),
    ("Scepter-A1", "Scepter-A", ("computer_science", "engineering", "chemical_engineering",
                                 "materials_science", "energy")),
    ("Scepter-A2", "Scepter-A", ("medicine", "dentistry", "nursing",
                                 "pharmacology_toxicology_and_pharmaceutics",
                                 "health_professions", "veterinary")),
    ("Scepter-I1", "Scepter-I", ("environmental_science",)),
]

# Legacy Wikipedia/arXiv domains -> OpenAlex field slugs, so that one taxonomy is used
# everywhere (same-domain penalty, void zones, colors).
LEGACY_TO_OPENALEX: dict[str, str] = {
    "physics":          "physics_and_astronomy",
    "biology":          "agricultural_and_biological_sciences",
    "economics":        "economics_econometrics_and_finance",
    "sociology":        "social_sciences",
    "philosophy":       "arts_and_humanities",
    "history":          "arts_and_humanities",
    "linguistics":      "arts_and_humanities",   # ASJC: Language and Linguistics
    # identical in both taxonomies
    "mathematics":      "mathematics",
    "psychology":       "psychology",
    "computer_science": "computer_science",
}


def canonical_domain(domain: str) -> str:
    """Map a legacy domain name to its OpenAlex field slug (other names pass through)."""
    return LEGACY_TO_OPENALEX.get(domain, domain)


def scepter_for(domain: str) -> str | None:
    """Return the Scepter responsible for a collector domain (None if unmapped)."""
    return DOMAIN_TO_SCEPTER.get(domain)


def domains_for(scepter_name: str) -> list[str]:
    """Return all collector domains covered by a Scepter."""
    return sorted(d for d, s in DOMAIN_TO_SCEPTER.items() if s == scepter_name)


def field_scepter_for(domain: str) -> str | None:
    """Return the field Scepter (e.g. "Scepter-N3") responsible for a domain (None if unmapped)."""
    domain = canonical_domain(domain)
    for name, _parent, fields in FIELD_GROUPS:
        if domain in fields:
            return name
    return None


def parent_of(name: str) -> str | None:
    """Parent Scepter of a field Scepter; a parent Scepter is its own parent."""
    for child, parent, _fields in FIELD_GROUPS:
        if child == name:
            return parent
    return name if name in set(DOMAIN_TO_SCEPTER.values()) else None
