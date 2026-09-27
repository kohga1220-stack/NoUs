"""
Base Scepter — a domain-specialized intelligence within Nous.
Each Scepter holds deep expertise in its academic field and can:
  1. generate() — produce a domain-specific hypothesis
  2. critique()  — evaluate another Scepter's hypothesis from its own lens
"""
from __future__ import annotations

from nous.config import DEFAULT_MODEL
from nous.domains import domains_for, scepter_for
from nous.llm import generate_json


class Scepter:
    name: str          # e.g. "Scepter-N"
    domain: str        # e.g. "Natural Sciences"
    subdomains: list[str]
    lens: str          # one-sentence description of its epistemic perspective

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    @property
    def collector_domains(self) -> list[str]:
        """Knowledge-base domains (physics, psychology, ...) owned by this Scepter."""
        return domains_for(self.name)

    def _format_context(self, context: list[dict], limit: int = 8) -> str:
        lines = []
        for r in context[:limit]:
            owner = scepter_for(r["domain"])
            tag = " (your field)" if owner == self.name else ""
            lines.append(f"- [{r['domain']}{tag}] {r['title']}: {r.get('summary','')[:150]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Generate a domain-specific hypothesis                               #
    # ------------------------------------------------------------------ #
    def generate(self, query: str, context: list[dict]) -> dict:
        """
        Args:
            query:   The concept or question to hypothesize about.
            context: Cross-domain search results from ChromaDB.

        Returns:
            dict with keys: scepter, domain, hypothesis, confidence,
                            structural_pattern, validation_question,
                            domains_connected
        """
        ctx_text = self._format_context(context)

        prompt = f"""You are {self.name}, the {self.domain} Scepter of NOUS — an AI that discovers novel knowledge by connecting distant fields.

Your epistemic lens: {self.lens}
Your sub-domains: {', '.join(self.subdomains)}

QUERY: {query}

CROSS-DOMAIN CONTEXT (concepts from all fields related to this query):
{ctx_text}

From your domain's unique perspective, generate a bold hypothesis that:
1. Applies your domain's core concepts and methods to this query
2. Connects your domain to at least ONE distant, unexpected field
3. Makes a claim that could actually be tested or computed

Respond ONLY with JSON:
{{
  "scepter": "{self.name}",
  "domain": "{self.domain}",
  "structural_pattern": "The deep structural pattern you see from your domain's lens",
  "hypothesis": "Your domain's hypothesis — bold, specific, cross-domain",
  "confidence": 0.0-1.0,
  "validation_question": "One concrete experiment or analysis that could test this",
  "domains_connected": ["domain1", "domain2"]
}}"""

        try:
            result = generate_json(prompt, model=self.model)
            result["scepter"] = self.name
            result["domain"]  = self.domain
            result["query"]   = query
            return result
        except Exception as ex:
            return {
                "scepter": self.name, "domain": self.domain,
                "hypothesis": f"[parse error: {ex}]",
                "confidence": 0.0, "query": query,
            }

    # ------------------------------------------------------------------ #
    #  Critique another Scepter's hypothesis                               #
    # ------------------------------------------------------------------ #
    def critique(self, own_hypothesis: dict, other: dict) -> dict:
        """
        Evaluate another Scepter's hypothesis through this Scepter's lens.

        Returns:
            dict with keys: from_scepter, to_scepter, link_type,
                            strength, rationale, challenge (optional)
        """
        prompt = f"""You are {self.name}, the {self.domain} Scepter of NOUS.
Your lens: {self.lens}

YOUR OWN HYPOTHESIS:
{own_hypothesis.get('hypothesis', '(none yet)')}

OTHER SCEPTER'S HYPOTHESIS ({other.get('scepter','?')} — {other.get('domain','?')}):
{other.get('hypothesis', '')}

Evaluate the other hypothesis from your domain's perspective:
- "supports":    It reinforces your hypothesis; if true, both are more likely true
- "contradicts": It conflicts with yours; one being true challenges the other
- "extends":     It is a special case, elaboration, or logical consequence of yours

Be rigorous. If the other hypothesis makes a claim your domain can directly challenge or validate, say so.

Respond ONLY with JSON:
{{
  "from_scepter": "{self.name}",
  "to_scepter": "{other.get('scepter','?')}",
  "link_type": "supports" | "contradicts" | "extends",
  "strength": 0.0-1.0,
  "rationale": "One sentence explaining the relationship from your domain's view",
  "challenge": "Optional: a specific empirical challenge your domain raises against the other hypothesis"
}}"""

        try:
            result = generate_json(prompt, model=self.model)
            result["from_scepter"] = self.name
            result["to_scepter"]   = other.get("scepter", "?")
            return result
        except Exception as ex:
            return {
                "from_scepter": self.name, "to_scepter": other.get("scepter", "?"),
                "link_type": "extends", "strength": 0.5,
                "rationale": f"[parse error: {ex}]",
            }
