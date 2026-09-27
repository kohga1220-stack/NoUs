"""
Structural abstraction — the part that makes Nous more than semantic search.

Each article is rewritten by the LLM into a *domain-free* description of its
structure (mechanism, dynamics, mathematical form) plus a set of motifs from a
fixed vocabulary. Two texts from unrelated fields can then match because they
share structure, even when they share no vocabulary.

structural_sim = 0.5 * cosine(abstraction embeddings) + 0.5 * Jaccard(motifs)
"""
from __future__ import annotations
import json
import sqlite3

from nous.config import CHROMA_PATH, DB_PATH, DEFAULT_MODEL

PROMPT_VERSION = "structure-v1"
STRUCT_COLLECTION = "nous_structures"

# Controlled vocabulary of domain-independent structural motifs.
MOTIFS: list[str] = [
    "positive_feedback", "negative_feedback", "threshold", "phase_transition",
    "tipping_point", "cascade", "power_law", "scale_free_network", "small_world",
    "hierarchy", "modularity", "diffusion", "contagion", "selection", "variation",
    "competition", "cooperation", "arms_race", "equilibrium", "multiple_equilibria",
    "oscillation", "path_dependence", "lock_in", "emergence", "self_organization",
    "optimization_under_constraint", "tradeoff", "exploration_exploitation",
    "information_compression", "signaling", "noise", "redundancy", "bottleneck",
    "symmetry_breaking", "conservation_law", "entropy_increase", "adaptation",
    "delay", "memory", "recursion", "coupling", "synchronization",
]
_MOTIF_SET = set(MOTIFS)


def normalize_motifs(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for m in raw:
        key = str(m).strip().lower().replace(" ", "_").replace("-", "_")
        if key in _MOTIF_SET and key not in out:
            out.append(key)
    return out


def motif_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def structural_sim(emb_a: list[float], motifs_a: list[str],
                   emb_b: list[float], motifs_b: list[str]) -> float:
    from nous.engine.connector import cosine_sim
    return 0.5 * cosine_sim(emb_a, emb_b) + 0.5 * motif_jaccard(motifs_a, motifs_b)


def abstract_structure(text: str, model: str = DEFAULT_MODEL) -> dict:
    """LLM → {"structure": str, "motifs": [..]}  (motifs restricted to MOTIFS)."""
    from nous.llm import generate_json

    prompt = f"""Rewrite the following text as a DOMAIN-FREE description of its underlying structure.

Rules:
- Do NOT use any field-specific nouns (no "neuron", "market", "protein", "language", ...).
  Use abstract roles instead: "units", "agents", "a quantity", "a resource", "a signal", "a population".
- Describe the mechanism, dynamics, or mathematical form in 2-3 sentences.
- Then choose 1-5 motifs from this list ONLY: {", ".join(MOTIFS)}

TEXT:
{text[:1500]}

Respond ONLY with JSON:
{{"structure": "...", "motifs": ["...", "..."]}}"""

    data = generate_json(prompt, model=model)
    return {
        "structure": str(data.get("structure", "")).strip(),
        "motifs": normalize_motifs(data.get("motifs")),
    }


# ------------------------------------------------------------------ #
#  Storage                                                            #
# ------------------------------------------------------------------ #

def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structures (
            doc_id         TEXT PRIMARY KEY,   -- same id as in the nous_knowledge collection
            structure      TEXT,
            motifs         TEXT,               -- JSON list
            model          TEXT,
            prompt_version TEXT,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_struct_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(STRUCT_COLLECTION)


def load_structures(ids: list[str] | None = None) -> dict[str, dict]:
    """{doc_id: {"embedding": [...], "motifs": [...], "structure": str}}"""
    try:
        col = get_struct_collection()
        if col.count() == 0:
            return {}
        res = col.get(ids=ids, include=["embeddings", "metadatas", "documents"]) if ids \
            else col.get(include=["embeddings", "metadatas", "documents"])
    except Exception:
        return {}
    out = {}
    for doc_id, emb, meta, doc in zip(res["ids"], res["embeddings"], res["metadatas"], res["documents"]):
        motifs = [m for m in (meta.get("motifs") or "").split(",") if m]
        out[doc_id] = {"embedding": list(emb), "motifs": motifs, "structure": doc}
    return out


def build_structure_index(model: str = DEFAULT_MODEL, limit: int | None = None,
                          verbose: bool = True) -> int:
    """Abstract every indexed article that has no structure yet (one LLM call each)."""
    from nous.engine.embedder import get_collection, get_model

    base = get_collection()
    struct = get_struct_collection()
    done = set(struct.get(include=[])["ids"])

    res = base.get(include=["documents", "metadatas"])
    todo = [(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
            if i not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        if verbose:
            print("All articles already abstracted.")
        return 0

    enc = get_model()
    conn = sqlite3.connect(DB_PATH)
    _ensure_table(conn)
    n = 0
    for doc_id, doc, meta in todo:
        try:
            s = abstract_structure(doc, model=model)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] {meta.get('title')}: {ex}")
            continue
        if not s["structure"]:
            continue
        struct.add(
            ids=[doc_id],
            embeddings=[enc.encode(s["structure"]).tolist()],
            documents=[s["structure"]],
            metadatas=[{"title": meta["title"], "domain": meta["domain"],
                        "motifs": ",".join(s["motifs"])}],
        )
        conn.execute("""INSERT OR REPLACE INTO structures
                        (doc_id, structure, motifs, model, prompt_version)
                        VALUES (?, ?, ?, ?, ?)""",
                     (doc_id, s["structure"], json.dumps(s["motifs"]), model, PROMPT_VERSION))
        conn.commit()
        n += 1
        if verbose:
            print(f"  [{meta['domain']}] {meta['title'][:50]:<50} → {', '.join(s['motifs'])}")
    conn.close()
    if verbose:
        print(f"Done. {n} structures added.")
    return n
