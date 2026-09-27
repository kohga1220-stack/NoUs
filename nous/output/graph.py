"""
Visualize the Nous knowledge graph.
Two modes:
  - query_graph:  shows connections for a single query concept
  - full_graph:   renders the entire knowledge network with bridge/void analysis
"""
import networkx as nx

from nous.config import DATA_DIR

OUTPUT_DIR = DATA_DIR

DOMAIN_COLORS = {
    "physics":          "#e74c3c",
    "biology":          "#2ecc71",
    "mathematics":      "#3498db",
    "psychology":       "#9b59b6",
    "economics":        "#f39c12",
    "philosophy":       "#1abc9c",
    "linguistics":      "#e67e22",
    "computer_science": "#34495e",
    "history":          "#95a5a6",
    "sociology":        "#d35400",
    "hypothesis":       "#f1c40f",
}
DEFAULT_COLOR = "#aaaaaa"

# Fallback for domains without their own color (e.g. OpenAlex fields): color by Scepter
SCEPTER_COLORS = {
    "Scepter-H": "#9b59b6",
    "Scepter-S": "#f39c12",
    "Scepter-N": "#3498db",
    "Scepter-A": "#34495e",
    "Scepter-I": "#2ecc71",
}


def domain_color(domain: str) -> str:
    from nous.domains import scepter_for
    return DOMAIN_COLORS.get(domain) or SCEPTER_COLORS.get(scepter_for(domain), DEFAULT_COLOR)

PYVIS_OPTIONS = """{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -8000,
      "springLength": 200,
      "springConstant": 0.04
    },
    "stabilization": {"iterations": 150}
  },
  "edges": {
    "color": {"inherit": "both"},
    "smooth": {"type": "continuous"},
    "scaling": {"min": 1, "max": 8}
  },
  "nodes": {
    "font": {"size": 13, "color": "#eeeeee"},
    "borderWidth": 2
  },
  "interaction": {"hover": true, "tooltipDelay": 100}
}"""


def _try_pyvis():
    try:
        from pyvis.network import Network
        return Network
    except ImportError:
        return None


def _add_nodes_edges(net, G: nx.Graph, highlight_ids: set | None = None):
    for node_id, data in G.nodes(data=True):
        domain = data.get("domain", "unknown")
        color  = domain_color(domain)
        size   = 15 + data.get("centrality", 0) * 80

        if highlight_ids and node_id in highlight_ids:
            color = "#ffffff"
            size  = max(size, 28)

        net.add_node(
            node_id,
            label   = data["title"][:35],
            title   = f"[{domain}] {data['title']}",
            color   = color,
            size    = size,
            font    = {"color": "#eeeeee"},
        )

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.3)
        net.add_edge(u, v, value=w, title=f"nous_score: {w:.3f}")


def render_full_graph(
    G: nx.Graph,
    bridges: list[dict],
    output_file: str = "nous_full_graph.html",
) -> str:
    Network = _try_pyvis()
    if not Network:
        return "pyvis not installed"

    bridge_ids = {b["id"] for b in bridges[:5]}

    net = Network(
        height="800px", width="100%",
        bgcolor="#0d1117", font_color="white",
        directed=False,
    )
    _add_nodes_edges(net, G, highlight_ids=bridge_ids)
    net.set_options(PYVIS_OPTIONS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / output_file
    net.save_graph(str(out))
    return str(out)


def render_query_graph(
    connections: list[dict],
    query: str,
    output_file: str = "nous_query_graph.html",
) -> str:
    Network = _try_pyvis()
    if not Network:
        return "pyvis not installed"

    net = Network(
        height="700px", width="100%",
        bgcolor="#0d1117", font_color="white",
    )

    net.add_node(
        "query",
        label=query[:40],
        title=f"Query: {query}",
        color="#ffffff",
        size=35,
    )

    for r in connections:
        domain = r.get("domain", "unknown")
        color  = domain_color(domain)
        score  = r.get("nous_score", 0.3)
        net.add_node(
            r["title"],
            label=r["title"][:35],
            title=f"[{domain}] score={score:.3f}\n{r.get('summary','')[:120]}",
            color=color,
            size=10 + max(score, 0) * 40,
        )
        net.add_edge("query", r["title"], value=max(score, 0.01), title=f"{score:.3f}")

    net.set_options(PYVIS_OPTIONS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / output_file
    net.save_graph(str(out))
    return str(out)


def render_text(connections: list[dict], query: str) -> str:
    lines = [f"\n=== Knowledge Network: '{query}' ===\n"]
    if connections and connections[0].get("source_domain"):
        lines.append(f"(query home domain: {connections[0]['source_domain']} — penalized)\n")
    by_domain: dict[str, list] = {}
    for r in connections:
        by_domain.setdefault(r["domain"], []).append(r)

    for domain, items in sorted(by_domain.items()):
        lines.append(f"[{domain.upper()}]")
        for item in items:
            score = item.get("nous_score", 0)
            st = item.get("structural_sim")
            st_s = f", structural: {st:.3f}" if st is not None else ""
            lines.append(f"  • {item['title']}  (nous_score: {score:.3f}{st_s})")
            lines.append(f"    {item.get('summary','')[:120]}...")
        lines.append("")

    return "\n".join(lines)


def render_void_report(voids: list[dict], bridges: list[dict]) -> str:
    lines = ["\n=== VOID ZONES (unexplored domain pairs) ===\n"]
    for v in voids[:10]:
        bar = "░" * v["connections"] if v["connections"] else "— NONE —"
        lines.append(f"  {v['domain_a']:20s} ↔ {v['domain_b']:20s}  [{bar}] {v['connections']} links")

    lines.append("\n=== BRIDGE NODES (cross-domain connectors) ===\n")
    for b in bridges:
        domains = ", ".join(b["domains_linked"])
        lines.append(f"  [{b['domain']}] {b['title']}")
        lines.append(f"    bridges → {domains}  (score: {b['bridge_score']:.5f})")

    return "\n".join(lines)
