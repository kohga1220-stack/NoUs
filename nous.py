"""
Nous CLI — entry point.

Usage:
    python nous.py collect             # Collect Wikipedia + arXiv
    python nous.py fillvoids           # Collect targeted bridge data for Void Zones
    python nous.py embed               # Build vector index
    python nous.py query "concept"     # Find cross-domain connections [text|graph|both]
    python nous.py graph               # Build full knowledge graph + void/bridge analysis
    python nous.py hyp   "concept"     # Generate hypothesis (requires Ollama)
    python nous.py memory              # Show recent hypotheses
    python nous.py link                # Auto-analyze & store links between all hypotheses
    python nous.py network             # Show the hypothesis relationship network
    python nous.py sync                # Sync hypotheses back into knowledge base
    python nous.py debate "concept"    # Run full 5-Scepter debate + NOUS synthesis
    python nous.py debates             # Show recent debate sessions
"""
import sys

from nous.config import DEFAULT_MODEL


def cmd_collect():
    from nous.collector.wikipedia import collect as wiki_collect
    from nous.collector.arxiv import collect as arxiv_collect
    from nous.collector.wikipedia import init_db
    init_db()
    print("=== Wikipedia ===")
    wiki_collect()
    print("\n=== arXiv ===")
    arxiv_collect()


def cmd_fillvoids():
    from nous.collector.wikipedia import collect_void_bridges as wiki_bridges
    from nous.collector.arxiv import collect_void_bridges as arxiv_bridges
    print("=== Wikipedia bridge articles ===")
    wiki_bridges()
    print("\n=== arXiv bridge papers ===")
    arxiv_bridges()


def cmd_embed():
    from nous.engine.embedder import embed_all
    embed_all()


def cmd_query(concept: str, output: str = "text"):
    from nous.engine.connector import find_cross_domain_connections
    from nous.output.graph import render_text, render_query_graph

    results = find_cross_domain_connections(concept, n_results=10)

    if output in ("text", "both"):
        print(render_text(results, concept))

    if output in ("graph", "both"):
        path = render_query_graph(results, concept)
        print(f"Graph saved: {path}")


def cmd_graph(threshold: float = 0.25):
    import webbrowser
    from pathlib import Path
    from nous.engine.knowledge_graph import build_graph, find_bridge_nodes, find_void_zones
    from nous.output.graph import render_full_graph, render_void_report

    print("Building full knowledge graph...")
    G = build_graph(threshold=threshold)

    print("Identifying bridge nodes...")
    bridges = find_bridge_nodes(G, top_n=10)

    print("Identifying void zones...")
    voids = find_void_zones(G)

    print(render_void_report(voids, bridges))

    path = render_full_graph(G, bridges)
    print(f"\nFull graph saved: {path}")
    if Path(path).exists():
        webbrowser.open(Path(path).resolve().as_uri())


def cmd_hypothesis(concept: str, model: str = DEFAULT_MODEL):
    import json
    from nous.engine.hypothesis import generate_hypothesis
    result = generate_hypothesis(concept, model=model)
    print("\n=== Nous Hypothesis ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_memory():
    from nous.memory.store import get_recent_hypotheses
    hyps = get_recent_hypotheses(10)
    if not hyps:
        print("No hypotheses stored yet.")
        return
    for h in hyps:
        print(f"\n[{h['timestamp']}] Query: {h['query']}")
        print(f"  Hypothesis: {h['hypothesis'][:200]}...")
        print(f"  Confidence: {h['confidence']}")


def cmd_link(model: str = DEFAULT_MODEL):
    from nous.memory.linker import link_all
    link_all(model=model)


def cmd_network():
    from nous.memory.linker import show_network
    show_network()


def cmd_sync():
    from nous.memory.store import sync_hypotheses_to_chroma
    sync_hypotheses_to_chroma()


def cmd_debate(query: str, model: str = DEFAULT_MODEL):
    from nous.core import debate
    debate(query, model=model)


def cmd_debates():
    from nous.core import show_debates
    show_debates()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "collect":
        cmd_collect()
    elif cmd == "fillvoids":
        cmd_fillvoids()
    elif cmd == "embed":
        cmd_embed()
    elif cmd == "query":
        if len(args) < 2:
            print("Usage: python nous.py query <concept> [text|graph|both]")
            return
        output = args[2] if len(args) > 2 else "text"
        cmd_query(args[1], output)
    elif cmd == "graph":
        threshold = float(args[1]) if len(args) > 1 else 0.25
        cmd_graph(threshold)
    elif cmd == "hyp":
        if len(args) < 2:
            print("Usage: python nous.py hyp <concept> [model]")
            return
        model = args[2] if len(args) > 2 else DEFAULT_MODEL
        cmd_hypothesis(args[1], model)
    elif cmd == "memory":
        cmd_memory()
    elif cmd == "link":
        model = args[1] if len(args) > 1 else DEFAULT_MODEL
        cmd_link(model)
    elif cmd == "network":
        cmd_network()
    elif cmd == "sync":
        cmd_sync()
    elif cmd == "debate":
        if len(args) < 2:
            print("Usage: python nous.py debate <concept> [model]")
            return
        model = args[2] if len(args) > 2 else DEFAULT_MODEL
        cmd_debate(args[1], model)
    elif cmd == "debates":
        cmd_debates()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
