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
    python nous.py debate "concept"    # Run full 5-Scepter debate + NOUS synthesis (refuses an identical repeated question; --force)
    python nous.py debates             # Show recent debate sessions

  Structure (domain-free abstractions; requires Ollama)
    python nous.py abstract [limit]    # Abstract indexed articles into structures + motifs

  Evaluation
    python nous.py evaluate [model]    # Score all hypotheses with 5 LLM judges + novelty
    python nous.py evaluate --judges m1,m2,m3   # Use several different models as judges
    python nous.py rate <name>         # Rate hypotheses yourself with the same rubric
    python nous.py scores              # Ranking of hypotheses
    python nous.py reliability         # Inter-rater agreement (ICC): LLM judges, and human vs. LLM
    python nous.py litcheck            # Compare each hypothesis with the closest OpenAlex papers
    python nous.py calibrate           # Measure lit-novelty of 10 famous ideas to set a "known" cut-off
    python nous.py combocheck          # Split each hypothesis into concepts; is the PAIRING rare in OpenAlex?
    python nous.py combocheck --report # Show all stored combination verdicts (no API calls)

  OpenAlex (set OPENALEX_API_KEY; free key gives $1/day)
    python nous.py openalex [n]        # Collect n cited + n recent works from each of 26 fields
    python nous.py voids [--refresh]   # Literature-grounded void zones between fields (lift)
    python nous.py fillvoids --openalex  # Collect works bridging the strongest void pairs
    python nous.py trends              # Growth & acceleration of each field

  Autonomous loop (needs OPENALEX_API_KEY and Ollama)
    python nous.py autoloop [cycles] [--per-pair N] [--abstract]   # void pair -> hypothesis -> checks -> accept/reject
    python nous.py autoloop --plan     # which field pairs would be explored next (no API calls); --min-shared N
    python nous.py autoloop --report   # log of past loop runs

  Maintenance
    python nous.py migrate-domains     # Rename legacy domains (physics, ...) to OpenAlex fields
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


def cmd_fillvoids(openalex: bool = False):
    if openalex:
        from nous.collector.openalex import collect_void_bridges
        collect_void_bridges()
        return
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


def cmd_debate(query: str, model: str = DEFAULT_MODEL, force: bool = False):
    from nous.core import debate, previous_debates
    prev = previous_debates(query)
    if prev and not force:
        print(f"This question was already debated (debate #{', #'.join(map(str, prev))}). "
              "Repeating it mostly yields near-duplicate hypotheses; use a different question, "
              "or add --force to run it again.")
        return
    debate(query, model=model)


def cmd_debates():
    from nous.core import show_debates
    show_debates()


def cmd_abstract(limit: int | None = None, model: str = DEFAULT_MODEL):
    from nous.engine.structure import build_structure_index
    build_structure_index(model=model, limit=limit)


def cmd_evaluate(model: str = DEFAULT_MODEL, judges: list[str] | None = None):
    from nous.evaluation.runner import evaluate, print_scores
    n = evaluate(model=model, judge_models=judges)
    print(f"\n{n} new judge ratings stored.")
    print_scores()


def cmd_litcheck():
    from nous.evaluation.literature import check_literature
    check_literature()


def cmd_combocheck(model: str = DEFAULT_MODEL, report_only: bool = False):
    from nous.evaluation.combination import check_combinations, print_report
    if report_only:
        print_report()
    else:
        check_combinations(model=model)


def cmd_calibrate():
    from nous.evaluation.literature import calibrate
    calibrate()


def cmd_autoloop(args: list[str]):
    from nous import loop
    min_shared = loop.MIN_SHARED
    if "--min-shared" in args:
        i = args.index("--min-shared")
        min_shared = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--plan" in args:
        loop.print_plan(min_shared=min_shared)
        return
    if "--report" in args:
        loop.print_report()
        return
    per_pair = 10
    if "--per-pair" in args:
        i = args.index("--per-pair")
        per_pair = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    rest = [a for a in args if not a.startswith("--")]
    loop.autoloop(cycles=int(rest[0]) if rest else 1, per_pair=per_pair,
                  abstract="--abstract" in args, min_shared=min_shared)


def cmd_migrate_domains():
    from nous.migrate import migrate_domains
    migrate_domains()


def cmd_rate(name: str):
    from nous.evaluation.runner import human_rate
    human_rate(name)


def cmd_scores():
    from nous.evaluation.runner import print_scores
    print_scores()


def cmd_reliability(prefix: str | None = None):
    from nous.evaluation.runner import print_reliability
    print_reliability(prefix)


def cmd_openalex(per_field: int = 20):
    from nous.collector.openalex import collect_fields
    collect_fields(per_field=per_field)


def cmd_voids(refresh: bool = False, top: int = 20):
    from nous.collector.openalex import field_cooccurrence, load_field_links
    if refresh or not load_field_links():
        print("Computing field co-occurrence across OpenAlex...")
        field_cooccurrence()
    links = load_field_links()
    print("\n=== LITERATURE VOID ZONES (lowest lift = fields that rarely meet) ===\n")
    for l in links[:top]:
        print(f"  lift={l['lift']:.3f}  obs={l['observed']:>9,}  "
              f"{l['name_a']}  ×  {l['name_b']}")
    print("\n=== HIGHWAYS (highest lift) ===\n")
    for l in links[-5:][::-1]:
        print(f"  lift={l['lift']:.3f}  obs={l['observed']:>9,}  "
              f"{l['name_a']}  ×  {l['name_b']}")


def cmd_trends():
    from nous.collector.openalex import field_trends
    field_trends()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "collect":
        cmd_collect()
    elif cmd == "fillvoids":
        cmd_fillvoids(openalex="--openalex" in args)
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
        force = "--force" in args
        args = [a for a in args if a != "--force"]
        if len(args) < 2:
            print("Usage: python nous.py debate <concept> [model] [--force]")
            return
        model = args[2] if len(args) > 2 else DEFAULT_MODEL
        cmd_debate(args[1], model, force)
    elif cmd == "debates":
        cmd_debates()
    elif cmd == "abstract":
        limit = int(args[1]) if len(args) > 1 else None
        cmd_abstract(limit)
    elif cmd == "evaluate":
        judges = None
        if "--judges" in args:
            i = args.index("--judges")
            judges = [m for m in args[i + 1].split(",") if m] if i + 1 < len(args) else None
            args = args[:i] + args[i + 2:]
        model = args[1] if len(args) > 1 else DEFAULT_MODEL
        cmd_evaluate(model, judges)
    elif cmd == "litcheck":
        cmd_litcheck()
    elif cmd == "calibrate":
        cmd_calibrate()
    elif cmd == "combocheck":
        rest = [a for a in args[1:] if not a.startswith("--")]
        cmd_combocheck(rest[0] if rest else DEFAULT_MODEL, report_only="--report" in args)
    elif cmd == "autoloop":
        cmd_autoloop(args[1:])
    elif cmd == "migrate-domains":
        cmd_migrate_domains()
    elif cmd == "rate":
        if len(args) < 2:
            print("Usage: python nous.py rate <your-name>")
            return
        cmd_rate(args[1])
    elif cmd == "scores":
        cmd_scores()
    elif cmd == "reliability":
        cmd_reliability(args[1] if len(args) > 1 else None)
    elif cmd == "openalex":
        cmd_openalex(int(args[1]) if len(args) > 1 else 20)
    elif cmd == "voids":
        cmd_voids(refresh="--refresh" in args)
    elif cmd == "trends":
        cmd_trends()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
