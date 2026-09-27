"""
Fetch papers from arXiv API and store to SQLite + raw text files.
"""
import sqlite3
import time
import xml.etree.ElementTree as ET
import requests

from nous.config import DB_PATH, RAW_PATH

ARXIV_API = "https://export.arxiv.org/api/query"

DOMAIN_QUERIES = {
    "physics":          "ti:emergence OR ti:complexity OR ti:entropy",
    "mathematics":      "ti:topology OR ti:information OR ti:graph",
    "computer_science": "ti:network OR ti:complexity OR ti:learning",
    "biology":          "ti:network OR ti:evolution OR ti:dynamics",
    "economics":        "ti:network OR ti:behavior OR ti:equilibrium",
}

# Void Zone bridge queries — targets underconnected domain pairs
VOID_BRIDGE_QUERIES: list[tuple[str, str]] = [
    # computer_science ↔ history
    ("history",           "ti:computational AND ti:history"),
    ("history",           "ti:digital AND ti:humanities"),
    ("computer_science",  "ti:historical AND ti:simulation"),
    # computer_science ↔ philosophy
    ("philosophy",        "ti:computationalism OR (ti:philosophy AND ti:mind)"),
    ("philosophy",        "ti:philosophy AND ti:language"),
    ("computer_science",  "ti:Turing AND ti:computation"),
    # economics ↔ mathematics
    ("economics",         "ti:econophysics OR (ti:mathematical AND ti:economics)"),
    ("mathematics",       "ti:stochastic AND ti:economics"),
    # economics ↔ linguistics
    ("economics",         "ti:language AND ti:economics"),
    ("linguistics",       "ti:computational AND ti:linguistics"),
    # history ↔ physics
    ("history",           "ti:entropy AND ti:history"),
    ("physics",           "ti:physics AND ti:social"),
    # psychology ↔ biology
    ("psychology",        "ti:evolutionary AND ti:psychology"),
    ("psychology",        "ti:behavioral AND ti:neuroscience"),
    ("psychology",        "ti:neuroplasticity"),
    # psychology ↔ physics / mathematics
    ("psychology",        "ti:psychophysics"),
    ("psychology",        "ti:computational AND ti:neuroscience"),
    ("psychology",        "ti:mathematical AND ti:psychology"),
    # psychology ↔ history
    ("psychology",        "ti:collective AND ti:memory"),
    ("psychology",        "ti:cultural AND ti:cognition"),
    # psychology ↔ linguistics
    ("psychology",        "ti:psycholinguistics OR (ti:language AND ti:cognition)"),
    # philosophy ↔ sociology
    ("philosophy",        "ti:social AND ti:ontology"),
    ("philosophy",        "ti:critical AND ti:theory"),
    ("sociology",         "ti:social AND ti:constructionism"),
    ("sociology",         "ti:normative AND ti:social"),
    # linguistics ↔ philosophy
    ("linguistics",       "ti:semantics AND ti:philosophy"),
    ("philosophy",        "ti:speech AND ti:act"),
]

NS = "http://www.w3.org/2005/Atom"


def parse_arxiv_id(entry_url: str) -> str:
    """
    'http://arxiv.org/abs/2401.01735v1'      -> '2401.01735v1'
    'http://arxiv.org/abs/cond-mat/0607151v1' -> 'cond-mat/0607151v1'
    Old-style IDs keep their archive prefix so they cannot collide.
    """
    url = entry_url.strip()
    if "/abs/" in url:
        return url.split("/abs/", 1)[1]
    return url.rsplit("/", 1)[-1]


def fetch_papers(domain: str, query: str, max_results: int = 10) -> list[dict]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    try:
        r = requests.get(ARXIV_API, params=params, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        papers = []
        for entry in root.findall(f"{{{NS}}}entry"):
            url     = entry.find(f"{{{NS}}}id").text.strip()
            title   = entry.find(f"{{{NS}}}title").text.strip().replace("\n", " ")
            summary = entry.find(f"{{{NS}}}summary").text.strip().replace("\n", " ")
            papers.append({
                "arxiv_id": parse_arxiv_id(url),
                "title":    title,
                "summary":  summary[:500],
                "text":     f"{title}\n\n{summary}",
                "url":      url,
                "domain":   domain,
            })
        return papers
    except Exception as e:
        print(f"  [WARN] {domain}: {e}")
        return []


def save_paper(paper: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO articles (title, domain, summary, full_text, page_id, url)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        paper["title"],
        paper["domain"],
        paper["summary"],
        paper["text"],
        None,
        paper["url"],
    ))
    conn.commit()
    conn.close()

    safe_id = paper["arxiv_id"].replace("/", "_")
    raw_file = RAW_PATH / f"arxiv_{safe_id}.txt"
    raw_file.write_text(paper["text"], encoding="utf-8")


def collect(domains: list[str] | None = None, max_per_domain: int = 10):
    RAW_PATH.mkdir(parents=True, exist_ok=True)

    targets = {k: v for k, v in DOMAIN_QUERIES.items() if domains is None or k in domains}
    total = 0

    for domain, query in targets.items():
        print(f"\n[{domain}]")
        papers = fetch_papers(domain, query, max_per_domain)
        for p in papers:
            save_paper(p)
            total += 1
            print(f"  OK  {p['title'][:60]}")
        time.sleep(1.0)

    print(f"\nDone. {total} papers collected.")


def collect_void_bridges(max_per_query: int = 5):
    """Fetch targeted arXiv papers for all Void Zone domain pairs."""
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    total = 0

    for domain, query in VOID_BRIDGE_QUERIES:
        print(f"  [{domain}] {query[:60]}")
        papers = fetch_papers(domain, query, max_per_query)
        for p in papers:
            save_paper(p)
            total += 1
            print(f"    OK  {p['title'][:55]}")
        time.sleep(1.0)

    print(f"\nDone. {total} bridge papers collected.")


if __name__ == "__main__":
    collect()
