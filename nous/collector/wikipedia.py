"""
Fetch articles from Wikipedia API and store to SQLite + raw text files.
"""
import sqlite3
import time
import requests

from nous.config import DATA_DIR, DB_PATH, RAW_PATH

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "Nous/0.1 (personal knowledge research tool; contact: nous-project)"}

DOMAIN_SEEDS = {
    "physics":        ["quantum mechanics", "thermodynamics", "relativity", "phase transition"],
    "biology":        ["evolution", "genetics", "immune system", "neuroscience"],
    "mathematics":    ["graph theory", "topology", "information theory", "chaos theory"],
    "psychology":     ["cognitive bias", "habit formation", "gestalt psychology"],
    "economics":      ["game theory", "behavioral economics", "network effects"],
    "philosophy":     ["epistemology", "emergence", "reductionism", "systems theory"],
    "linguistics":    ["language acquisition", "semiotics", "morphology"],
    "computer_science": ["machine learning", "complexity theory", "cellular automata"],
    "history":        ["scientific revolution", "cultural diffusion", "collapse of civilizations"],
    "sociology":      ["social network", "collective behavior", "paradigm shift"],
}

# Targeted bridge seeds for Void Zones — list of (domain, title) pairs
VOID_BRIDGE_SEEDS: list[tuple[str, str]] = [
    # computer_science ↔ history
    ("history",           "digital humanities"),
    ("history",           "history of computing"),
    ("history",           "information age"),
    ("history",           "cliodynamics"),
    ("computer_science",  "computational social science"),
    # computer_science ↔ philosophy
    ("philosophy",        "philosophy of mind"),
    ("philosophy",        "computationalism"),
    ("philosophy",        "Chinese room"),
    ("philosophy",        "philosophy of economics"),
    ("computer_science",  "Turing test"),
    ("computer_science",  "artificial intelligence"),
    # economics ↔ mathematics
    ("economics",         "econophysics"),
    ("economics",         "mathematical economics"),
    ("economics",         "operations research"),
    ("mathematics",       "stochastic process"),
    # economics ↔ linguistics
    ("linguistics",       "computational linguistics"),
    ("linguistics",       "semiotics"),
    # economics ↔ philosophy
    ("philosophy",        "linguistic relativity"),
    ("philosophy",        "philosophy of language"),
    # history ↔ physics
    ("history",           "entropy and civilization"),
    # psychology ↔ biology
    ("psychology",        "behavioral neuroscience"),
    ("psychology",        "evolutionary psychology"),
    ("psychology",        "neuroplasticity"),
    ("psychology",        "epigenetics and behavior"),
    # psychology ↔ physics / mathematics
    ("psychology",        "psychophysics"),
    ("psychology",        "computational neuroscience"),
    ("psychology",        "mathematical psychology"),
    ("psychology",        "signal detection theory"),
    # psychology ↔ history
    ("psychology",        "collective memory"),
    ("psychology",        "social psychology"),
    ("psychology",        "cultural psychology"),
    ("psychology",        "history of psychology"),
    # psychology ↔ linguistics
    ("psychology",        "psycholinguistics"),
    ("psychology",        "cognitive linguistics"),
    ("linguistics",       "language and thought"),
    # psychology ↔ sociology
    ("psychology",        "social influence"),
    ("psychology",        "conformity"),
    # philosophy ↔ sociology
    ("philosophy",        "social ontology"),
    ("philosophy",        "philosophy of social science"),
    ("philosophy",        "critical theory"),
    ("philosophy",        "structuralism"),
    ("sociology",         "social constructionism"),
    ("sociology",         "phenomenological sociology"),
    ("sociology",         "normative social theory"),
    ("sociology",         "Frankfurt School"),
    # history ↔ linguistics
    ("linguistics",       "historical linguistics"),
    ("linguistics",       "language change"),
    # linguistics ↔ philosophy
    ("linguistics",       "speech act theory"),
    ("philosophy",        "truth-conditional semantics"),
]


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT UNIQUE,
            domain      TEXT,
            summary     TEXT,
            full_text   TEXT,
            page_id     INTEGER,
            url         TEXT,
            fetched_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS explorations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT,
            domain_a    TEXT,
            domain_b    TEXT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS hypotheses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            exploration_id  INTEGER,
            hypothesis_text TEXT,
            confidence      REAL,
            source_ids      TEXT,
            validated       INTEGER DEFAULT 0,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS hypothesis_links (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            hyp_id_a    INTEGER,
            hyp_id_b    INTEGER,
            link_type   TEXT,
            rationale   TEXT,
            strength    REAL
        );
    """)
    conn.commit()
    conn.close()


def fetch_article(title: str) -> dict | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "exintro": False,
        "explaintext": True,
        "inprop": "url",
        "format": "json",
        "redirects": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return {
            "title":    page.get("title", title),
            "page_id":  page.get("pageid"),
            "url":      page.get("fullurl", ""),
            "text":     page.get("extract", ""),
            "summary":  page.get("extract", "")[:500],
        }
    except Exception as e:
        print(f"  [WARN] {title}: {e}")
        return None


def save_article(domain: str, article: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO articles (title, domain, summary, full_text, page_id, url)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        article["title"],
        domain,
        article["summary"],
        article["text"],
        article["page_id"],
        article["url"],
    ))
    conn.commit()
    conn.close()

    raw_file = RAW_PATH / f"wiki_{article['page_id']}.txt"
    raw_file.write_text(article["text"], encoding="utf-8")


def collect(domains: list[str] | None = None, limit_per_domain: int = 5):
    init_db()
    RAW_PATH.mkdir(parents=True, exist_ok=True)

    targets = {k: v for k, v in DOMAIN_SEEDS.items() if domains is None or k in domains}
    total = 0

    for domain, seeds in targets.items():
        print(f"\n[{domain}]")
        for seed in seeds[:limit_per_domain]:
            print(f"  Fetching: {seed}")
            article = fetch_article(seed)
            if article:
                save_article(domain, article)
                total += 1
                print(f"  OK  {article['title']} ({len(article['text'])} chars)")
            time.sleep(1.5)

    print(f"\nDone. {total} articles collected.")


def collect_void_bridges():
    """Fetch targeted bridge articles for all Void Zone domain pairs."""
    init_db()
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    total = 0
    seen = set()

    for domain, title in VOID_BRIDGE_SEEDS:
        if title in seen:
            continue
        seen.add(title)
        print(f"  [{domain}] {title}")
        article = fetch_article(title)
        if article:
            save_article(domain, article)
            total += 1
            print(f"    OK  {article['title']} ({len(article['text'])} chars)")
        else:
            print("    SKIP (not found or error)")
        time.sleep(1.5)

    print(f"\nDone. {total} bridge articles collected.")


if __name__ == "__main__":
    collect()
