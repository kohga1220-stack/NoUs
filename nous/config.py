"""
Shared paths and defaults for all Nous modules.
"""
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data"
DB_PATH     = DATA_DIR / "nous.db"
RAW_PATH    = DATA_DIR / "raw"
CHROMA_PATH = DATA_DIR / "chroma"

DEFAULT_MODEL = "gemma4:e2b"
