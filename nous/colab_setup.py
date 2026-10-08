"""
Colab helpers: load the data from Drive without ever overwriting good data with an empty copy.

A fresh Colab runtime has no data. Running any command first creates an EMPTY `nous.db`
(0 bytes); a later `save()` of that would overwrite the real database on Drive. So
  - data is loaded only when the local copy is missing/empty (never over unsaved work), and
  - `save()` refuses to write a database smaller than MIN_DB_BYTES.
"""
from __future__ import annotations
import os
import shutil

MIN_DB_BYTES = 1_000_000      # the real database is ~2.8 MB; an empty one is 0 bytes


def _db_size(data_dir: str) -> int:
    p = os.path.join(data_dir, "nous.db")
    return os.path.getsize(p) if os.path.exists(p) else 0


def prepare_data(local: str, drive: str, old: str | None = None) -> str:
    """
    Fill `local` from `drive` (or from `old`, the pre-NoUs folder, the first time).
    Returns a one-line status message.
    """
    if _db_size(local) >= MIN_DB_BYTES:
        return f"local data already present ({_db_size(local):,} bytes) — left untouched"
    src = next((s for s in (drive, old) if s and _db_size(s) >= MIN_DB_BYTES), None)
    if src is None:
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)
        return "no usable data on Drive — starting from an empty state"
    shutil.rmtree(local, ignore_errors=True)       # drop the empty/partial local copy
    shutil.copytree(src, local)
    return f"loaded {src} ({_db_size(local):,} bytes)"


def make_save(local: str, drive: str):
    def save():
        """Copy the local data to Drive (overwrite). Refuses to save an empty database."""
        size = _db_size(local)
        if size < MIN_DB_BYTES:
            print(f"STOPPED: local nous.db is only {size:,} bytes — not overwriting Drive")
            return False
        shutil.copytree(local, drive, dirs_exist_ok=True)
        print("saved →", drive)
        return True
    return save
