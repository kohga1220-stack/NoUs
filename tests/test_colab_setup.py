import os

from nous import colab_setup as cs


def _make(dir_, size):
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, "nous.db"), "wb") as f:
        f.write(b"x" * size)
    with open(os.path.join(dir_, "extra.txt"), "w") as f:
        f.write("e")


def test_fresh_runtime_loads_from_drive_and_replaces_the_empty_db(tmp_path):
    local, drive = str(tmp_path / "local"), str(tmp_path / "drive")
    _make(drive, 2_000_000)
    os.makedirs(local)
    open(os.path.join(local, "nous.db"), "wb").close()                 # the 0-byte db that bit us
    msg = cs.prepare_data(local, drive)
    assert "loaded" in msg and os.path.getsize(os.path.join(local, "nous.db")) == 2_000_000


def test_existing_local_data_is_never_overwritten(tmp_path):
    local, drive = str(tmp_path / "local"), str(tmp_path / "drive")
    _make(local, 3_000_000)
    _make(drive, 2_000_000)
    assert "untouched" in cs.prepare_data(local, drive)
    assert os.path.getsize(os.path.join(local, "nous.db")) == 3_000_000


def test_falls_back_to_old_data_then_to_empty(tmp_path):
    local, drive, old = (str(tmp_path / n) for n in ("local", "drive", "old"))
    _make(old, 2_000_000)
    assert "old" in cs.prepare_data(local, drive, old)
    empty = str(tmp_path / "e")
    assert "empty" in cs.prepare_data(empty, str(tmp_path / "nope"), None)
    assert os.path.isdir(empty)


def test_save_refuses_an_empty_db_and_saves_a_real_one(tmp_path, capsys):
    local, drive = str(tmp_path / "local"), str(tmp_path / "drive")
    _make(drive, 2_000_000)
    _make(local, 0)
    save = cs.make_save(local, drive)
    assert save() is False
    assert os.path.getsize(os.path.join(drive, "nous.db")) == 2_000_000     # Drive untouched
    _make(local, 2_500_000)
    assert save() is True
    assert os.path.getsize(os.path.join(drive, "nous.db")) == 2_500_000
