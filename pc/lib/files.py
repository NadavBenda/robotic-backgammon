"""Where the PC's own data files live: data/ next to the scripts (Nadav 14.11: the scripts
folder holds only the scripts to double-click; everything else in sub-folders).

    from files import data
    PATH = data("calib.json")       # <PC code>/data/calib.json

A file still at the old place (next to the scripts) is moved into data/ the first time it is asked for.
"""
import shutil
from pathlib import Path

TOP = Path(__file__).resolve().parents[1]
DATA = TOP / "data"


def data(name):
    new = DATA / name
    old = TOP / name
    try:
        DATA.mkdir(exist_ok=True)
    except OSError:
        pass
    if not new.exists() and old.exists():
        try:
            shutil.move(str(old), str(new))
        except OSError:
            return old
    return new


BACKUP = TOP / "backup"


def backup(log=print):
    """Copy the data files that cannot be made again without the board (data/*.json:
    calibration, layout, ...) to backup/<date-time>/, when any of them differs from the newest
    copy. Called when the game or calibrate.py starts, so there is always a copy from before
    the last change. Returns the new folder, or None when nothing changed."""
    from datetime import datetime
    try:
        files = sorted(DATA.glob("*.json"))
        if not files:
            return None
        old = sorted(p for p in BACKUP.glob("*") if p.is_dir()) if BACKUP.exists() else []
        last = old[-1] if old else None
        if last and all((last / f.name).exists() and (last / f.name).read_bytes() == f.read_bytes() for f in files):
            return None
        new = BACKUP / f"{datetime.now():%Y%m%d-%H%M%S}"
        new.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(str(f), str(new / f.name))
        log(f"Backup of {', '.join(f.name for f in files)} in backup/{new.name}")
        return new
    except OSError as e:      # a backup must never stop the game
        log(f"Backup failed: {e}")
        return None
