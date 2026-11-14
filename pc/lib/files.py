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
