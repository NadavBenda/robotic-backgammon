"""Every script the user runs must at least import (catches a missing name in lib/)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]


def test_scripts_import():
    for f in sorted(ROOT.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f.stem + "_check", f)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == "__main__":
    test_scripts_import()
    print("ok test_scripts_import")
