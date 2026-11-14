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


def test_detector_saves_and_loads(tmp_path=None):
    """detector.json written by training must load again (game, collect, arrange_opening use it)."""
    import tempfile
    import numpy as np
    import learned
    m = learned.MLP(5, hidden=3)
    m.mu, m.sd = np.zeros(5), np.ones(5)
    p = Path(tempfile.mkdtemp()) / "d.json"
    learned.Detector(m, m).save(p)
    d = learned.Detector.load(p)
    x = np.random.default_rng(0).normal(size=(4, 5))
    assert np.allclose(d.presence.prob(x), m.prob(x))


if __name__ == "__main__":
    test_scripts_import()
    test_detector_saves_and_loads()
    print("ok test_scripts_import")
