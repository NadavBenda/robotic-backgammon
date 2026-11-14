"""Why does the trained detector misread? Checks the photos in dataset/ (no board needed):

    py diag_detector.py

  1. How one-sided the photos are: per column, the share of photos where its checkers
     are the robot's. Near 0% or 100% = the network can learn "this column is always
     that colour" instead of looking at the checkers.
  2. Trains on all runs but one and reads the one left out, for every run, three ways:
     as now / the colour network without the column's position / both networks without
     position. Prints checkers wrong per photo and how many mistakes are colour or count.
The result is also written to dataset/diag.txt.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import numpy as np

import learned
import train_detector as td


def main():
    import calib
    from layout import Layout
    L = Layout.load()
    td.USE_CNN = False
    lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    runs = td.load_runs(calib.load(), L, log)
    if len(runs) < 2:
        log("Need at least 2 runs.")
        return
    allp = [p for r in runs for p in r["photos"]]
    log(f"\n{len(allp)} photos in {len(runs)} runs.")
    log("\n1. Share of photos where the column's checkers are the robot's (robot numbering 1..24):")
    row = []
    for p in range(1, 25):
        v = [ph["board"][p] for ph in allp if ph["board"][p] != 0]
        row.append(f"{p}:{100 * np.mean([x > 0 for x in v]):.0f}%({len(v)})" if v else f"{p}:-")
    log("   " + "  ".join(row[:12]))
    log("   " + "  ".join(row[12:]))

    log("\n2. Read the run left out (checkers wrong per photo; mistakes = photo + column):")
    for name, pp, pc in (("as now", True, True), ("colour without position", True, False),
                         ("both without position", False, False)):
        learned.POS_PRESENCE, learned.POS_COLOUR = pp, pc
        tot, kinds, per_run = [], [], []
        for k, r in enumerate(runs):
            train = [p for j, o in enumerate(runs) if j != k for p in o["photos"]]
            det = learned.Detector.train(td.samples(train), L)
            el = []
            for p in r["photos"]:
                b = det.board(p["F0"], L)
                el.append(td.wrong(b, p["board"]))
                kinds += list(td.mistakes(b, p["board"]).values())
            tot += el
            per_run.append(f"{r['name'][-11:]} {np.mean(el):.1f}")
        log(f"   {name}: {np.mean(tot):.2f} wrong/photo, {kinds.count('colour')} colour mistakes, "
            f"{kinds.count('count')} count mistakes")
        log("      per run: " + ", ".join(per_run))
    (td.DATA / "diag.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
