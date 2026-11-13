"""Teach the checker detector from the photos collect.py saved, and compare it with the
current method (norm). No installs needed.

    py train_detector.py     (double-click)

What it does:
  1. Reads every run in dataset/ (the photos + labels.jsonl).
  2. Checks itself honestly: it learns from all runs but one and reads the photos of
     the one it did not see, for every run in turn (with a single run: learns from the
     first 70% of its photos, reads the rest).
  3. Prints, per run: wrong checkers per photo, the learned detector vs norm.
  4. Learns again from everything and saves detector.json. The game uses it only when
     you choose it ("learned" in the vision setting), never by itself.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import json
from pathlib import Path

import cv2
import numpy as np

import learned
import ui
import vision

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # works with Hebrew paths


def wrong(a, b):
    return int(sum(abs(x - y) for x, y in zip(a[:26], b[:26])))


def load_runs(c, L, log=print):
    """[{name, photos: [(img, board, F, F_no_empty, y)]}] every run in dataset/."""
    runs = []
    for lab in sorted(DATA.glob("*/labels.jsonl")):
        folder = lab.parent
        e = folder / "empty_0.jpg"
        empty = imread(e) if e.exists() else None
        photos = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            img = imread(folder / rec["photo"])
            if img is None:
                continue
            y = learned.labels(rec["checkers"], L)
            F0 = learned.features(img, c, L, None)
            F = learned.features(img, c, L, empty) if empty is not None else F0
            photos.append({"img": img, "board": rec["board"], "F": F, "F0": F0, "y": y,
                           "note": rec.get("note", "")})
        if photos:
            runs.append({"name": folder.name, "photos": photos, "empty": empty is not None})
            log(f"   {folder.name}: {len(photos)} photos" + ("" if empty is not None else "  (no empty-board photo)"))
    return runs


def samples(photos):
    out = []
    for p in photos:
        out.append((p["F"], p["y"]))
        if p["F"] is not p["F0"]:
            out.append((p["F0"], p["y"]))   # also without the empty photo (the game has none)
    return out


def norm_errors(photos, c, L):
    """The current method: Learn colours on the run's first photo (a known board), read the rest."""
    first = photos[0]
    n = vision.learn_norm(first["img"], c, L, first["board"])
    model = type("M", (), {"norm": n})()
    out = []
    for p in photos:
        ck, _ = vision.find_norm(p["img"], c, L, model)
        out.append(wrong(vision.summarize(ck, L).board, p["board"]))
    return out


def evaluate(runs, c, L, log=print):
    tests = []
    if len(runs) >= 2:
        for k, r in enumerate(runs):
            train = [p for j, o in enumerate(runs) if j != k for p in o["photos"]]
            tests.append((r["name"], train, r["photos"]))
    else:
        ph = runs[0]["photos"]
        cut = max(1, int(len(ph) * 0.7))
        tests.append((runs[0]["name"] + " (last 30%)", ph[:cut], ph[cut:] or ph[-1:]))
    tot_l, tot_n = [], []
    for name, train, test in tests:
        det = learned.Detector.train(samples(train), L)
        el = [wrong(det.board(p["F0"], L), p["board"]) for p in test]
        en = norm_errors([train[0]] + test, c, L)[1:] if len(runs) < 2 else norm_errors(test, c, L)
        tot_l += el
        tot_n += en
        log(f"   {name}: {len(test)} photos   learned {np.mean(el):.1f} wrong/photo "
            f"({sum(e == 0 for e in el)} perfect)   norm {np.mean(en):.1f} ({sum(e == 0 for e in en)} perfect)")
    return float(np.mean(tot_l)), float(np.mean(tot_n))


def main():
    import calib
    from layout import Layout
    L = Layout.load()
    report(calib.load(), L)


def report(c, L, save=True):
    print("Reading the photos...")
    runs = load_runs(c, L)
    if not runs:
        print("No photos yet: run collect.py first.")
        return
    n = sum(len(r["photos"]) for r in runs)
    print(f"{n} photos in {len(runs)} run(s). Checking on photos it did not learn from:")
    ml, mn = evaluate(runs, c, L)
    det = learned.Detector.train(samples([p for r in runs for p in r["photos"]]), L)
    if save:
        det.save(photos=n, runs=len(runs), learned_wrong=ml, norm_wrong=mn)
    print(f"\nAverage wrong checkers per photo: learned {ml:.1f}, norm {mn:.1f}.")
    print(("Saved detector.json." if save else "") + ("  The learned detector is better." if ml < mn else
                                     "  It is NOT better than norm yet: more photos needed."))


if __name__ == "__main__":
    ui.keep_open()
    main()
