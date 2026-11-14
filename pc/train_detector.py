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

import cnn
import ensemble
import learned
import ui
import vision

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
USE_CNN = cnn.available()


def cnn_samples(photos, L, with_empty=True):
    """[(images, extra, labels)] for cnn.Detector.train / board."""
    out = []
    for p in photos:
        a = p["crop"].astype(np.float32)
        has = with_empty and p["empty"] is not None
        b = p["empty"].astype(np.float32) if has else np.zeros_like(a)
        e = (p["F"] if has else p["F0"]).astype(np.float32)
        out.append((np.stack([a, b], 1), e, p["y"]))
    return out


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)   # works with Hebrew paths


def wrong(a, b):
    """Checkers that differ between two 28-number boards (board + both trays)."""
    return int(sum(abs(x - y) for x, y in zip(a, b)))


def mistakes(a, b):
    """{place: 'colour' | 'count'} where board a differs from the true board b
    (place = the person's column number, or 'bar')."""
    out = {}
    for i in range(len(b)):
        if a[i] == b[i]:
            continue
        name = "bar" if i in (0, 25) else "off" if i >= 26 else 25 - i
        out[name] = "colour" if a[i] * b[i] < 0 else "count"
    return out


def own_calib(folder):
    """The calibration saved with a run by collect.py, or None (then the current one is used)."""
    import calib
    try:
        return calib.load(folder / "calib.json")
    except (OSError, ValueError, KeyError):
        return None


def load_runs(c0, L, log=print):
    """[{name, photos: [(img, board, F, F_no_empty, y)]}] every run in dataset/."""
    runs = []
    for lab in sorted(DATA.glob("*/labels.jsonl")):
        folder = lab.parent
        c = own_calib(folder) or c0
        e = folder / "empty_0.jpg"
        empty = imread(e) if e.exists() else None
        ecrop = cnn.crops(empty, c, L).astype(np.float16) if USE_CNN and empty is not None else None
        photos = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            img = imread(folder / rec["photo"])
            if img is None:
                continue
            y = learned.labels(rec["checkers"], L)
            F0 = learned.features(img, c, L, None)
            F = learned.features(img, c, L, empty) if empty is not None else F0
            board = rec["board"] + [sum(1 for k in rec["checkers"] if k[3] == f"off_{s}") for s in ("robot", "human")]
            ph = {"img": img, "board": board, "F": F, "F0": F0, "y": y, "note": rec.get("note", "")}
            if USE_CNN:
                ph["crop"] = cnn.crops(img, c, L).astype(np.float16)
                ph["empty"] = ecrop
            photos.append(ph)
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
    n = vision.learn_norm(first["img"], c, L, first["board"][:26])
    model = type("M", (), {"norm": n})()
    out = []
    for p in photos:
        ck, _ = vision.find_norm(p["img"], c, L, model)
        out.append(wrong(learned.full(vision.summarize(ck, L)), p["board"]))
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
    tot_l, tot_n, tot_c, tot_p, tot_e = [], [], [], [], []
    errs = {"features-net": [], "pixel+features-net": [], "both together": []}   # [(photo id, {place: kind})]
    for name, train, test in tests:
        det = learned.Detector.train(samples(train), L)
        el = [wrong(det.board(p["F0"], L), p["board"]) for p in test]
        errs["features-net"] += [((name, j), mistakes(det.board(p["F0"], L), p["board"])) for j, p in enumerate(test)]
        seeds = [np.mean([wrong(learned.Detector.train(samples(train), L, seed=sd).board(p["F0"], L), p["board"])
                          for p in test]) for sd in (1, 2)]
        en = norm_errors([train[0]] + test, c, L)[1:] if len(runs) < 2 else norm_errors(test, c, L)
        tot_l += el
        tot_n += en
        line = (f"   {name}: {len(test)} photos   features-net {np.mean(el):.1f} wrong/photo "
                f"({sum(e == 0 for e in el)} perfect; trained again: {seeds[0]:.1f}, {seeds[1]:.1f})"
                f"   norm {np.mean(en):.1f} ({sum(e == 0 for e in en)} perfect)")
        if USE_CNN:
            log("   training the pixel network (a few minutes)...")
            cd = cnn.Detector.train(cnn_samples(train, L))
            ec = [wrong(cd.board(x, e, L), p["board"])
                  for (x, e, _), p in zip(cnn_samples(test, L, with_empty=False), test)]
            tot_c += ec
            line += f"   pixel+features-net {np.mean(ec):.1f} ({sum(e == 0 for e in ec)} perfect)"
            cs = cnn_samples(test, L, with_empty=False)
            errs["pixel+features-net"] += [((name, j), mistakes(cd.board(x, e, L), p["board"]))
                                           for j, ((x, e, _), p) in enumerate(zip(cs, test))]
            eb = [learned.full(vision.summarize(ensemble.find(det, cd, p["F0"], x, e, L), L))
                  for (x, e, _), p in zip(cs, test)]
            ee = [wrong(b, p["board"]) for b, p in zip(eb, test)]
            tot_e += ee
            errs["both together"] += [((name, j), mistakes(b, p["board"])) for j, (b, p) in enumerate(zip(eb, test))]
            line += f"   both together {np.mean(ee):.1f} ({sum(e == 0 for e in ee)} perfect)"
            pd = cnn.Detector.train(cnn_samples(train, L), pixels_only=True)
            ep = [wrong(pd.board(x, e, L), p["board"])
                  for (x, e, _), p in zip(cnn_samples(test, L, with_empty=False), test)]
            tot_p += ep
            line += f"   pixels-only {np.mean(ep):.1f}"
        log(line)
    explain(errs, log)
    return (float(np.mean(tot_l)), float(np.mean(tot_n)), (float(np.mean(tot_c)) if tot_c else None),
            (float(np.mean(tot_p)) if tot_p else None), (float(np.mean(tot_e)) if tot_e else None), weak(errs))


def weak(errs):
    """{place: mistakes} of the features-net on photos it did not learn from, for
    collect.py to put more checkers where the detector goes wrong. Place: 1..24 (robot
    numbering, as collect uses), 'bar' or 'off'."""
    out = {}
    for _, d in errs["features-net"]:
        for name in d:
            k = 25 - name if isinstance(name, int) else name
            out[k] = out.get(k, 0) + 1
    return out


def explain(errs, log=print):
    """Where each method goes wrong, and whether the two networks make the same mistakes."""
    log("\nWhere the mistakes are (photo + column counted once):")
    for m, lst in errs.items():
        if not lst:
            continue
        kinds = [k for _, d in lst for k in d.values()]
        cols = {}
        for _, d in lst:
            for place in d:
                cols[place] = cols.get(place, 0) + 1
        top = ", ".join(f"{p}: {n}" for p, n in sorted(cols.items(), key=lambda t: -t[1])[:6])
        log(f"   {m}: {len(kinds)} mistakes, {kinds.count('count')} wrong count, "
            f"{kinds.count('colour')} wrong colour.  Most in columns: {top}")
    a = {(pid, pl) for pid, d in errs["features-net"] for pl in d}
    b = {(pid, pl) for pid, d in errs["pixel+features-net"] for pl in d}
    if a and b:
        log(f"   Same mistake in both networks: {len(a & b)}; only features-net: {len(a - b)}; "
            f"only pixel+features-net: {len(b - a)}")


def main():
    import calib
    from layout import Layout
    L = Layout.load()
    k = input("Compare all the methods on photos they did not learn from (slow: an hour or more)?\n"
              "y = yes   Enter = no, just train the features network (learned) on all the photos: ").strip().lower()
    if k == "y":
        report(calib.load(), L)
    else:
        quick(calib.load(), L)


def quick(c, L):
    """Train the features network (learned) once on all the photos and save it (no comparison). The
    comparison is for choosing a method; once chosen, this is all a new batch of photos needs."""
    import time
    t0 = time.time()
    print("Reading the photos...")
    runs = load_runs(c, L)
    if not runs:
        print("No photos yet: run collect.py first.")
        return
    allp = [p for r in runs for p in r["photos"]]
    old = {}
    try:
        old = json.loads(learned.PATH.read_text())
    except (OSError, ValueError):
        pass
    print(f"{len(allp)} photos in {len(runs)} run(s). Training the features network...")
    det = learned.Detector.train(samples(allp), L)
    det.save(photos=len(allp), runs=len(runs), weak=old.get("weak"),
             learned_wrong=old.get("learned_wrong"), norm_wrong=old.get("norm_wrong"), cnn_wrong=old.get("cnn_wrong"))
    print(f"   saved detector.json (stopped after {det.presence.stopped_at} steps)")
    # Nadav 14.11: the game reads with learned only; the pixel network is no longer trained here
    print(f"Done in {(time.time() - t0) / 60:.0f} min. The game uses them from the next start.")


def report(c, L, save=True):
    print("Reading the photos...")
    runs = load_runs(c, L)
    if not runs:
        print("No photos yet: run collect.py first.")
        return
    n = sum(len(r["photos"]) for r in runs)
    print(f"{n} photos in {len(runs)} run(s). Checking on photos it did not learn from:")
    ml, mn, mc, mp, me, wk = evaluate(runs, c, L)
    allp = [p for r in runs for p in r["photos"]]
    det = learned.Detector.train(samples(allp), L)
    if save:
        det.save(photos=n, runs=len(runs), learned_wrong=ml, norm_wrong=mn, cnn_wrong=mc, weak=wk)
    print(f"\nAverage wrong checkers per photo (on photos it did not learn from):")
    print(f"   norm (now in use)   {mn:.1f}")
    print(f"   features-net        {ml:.1f}   -> vision = learned")
    if mc is not None:
        print(f"   pixel+features-net  {mc:.1f}   -> vision = cnn")
        print(f"   pixels-only         {mp:.1f}   (a control: what the computed features add)")
        print(f"   both together       {me:.1f}   (features-net + pixel+features-net averaged)")
        print("Training the pixel network on everything...")
        cd = cnn.Detector.train(cnn_samples(allp, L))
        if save:
            cd.save()
    else:
        print("   pixel-net: PyTorch is not installed, skipped")
    best = min([("norm", mn), ("learned", ml)] + ([("cnn", mc)] if mc is not None else []), key=lambda t: t[1])
    print(("Saved." if save else "") + f"  Best: {best[0]}. Choose it in the game page (vision).")


if __name__ == "__main__":
    ui.keep_open()
    main()
