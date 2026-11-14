"""Collect labelled photos for checker detection: the robot moves checkers to random
places and photographs each position. It knows where every checker is (it put them
there), so every photo comes with its answer.

    py collect.py     (double-click)

Steps (the photo window tells you what to press at each one):
  1. How many moves this run (Enter = 10, a short first try).
  2. Empty board: take every checker off the glass, press Enter -> 2 photos of the
     bare glass (s = skip, e.g. when you already took them today).
  3. Start position: the opening (set it up by hand or with place_opening.py), or,
     when you skipped step 2, the position the last run ended in.
  4. Check: the photo with a circle where every checker should be. Enter = right.
     If a checker is outside its circle: push it into the circle by hand, press f
     (the robot looks again).
  5. The robot moves one checker at a time and takes a photo after each move. Before
     each move the window shows a yellow arrow: which checker, and the way it will go. Some
     are put a few mm off centre on purpose (a person's sloppy placement). Some go to the
     bar or a bear-off tray, and come back from there.
     Where to: 3 moves in 10 at random, the rest where dataset/ has the fewest photos
     (that colour at that spot of the column), more where the last train_detector.py
     made mistakes.
     After each move it compares the photo with the one before: if more changed than
     the move explains (a checker dropped, pushed or left behind), the trained detector
     (detector.json, if there is one) reads the photo too. If it reads exactly the
     expected position, the robot goes on (a false alarm). If it sees a checker where
     none should be and one missing elsewhere (a pushed checker), the robot drags it
     back to its circle once and looks again; when that fixed it, it goes on.
     Otherwise it stops and shows step 4 again. Marked on the photo: a thick red ring
     (+ how much it changed) on every spot that changed without the move explaining it,
     blue on a spot that should have changed and did not, the move itself in yellow, and
     in a corner a small picture of the difference from the photo before. The camera's
     reading is shown as dots, with purple rings where it differs:
     d = the dots are right (the robot goes on from what the camera sees; you confirm
     the new circles with Enter), r = the robot puts back along the yellow arrows (when
     it has an offer). q in the photo window = stop after this move.
  6. At the end: Enter = the robot puts everything back into the opening (for the next
     run), n = leave it as it is.

Saved in dataset/<date-time>/: empty_0.jpg, empty_1.jpg, 000.jpg, 001.jpg, ... and
labels.jsonl (one line per photo: every checker's x, y, colour and place).
dataset/last_state.json = where the checkers are now (so the next run can go on), and
the drag in progress, if any.
log.txt = everything the run printed, with the time. If an error stops the run:
crash.log (the error, and which move it was in the middle of) and crash_last_photo.jpg.
The next run then shows that move as a yellow arrow at the start check.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import rules
import ui
import vision

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
STATE = DATA / "last_state.json"
WIN = "collect"
MAX_STACK = 6          # checkers per point (the 7th reaches into the lane)
OFF_CENTRE_P = 0.3     # share of moves that put the checker a few mm off centre
OFF_CENTRE_MM = (3.0, 7.0)
BAR_P = 0.12           # share of moves to the bar (hit checkers)
TRAY_P = 0.12          # share of moves to a bear-off tray
BAR_MAX = 4            # checkers per bar half (the detector looks at 4 places)
RANDOM_P = 0.3         # share of moves picked at random; the rest go where the dataset is thinnest
STOP_COST = 1          # any place that changed more / less than the move explains -> second opinion, then stop
COLOURS = {"robot": (80, 220, 80), "human": (0, 165, 255)}
DIFF = (220, 60, 220)  # purple: where the camera (the trained detector) reads something else
MOVED = (40, 40, 255)  # red: a spot that changed without the move explaining it
QUIET = (255, 170, 0)  # blue: a spot the move should have changed, and did not
FIX_MATCH = 14.0       # mm: a checker seen this near an expected one is that checker
FIX_MAX = 3            # the robot puts back at most this many checkers by itself


def add_counts(cnt, checkers):
    """cnt[(place, i, side)] += 1 for every checker of one photo (i = its place in the
    column / bar / tray, 0 = the edge)."""
    seen = {}
    for _, _, side, where in checkers:
        i = seen[where] = seen.get(where, -1) + 1
        cnt[(where, i, side)] = cnt.get((where, i, side), 0) + 1


def coverage():
    """{(place, i, side): photos} over everything in dataset/ so far."""
    cnt = {}
    for lab in DATA.glob("*/labels.jsonl"):
        for line in lab.read_text(encoding="utf-8").splitlines():
            if line.strip():
                add_counts(cnt, json.loads(line)["checkers"])
    return cnt


def weak_places():
    """{place: mistakes} where the last training's detector went wrong (detector.json)."""
    try:
        import learned
        w = json.loads(learned.PATH.read_text()).get("weak") or {}
    except (OSError, ValueError, ImportError):
        return {}
    return {(int(k) if str(k).isdigit() else k): v for k, v in w.items()}


def where_of(L, x, y):
    """Place key of a checker at magnet (x, y): 1..24, bar_*, off_*, or None."""
    k = L.nearest(x, y)
    if k is None:
        return None
    return k[1] if k[0] == "point" else f"{k[0]}_{k[1]}"


def full_board(board, checkers):
    """The 26-number board + (robot, human) in the bear-off trays."""
    return list(board) + [sum(1 for c in checkers if c[3] == f"off_{s}") for s in ("robot", "human")]


def place_name(i):
    """The person's name for board index i (0..27)."""
    if i in (0, 25):
        return "bar"
    if i in (26, 27):
        return "tray " + ("robot" if i == 26 else "human")
    return f"column {25 - i}"


class Collector:
    def __init__(self, b, c, L, folder, auto=False, seed=None, log=print):
        from executor import Executor
        self.b, self.c, self.L, self.folder, self.auto, self.log = b, c, L, Path(folder), auto, log
        self.rng = random.Random(seed)
        self.ex = Executor(b, L, log=lambda *a: None, grab_s=0.0 if auto else 0.3)
        self.n = 0
        self.prev = None
        self.cov = {} if auto else coverage()
        self.weak = {} if auto else weak_places()
        self._det = None
        self.folder.mkdir(parents=True, exist_ok=True)

    # -- position ------------------------------------------------------------
    def set_board(self, board):
        self.ex.set_from_board(board)

    def set_state(self, checkers):
        self.ex.places, self.ex.side_of = {}, {}
        for x, y, side, where in checkers:
            self.ex.places.setdefault(where, []).append([x, y])
            if isinstance(where, int):
                self.ex.side_of[where] = side

    def checkers(self):
        out = []
        for where, lst in self.ex.places.items():
            side = self.ex.side_of.get(where) if isinstance(where, int) else str(where).split("_")[1]
            out += [[round(p[0], 1), round(p[1], 1), side, where] for p in lst]
        return out

    def board(self):
        b = [0] * 26
        for where, lst in self.ex.places.items():
            if isinstance(where, int) and lst:
                b[where] = len(lst) if self.ex.side_of[where] == "robot" else -len(lst)
            elif where == "bar_robot":
                b[0] = len(lst)
            elif where == "bar_human":
                b[25] = -len(lst)
        return b

    def save_state(self, moving=None):
        """Where every checker is, for the next run. moving = the drag about to start
        ({what, src, goal}): if the run dies in the middle of it, the next run knows which
        checker may be anywhere between the two."""
        STATE.parent.mkdir(parents=True, exist_ok=True)
        d = {"checkers": self.checkers()}
        if moving:
            d["moving"] = moving
        STATE.write_text(json.dumps(d))

    # -- photos --------------------------------------------------------------
    def photo(self):
        from helpers import photo
        self.ex.park()
        if not self.auto:
            time.sleep(0.3)
        img = photo(self.b)[0]
        self.c.check(img)
        return img

    def save(self, img, name, note=""):
        cv2.imencode(".jpg", img)[1].tofile(str(self.folder / name))   # works with Hebrew paths
        if name.startswith("empty"):
            return
        rec = {"photo": name, "time": datetime.now().isoformat(timespec="seconds"),
               "board": self.board(), "checkers": self.checkers(), "note": note}
        add_counts(self.cov, rec["checkers"])
        with open(self.folder / "labels.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def unexplained(self, img, moved):
        """What the change check holds against this photo, slot by slot:
        (extra [(xy, score)] changed although the move does not explain it,
         quiet [xy] the move should have changed them and they did not, names of the places).
        Only places whose number of changed slots is off count (as in vision.change_cost)."""
        changed, score = vision.slot_changes(self.prev, img, self.c, self.L)
        exp = vision._expected(self.L, moved["b0"], moved["b1"])
        seen, want = {}, {}
        for (w, _), ch in changed.items():
            seen[w] = seen.get(w, 0) + int(ch)
        for w, _ in exp:
            want[w] = want.get(w, 0) + 1
        bad = {w for w in set(seen) | set(want) if seen.get(w, 0) != want.get(w, 0)}
        extra = [(self.L.slot(*k), float(score[k])) for k, ch in changed.items() if ch and k[0] in bad and k not in exp]
        quiet = [self.L.slot(*k) for k in exp if k[0] in bad and not changed.get(k, False)]
        cols = sorted(25 - w for w in bad if isinstance(w, int))
        names = ([("columns " if len(cols) > 1 else "column ") + ", ".join(map(str, cols))] if cols else []) \
            + sorted(w.replace("off_", "tray ").replace("_", " ") for w in bad if not isinstance(w, int))
        return extra, quiet, names

    def inset(self, top, img, f, avoid):
        """A small picture of what differs between the photo before and this one (bright =
        changed), in the free middle lane, at the end farther from the marks."""
        import calib
        a = cv2.cvtColor(calib.view(self.prev, self.c, xmax=self.L.xmax, ymax=self.L.ymax, board=False)[0], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(calib.view(img, self.c, xmax=self.L.xmax, ymax=self.L.ymax, board=False)[0], cv2.COLOR_BGR2GRAY)
        d = cv2.GaussianBlur(cv2.absdiff(a, b), (0, 0), 2).astype(np.float32)
        d = np.clip(d / max(40.0, float(np.percentile(d, 99.9))) * 255, 0, 255).astype(np.uint8)
        lane0, lane1 = self.L.lane_x()
        u0, u1 = sorted(float(f.mm_to_px(*self.L.to_magnet(x, self.L.ymax / 2))[0]) for x in (lane0, lane1))
        w = int(max(120, u1 - u0 - 80))          # between the column numbers
        h = int(w * top.shape[0] / top.shape[1])
        small = cv2.applyColorMap(cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA), cv2.COLORMAP_HOT)
        cv2.rectangle(small, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        cv2.putText(small, "what changed", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x = int((u0 + u1) / 2 - w / 2)
        P = [f.mm_to_px(*q) for q in avoid]
        far = lambda y: min([np.hypot(x + w / 2 - float(u), y + h / 2 - float(v)) for u, v in P] or [0.0])  # noqa: E731
        y = max((top.shape[0] - h - 90, 30), key=far)      # above the parked cart, or at the top
        top[y:y + h, x:x + w] = small

    def drawing(self, img, mark=None, arrow=None, seen=None, fixes=(), moved=None):
        """Top view with a circle where every checker should be (the moved one thicker,
        a small cross at each centre), arrow = [xy, ...]: the drag about to be made (its
        planned path), seen = the camera's reading (dots; purple ring where it differs),
        moved = {b0, b1, src, goal}: what the change check objects to (Nadav 13.11): a
        thick red ring + how much it changed on every spot that changed without the move
        explaining it, blue on a spot that should have changed and did not, the move
        itself in yellow, and a small before/after difference picture in a corner."""
        import calib
        top, f = calib.view(img, self.c, xmax=self.L.xmax, ymax=self.L.ymax)
        r = int(round(self.L.checker / 2 * f.k))
        if moved is not None and self.prev is not None:
            extra, quiet, _ = self.unexplained(img, moved)
            px = lambda q: tuple(int(t) for t in f.mm_to_px(*q))  # noqa: E731
            cv2.circle(top, px(moved["src"]), r + 4, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(top, px(moved["src"]), px(moved["goal"]), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.08)
            for q, sc in extra:
                u, v = px(q)
                cv2.circle(top, (u, v), r + 10, MOVED, 5, cv2.LINE_AA)
                cv2.putText(top, f"{sc:.2f}", (u + r + 12, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MOVED, 2, cv2.LINE_AA)
            for q in quiet:
                u, v = px(q)
                cv2.circle(top, (u, v), r + 10, QUIET, 4, cv2.LINE_AA)
                cv2.putText(top, "no change", (u + r + 12, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUIET, 2, cv2.LINE_AA)
        for x, y, side, _ in self.checkers():
            u, v = (int(t) for t in f.mm_to_px(x, y))
            thick = 4 if mark is not None and np.hypot(x - mark[0], y - mark[1]) < 1 else 2
            cv2.circle(top, (u, v), r, COLOURS[side], thick, cv2.LINE_AA)
            cv2.drawMarker(top, (u, v), COLOURS[side], cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        if seen is not None:
            bad = set(seen["diff"])
            for x, y, side in seen["checkers"]:
                u, v = (int(t) for t in f.mm_to_px(x, y))
                cv2.circle(top, (u, v), 5, COLOURS[side], -1, cv2.LINE_AA)
            both = [(x, y) for x, y, _ in seen["checkers"]] + [(x, y) for x, y, _, _ in self.checkers()]
            for x, y in both:      # red ring: every checker (expected or seen) of a place that differs
                if where_of(self.L, x, y) in bad:
                    u, v = (int(t) for t in f.mm_to_px(x, y))
                    cv2.circle(top, (u, v), r + 6, DIFF, 2, cv2.LINE_AA)
        for a, z in fixes:     # the robot's offer: drag this checker back to its circle
            cv2.arrowedLine(top, tuple(int(t) for t in f.mm_to_px(*a)), tuple(int(t) for t in f.mm_to_px(*z)),
                            (0, 230, 255), 3, cv2.LINE_AA, tipLength=0.25)
        if arrow is not None:
            P = [tuple(int(t) for t in f.mm_to_px(*q)) for q in arrow]
            cv2.circle(top, P[0], r + 4, (0, 230, 255), 3, cv2.LINE_AA)
            for a, z in zip(P[:-2], P[1:-1]):
                cv2.line(top, a, z, (0, 230, 255), 4, cv2.LINE_AA)
            cv2.arrowedLine(top, P[-2], P[-1], (0, 230, 255), 4, cv2.LINE_AA,
                            tipLength=min(0.5, 30 / max(1.0, float(np.hypot(P[-1][0] - P[-2][0], P[-1][1] - P[-2][1])))))
        if moved is not None and self.prev is not None:
            self.inset(top, img, f, [q for q, _ in extra] + quiet + [moved["src"], moved["goal"]])
        return top

    def key(self, view, lines, keys, default):
        """Show the view with text above it and wait for one of keys (Enter = '\\r')."""
        if self.auto:
            return default
        ui.show(WIN, ui.with_bar(view, lines))
        while True:
            k = cv2.waitKey(50) & 0xFF
            ch = "\r" if k in (13, 10) else chr(k) if k < 128 else ""
            if ch in keys:
                return ch

    def peek(self, view, lines):
        """Show without waiting; True when q was pressed."""
        if self.auto:
            return False
        ui.show(WIN, ui.with_bar(view, lines))
        return (cv2.waitKey(30) & 0xFF) == ord("q")

    def check(self, img, why, mark=None, seen=None, moved=None, arrow=None):
        """Ask until the photo matches the drawing. Returns the photo to keep, or None (stop).
        seen = the camera's reading (read()): shown as dots, and d takes it as the truth
        (the drawing then shows it as circles, to be confirmed with Enter like any other).
        moved = {b0, b1, src, goal}: mark what changed beyond this move (drawing())."""
        while True:
            lines = [why]
            if moved is not None:
                extra, quiet, names = self.unexplained(img, moved)
                lines.append((f"Red: changed, not by the move ({len(extra)})  " if extra else "")
                             + (f"Blue: no change ({len(quiet)})  " if quiet else "")
                             + ("at " + ", ".join(names) if names else "Now only the move itself changed"))
            if seen is not None and seen["diff"]:
                lines.append("Camera (dots, purple) differs: " + ", ".join(seen["text"][:2])
                             + (f" +{len(seen['text']) - 2}" if len(seen["text"]) > 2 else ""))
            fixes = self.plan_fix(img, seen) if seen is not None and seen["diff"] else []
            lines.append("Enter=the circles are right   f=I fixed it by hand, look again   q=stop")
            if seen is not None and seen["diff"]:
                lines.append("d=the dots are right" + ("   r=robot puts back (yellow arrows)" if fixes else ""))
            k = self.key(self.drawing(img, mark, arrow=arrow, seen=seen, fixes=fixes, moved=moved), lines, "\rfdrq", "\r")
            if k == "\r":
                return img
            if k == "q":
                return None
            if k == "r" and fixes:
                self.fix(fixes)
            if k == "d" and seen is not None and seen["diff"]:
                self.take(seen["checkers"])
                why, seen, mark, moved = "The camera's reading is now the position: are the circles right?", None, None, None
                continue
            img = self.photo()
            seen = self.read(img) if seen is not None else None

    # -- the trained detector ------------------------------------------------
    def detect(self, img):
        """[(x, y, side)] the trained detector's checkers, or None without detector.json."""
        if self._det is None:
            try:
                import learned
                self._det = (learned.Detector.load(),)
            except (OSError, ValueError, KeyError):
                self._det = ()
        if not self._det:
            return None
        e = self.folder / "empty_0.jpg"
        empty = cv2.imdecode(np.fromfile(str(e), np.uint8), cv2.IMREAD_COLOR) if e.exists() else None
        return self._det[0].find(img, self.c, self.L, empty)[0]

    def read(self, img):
        """The trained detector's reading of a photo, against where the checkers should be:
        {checkers: [(x, y, side)], diff: [place keys], text: [what differs]}, or None when
        there is no detector.json yet (run train_detector.py)."""
        ck = self.detect(img)
        if ck is None:
            return None
        r = vision.summarize(ck, self.L)
        got, want = list(r.board) + list(r.off), full_board(self.board(), self.checkers())
        diff, text = [], []
        for i, (a, b) in enumerate(zip(got, want)):
            if a == b:
                continue
            key = (i if 1 <= i <= 24 else {0: "bar_robot", 25: "bar_human", 26: "off_robot", 27: "off_human"}[i])
            diff.append(key)
            tag = (lambda n: f"{abs(n)}{'R' if n > 0 else 'H' if n < 0 else ''}") if i < 26 else str
            text.append(f"{place_name(i)} {tag(b)}->{tag(a)}")
        return {"checkers": ck, "diff": diff, "text": text}

    # -- the robot puts a pushed checker back (Nadav 13.11) -------------------
    def plan_fix(self, img, seen):
        """[(from xy, to xy)] drags that put the checkers back where they should be: every
        expected checker the camera does not see, paired with the nearest checker it sees
        where none should be. A stray is a place the detector calls a checker, or (between
        the columns, where the detector does not look) a round shape that was not in the
        photo before; its position is the round shape's centre when there is one."""
        import learned
        E = [(x, y) for x, y, _, _ in self.checkers()]
        S = [(x, y) for x, y, _ in seen["checkers"]]
        dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))  # noqa: E731
        pairs = sorted((dist(e, q), i, j) for i, e in enumerate(E) for j, q in enumerate(S) if dist(e, q) <= FIX_MATCH)
        used_e, used_s = set(), set()
        for _, i, j in pairs:
            if i not in used_e and j not in used_s:
                used_e.add(i)
                used_s.add(j)
        H = [tuple(h) for h in learned.hough_centres(img, self.c, self.L)]
        H0 = [tuple(h) for h in learned.hough_centres(self.prev, self.c, self.L)] if self.prev is not None else []
        missing = [e for i, e in enumerate(E) if i not in used_e and not any(dist(e, h) < 8 for h in H)]
        extra = []
        for j, q in enumerate(S):
            if j in used_s:
                continue
            near = [h for h in H if dist(q, h) <= 13]
            extra.append(min(near, key=lambda h: dist(q, h)) if near else q)
        # between the columns the detector does not look: a new round shape far from every checker
        loose = [h for h in H if all(dist(h, e) > 20 for e in E) and all(dist(h, x) > 20 for x in extra)
                 and all(dist(h, o) > 8 for o in H0)]
        # the photo before was right, so a real difference is at a spot that changed since
        # then; anything else is the detector's own mistake and is left alone
        if self.prev is None:
            return []
        keep = lambda pts: [q for q, yes in zip(pts, vision.changed_at(self.prev, img, self.c, self.L, pts)) if yes]  # noqa: E731
        missing, extra, loose = keep(missing), keep(extra), keep(loose)
        out = []
        for pool in (extra, loose):
            while missing and pool and len(out) < FIX_MAX:
                _, m, x = min((dist(m, x), m, x) for m in missing for x in pool)
                missing.remove(m)
                pool.remove(x)
                out.append((x, m))
        return out if not missing and not extra else []     # only when it explains everything

    def fix(self, fixes):
        """Drag each stray checker to its place. The position the robot believes in does
        not change: this only makes the glass match it again."""
        import paths
        ex = self.ex
        targets = [t for _, t in fixes]
        done = []
        for k, (src, goal) in enumerate(fixes):
            others = [p for p in ex.all_xy() if p not in targets] + done + [s for s, _ in fixes[k + 1:]]
            path, _ = ex.plan(tuple(src), tuple(goal), others)
            ex.goto(tuple(src), others)
            paths.drag(ex.b, path, overshoot=ex._overshoot(path, others), grab_s=ex.grab_s,
                       xmax=self.L.xmax, ymax=self.L.ymax, settle_s=ex.settle_s)
            ex.cart = tuple(goal)
            done.append(tuple(goal))

    def take(self, ck):
        """Make the camera's reading [(x, y, side)] the position the robot works from."""
        state = []
        for x, y, side in ck:
            w = where_of(self.L, x, y)
            if w is not None:
                state.append([x, y, side, w])
        self.set_state(state)
        self.save_state()

    # -- steps ----------------------------------------------------------------
    def empty_board(self):
        img = self.photo()
        k = self.key(img if self.auto else self.drawing(img),
                     ["Empty board: take every checker off the glass, then Enter",
                      "s = skip (empty photos already taken today)"], "\rs", "s")
        if k == "s":
            return False
        for i in range(2):
            self.save(self.photo(), f"empty_{i}.jpg")
        self.log("   2 photos of the empty board saved")
        return True

    def start(self, board=None, state=None, moving=None):
        """moving: the drag the last run died in ({what, src, goal}); shown as a yellow
        arrow, that checker may be anywhere along it."""
        if state:
            self.set_state(state)
        else:
            self.set_board(board or rules.OPENING)
        why = "Start: put the checkers inside the circles (f = look again)"
        arrow = None
        if state and moving:
            why = f"Last run stopped in {moving['what']}: put that checker (yellow) back in its circle"
            arrow = [tuple(moving["src"]), tuple(moving["goal"])]
        img = self.check(self.photo(), why, arrow=arrow)
        if img is None:
            return False
        self.save(img, f"{self.n:03d}.jpg", "start")
        self.prev, self.n = img, self.n + 1
        self.save_state()
        return True

    def moves(self):
        """[(from, to, side)] every move collect may make: one colour per column, at most
        MAX_STACK high, BAR_MAX on a bar half, 15 in a tray."""
        ex = self.ex
        out = []
        for frm, lst in ex.places.items():
            if not lst or str(frm).startswith("_"):
                continue
            side = ex.side_of[frm] if isinstance(frm, int) else frm.split("_")[1]
            for to in range(1, 25):
                if to != frm and len(ex.places.get(to, [])) < MAX_STACK and \
                        (not ex.places.get(to) or ex.side_of.get(to) == side):
                    out.append((frm, to, side))
            for to, cap in ((f"bar_{side}", BAR_MAX), (f"off_{side}", 15)):
                if to != frm and len(ex.places.get(to, [])) < cap:
                    out.append((frm, to, side))
        return out

    def need(self, to, side):
        """How much the dataset wants a checker of side as the next one on place to: few
        photos of that spot with that colour (and where the detector made mistakes) = more."""
        i = len(self.ex.places.get(to, []))
        w = 1.0 / np.sqrt(1.0 + self.cov.get((to, i, side), 0))
        if self.weak:
            name = to if isinstance(to, int) else to.split("_")[0]
            w *= 1.0 + self.weak.get(name, 0) / (1.0 + np.mean(list(self.weak.values())))
        return w

    def random_move(self):
        """(from, to, side, goal xy, off-centre mm or 0). Where to: BAR_P to the bar,
        TRAY_P to a bear-off tray, the rest to a column; RANDOM_P of the moves picked at
        random, the others weighted by need() (Nadav 13.11: fill what the dataset lacks)."""
        ex, L = self.ex, self.L
        allm = self.moves()
        if not allm:
            raise RuntimeError("no possible random move")
        r = self.rng.random()
        kind = "bar" if r < BAR_P else "off" if r < BAR_P + TRAY_P else "point"
        cands = [m for m in allm if (isinstance(m[1], int) if kind == "point" else str(m[1]).startswith(kind))]
        cands = cands or allm
        if self.rng.random() < RANDOM_P:
            frm = self.rng.choice(sorted({m[0] for m in cands}, key=str))
            frm, to, side = self.rng.choice([m for m in cands if m[0] == frm])
        else:
            w = [self.need(m[1], m[2]) for m in cands]
            frm, to, side = self.rng.choices(cands, weights=w)[0]
        goal = ex._free_slot(to)
        off = 0.0
        if self.rng.random() < OFF_CENTRE_P:
            others = [p for p in ex.all_xy()]
            for _ in range(20):
                r = self.rng.uniform(*OFF_CENTRE_MM)
                a = self.rng.uniform(0, 2 * np.pi)
                q = (goal[0] + r * np.cos(a), goal[1] + r * np.sin(a))
                if (4 <= q[0] <= L.xmax - 4 and 4 <= q[1] <= L.ymax - 4
                        and all(np.hypot(q[0] - o[0], q[1] - o[1]) >= L.checker + 1 for o in others)):
                    goal, off = q, r
                    break
        return frm, to, side, tuple(goal), off

    def step(self):
        """One random move + photo + check. False = stop."""
        frm, to, side, goal, off = self.random_move()
        what = f"move {self.n}: {frm} -> {to}" + (f" ({off:.0f} mm off centre)" if off else "")
        # show the move about to be made (Nadav 13.11): the same checker move_checker takes,
        # the one of the place nearest the middle lane
        lst = self.ex.places[frm]
        mid = self.L.xmax / 2
        src = tuple(min(lst, key=lambda q: abs(self.L.from_magnet(*q)[0] - mid)))
        others = [p for p in self.ex.all_xy() if np.hypot(p[0] - src[0], p[1] - src[1]) > 1]
        path = self.ex.plan(src, goal, others)[0]
        if self.peek(self.drawing(self.prev, arrow=path), ["next: " + what, "q = stop (before this move)"]):
            return False
        b0 = self.board()
        self.log(f"   {what}")       # before the drag, so the log shows what was going on if it dies here
        self.save_state(moving={"what": what, "src": [float(src[0]), float(src[1])],
                                "goal": [float(goal[0]), float(goal[1])]})
        self.ex.move_checker(frm, to, side, goal=goal)
        b1 = self.board()
        self.save_state()
        img = self.photo()
        changed, _ = vision.slot_changes(self.prev, img, self.c, self.L)
        cost = vision.change_cost(changed, self.L, b0, b1)
        self.log(f"      change check {cost}")
        note = ""
        if cost >= STOP_COST:
            # a second opinion before stopping (Nadav 13.11): the trained detector reads the
            # photo; if it sees exactly the expected position, the change check was a false alarm
            seen = self.read(img)
            ok = seen is not None and not seen["diff"]
            if ok:
                note = "change check stopped, detector agreed"
                self.log(f"   change check {cost}, but the detector reads exactly this position: going on")
            elif seen is not None:
                fixes = self.plan_fix(img, seen)
                if fixes:      # a checker was pushed: the robot puts it back once, then looks again
                    self.log(f"   change check {cost}: putting {len(fixes)} checker(s) back")
                    self.fix(fixes)
                    img = self.photo()
                    changed, _ = vision.slot_changes(self.prev, img, self.c, self.L)
                    cost = vision.change_cost(changed, self.L, b0, b1)
                    seen = self.read(img)
                    ok = cost < STOP_COST
                    if ok:
                        note = "put back by the robot"
                        self.log(f"   now the change check is {cost}: going on")
            if not ok:
                img = self.check(img, f"{what}: more changed than this move",
                                 mark=goal, seen=seen, moved={"b0": b0, "b1": b1, "src": src, "goal": goal})
                if img is None:
                    return False
                note = "checked by hand"
        self.save(img, f"{self.n:03d}.jpg", note)
        self.prev, self.n = img, self.n + 1
        return not self.peek(self.drawing(img, mark=goal), [what, "q = stop after this move"])

    def back_to_opening(self):
        self.ex.arrange(rules.OPENING)
        img = self.check(self.photo(), "Back in the opening")
        if img is not None:
            self.save(img, f"{self.n:03d}.jpg", "opening again")
            self.prev, self.n = img, self.n + 1
        self.save_state()


def ask_int(prompt, default):
    s = input(prompt).strip()
    try:
        return max(1, int(s)) if s else default
    except ValueError:
        return default


def run_log(folder):
    """A log(...) that prints and also writes every line, with the time, to <run>/log.txt.
    A hard crash of Python itself (no error message) is written there too."""
    import faulthandler
    folder.mkdir(parents=True, exist_ok=True)
    f = open(folder / "log.txt", "a", encoding="utf-8", buffering=1)
    faulthandler.enable(f)

    def log(*a):
        print(*a)
        f.write(f"{datetime.now():%H:%M:%S} " + " ".join(str(x) for x in a) + "\n")
    return log


def crashed(folder, col, log):
    """An error stopped the run: say why, and keep it in <run>/crash.log (Nadav 13.11)."""
    import traceback
    tb = traceback.format_exc()
    last = ""
    try:
        if STATE.exists():
            last = (json.loads(STATE.read_text()).get("moving") or {}).get("what", "")
    except (OSError, ValueError):
        pass
    text = (f"{datetime.now().isoformat(timespec='seconds')}\n"
            + (f"In the middle of: {last}\n" if last else "Not in the middle of a move.\n")
            + (f"Photos saved so far: {col.n}\n" if col is not None else "Before the first photo.\n")
            + "\n" + tb)
    try:
        (folder / "crash.log").write_text(text, encoding="utf-8")
        if col is not None and col.prev is not None:
            cv2.imencode(".jpg", col.prev)[1].tofile(str(folder / "crash_last_photo.jpg"))
    except OSError:
        pass
    log("\n" + tb)
    log("collect stopped on an error (the last line above says which).")
    if last:
        log(f"It was in the middle of: {last}")
    log(f"Written in {folder / 'crash.log'}")
    log("The position is saved. To go on: run collect.py again, s (skip the empty photos), Enter.")


def main():
    import calib
    from board import Board
    from layout import Layout
    L = Layout.load()
    c = calib.load()
    moves = ask_int("How many moves this run? Enter = 10 (a short first try): ", 10)
    folder = DATA / f"{datetime.now():%Y%m%d-%H%M%S}"
    log = run_log(folder)
    try:      # the calibration of this run, so a later recalibration does not spoil these photos
        import shutil
        shutil.copyfile(calib.PATH, folder / "calib.json")
    except OSError:
        pass
    col = None
    try:
        with Board() as b:
            b.home_if_needed()
            col = Collector(b, c, L, folder, log=log)
            state = moving = None
            if not col.empty_board() and STATE.exists():
                k = col.key(col.photo(), ["Are the checkers where the last run left them?",
                                          "Enter = yes   o = no, I will set up the opening"], "\ro", "o")
                if k == "\r":
                    saved = json.loads(STATE.read_text())
                    state, moving = saved["checkers"], saved.get("moving")
            if not col.start(state=state, moving=moving):
                log("Stopped.")
                return
            for _ in range(moves):
                if not col.step():
                    break
            k = col.key(col.drawing(col.prev), ["Done.  Enter = put everything back in the opening",
                                                "n = leave the checkers as they are"], "\rn", "n")
            if k == "\r":
                col.back_to_opening()
            col.ex.park()
    except Exception:
        crashed(folder, col, log)
        sys.exit(1)
    cv2.destroyAllWindows()
    log(f"Done: {col.n} photos in {folder}")


if __name__ == "__main__":
    ui.keep_open()
    main()
