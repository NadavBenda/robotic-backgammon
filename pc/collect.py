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
  5. The robot moves one checker at a time and takes a photo after each move. Some
     are put a few mm off centre on purpose (a person's sloppy placement).
     After each move it compares the photo with the one before: if more changed than
     the move explains (a checker dropped, pushed or left behind) it stops and shows
     step 4 again. q in the photo window = stop after this move.
  6. At the end: Enter = the robot puts everything back into the opening (for the next
     run), n = leave it as it is.

Saved in dataset/<date-time>/: empty_0.jpg, empty_1.jpg, 000.jpg, 001.jpg, ... and
labels.jsonl (one line per photo: every checker's x, y, colour and place).
dataset/last_state.json = where the checkers are now (so the next run can go on).
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
BAR_P = 0.08           # share of moves to the bar
STOP_COST = 2          # places that changed more / less than the move explains -> stop
COLOURS = {"robot": (80, 220, 80), "human": (0, 165, 255)}


class Collector:
    def __init__(self, b, c, L, folder, auto=False, seed=None, log=print):
        from executor import Executor
        self.b, self.c, self.L, self.folder, self.auto, self.log = b, c, L, Path(folder), auto, log
        self.rng = random.Random(seed)
        self.ex = Executor(b, L, log=lambda *a: None, grab_s=0.0 if auto else 0.3)
        self.n = 0
        self.prev = None
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

    def save_state(self):
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"checkers": self.checkers()}))

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
        with open(self.folder / "labels.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def drawing(self, img, mark=None):
        """Top view with a circle where every checker should be (the moved one thicker)."""
        import calib
        top, f = calib.view(img, self.c, xmax=self.L.xmax, ymax=self.L.ymax)
        r = int(round(self.L.checker / 2 * f.k))
        for x, y, side, _ in self.checkers():
            u, v = f.mm_to_px(x, y)
            thick = 4 if mark is not None and np.hypot(x - mark[0], y - mark[1]) < 1 else 2
            cv2.circle(top, (int(u), int(v)), r, COLOURS[side], thick, cv2.LINE_AA)
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

    def check(self, img, why, mark=None):
        """Ask until the photo matches the drawing. Returns the photo to keep, or None (stop)."""
        while True:
            lines = [why, "Every checker inside its circle?  Enter=yes   "
                     "f=I moved them into the circles, look again   q=stop"]
            k = self.key(self.drawing(img, mark), lines, "\rfq", "\r")
            if k == "\r":
                return img
            if k == "q":
                return None
            img = self.photo()

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

    def start(self, board=None, state=None):
        if state:
            self.set_state(state)
        else:
            self.set_board(board or rules.OPENING)
        img = self.check(self.photo(), "Start: put the checkers inside the circles (f = look again)")
        if img is None:
            return False
        self.save(img, f"{self.n:03d}.jpg", "start")
        self.prev, self.n = img, self.n + 1
        self.save_state()
        return True

    def random_move(self):
        """(from, to, side, goal xy, off-centre mm or 0) of a random move that keeps each
        point one colour and at most MAX_STACK high."""
        ex, L = self.ex, self.L
        sources = [w for w, lst in ex.places.items() if lst and not str(w).startswith(("off", "_"))]
        for _ in range(100):
            frm = self.rng.choice(sources)
            side = ex.side_of[frm] if isinstance(frm, int) else frm.split("_")[1]
            bar = f"bar_{side}"
            if frm != bar and self.rng.random() < BAR_P and len(ex.places.get(bar, [])) < 3:
                to = bar
            else:
                tos = [p for p in range(1, 25) if p != frm and len(ex.places.get(p, [])) < MAX_STACK
                       and (not ex.places.get(p) or ex.side_of.get(p) == side)]
                if not tos:
                    continue
                to = self.rng.choice(tos)
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
        raise RuntimeError("no possible random move")

    def step(self):
        """One random move + photo + check. False = stop."""
        frm, to, side, goal, off = self.random_move()
        b0 = self.board()
        self.ex.move_checker(frm, to, side, goal=goal)
        b1 = self.board()
        self.save_state()
        img = self.photo()
        changed, _ = vision.slot_changes(self.prev, img, self.c, self.L)
        cost = vision.change_cost(changed, self.L, b0, b1)
        what = f"move {self.n}: {frm} -> {to}" + (f" ({off:.0f} mm off centre)" if off else "")
        self.log(f"   {what}   change check {cost}")
        note = ""
        if cost >= STOP_COST:
            img = self.check(img, f"{what}: more changed than this move. Look at the circles", mark=goal)
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


def main():
    import calib
    from board import Board
    from layout import Layout
    L = Layout.load()
    c = calib.load()
    moves = ask_int("How many moves this run? Enter = 10 (a short first try): ", 10)
    folder = DATA / f"{datetime.now():%Y%m%d-%H%M%S}"
    with Board() as b:
        b.home_if_needed()
        col = Collector(b, c, L, folder)
        state = None
        if not col.empty_board() and STATE.exists():
            k = col.key(col.photo(), ["Are the checkers where the last run left them?",
                                      "Enter = yes   o = no, I will set up the opening"], "\ro", "o")
            if k == "\r":
                state = json.loads(STATE.read_text())["checkers"]
        if not col.start(state=state):
            print("Stopped.")
            return
        for _ in range(moves):
            if not col.step():
                break
        k = col.key(col.drawing(col.prev), ["Done.  Enter = put everything back in the opening",
                                            "n = leave the checkers as they are"], "\rn", "n")
        if k == "\r":
            col.back_to_opening()
        col.ex.park()
    cv2.destroyAllWindows()
    print(f"Done: {col.n} photos in {folder}")


if __name__ == "__main__":
    ui.keep_open()
    main()
