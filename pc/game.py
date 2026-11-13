"""A whole game of backgammon on the robot board. Double-click, a browser page opens.

    py game.py            # real board if it is connected, otherwise the simulator
    (page: http://localhost:8765, also from a phone on the same WiFi: http://<PC IP>:8765)

Everything is chosen on the page (Settings), not with flags:
  Mode      robot vs human  - the robot plays its side, you move yours by hand
            robot vs robot  - demo / video: the robot plays and moves both sides
            watch           - two people play; the robot reads, checks and logs every move
                              (and, with Coach on, shows what gnubg would have played)
  Engine    gnubg (strong, ~1 s) -> simple (built in, weak) if gnubg is missing -> random
  Dice      digital (the PC rolls) / typed (real dice, you type them) / camera (real dice
            in the dice zone, read from below; you can correct it)
  Your move by hand + "Done" button / by hand + automatic (the robot watches until the
            board shows a legal move and stays still) / pick it on the page and the robot
            moves your checkers (remote play, or nobody wants to touch the board)
  Vision    norm (default, measured on the real board 12.11) / scan / slots / hough  (see vision.py)

How a human move is read: park the magnet in the lane, photo. First by change: compare
with the photo before the move (vision.closest_by_change); if exactly one legal move explains
the slots that changed, that is the move (no colours needed; notes, light and hidden spots
cancel out). Otherwise: vision.read, then the
board that fits best among the legal results of this roll (vision.closest). Exact fit ->
accepted. Not exact but the photo is consistent (15 + 15) -> accepted with a warning
(legality is only a warning, Nadav 7.11). Not consistent -> the page asks: read again
after fixing / take the closest legal move.
Every game: games/<date-time>.jsonl (one line per turn) and the photos next to it.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import json
import queue
import random
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import dice as dice_mod
import rules
import vision
from gnubg import BAR, OFF, Step  # noqa: F401
from layout import Layout

GAMES = Path(__file__).parent / "games"

DEFAULTS = {
    "mode": "robot_vs_human",    # robot_vs_human | robot_vs_robot | watch
    "engine": "gnubg",           # gnubg | simple | random
    "dice": "digital",           # digital | typed | camera
    "human_move": "done_button",  # done_button | auto | robot_moves
    "vision": "norm",            # norm | learned | cnn (train_detector.py) | scan | slots | hough
    "coach": False,              # watch / human turns: show gnubg's choice
    "first": "roll",             # roll (opening roll) | robot | human
    "board": "auto",             # auto (real, else simulator) | real | sim
}


class Restart(Exception):
    pass


class Game:
    def __init__(self, settings=None, log=print):
        self.s = dict(DEFAULTS, **(settings or {}))
        self.inbox = queue.Queue()
        self.lock = threading.Lock()
        self.log_fn = log
        self.L = Layout.load()
        self.b = None
        self.c = None
        self.is_sim = False
        self.model = None
        self.ex = None
        self.reset_state()

    # -- state shown on the page ---------------------------------------------
    def reset_state(self):
        self.board = list(rules.OPENING)   # robot's side
        self.turn = None                   # 'robot' | 'human'
        self.dice = None
        self.phase = "idle"
        self.prompt = "Press New game"
        self.asks = []                     # buttons the page should show
        self.legal = []                    # [(text)] for the move picker
        self.history = []
        self.warnings = []
        self.coach = None
        self.photo = None                  # jpeg bytes with what vision found
        self.last_img = None               # the last photo (of the board as it was known), for reading by change
        self.seen = None                   # the board the camera read last (the page can edit it)
        self.winner = None
        self.logfile = None

    def snapshot(self):
        with self.lock:
            own_off, opp_off = rules.off_count(self.board)
            return {"board": self.board, "turn": self.turn, "dice": self.dice, "phase": self.phase,
                    "prompt": self.prompt, "asks": self.asks, "legal": self.legal[:400],
                    "history": self.history[-40:], "warnings": self.warnings[-6:],
                    "coach": self.coach, "settings": self.s, "sim": self.is_sim,
                    "off": [own_off, opp_off], "pips": rules.pips(self.board), "winner": self.winner,
                    "has_photo": self.photo is not None, "photo_t": getattr(self, "photo_t", 0),
                    "pieces": self.pieces(), "labels": self.labels(), "seen": self.seen,
                    "bar_y": round(self.L._row_y(6), 1)}

    def pieces(self):
        """Checkers in the picture frame (x right, y from the near side), for the page."""
        out = []
        for (where, i), (xy, side) in self.L.place(self.board).items():
            x, y = self.L._nominal(where, i)
            out.append([round(x, 1), round(y, 1), side])
        return out

    def labels(self):
        """Point numbers as the person sees them (their 1 = the robot's 24)."""
        out = []
        for p in range(1, 25):
            x, y = self.L._nominal(p, 0)
            out.append([round(x, 1), round(y, 1), 25 - p])
        return out

    def say(self, prompt, phase=None, asks=()):
        with self.lock:
            self.prompt = prompt
            if phase:
                self.phase = phase
            self.asks = list(asks)
        self.log_fn(f"[{self.phase}] {prompt}")

    def warn(self, text):
        with self.lock:
            self.warnings.append(text)
        self.log_fn("WARNING " + text)

    # -- inbox: actions from the page / terminal ----------------------------
    def post(self, action):
        """action: {'name': 'new' | 'done' | 'dice' | 'accept' | 'reread' | 'closest' |
        'move' | 'settings' | 'learn' | 'stop', ...}"""
        if action.get("name") == "settings":
            with self.lock:
                self.s.update({k: v for k, v in action.get("values", {}).items() if k in DEFAULTS})
            return
        self.inbox.put(action)

    def drain(self):
        """Forget clicks made before the question was asked (a double-click on Done)."""
        keep = []
        while True:
            try:
                a = self.inbox.get_nowait()
            except queue.Empty:
                break
            if a["name"] in ("new", "stop"):
                keep.append(a)
        for a in keep:
            self.inbox.put(a)

    def wait(self, *names, timeout=None):
        """Next action whose name is in names. 'new' / 'stop' always interrupt."""
        end = None if timeout is None else time.time() + timeout
        while True:
            left = None if end is None else max(0.0, end - time.time())
            try:
                a = self.inbox.get(timeout=left)
            except queue.Empty:
                return None
            if a["name"] in ("new", "stop"):
                self.inbox.put(a)
                raise Restart()
            if a["name"] in names:
                return a

    # -- hardware ---------------------------------------------------------------
    def connect(self):
        if self.b is not None:
            return
        want = self.s["board"]
        if want in ("auto", "real"):
            try:
                import calib
                from board import Board
                self.c = calib.load()
                self.b = Board()
                self.b.home_if_needed()
                self.is_sim = False
                self.log_fn("Connected to the real board.")
            except Exception as e:
                if want == "real":
                    raise
                self.log_fn(f"Real board not available ({e}); using the simulator.")
                self.b = None
        if self.b is None:
            from sim import SimBoard
            self.b = SimBoard.with_board(rules.OPENING, self.L, noise_mm=0.5)
            self.c = self.b.calib
            self.is_sim = True
            import paths
            paths.GRAB_S = 0.0
        from executor import Executor
        self.ex = Executor(self.b, self.L, log=self.log_fn,
                           grab_s=0.0 if self.is_sim else 0.3)

    def photo_img(self):
        if getattr(self, "ex", None) is not None:
            self.ex.park()   # around the checkers, magnet off
        else:
            self.b.move(*self.L.park())
        if not self.is_sim:
            time.sleep(0.3)
        from helpers import photo
        img = photo(self.b)[0]   # a broken frame is taken again by itself
        self.c.check(img)   # calib.json made for another camera resolution: say so plainly
        return img

    def look(self):
        """Photo + vision. Keeps an annotated copy for the page."""
        img = self.photo_img()
        r = vision.read(img, self.c, self.L, self.model, self.s["vision"])
        view = vision.draw(img, self.c, r)
        import calib
        top = calib.view(view, self.c, xmax=self.L.xmax, ymax=self.L.ymax)[0]
        with self.lock:
            self.photo = cv2.imencode(".jpg", view)[1].tobytes()
            self.photo_top = cv2.imencode(".jpg", top)[1].tobytes()
            self.photo_t = time.time()
        if self.logfile:
            n = len(self.history)
            cv2.imencode(".jpg", img)[1].tofile(str(self.logfile.with_suffix("")) + f"_{n:03d}.jpg")  # works with Hebrew paths
        self.last_img = img
        with self.lock:
            self.seen = list(r.board) if r.board else list(self.board)
        return r, img

    def learn_colours(self):
        """Learn checker colours from the opening position (or load the saved model)."""
        img = self.photo_img()
        self.model = vision.learn(img, self.c, self.L, rules.OPENING)
        if not self.is_sim:
            self.model.save()
        r = vision.read(img, self.c, self.L, self.model, self.s["vision"])
        return r

    # -- engine -------------------------------------------------------------------
    def choose(self, board, dice_):
        """Best steps for the side on roll of `board` (that side's view)."""
        eng = self.s["engine"]
        if eng == "gnubg":
            try:
                import gnubg
                return gnubg.best_move(board, dice_), "gnubg"
            except Exception as e:
                self.warn(f"gnubg failed ({str(e)[:80]}), using the simple engine")
                eng = "simple"
        if eng == "random":
            return random.choice(rules.legal_moves(board, dice_))[0], "random"
        return rules.best_move_simple(board, dice_), "simple"

    # -- dice -----------------------------------------------------------------------
    def roll(self, who):
        how = self.s["dice"]
        if how == "digital":
            d = dice_mod.digital()
        elif how == "camera":
            self.say(f"{who}: roll the dice in the dice zone, then press Rolled", f"{who}_dice",
                     ["rolled", "dice_input"])
            while True:
                a = self.wait("rolled", "dice")
                if a["name"] == "dice":
                    d = dice_mod.parse(a.get("value", ""))
                    if d:
                        break
                    continue
                img = self.photo_img()
                d = dice_mod.from_photo(img, self.c, self.L)
                if d:
                    self.say(f"Camera reads {d[0]}-{d[1]}. OK?", asks=["accept", "dice_input"])
                    a = self.wait("accept", "dice")
                    if a["name"] == "dice" and dice_mod.parse(a.get("value", "")):
                        d = dice_mod.parse(a["value"])
                    break
                self.say("Could not read the dice. Type them, or roll again and press Rolled",
                         asks=["rolled", "dice_input"])
        else:
            self.say(f"{who}: roll and type the dice", f"{who}_dice", ["dice_input"])
            while True:
                a = self.wait("dice")
                d = dice_mod.parse(a.get("value", ""))
                if d:
                    break
                self.say("Two numbers 1-6, like 5 2", asks=["dice_input"])
        with self.lock:
            self.dice = list(d)
        return tuple(d)

    # -- turns ---------------------------------------------------------------------
    def record(self, who, d, steps, before, after, note=""):
        txt = rules.move_text(steps) if steps is not None else "?"
        item = {"n": len(self.history) + 1, "who": who, "dice": list(d), "move": txt,
                "board_before": before, "board_after": after, "note": note,
                "t": datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            self.history.append(item)
        if self.logfile:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")

    def robot_turn(self, side="robot"):
        """The robot plays `side` ('robot', or 'human' in robot vs robot / robot_moves)."""
        d = self.roll(side)
        before = list(self.board)
        view = self.board if side == "robot" else rules.flip(self.board)
        steps, eng = self.choose(view, d)
        if not steps:
            self.say(f"{side} rolled {d[0]}-{d[1]}: no move", "robot_move")
            self.record(side, d, [], self.board, self.board, "no legal move")
            return
        after_view = self.after(view, d, steps)
        txt = rules.move_text(steps)
        self.say(f"{side} rolled {d[0]}-{d[1]}, plays {txt} ({eng})", "robot_move")
        self.ex.play(steps, side=side)
        expected = after_view if side == "robot" else rules.flip(after_view)
        expected = self.check_after_robot(expected)
        with self.lock:
            self.board = expected
        self.record(side, d, steps, before, expected)

    def after(self, view, d, steps):
        """Board after steps (mover's view): from the legal move with the same steps."""
        for s, nb in rules.legal_moves(view, d):
            if sorted((x.frm, x.to) for x in s) == sorted((x.frm, x.to) for x in steps):
                return list(nb)
        from executor import expected_after
        return expected_after(view, steps)

    def check_after_robot(self, expected):
        """Photo after a robot move. Returns the board to go on with: expected, or the one
        fixed on the page (Edit)."""
        before, board0 = self.last_img, list(self.board)
        r, img = self.look()
        if r.board == expected:
            self.ex.set_from_reading(r)
            return expected
        if before is not None:   # what changed since the last photo is exactly this move: fine
            changed, _ = vision.slot_changes(before, img, self.c, self.L)
            if vision.change_cost(changed, self.L, board0, expected) == 0:
                self.log_fn("after the robot's move: read by change (the photo matches the move)")
                self.ex.set_from_board(expected)
                return expected
        diff = [i for i in range(26) if r.board[i] != expected[i]]
        self.warn(f"after the robot's move the camera sees something else at {diff[:6]}"
                  + (f": {r.warnings[0]}" if r.warnings else ""))
        self.say("The board does not look as expected. Fix the checkers by hand and press Read again, "
                 "Accept to go on, or fix the reading on the board below and press Use this board",
                 "check", ["reread", "accept", "edit"])
        while True:
            a = self.wait("reread", "accept", "board")
            if a["name"] == "accept":
                self.ex.set_from_board(expected)
                return expected
            if a["name"] == "board" and self._valid(a.get("value")):
                b = [int(v) for v in a["value"]]
                self.ex.set_from_board(b)
                if b != expected:
                    self.warn("board fixed on the page: it differs from the robot's move")
                return b
            r, _ = self.look()
            if r.board == expected:
                self.ex.set_from_reading(r)
                return expected

    @staticmethod
    def _valid(b):
        """A board from the page: 26 numbers, at most 15 checkers a side."""
        try:
            b = [int(v) for v in b]
        except (TypeError, ValueError):
            return False
        return (len(b) == 26 and sum(v for v in b if v > 0) <= 15 and -sum(v for v in b if v < 0) <= 15
                and b[0] >= 0 and b[25] <= 0)

    def human_turn(self, side="human"):
        d = self.roll(side)
        hv = rules.flip(self.board) if side == "human" else list(self.board)
        plays = rules.legal_moves(hv, d)
        if self.s["coach"]:
            steps, eng = self.choose(hv, d)
            with self.lock:
                self.coach = f"{eng} would play {rules.move_text(steps)}"
        if not plays[0][0]:
            self.say(f"{side} rolled {d[0]}-{d[1]}: no legal move, turn passes", "human_move")
            self.record(side, d, [], self.board, self.board, "no legal move")
            time.sleep(1.5)
            return
        texts = sorted(rules.move_text(s) for s, _ in plays)
        with self.lock:
            self.legal = texts
        to_robot = (lambda nb: rules.flip(nb)) if side == "human" else (lambda nb: list(nb))
        cands = [to_robot(nb) for _, nb in plays]
        how = self.s["human_move"]
        before = self.last_img   # photo of the board before this move (after the last one)

        if how == "robot_moves":
            self.drain()
            self.say(f"{side} rolled {d[0]}-{d[1]}: pick your move", "human_move", ["move_pick"])
            while True:
                a = self.wait("move")
                hit = [(s, nb) for s, nb in plays if rules.move_text(s) == a.get("value")]
                if hit:
                    break
            steps, nb = hit[0]
            self.ex.play(steps, side=side)
            got = self.check_after_robot(to_robot(nb))
            self.finish_human(side, d, steps, got, "moved by the robot")
            return

        self.drain()
        if self.is_sim:   # nobody's hand in the simulator: the page's picker moves "by hand"
            self.say(f"{side} rolled {d[0]}-{d[1]}. Simulator: pick the move the 'person' makes "
                     "(sloppy, by hand), then Done", "human_move", ["move_pick", "done"])
        else:
            self.say(f"{side} rolled {d[0]}-{d[1]}. Move your checkers, then press Done"
                     + (" (or just wait: auto)" if how == "auto" else ""), "human_move", ["done"])
        if how == "auto" and not self.is_sim:
            r = self.wait_still(cands)
        else:
            while True:
                a = self.wait("done", "move")
                if a["name"] == "move" and self.is_sim:
                    self.sim_hand_move(hv, d, a.get("value"), side)
                    continue
                if a["name"] == "done":
                    break
            r, _ = self.look()
        if before is not None and self.last_img is not before:
            best, cost, margin = vision.closest_by_change(before, self.last_img, self.c, self.L,
                                                          self.board, cands)
            agree = vision.closest(r, cands)[0] == best
            # one legal move explains what changed: clearly better than the rest, or the
            # second best is just one slot worse and the colour reading picks the same move
            if cost == 0 and (margin >= 2 or (margin >= 1 and agree)):
                steps = next(s for s, nb in plays if to_robot(nb) == best)
                self.log_fn(f"{side}'s move read by change: {rules.move_text(steps)}")
                self.finish_human(side, d, steps, best, "read by change")
                self.ex.set_from_board(best)
                return
        self.accept_reading(r, cands, plays, hv, d, side, to_robot)

    def sim_hand_move(self, hv, d, text, side):
        for s, nb in rules.legal_moves(hv, d):
            if rules.move_text(s) == text:
                from executor import Executor
                ghost = Executor(self.b, self.L, drag=None, log=lambda *a: None)
                ghost.places, ghost.side_of = self.ex.places, self.ex.side_of
                opp = "robot" if side == "human" else "human"
                for st in s:
                    to_key = ghost._key(st.to, side)
                    if st.hit:
                        self._hand(ghost, to_key, f"bar_{opp}")
                    self._hand(ghost, ghost._key(st.frm, side), to_key)
                self.say("Moved (simulated hand). Press Done", asks=["done"])
                return

    def _hand(self, ex, frm, to):
        lst = ex.places[frm]
        mid = self.L.xmax / 2
        top = min(range(len(lst)), key=lambda i: abs(self.L.from_magnet(*lst[i])[0] - mid))
        p = lst.pop(top)
        goal = ex._free_slot(to)
        self.b.human_move(p, goal, sloppy=2.5)
        ex.places.setdefault(to, []).append(list(goal))

    def wait_still(self, cands, period=1.0):
        """Auto 'done': read every second; done when 2 reads in a row show the same legal result."""
        last = None
        while True:
            if self.wait("done", timeout=period):
                r, _ = self.look()
                return r
            r, _ = self.look()
            best, cost = vision.closest(r, cands)
            if cost == 0 and best != self.board:
                if last == best:
                    return r
                last = best
            else:
                last = None

    def accept_reading(self, r, cands, plays, hv, d, side, to_robot):
        while True:
            best, cost = vision.closest(r, cands)
            steps = next(s for s, nb in plays if to_robot(nb) == best)
            if cost == 0:
                self.finish_human(side, d, steps, best)
                self.ex.set_from_reading(r)
                return
            consistent = not r.warnings
            if consistent and r.board != self.board:
                self.warn(f"{side}'s move is not legal for {d[0]}-{d[1]} (closest legal: "
                          f"{rules.move_text(steps)}). Taken as it is on the board.")
                self.finish_human(side, d, None, r.board, "not legal, taken as seen")
                self.ex.set_from_reading(r)
                return
            with self.lock:
                self.legal = sorted(rules.move_text(s) for s, _ in plays)
            self.say("I can't read the move: " + "; ".join(r.warnings[:2] or ["board unchanged"])
                     + f". Closest legal move: {rules.move_text(steps)}. Or pick the move you made, "
                     "or fix the reading on the board below", "check",
                     ["reread", "closest", "move_pick", "edit"])
            a = self.wait("reread", "closest", "done", "move", "board")
            if a["name"] == "move":
                hit = [(s, nb) for s, nb in plays if rules.move_text(s) == a.get("value")]
                if hit:
                    self.finish_human(side, d, hit[0][0], to_robot(hit[0][1]), "move picked on the page")
                    self.ex.set_from_board(to_robot(hit[0][1]))
                    return
                continue
            if a["name"] == "board" and self._valid(a.get("value")):
                b = [int(v) for v in a["value"]]
                legal = [s for s, nb in plays if to_robot(nb) == b]
                self.finish_human(side, d, legal[0] if legal else None, b,
                                  "fixed on the page" + ("" if legal else " (not a legal move)"))
                self.ex.set_from_board(b)
                return
            if a["name"] == "closest":
                self.finish_human(side, d, steps, best, "closest legal move, chosen on the page")
                self.ex.set_from_board(best)
                return
            r, _ = self.look()

    def finish_human(self, side, d, steps, after, note=""):
        before = self.board
        with self.lock:
            self.board = list(after)
            self.legal = []
        self.record(side, d, steps, before, list(after), note)

    # -- game ---------------------------------------------------------------------
    def setup(self):
        self.connect()
        GAMES.mkdir(exist_ok=True)
        self.logfile = GAMES / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        if self.model is None:
            try:
                if self.is_sim:
                    raise OSError
                self.model = vision.Model.load()
                if self.s["vision"] == "norm" and not self.model.norm:
                    raise ValueError("colours saved before the norm method: learn again")
            except (OSError, ValueError, KeyError):
                self.say("Put the checkers in the opening position, then press Learn colours",
                         "setup", ["learn"])
                if not self.is_sim:
                    self.wait("learn")
                self.learn_colours()
        r, _ = self.look()
        if r.board != rules.OPENING:
            self.say("The board is not in the opening position: " + "; ".join(r.warnings[:2])
                     + ". If it is, press Learn colours", "setup", ["learn", "arrange", "reread", "accept"])
            while r.board != rules.OPENING:
                a = self.wait("arrange", "reread", "accept", "learn")
                if a["name"] == "accept":
                    break
                if a["name"] == "learn":
                    r = self.learn_colours()
                    continue
                if a["name"] == "arrange":
                    self.say("Arranging the opening position...", "setup")
                    self.ex.set_from_reading(r)
                    n = self.ex.arrange(rules.OPENING)
                    self.log_fn(f"arranged with {n} drags")
                r, _ = self.look()
        self.ex.set_from_reading(r) if r.board == rules.OPENING else self.ex.set_from_board(rules.OPENING)
        with self.lock:
            self.board = list(rules.OPENING)

    def play(self):
        self.setup()
        first = self.s["first"]
        if first == "roll":
            who, d = dice_mod.opening_roll()
            self.say(f"Opening roll: robot {max(d) if who == 'robot' else min(d)}, "
                     f"human {max(d) if who == 'human' else min(d)}. {who} starts", "opening")
            time.sleep(1.0)
        else:
            who = first
        mode = self.s["mode"]
        while True:
            with self.lock:
                self.turn = who
                self.coach = None
            if who == "robot":
                if mode == "watch":
                    self.human_turn("robot")      # 'robot' side played by a second person
                else:
                    self.robot_turn("robot")
            else:
                if mode == "robot_vs_robot":
                    self.robot_turn("human")
                else:
                    self.human_turn("human")
            w = rules.winner(self.board)
            if w:
                self.game_over("robot" if w == "own" else "human")
                return
            who = "human" if who == "robot" else "robot"

    def game_over(self, who):
        b = self.board if who == "robot" else rules.flip(self.board)
        loser_off = rules.off_count(b)[1]
        kind, pts = "single", 1
        if loser_off == 0:
            kind, pts = "gammon", 2
            if b[25] < 0 or any(n < 0 for n in b[1:7]):
                kind, pts = "backgammon", 3
        with self.lock:
            self.winner = {"who": who, "kind": kind, "points": pts}
        self.say(f"{who} wins ({kind}, {pts} point{'s' if pts > 1 else ''})", "over", ["new"])

    def run_forever(self):
        """Game thread: wait for New game, play, repeat."""
        while True:
            try:
                a = self.inbox.get()
                if a["name"] == "stop":
                    return
                if a["name"] != "new":
                    continue
                with self.lock:
                    self.reset_state()
                self.play()
            except Restart:
                a = self.inbox.get()
                if a["name"] == "stop":
                    return
                self.inbox.put(a)
            except Exception as e:
                traceback.print_exc()
                self.say(f"Error: {e}", "error", ["new"])


if __name__ == "__main__":
    import ui
    ui.keep_open()
    import webui
    webui.main()
