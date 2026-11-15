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
  Your move by hand + "I'm done" button / pick it on the page and the robot moves your
            checkers (remote play, or nobody wants to touch the board)

The game never looks for checkers in the photo (Nadav 15.11). It starts from the opening
position (you set it up and press Start, or the robot arranges it from the trays) and from
then on only updates the known board with every move:
  the robot's move   is taken as carried out (no check, no stop).
  a person's move    the "depressive" reading (Nadav 14.11): park the magnet, photo,
                     compare with the photo before the move (vision.rank_by_change). Every
                     legal result of the roll gets a cost: how far what changed is from what
                     that move changes (soft counts per place; the bear-off trays are not
                     judged, so checkers may leave the glass). The best is taken only when
                     its cost <= CHANGE_OK and the next one is at least CHANGE_MARGIN worse;
                     otherwise the page asks, the most likely moves first (or fix the
                     checkers and read again, or fix the board on the page).
The drawn board and the circles on the camera's picture are the same known board.
Every game has its own folder: games/<date-time>/game.jsonl (one line per turn) and the
photos next to it, every one kept: 000_move00.jpg, 001_move00.jpg, 002_move01.jpg, ...
(running number + how many moves were played). The folder appears at the first move.
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
# The depressive reading takes a move only if it explains the photos this well ... (3 games on
# 14-15.11, 33 moves: the read ones were <= 2.0; of the 6 stops the right move was always first,
# three of them at 2.2-3.0 with the next one 2+ behind, so 2.0 -> 3.5, Nadav 15.11)
CHANGE_OK = 3.5
CHANGE_MARGIN = 1.0   # ... and the next legal move is at least this much worse (else: ask)
TRAYS = ("off_robot", "off_human")   # not judged: a person may take checkers off the glass altogether
ROBOT_FIT_WARN = 6.0  # the photo after a robot move fits it worse than this: a quiet note at the bottom
NO_MOVE_S = 2.5       # how long "no legal move" stays on the page when the robot cannot play
ALL_OUT = [0] * 26    # every checker in the bear-off trays (as place_trays.py leaves them)

DEFAULTS = {
    "mode": "robot_vs_human",    # robot_vs_human | robot_vs_robot | watch
    "engine": "gnubg",           # gnubg | simple | random
    "dice": "digital",           # digital | typed | camera
    "human_move": "done_button",  # done_button | robot_moves
    "coach": False,              # watch / human turns: show gnubg's choice
    "first": "human",            # human (always, Nadav 14.11) | robot | roll (opening roll)
    "board": "auto",             # auto (real, else simulator) | real | sim
}


class Restart(Exception):
    pass


class _Upside:
    """A top view's pixel map (calib.Flat) for the same picture turned upside down."""

    def __init__(self, f, height):
        self.f, self.h, self.k = f, height, f.k

    def mm_to_px(self, x, y):
        u, v = self.f.mm_to_px(x, y)
        return u, self.h - 1 - np.asarray(v, float)


class Game:
    def __init__(self, settings=None, log=print):
        self.s = dict(DEFAULTS, **{k: v for k, v in (settings or {}).items() if k in DEFAULTS})
        self.inbox = queue.Queue()
        self.robot_white = False    # whose checkers are the white ones (found at the opening; the page's swap button turns it)
        self.busy = False           # the game is working (not waiting for a click): the page shows it
        self.lock = threading.Lock()
        self.log_fn = log
        self.L = Layout.load()
        self.b = None
        self.c = None
        self.is_sim = False
        self.ex = None
        self.last_board = None      # where the last game stopped (the robot can arrange the opening from it)
        self.reset_state()

    # -- state shown on the page ---------------------------------------------
    def reset_state(self):
        self.board = list(rules.OPENING)   # robot's side
        self.turn = None                   # 'robot' | 'human'
        self.dice = None
        self.phase = "idle"
        self.prompt = "Press New game"
        self.info = ""                     # the small line under the board (what the robot played)
        self.asks = []                     # buttons the page should show
        self.legal = []                    # [(text)] for the move picker
        self.likely = []                   # the most likely moves, when the reading is not sure
        self.history = []
        self.warnings = []
        self.coach = None
        self.photo = None                  # jpeg bytes: the camera photo (shown upside down, like the drawing)
        self.photo_top = None              # jpeg bytes: the top view with the known board circled
        self.top_base = None               # the top view before anything is drawn on it
        self.marks = []                    # [((x, y), side)] the known checkers circled on the top view
        self.paths = []                    # the robot's drag paths of this turn (magnet mm), drawn on the page
        self.last_img = None               # the last photo (of the board as it is known), for reading by change
        self.winner = None
        self.logfile = None
        self.shots = []                    # photos taken before the first move (the folder is made then)
        self.shot_n = 0                    # running number of the game's photos

    def name(self, side):
        mode = self.s["mode"]
        if mode == "robot_vs_robot":
            return "Robot A" if side == "robot" else "Robot B"
        if mode == "watch":
            return "Player 1" if side == "robot" else "Player 2"
        return "Robot" if side == "robot" else "You"

    def players(self):
        mode = self.s["mode"]
        return {"robot": {"name": self.name("robot"), "kind": "person" if mode == "watch" else "computer"},
                "human": {"name": self.name("human"), "kind": "computer" if mode == "robot_vs_robot" else "person"}}

    def snapshot(self):
        with self.lock:
            own_off, opp_off = rules.off_count(self.board)
            return {"board": self.board, "turn": self.turn, "dice": self.dice, "phase": self.phase,
                    "prompt": self.prompt, "info": self.info, "asks": self.asks, "legal": self.legal[:400],
                    "likely": self.likely, "busy": self.busy,
                    "history": self.history[-40:], "warnings": self.warnings[-6:],
                    "coach": self.coach, "settings": self.s, "sim": self.is_sim,
                    "off": [own_off, opp_off], "pips": rules.pips(self.board), "winner": self.winner,
                    "has_photo": self.photo is not None, "photo_t": getattr(self, "photo_t", 0),
                    "pieces": self.pieces(), "labels": self.labels(), "seen": self.board,
                    "bar_y": round(self.L._row_y(6), 1), "frame": self.frame(),
                    "paths": [[[round(x, 1), round(y, 1)] for x, y in p] for p in self.paths],
                    "players": self.players(), "robot_white": self.robot_white}

    def frame(self):
        """Picture frame mm -> magnet mm (= the corrected photo's axes) as a 2x3 matrix, for the page."""
        o, ex, ey = (np.subtract(self.L.to_magnet(*p), 0) for p in ((0, 0), (1, 0), (0, 1)))
        return [[round(ex[0] - o[0], 6), round(ey[0] - o[0], 6), round(o[0], 3)],
                [round(ex[1] - o[1], 6), round(ey[1] - o[1], 6), round(o[1], 3)]]

    def where(self):
        """[((x, y) magnet mm, side)] every checker of the known board: where the robot left the
        ones it moved (so the page follows a move checker by checker), the layout's places otherwise."""
        ex = self.ex
        if ex is not None and ex.places:
            try:
                out = []
                for key, lst in list(ex.places.items()):
                    side = (ex.side_of.get(key) if isinstance(key, int)
                            else key.split("_")[1] if key.startswith(("bar_", "off_")) else self.turn)
                    out += [((float(x), float(y)), side or "robot") for x, y in list(lst)]
                return out
            except Exception:   # the game thread changed it meanwhile: the plain board will do
                pass
        return [((float(xy[0]), float(xy[1])), side) for xy, side in self.L.place(self.board).values()]

    def pieces(self):
        """Checkers in the picture frame (x right, y from the near side), for the page."""
        out = []
        for (x, y), side in self.where():
            fx, fy = self.L.from_magnet(x, y)
            out.append([round(fx, 1), round(fy, 1), side])
        return out

    def labels(self):
        """Point numbers as the person sees them (their 1 = the robot's 24)."""
        out = []
        for p in range(1, 25):
            x, y = self.L._nominal(p, 0)
            out.append([round(x, 1), round(y, 1), 25 - p])
        return out

    def say(self, prompt, phase=None, asks=(), info=None):
        with self.lock:
            self.prompt = prompt
            if phase:
                self.phase = phase
            if info is not None:
                self.info = info
            self.asks = list(asks)
        self.log_fn(f"[{self.phase}] {prompt}" + (f"  ({info})" if info else ""))

    def warn(self, text):
        with self.lock:
            self.warnings.append(text)
        self.log_fn("WARNING " + text)

    # -- inbox: actions from the page / terminal ----------------------------
    def post(self, action):
        """action: {'name': 'new' | 'start' | 'arrange' | 'done' | 'dice' | 'rolled' | 'accept' |
        'reread' | 'move' | 'board' | 'settings' | 'swap_colours' | 'stop', ...}"""
        if action.get("name") == "swap_colours":   # the page's swap button: whose checkers are white
            self.robot_white = not self.robot_white
            self.redraw()
            return
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
        self.busy = False
        try:
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
        finally:
            self.busy = True

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
            self.sim_board(rules.OPENING)
        self.new_executor()

    def sim_board(self, board):
        from sim import SimBoard
        self.b = SimBoard.with_board(board, self.L, noise_mm=0.5)
        self.c = self.b.calib
        self.is_sim = True
        import paths
        paths.GRAB_S = 0.0

    def new_executor(self):
        from executor import Executor
        self.ex = Executor(self.b, self.L, log=self.log_fn, grab_s=0.0 if self.is_sim else 0.3)
        self.ex.on_path = self.show_path

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

    def show_path(self, path):
        """The executor is about to drag a checker along path: draw it on the page (like collect)."""
        with self.lock:
            self.paths = self.paths[-7:] + [[(float(x), float(y)) for x, y in path]]
        self.redraw()

    def mark(self):
        """Circle the known board on the camera's picture (no checker is looked for in the photo)."""
        self.marks = self.where()
        self.redraw()

    def redraw(self):
        """The page's pictures. Top view, upside down like the drawing (the numbers drawn after
        the turn, so they read the right way): every checker of the known board circled in its
        own colour (black / white) and the robot's drag paths of this turn in yellow."""
        if self.top_base is None:
            return
        import calib
        top = self.top_base.copy()
        f = _Upside(calib.Flat(self.c), top.shape[0])
        px = lambda xy: tuple(int(round(float(t))) for t in f.mm_to_px(*xy))  # noqa: E731
        r = int(round(self.L.checker / 2 * f.k))
        white, black = (240, 242, 242), (24, 24, 24)
        for xy, side in self.marks:
            is_white = (side == "robot") == self.robot_white
            cv2.circle(top, px(xy), r, black if is_white else white, 5, cv2.LINE_AA)
            cv2.circle(top, px(xy), r, white if is_white else black, 3, cv2.LINE_AA)
        for n, path in enumerate(self.paths):
            pts = [px(p) for p in path]
            last = n == len(self.paths) - 1
            col = (0, 230, 255) if last else (0, 150, 170)
            if len(pts) > 2:
                cv2.polylines(top, [np.array(pts[:-1], np.int32)], False, col, 3, cv2.LINE_AA)
            if len(pts) > 1:
                cv2.arrowedLine(top, pts[-2], pts[-1], col, 3, cv2.LINE_AA,
                                tipLength=min(0.5, 18.0 / max(1.0, float(np.hypot(*np.subtract(pts[-1], pts[-2]))))))
                cv2.circle(top, pts[0], 5, col, -1, cv2.LINE_AA)
        with self.lock:
            self.photo_top = cv2.imencode(".jpg", top, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            self.photo_t = time.time()

    def look(self):
        """A photo of the board as it is known: kept in the game's folder, and it becomes the
        'before' of the next move read by change. Nothing is detected in it."""
        img = self.photo_img()
        if self.logfile:
            # every photo is kept: its running number + the number of moves played when it was taken
            name = f"{self.shot_n:03d}_move{len(self.history):02d}.jpg"
            self.shot_n += 1
            self.shots = (self.shots + [(name, cv2.imencode(".jpg", img)[1])])[-6:]
            self.flush_shots()
        self.last_img = img
        import calib
        top, f = calib.view(img, self.c, xmax=self.L.xmax, ymax=self.L.ymax, board=False)
        top = cv2.flip(top, 0)
        try:
            top = calib.draw_board(top, _Upside(f, top.shape[0]), self.L)
        except Exception:   # never lose the photo over the drawing
            pass
        with self.lock:
            self.top_base = top
            self.paths = []
            self.photo = cv2.imencode(".jpg", cv2.flip(img, 0))[1].tobytes()
        self.mark()
        return img

    def set_board(self, board):
        """The known board after a move that the robot did not carry out itself."""
        with self.lock:
            self.board = list(board)
            self.legal, self.likely = [], []
        self.ex.set_from_board(board)
        self.mark()

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
        who = self.name(who)
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
    def flush_shots(self):
        """Write the waiting photos, once the game has its folder (made at the first move, so
        a game that was only opened leaves no folder behind)."""
        if self.logfile and self.logfile.parent.exists():
            for name, buf in self.shots:
                buf.tofile(str(self.logfile.parent / name))   # works with Hebrew paths
            self.shots = []

    def shown(self, steps, side):
        """The move in the numbers written on the page's board (the 'human' side's): the 'robot'
        side's own numbers run the other way (its 24 is the page's 1)."""
        if steps is None:
            return "?"
        if side == "human":
            return rules.move_text(steps)
        name = lambda p: "bar" if p == BAR else "off" if p == OFF else str(25 - p)  # noqa: E731
        return " ".join(f"{name(x.frm)}/{name(x.to)}{'*' if x.hit else ''}" for x in steps)

    def record(self, who, d, steps, before, after, note=""):
        txt = rules.move_text(steps) if steps is not None else "?"
        item = {"n": len(self.history) + 1, "who": who, "dice": list(d), "move": txt,
                "shown": self.shown(steps, who),
                "board_before": before, "board_after": after, "note": note,
                "t": datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            self.history.append(item)
        if self.logfile:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)
            self.flush_shots()
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")

    def robot_turn(self, side="robot"):
        """The robot plays `side` ('robot', or 'human' in robot vs robot)."""
        d = self.roll(side)
        who = self.name(side)
        before = list(self.board)
        view = self.board if side == "robot" else rules.flip(self.board)
        steps, eng = self.choose(view, d)
        if not steps:
            # nothing to play: leave it on the page for a moment before the turn passes (Nadav 15.11)
            self.say(f"{who}: no legal move", "robot_move", info=f"{who} rolled {d[0]}-{d[1]}: no legal move")
            self.record(side, d, [], self.board, self.board, "no legal move")
            time.sleep(NO_MOVE_S)
            return
        after_view = self.after(view, d, steps)
        self.say(f"{who} is playing…", "robot_move",
                 info=f"{who} rolled {d[0]}-{d[1]}, plays {self.shown(steps, side)} ({eng})")
        self.ex.play(steps, side=side)
        expected = after_view if side == "robot" else rules.flip(after_view)
        # the robot's move is taken as carried out (Nadav 15.11): no reading, no stop
        note = self.after_robot(before, expected)
        self.record(side, d, steps, before, expected, note)

    def after_robot(self, board0, expected):
        """The known board becomes `expected`; a photo of it (the 'before' of the next move).
        How well the photos fit the move only goes into the log (and a quiet note if badly)."""
        last = self.last_img
        with self.lock:
            self.board = list(expected)
        img = self.look()
        try:
            fit = vision.rank_by_change(last, img, self.c, self.L, board0, [expected], TRAYS)[0][0]
        except Exception as e:
            self.log_fn(f"fit after the robot's move: {e}")
            return ""
        if fit > ROBOT_FIT_WARN:
            self.warn(f"move {len(self.history) + 1}: the photo after the robot's move fits it poorly "
                      f"({fit:.1f}). The game goes on as if it was carried out; look at the board if it gets stuck")
        return f"photo fit {fit:.1f}"

    def after(self, view, d, steps):
        """Board after steps (mover's view): from the legal move with the same steps."""
        for s, nb in rules.legal_moves(view, d):
            if sorted((x.frm, x.to) for x in s) == sorted((x.frm, x.to) for x in steps):
                return list(nb)
        from executor import expected_after
        return expected_after(view, steps)

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
        who = self.name(side)
        yours = "Your" if who == "You" else who + "'s"
        hv = rules.flip(self.board) if side == "human" else list(self.board)
        plays = rules.legal_moves(hv, d)
        if self.s["coach"]:
            steps, eng = self.choose(hv, d)
            with self.lock:
                self.coach = f"{eng} would play {rules.move_text(steps)}"
        self.drain()
        if not plays[0][0]:
            # nothing to play: the person still presses Done, so the turn never passes unseen (Nadav 15.11)
            self.say(f"No legal move for {d[0]}-{d[1]}. Press I'm done", "human_move", ["done"])
            self.wait("done")
            self.record(side, d, [], self.board, self.board, "no legal move")
            return
        texts = sorted(rules.move_text(s) for s, _ in plays)
        with self.lock:
            self.legal = texts
        to_robot = (lambda nb: rules.flip(nb)) if side == "human" else (lambda nb: list(nb))
        cands = [to_robot(nb) for _, nb in plays]
        before = self.last_img   # photo of the board before this move (after the last one)

        if self.s["human_move"] == "robot_moves":
            self.say(f"{yours} turn: pick the move", "human_move", ["move_pick"])
            while True:
                a = self.wait("move")
                hit = [(s, nb) for s, nb in plays if rules.move_text(s) == a.get("value")]
                if hit:
                    break
            steps, nb = hit[0]
            board0 = list(self.board)
            with self.lock:
                self.legal = []
            self.say(f"The robot moves {yours.lower() if who == 'You' else yours} checkers…", "robot_move",
                     info=f"{who} rolled {d[0]}-{d[1]}, plays {self.shown(steps, side)} (moved by the robot)")
            self.ex.play(steps, side=side)
            note = self.after_robot(board0, to_robot(nb))
            self.record(side, d, steps, board0, to_robot(nb), ("moved by the robot; " + note).rstrip("; "))
            return

        if self.is_sim:   # nobody's hand in the simulator: the page's picker moves "by hand"
            self.say(f"{yours} turn (simulator): pick the move, then I'm done", "human_move", ["move_pick", "done"])
        else:
            self.say(f"{yours} turn", "human_move", ["done"])
        while True:
            a = self.wait("done", "move")
            if a["name"] == "move" and self.is_sim:
                self.sim_hand_move(hv, d, a.get("value"), side)
                continue
            if a["name"] == "done":
                break
        self.say("Reading the move…", "reading")
        self.look()
        self.read_by_change(before, cands, plays, d, side, to_robot)

    def read_by_change(self, before, cands, plays, d, side, to_robot):
        """The "depressive" reading (Nadav 14.11): the new board is always one of the legal
        results of the roll, the one that best explains what changed between the photos.
        Taken only when it fits well (cost <= CHANGE_OK) AND is clearly ahead of the next one
        (by CHANGE_MARGIN); otherwise the game stops and asks, offering the best few first."""
        while True:
            ranked = vision.rank_by_change(before, self.last_img, self.c, self.L, self.board, cands, TRAYS)
            c1, best = ranked[0]
            c2 = ranked[1][0] if len(ranked) > 1 else 99.0
            text = lambda cb: next(rules.move_text(s) for s, nb in plays if to_robot(nb) == cb)  # noqa: E731
            self.log_fn(f"{side}'s move by change: {text(best)} cost {c1:.2f}, next {c2:.2f}")
            if c1 <= CHANGE_OK and c2 - c1 >= CHANGE_MARGIN:
                steps = next(s for s, nb in plays if to_robot(nb) == best)
                self.finish_human(side, d, steps, best, f"read by change ({c1:.1f} / {c2:.1f})")
                return
            top = [text(cb) for _, cb in ranked[:3]]
            with self.lock:
                self.likely = top
                self.legal = top + sorted(t for t in (rules.move_text(s) for s, _ in plays) if t not in top)
            self.say("Which move did you make?" if side == "human" and self.s["mode"] != "watch"
                     else f"Which move did {self.name(side)} make?", "check",
                     ["likely", "move_pick", "reread", "edit"])
            a = self.wait("move", "reread", "board")
            if a["name"] == "move":
                hit = [(s, nb) for s, nb in plays if rules.move_text(s) == a.get("value")]
                if hit:
                    rank = next((i for i, (_, cb) in enumerate(ranked) if cb == to_robot(hit[0][1])), -1)
                    self.finish_human(side, d, hit[0][0], to_robot(hit[0][1]),
                                      f"picked on the page (was no. {rank + 1} by change; {c1:.1f} / {c2:.1f})")
                    return
                continue
            if a["name"] == "board" and self._valid(a.get("value")):
                b = [int(v) for v in a["value"]]
                legal = [s for s, nb in plays if to_robot(nb) == b]
                self.finish_human(side, d, legal[0] if legal else None, b,
                                  "fixed on the page" + ("" if legal else " (not a legal move)"))
                return
            self.say("Reading the move…", "reading")
            self.look()

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
                self.say("Moved (simulated hand). Press I'm done", asks=["done"])
                return

    def _hand(self, ex, frm, to):
        lst = ex.places[frm]
        mid = self.L.xmax / 2
        top = min(range(len(lst)), key=lambda i: abs(self.L.from_magnet(*lst[i])[0] - mid))
        p = lst.pop(top)
        goal = ex._free_slot(to)
        self.b.human_move(p, goal, sloppy=2.5)
        ex.places.setdefault(to, []).append(list(goal))

    def finish_human(self, side, d, steps, after, note=""):
        before = self.board
        self.set_board(after)
        self.record(side, d, steps, before, list(after), note)

    # -- game ---------------------------------------------------------------------
    def setup(self):
        """The opening position is taken as it stands (Nadav 15.11): Start, or the robot
        arranges it from the trays first. Nothing is read from the photo."""
        self.say("Connecting…", "setup")
        self.connect()
        # one folder per game (Nadav 14.11), made when the first move is recorded
        self.logfile = GAMES / f"{datetime.now():%Y%m%d-%H%M%S}" / "game.jsonl"
        with self.lock:
            self.board = list(rules.OPENING)
        self.ex.set_from_board(rules.OPENING)
        self.look()
        while True:
            self.drain()
            again = bool(self.last_board) and self.last_board != rules.OPENING
            self.say("Set up the opening position, then press Start", "setup",
                     ["start", "arrange"] + (["rearrange"] if again else []))
            a = self.wait("start", "arrange", "rearrange")
            if a["name"] == "start":
                break
            if a["name"] == "rearrange":
                self.arrange_from_last()
            else:
                self.arrange_opening()
        self.say("Starting…", "setup")
        self.look()
        self.find_side_colours()
        self.redraw()

    def arrange_opening(self):
        """The robot drags all 30 checkers from the bear-off trays (where place_trays.py or the
        end of a game leaves them) to the opening position."""
        import arrange_opening as ao
        if self.is_sim:
            self.sim_board(ALL_OUT)
            self.new_executor()
        with self.lock:
            self.board = list(ALL_OUT)
        self.ex.set_from_board(ALL_OUT)
        self.say("The robot is arranging the opening position…", "setup")
        self.look()
        _, n = ao.arrange(self.b, self.L, ALL_OUT, rules.OPENING, log=self.log_fn, ex=self.ex)
        self.log_fn(f"arranged with {n} drags")
        with self.lock:
            self.board = list(rules.OPENING)
        self.ex.set_from_board(rules.OPENING)
        self.look()

    def arrange_from_last(self):
        """The robot arranges the opening from where the last game stopped (Nadav 15.11): the
        page shows that position, the person puts back whatever is not there (checkers taken
        off the glass go to their tray places) and confirms; then executor.arrange (matched
        for a small total distance)."""
        last = list(self.last_board)
        if self.is_sim:
            self.sim_board(last)
            self.new_executor()
        with self.lock:
            self.board = last
        self.ex.set_from_board(last)
        self.look()
        self.say("Is every checker where the page shows it?", "setup", ["go", "cancel"],
                 info="Checkers that were taken off go to their tray places. Then the robot arranges the opening")
        a = self.wait("go", "cancel")
        if a["name"] == "go":
            self.say("The robot is arranging the opening position…", "setup", info="")
            n = self.ex.arrange(rules.OPENING)
            self.log_fn(f"arranged from the last game with {n} drags, {self.ex.dragged_mm:.0f} mm")
            self.last_board = None
        with self.lock:
            self.board = list(rules.OPENING)
            self.info = ""
        self.ex.set_from_board(rules.OPENING)
        self.look()

    def find_side_colours(self):
        """Which side has the white checkers and which the black, from the brightness of the
        opening photo at the places the opening puts them (no detector). The page draws both
        the board and the circles on the photo in those colours; its swap button turns them."""
        if self.is_sim or getattr(self, "last_img", None) is None:
            return
        try:
            col = vision.side_colours(self.last_img, self.c, self.L, rules.OPENING)
        except Exception as e:   # never lose the game over a colour
            self.log_fn(f"side colours: {e}")
            return
        if col:
            self.robot_white = col["robot"] == "#f2f2ea"
            self.log_fn(f"side colours: robot {'white' if self.robot_white else 'black'}")

    def play(self):
        self.setup()
        first = self.s["first"]
        if first == "roll":
            who, d = dice_mod.opening_roll()
            self.say(f"{self.name(who)} start{'' if self.name(who) == 'You' else 's'}", "opening",
                     info=f"Opening roll: {self.name('robot')} {max(d) if who == 'robot' else min(d)}, "
                          f"{self.name('human')} {max(d) if who == 'human' else min(d)}")
            time.sleep(1.5)
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
        # the page plays a fanfare, or a sad tune when a person lost to the robot
        lost = self.s["mode"] == "robot_vs_human" and who == "robot"
        with self.lock:
            self.winner = {"who": who, "kind": kind, "points": pts, "lost": lost}
        n = self.name(who)
        self.say(f"{n} win{'' if n == 'You' else 's'}!", "over", ["new"],
                 info=f"{n} win{'' if n == 'You' else 's'}: {kind}, {pts} point{'s' if pts > 1 else ''}")

    def run_forever(self):
        """Game thread: wait for New game, play, repeat."""
        while True:
            try:
                self.busy = False
                a = self.inbox.get()
                if a["name"] == "stop":
                    return
                if a["name"] != "new":
                    continue
                self.busy = True
                with self.lock:
                    if self.turn:   # a game was played: remember where it stopped
                        self.last_board = list(self.board)
                    self.reset_state()
                self.play()
            except Restart:
                self.busy = False
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
