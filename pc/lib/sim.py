"""A board without hardware: same calls as board.Board, checkers that follow the magnet,
and photos drawn from the checkers' positions, so game.py, executor.py and vision.py
can be run and tested on any PC.

    from sim import SimBoard, sim_calib
    b = SimBoard.with_board(rules.OPENING)      # checkers placed by layout.json
    b.move(0, 360); b.magnet(True); b.move(80, 360); b.magnet(False)
    img = cv2.imdecode(np.frombuffer(b.capture(), np.uint8), 1)
    c = sim_calib()                              # the pixel <-> mm map of the fake camera

Physics (simple on purpose): when the magnet turns on under a checker (within GRAB_MM)
the checker follows it; it stops LAG_MM behind the magnet; when another checker comes
closer than 26 mm the bump is counted (b.bumps) and the other one is pushed aside.
"""
import random

import cv2
import numpy as np

import calib
from layout import Layout

GRAB_MM = 9.0
LAG_MM = 2.0
COLORS = {"robot": (225, 225, 215), "human": (40, 40, 170)}  # BGR rim colours
STEEL = (150, 150, 150)
BACKGROUND = (70, 90, 105)


def sim_calib():
    """A fixed camera: mirror flip + mild perspective, like the real view, no distortion."""
    src = np.float32([[0, 0], [410, 0], [410, 420], [0, 420]])
    dst = np.float32([[860, 40], [170, 60], [140, 650], [890, 635]])
    H = cv2.getPerspectiveTransform(src, dst)
    zero = np.zeros((1, 2))
    return calib.Calib(calib._Map(np.linalg.inv(H), zero, 400.0, 0),
                       calib._Map(H, zero, 1024.0, 0), 0.0, 0.0)


class SimBoard:
    def __init__(self, checkers=(), layout=None, noise_mm=0.0, seed=1, dice=None):
        """checkers: [(x, y, 'robot' | 'human')]."""
        self.ck = [[float(x), float(y), s] for x, y, s in checkers]
        self.layout = layout or Layout.load()
        self.pos = [0.0, 0.0]
        self.mag = False
        self.held = None
        self.homed = False
        self.drag = (180, 800)
        self.free = (150, 1200)
        self.bumps = 0
        self.under = 0       # times the cart passed right under a checker with the magnet off
        self.bump_log = []
        self.moves = 0
        self.travel = 0.0
        self.calib = sim_calib()
        self.rng = random.Random(seed)
        self.noise_mm = noise_mm
        self.dice = dice  # (d1, d2) drawn in the dice zone when set, for vision tests

    @classmethod
    def with_board(cls, b, layout=None, **kw):
        L = layout or Layout.load()
        return cls([(*xy, side) for xy, side in L.place(b).values()], L, **kw)

    # -- Board API ---------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.magnet(False)

    def command(self, line, timeout=30):
        t = line.split()
        if t[0] == "H":
            self.home()
        elif t[0] == "G":
            self.move(float(t[1]), float(t[2]))
        elif t[0] == "M":
            self.magnet(t[1] == "1")
        elif t[0] == "?":
            return [self._status_line()]
        return []

    def home(self):
        self.pos, self.homed = [0.0, 0.0], True

    def home_if_needed(self, force=False):
        if force or not self.homed:
            self.home()

    def move(self, x, y):
        x = min(max(float(x), 0), self.layout.xmax)
        y = min(max(float(y), 0), self.layout.ymax)
        a = np.array(self.pos)
        z = np.array([x, y])
        self.travel += float(np.hypot(*(z - a)))
        self.moves += 1
        if self.held is not None:
            c = self.ck[self.held]
            # sample the way so bumps along it are caught
            d = z - a
            L = np.hypot(*d)
            lag = d / L * LAG_MM if L > 1 else np.zeros(2)
            n = max(2, int(L / 3))
            for t in np.linspace(0, 1, n)[1:]:
                p = a + t * d - lag
                c[0], c[1] = float(p[0]), float(p[1])
                self._push(self.held)
        elif L0 := float(np.hypot(*(z - a))):
            # magnet off: count checkers the cart passes right under (they may shift on the real board)
            p = np.asarray([c[:2] for c in self.ck], float).reshape(-1, 2)
            if len(p):
                ab = z - a
                t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
                d = np.hypot(*(p - (a + t[:, None] * ab)).T)
                ends = np.minimum(np.hypot(*(p - a).T), np.hypot(*(p - z).T))
                self.under += int(np.sum((d < 8) & (ends > 8)))
        self.pos = [x, y]

    def _push(self, i):
        x, y, _ = self.ck[i]
        for j, o in enumerate(self.ck):
            if j == i:
                continue
            d = np.hypot(o[0] - x, o[1] - y)
            if d < 25.8:
                if d < 24.5:   # more than a touch: a real knock
                    self.bumps += 1
                    self.bump_log.append(((round(x), round(y)), (round(o[0]), round(o[1]))))
                v = np.array([o[0] - x, o[1] - y]) / max(d, 1e-6)
                o[0], o[1] = float(x + v[0] * 26), float(y + v[1] * 26)

    def magnet(self, on):
        self.mag = bool(on)
        if on:
            near = [(np.hypot(c[0] - self.pos[0], c[1] - self.pos[1]), i) for i, c in enumerate(self.ck)]
            near = [n for n in near if n[0] <= GRAB_MM]
            self.held = min(near)[1] if near else None
        else:
            self.held = None

    def drag_profile(self, v, a):
        self.drag = (v, a)

    def free_profile(self, v, a):
        self.free = (v, a)

    def status(self):
        return {"POS": tuple(self.pos), "HOMED": int(self.homed), "MAG": int(self.mag), "HEAT": 0,
                "CAM": 1, "DRAG_V": self.drag[0], "DRAG_A": self.drag[1],
                "FREE_V": self.free[0], "FREE_A": self.free[1]}

    def _status_line(self):
        s = self.status()
        return f"POS {s['POS'][0]:.2f} {s['POS'][1]:.2f} HOMED {s['HOMED']} MAG {s['MAG']}"

    # -- camera ------------------------------------------------------------
    def render(self):
        img = np.full((768, 1024, 3), BACKGROUND, np.uint8)
        c = self.calib
        # faint board: the rows of points
        L = self.layout
        for p in range(1, 25):
            a = c.mm_to_px(*L.slot(p, 0))
            z = c.mm_to_px(*L.slot(p, 5))
            cv2.line(img, (int(a[0]), int(a[1])), (int(z[0]), int(z[1])), (90, 110, 125), 18)
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        for x, y, side in self.ck:
            if self.noise_mm:
                x += self.rng.gauss(0, self.noise_mm)
                y += self.rng.gauss(0, self.noise_mm)
            for r, col in ((13, COLORS[side]), (9, STEEL)):
                u, v = c.mm_to_px(x + r * np.cos(t), y + r * np.sin(t))
                cv2.fillPoly(img, [np.int32(np.stack([u, v], -1))], col, cv2.LINE_AA)
        if self.dice:
            from dice import draw_dice
            draw_dice(img, c, self.dice, self.layout)
        # the magnet cart is under the glass: a dark blob where it is
        u, v = c.mm_to_px(*self.pos)
        cv2.circle(img, (int(u), int(v)), 12, (30, 30, 30), -1)
        noise = np.random.default_rng(self.rng.randrange(1 << 30)).normal(0, 4, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def capture(self):
        return cv2.imencode(".jpg", self.render(), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    # -- helpers for tests --------------------------------------------------
    def human_move(self, frm_xy, to_xy, sloppy=4.0):
        """A person picks the checker nearest frm_xy and puts it near to_xy."""
        i = min(range(len(self.ck)), key=lambda i: np.hypot(self.ck[i][0] - frm_xy[0], self.ck[i][1] - frm_xy[1]))
        # a hand does not put a checker on top of another: less sloppy until it fits
        for k in range(20):
            s = sloppy * (1 - k / 20)
            x = float(to_xy[0] + self.rng.uniform(-s, s))
            y = float(to_xy[1] + self.rng.uniform(-s, s))
            if all(np.hypot(o[0] - x, o[1] - y) >= 25.5 for j, o in enumerate(self.ck) if j != i):
                break
        self.ck[i][0], self.ck[i][1] = x, y
        return i

    def checkers(self):
        return [(x, y, s) for x, y, s in self.ck]
