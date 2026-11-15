"""A film of a game as the page showed it, to put next to the real video (Nadav 15.11).

    py make_video.py     (double-click)

Takes the newest game in games/ that has events.jsonl (game.py writes it by itself) and
makes games/<that game>/game.mp4: the drawn board, the dice, whose turn it is and what the
robot plays, 1080 x 1080, 30 pictures a second. It runs at exactly the speed the game was
played, so in a video editor it lines up with the real film by sliding it once. When the
robot drags a checker, the checker slides along the yellow path.

Things to change here, at the top of the file:
  SPEED      1.0 = real time. 2.0 = twice as fast.
  MAX_WAIT   None = keep every pause. A number = no pause longer than that many seconds
             (shorter film, but it no longer lines up with the real one).
  SIZE       side of the square film, in pixels.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

import ui
from layout import Layout

SPEED = 1.0
MAX_WAIT = None
SIZE = 1080
FPS = 30
GAMES = Path(__file__).parent / "games"

GOLD, INK, MUTE = (83, 176, 226), (221, 235, 243), (130, 152, 165)     # BGR
FELT, WOOD, WOOD2 = (63, 77, 29), (39, 68, 107), (20, 36, 58)
LIGHT, DARK = (143, 194, 227), (46, 69, 168)
PIPS = {1: [(2, 2)], 2: [(1, 1), (3, 3)], 3: [(1, 1), (2, 2), (3, 3)], 4: [(1, 1), (3, 1), (1, 3), (3, 3)],
        5: [(1, 1), (3, 1), (2, 2), (1, 3), (3, 3)], 6: [(1, 1), (3, 1), (1, 2), (3, 2), (1, 3), (3, 3)]}


class Painter:
    """Draws one picture of the film. Magnet mm -> pixels, upside down like the page."""

    def __init__(self, L, size=SIZE):
        self.L, self.size = L, size
        self.top = int(size * 0.17)                         # the strip with the players, the text and the dice
        self.k = (size - self.top - int(size * 0.045)) / (L.ymax + 40.0)
        self.x0 = (size - (L.xmax + 40.0) * self.k) / 2
        self.back = self.board()

    def px(self, x, y):
        return (int(round(self.x0 + (x + 20.0) * self.k)), int(round(self.top + (self.L.ymax - y + 20.0) * self.k)))

    def text(self, img, s, xy, h, col, bold=False, right=False):
        scale = h / 22.0
        th = max(1, int(round(scale * (2.2 if bold else 1.3))))
        if right:
            xy = (xy[0] - cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, scale, th)[0][0], xy[1])
        cv2.putText(img, s, (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_DUPLEX, scale, col, th, cv2.LINE_AA)

    def board(self):
        L, s = self.L, self.size
        img = np.zeros((s, s, 3), np.uint8)
        img[:] = (14, 16, 18)
        a, b = self.px(-20, L.ymax + 20), self.px(L.xmax + 20, -20)
        cv2.rectangle(img, a, b, WOOD, -1)
        cv2.rectangle(img, a, b, WOOD2, max(2, s // 300))
        cv2.rectangle(img, self.px(-6, L.ymax + 6), self.px(L.xmax + 6, -6), FELT, -1)
        r = L.checker / 2
        lane0, lane1 = L.lane_x()
        mg = lambda x, y: self.px(*L.to_magnet(x, y))  # noqa: E731
        by = L._row_y(6)
        p, q = mg(0, by - 14), mg(L.xmax, by + 14)
        cv2.rectangle(img, (min(p[0], q[0]), min(p[1], q[1])), (max(p[0], q[0]), max(p[1], q[1])), (48, 60, 22), -1)
        for pt in range(1, 25):
            x, y = L._nominal(pt, 0)
            left = x < L.xmax / 2
            base = x - r if left else x + r
            tip = lane0 - 4 if left else lane1 + 4
            tri = np.array([mg(base, y - r + 1), mg(base, y + r - 1), mg(tip, y)], np.int32)
            cv2.fillPoly(img, [tri], LIGHT if (25 - pt) % 2 else DARK, cv2.LINE_AA)
            u, v = mg(tip + 12 if left else tip - 12, y)
            n = str(25 - pt)
            w = cv2.getTextSize(n, cv2.FONT_HERSHEY_DUPLEX, s / 2200, 1)[0][0]
            cv2.putText(img, n, (u - w // 2, v + s // 200), cv2.FONT_HERSHEY_DUPLEX, s / 2200, (218, 227, 207), 1, cv2.LINE_AA)
        return img

    def checker(self, img, xy, white):
        c, r = self.px(*xy), int(round(self.L.checker / 2 * self.k * 0.96))
        cv2.circle(img, (c[0] + 2, c[1] + 3), r, (10, 25, 8), -1, cv2.LINE_AA)
        body, rim, ring = ((228, 234, 236), (122, 135, 141), (164, 179, 185)) if white else ((30, 30, 30), (0, 0, 0), (77, 77, 77))
        cv2.circle(img, c, r, body, -1, cv2.LINE_AA)
        cv2.circle(img, c, r, rim, max(1, r // 14), cv2.LINE_AA)
        cv2.circle(img, c, int(r * 0.64), ring, max(1, r // 14), cv2.LINE_AA)
        cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(2, r // 5), (255, 255, 255) if white else (90, 90, 90), -1, cv2.LINE_AA)

    def die(self, img, x, y, d, v, red):
        cv2.rectangle(img, (x, y), (x + d, y + d), (40, 40, 198) if red else (239, 248, 251), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + d, y + d), (21, 21, 127) if red else (160, 178, 185), 2, cv2.LINE_AA)
        for i, j in PIPS.get(v, []):
            cv2.circle(img, (x + i * d // 4, y + j * d // 4), max(2, d // 11), (255, 255, 255) if red else (27, 27, 27), -1, cv2.LINE_AA)

    def frame(self, ev, moving=None):
        """ev: one line of events.jsonl. moving: (index of the checker that is being dragged,
        how far along the path 0..1)."""
        img, s = self.back.copy(), self.size
        rw = bool(ev.get("robot_white"))
        path = ev.get("path")
        if path and len(path) > 1:
            pts = np.array([self.px(*p) for p in path], np.int32)
            cv2.polylines(img, [pts], False, (77, 225, 255), max(2, s // 300), cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in pts[-1]), max(4, s // 160), (77, 225, 255), -1, cv2.LINE_AA)
        late = None
        for i, (x, y, side) in enumerate(ev["pieces"]):
            if moving and moving[0] == i:
                late = (along(path, moving[1]), (side == "robot") == rw)
                continue
            self.checker(img, (x, y), (side == "robot") == rw)
        if late:
            self.checker(img, *late)
        # the strip on top: the two players, the text, the dice
        pl, turn, win = ev.get("players", {}), ev.get("turn"), ev.get("winner")
        m = int(s * 0.035)
        for k, side in enumerate(("robot", "human")):
            y = int(s * 0.046) + k * int(s * 0.048)
            on = (win or {}).get("who") == side if win else turn == side
            white = (side == "robot") == rw
            cv2.circle(img, (m + s // 60, y - s // 110), s // 70, (228, 234, 236) if white else (30, 30, 30), -1, cv2.LINE_AA)
            cv2.circle(img, (m + s // 60, y - s // 110), s // 70, GOLD if on else MUTE, 2, cv2.LINE_AA)
            name = (pl.get(side) or {}).get("name", side)
            self.text(img, name, (m + s // 22, y), s / 42, GOLD if on else INK, bold=on)
            i = 0 if side == "robot" else 1
            more = f"{ev['pips'][i]} to go" + (f"   {ev['off'][i]} off" if ev["off"][i] else "")
            self.text(img, more, (m + int(s * 0.19), y), s / 60, MUTE)
        self.text(img, ev.get("prompt") or "", (m, int(s * 0.15)), s / 36, GOLD if win else INK, bold=True)
        info = ev.get("info") or ""
        if info:
            self.text(img, info[:70], (s - m, s - int(s * 0.014)), s / 62, MUTE, right=True)
        d = int(s * 0.062)
        for k, v in enumerate(ev.get("dice") or []):
            self.die(img, s - m - (2 - k) * (d + d // 5) + d // 5, int(s * 0.03), d, v, k == 1)
        return img


def along(path, f):
    """The point at fraction f (0..1) of the way along path."""
    p = np.array(path, float)
    seg = np.hypot(*(p[1:] - p[:-1]).T)
    want, total = f * seg.sum(), 0.0
    for a, b, n in zip(p[:-1], p[1:], seg):
        if total + n >= want and n > 0:
            return tuple(a + (b - a) * (want - total) / n)
        total += n
    return tuple(p[-1])


def load(folder):
    ev = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((i for i, e in enumerate(ev) if e.get("turn")), 0)
    return ev[max(0, start - 1):]


def plan(ev):
    """[(film time, event index, moving)] one entry per picture of the film."""
    t0 = ev[0]["t"]
    at = [0.0]
    for a, b in zip(ev, ev[1:]):                 # the film's clock: pauses cut to MAX_WAIT
        gap = (b["t"] - a["t"]) / SPEED
        at.append(at[-1] + (gap if MAX_WAIT is None else min(gap, MAX_WAIT)))
    del t0
    drag = {}                                    # event index -> (checker index, index of the event that ends the drag)
    for i, e in enumerate(ev):
        p = e.get("path")
        if not p or len(p) < 2 or (i and ev[i - 1].get("path") == p):
            continue
        d = [np.hypot(x - p[0][0], y - p[0][1]) for x, y, _ in e["pieces"]]
        j = int(np.argmin(d)) if d else -1
        if j < 0 or d[j] > 4:
            continue
        end = next((k for k in range(i + 1, len(ev)) if ev[k]["pieces"] != e["pieces"]), None)
        if end is not None:
            for k in range(i, end):
                drag[k] = (j, at[i], at[end])
    out, n, i = [], int((at[-1] + 3.0) * FPS), 0
    for f in range(n):
        t = f / FPS
        while i + 1 < len(ev) and at[i + 1] <= t:
            i += 1
        mv = None
        if i in drag:
            j, a, b = drag[i]
            mv = (j, min(1.0, max(0.0, (t - a) / max(0.05, b - a))))
        out.append((i, mv))
    return out


def render(folder, log=print):
    ev = load(folder)
    if len(ev) < 2:
        log("This game has no moves to show.")
        return None
    L = Layout.load()
    L.mirror = bool(ev[-1].get("mirror"))   # the robot played the other side (the page's swap button)
    P = Painter(L)
    frames = plan(ev)
    tmp = Path(tempfile.mkdtemp()) / "game.mp4"      # OpenCV cannot write a video to a Hebrew path
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    if not vw.isOpened():
        log("Could not start the video file (no mp4 writer in this OpenCV).")
        return None
    last_key, img = None, None
    for n, (i, mv) in enumerate(frames):
        key = (i, mv)
        if key != last_key:
            img, last_key = P.frame(ev[i], mv), key
        vw.write(img)
        if n % (FPS * 30) == 0:
            log(f"   {n / FPS / 60:.1f} of {len(frames) / FPS / 60:.1f} minutes")
    vw.release()
    out = folder / "game.mp4"
    shutil.move(str(tmp), str(out))
    return out


def main():
    games = sorted(p for p in GAMES.glob("*") if (p / "events.jsonl").exists()) if GAMES.exists() else []
    if not games:
        print("No recorded game yet: play one with game.py first (it records by itself).")
        return
    print(f"Making the film of the game {games[-1].name} ...")
    out = render(games[-1])
    if out:
        print(f"Done: {out}")


if __name__ == "__main__":
    ui.keep_open()
    main()
