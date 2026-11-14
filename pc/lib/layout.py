"""Where every checker goes on the board, in mm (the same mm as the G command).

    from layout import Layout
    L = Layout.load()                  # layout.json next to this file, or the defaults below
    L.slot(13, 0)                      # (x, y) of the first checker on point 13 (robot numbering)
    L.slot("bar_robot", 1)             # second robot checker on the bar
    L.slot("off_human", 4)             # fifth human checker in the human bear-off tray
    L.place(board)                     # {(where, i): (x, y)} for every checker of a board
    L.nearest(x, y)                    # ("point", 7) / ("bar", side) / ("off", side) / None

Plan v4 (rotated, see checkers/board-sketch.png), scaled to the real travel 410 x 420
and the real checker (26 mm). Points are rows; checkers in a point stand in a line from
the board edge toward the middle lane, 6 fit before the lane, the 7th spills over:

        x=0 ............ lane ............ x=410
        12  o o o o o o           o o o o o o  13      far side
        ..                                     ..
         7                                     18
        BAR: hit checkers of both colours stand in one column on the centre line (x = 205)
         6                                     19
        ..                                     ..
         1  o o o o o o           o o o o o o  24
        OFF robot 2x7      |      OFF human 2x7      near side (motors / camera)

Robot numbering: the robot moves 24 -> 1 and bears off from 1..6 (left, near side);
the human's point p is the robot's 25 - p. Which corner of that picture is (0, 0) for
the magnet is set by flip_x / flip_y, or measured once with teach_layout.py (an affine
fit from 4+ clicked reference slots, stored in layout.json as "affine").
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from files import data

import numpy as np

from gnubg import BAR, OFF  # noqa: F401

PATH = data("layout.json")   # <PC code>/data/


@dataclass
class Layout:
    xmax: float = 410.0
    ymax: float = 420.0
    checker: float = 26.0
    pitch_x: float = 27.0          # centre to centre along a point
    row_pitch: float = 29.0        # centre to centre between points (rows)
    first_row_y: float = 62.0      # y of points 1 / 24 (the row next to the bear-off strip)
    off_rows_y: tuple = (2.0, 31.0)  # the 2 rows of each bear-off tray
    edge_x: float = 2.0            # x of the first checker on the left points (right: xmax - edge_x);
                                   # 2 mm inside the travel, so the 3 mm overshoot still works there
    per_row_off: int = 7
    flip_x: bool = False           # True: the picture above is mirrored left-right for the magnet
    flip_y: bool = False           # True: points 12/13 are at y=0 instead of y=ymax
    affine: list = field(default=None)  # 2x3, from teach_layout.py; replaces flip_x/flip_y
    keepout: list = field(default_factory=list)  # [[x0, y0, x1, y1]] the magnet must not enter
    dice_zone: tuple = (172.0, -44.0, 238.0, -14.0)  # picture frame; beyond the near trays

    # -- file -------------------------------------------------------------
    @classmethod
    def load(cls, path=PATH):
        try:
            d = json.loads(Path(path).read_text())
        except OSError:
            return cls()
        for k in ("off_rows_y", "dice_zone"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path=PATH):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    # -- geometry in the "picture" frame, then mapped to magnet mm -------------
    def _row_y(self, k):
        """k = 0 (row of points 1/24) .. 12 (points 12/13); k = 6 is the bar."""
        return self.first_row_y + k * self.row_pitch

    def _nominal(self, where, i):
        left = lambda j: self.edge_x + j * self.pitch_x
        right = lambda j: self.xmax - self.edge_x - j * self.pitch_x
        if isinstance(where, int) or str(where).isdigit():
            p = int(where)
            if not 1 <= p <= 24:
                raise ValueError(f"no point {p}")
            if p <= 12:   # left column, 1 at the near side
                k = p - 1 if p <= 6 else p
                return left(i), self._row_y(k)
            q = 25 - p    # 13..24 on the right: 24 near, 13 far
            k = q - 1 if q <= 6 else q
            return right(i), self._row_y(k)
        if where in ("bar_robot", "bar_human"):
            # hit checkers of both colours stand in one column on the centre line of the board
            # (x = xmax / 2, Nadav 14.11), out from the middle of the bar row: the robot's 1st at the
            # middle, the person's 1st one pitch toward the far side, the robot's 2nd one pitch the
            # other way, and so on (each colour has its own places, so the camera knows whose is whose)
            k = 2 * i + (1 if where == "bar_human" else 0)
            step = (k + 1) // 2 * (-1 if k % 2 else 1)
            return self.xmax / 2, self._row_y(6) + step * self.row_pitch
        if where in ("off_robot", "off_human"):
            r, j = divmod(i, self.per_row_off)
            if r >= len(self.off_rows_y):   # 15th checker: past the end of the last row
                r = len(self.off_rows_y) - 1
                j = i - r * self.per_row_off
            y = self.off_rows_y[r]
            return (left(j), y) if where == "off_robot" else (right(j), y)
        raise ValueError(f"unknown place {where!r}")

    def to_magnet(self, x, y):
        if self.affine is not None:
            A = np.asarray(self.affine, float)
            return float(A[0, 0] * x + A[0, 1] * y + A[0, 2]), float(A[1, 0] * x + A[1, 1] * y + A[1, 2])
        return (self.xmax - x if self.flip_x else x), (y if self.flip_y else self.ymax - y)

    def from_magnet(self, x, y):
        if self.affine is not None:
            A = np.vstack([np.asarray(self.affine, float), [0, 0, 1]])
            v = np.linalg.solve(A, [x, y, 1.0])
            return float(v[0]), float(v[1])
        return (self.xmax - x if self.flip_x else x), (y if self.flip_y else self.ymax - y)

    def slot(self, where, i):
        """Magnet mm of checker number i (0 = at the edge) in a point, bar or tray."""
        return self.to_magnet(*self._nominal(where, i))

    # -- whole boards -----------------------------------------------------
    def place(self, b):
        """{(where, i): ((x, y), 'robot' | 'human')} for a board from the robot's side."""
        from rules import off_count
        out = {}
        for p in range(1, 25):
            for i in range(abs(b[p])):
                out[(p, i)] = (self.slot(p, i), "robot" if b[p] > 0 else "human")
        for i in range(b[0]):
            out[("bar_robot", i)] = (self.slot("bar_robot", i), "robot")
        for i in range(-b[25]):
            out[("bar_human", i)] = (self.slot("bar_human", i), "human")
        own_off, opp_off = off_count(b)
        for i in range(own_off):
            out[("off_robot", i)] = (self.slot("off_robot", i), "robot")
        for i in range(opp_off):
            out[("off_human", i)] = (self.slot("off_human", i), "human")
        return out

    def nearest(self, x, y, max_mm=None):
        """Which place a checker at magnet (x, y) belongs to: (kind, key) where kind is
        'point' (key 1..24), 'bar' or 'off' (key 'robot' / 'human'), or None if it is
        farther than max_mm (default: one checker) from every row."""
        nx, ny = self.from_magnet(x, y)
        max_mm = self.checker if max_mm is None else max_mm
        mid = self.xmax / 2
        left = nx < mid
        cands = []
        for w in ("bar_robot", "bar_human"):          # a checker on the centre line near a bar place
            for i in range(4):                       # (a long point's spill reaches 14 mm from the line)
                bx, by = self._nominal(w, i)
                if np.hypot(nx - bx, ny - by) <= 11.0:
                    return ("bar", w.split("_")[1])
        for k in range(13):
            if k == 6:
                continue
            d = abs(ny - self._row_y(k))
            q = k + 1 if k < 6 else k
            cands.append((d, ("point", q if left else 25 - q)))
        for ry in self.off_rows_y:
            cands.append((abs(ny - ry), ("off", "robot" if left else "human")))
        d, key = min(cands, key=lambda c: c[0])
        return key if d <= max_mm else None

    def all_slots(self, depth=7):
        """[(where, i, (x, y))] for drawing / teaching."""
        out = [(p, i, self.slot(p, i)) for p in range(1, 25) for i in range(depth)]
        out += [(w, i, self.slot(w, i)) for w in ("bar_robot", "bar_human") for i in range(4)]
        out += [(w, i, self.slot(w, i)) for w in ("off_robot", "off_human") for i in range(15)]
        return out

    def park(self):
        """Where the magnet waits during a photo: middle of the lane, at the far end of its
        travel (y max). Checked on the real board 12.11: in the middle of the board the cart
        and its rail hide a whole band from the camera; at y max the rail is beyond every
        checker. x in the lane so the cart is not under a bear-off tray slot."""
        lo, hi = self.lane_x()
        x = float(self.to_magnet((lo + hi) / 2, self._row_y(6))[0])
        return (x, float(self.ymax))

    def lane_x(self):
        """x range of the free middle lane in the picture frame (no checker of a full
        point of 6 reaches into it)."""
        return (self.edge_x + 5 * self.pitch_x + self.checker / 2 + 2,
                self.xmax - self.edge_x - 5 * self.pitch_x - self.checker / 2 - 2)


def fit_affine(nominal, measured):
    """2x3 affine from picture-frame points to measured magnet mm (least squares, >= 3 points)."""
    P = np.asarray(nominal, float)
    Q = np.asarray(measured, float)
    A = np.hstack([P, np.ones((len(P), 1))])
    M = np.linalg.lstsq(A, Q, rcond=None)[0].T
    err = np.hypot(*(A @ M.T - Q).T)
    return M.tolist(), float(np.sqrt(np.mean(err ** 2)))


if __name__ == "__main__":
    L = Layout.load()
    for w, i in [(1, 0), (6, 5), (7, 0), (12, 0), (13, 0), (24, 0), ("bar_robot", 0),
                 ("bar_human", 0), ("off_robot", 0), ("off_robot", 13), ("off_human", 0)]:
        print(f"{str(w):>10} {i:2d}  ->  ({L.slot(w, i)[0]:5.0f}, {L.slot(w, i)[1]:5.0f})")
