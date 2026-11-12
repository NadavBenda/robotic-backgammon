"""Read the board from one photo: where the checkers are and whose they are.

    import vision
    model = vision.learn(img, c, L, rules.OPENING)    # once per game, board in a known position
    r = vision.read(img, c, L, model)                  # method "scan" (default), "slots" or "hough"
    r.board        # 26 numbers, robot's side (like gnubg.py); r.off = (robot off, human off)
    r.checkers     # [(x_mm, y_mm, 'robot' | 'human')] measured positions
    r.warnings     # ["point 7: both colours", "robot has 14 checkers", ...]
    board, cost = vision.closest(r, candidates)        # the candidate board that fits best

How a spot is judged: the camera sees the checker from below, so the centre is the
steel washer (the same for both players) and the colour shows in the rim around it.
For a spot (x, y) in mm, the pixels of a small disc (the steel) and of a ring (the rim)
are averaged in Lab colour -> 6 numbers. learn() collects these numbers on a known board
(empty spots too) and keeps mean and spread per class: empty / robot / human.
A spot is the class with the smallest normalised distance.

Three ways to find the checkers (all use the same classes):
  slots  - judge only the slots of the layout (fast; needs checkers placed on their slots,
           good after the robot moved; a person's sloppy placement may be missed)
  scan   - slide along every row (each point is a row) in 3 mm steps and keep the peaks
           (default: finds checkers where a person really left them, gives their position)
  hough  - Hough circles in the photo, then judge each one (independent of the layout,
           good for odd positions; misses some on a busy background)
Plus closest(): out of the boards that are possible after this roll (rules.legal_moves),
take the one that best fits the photo. One misread checker then usually does not matter.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "vision_model.json"
CLASSES = ("empty", "robot", "human")
STEEL_R, RIM = 7.0, (10.0, 12.5)   # mm: steel disc radius, rim ring from-to


def _offsets():
    pts = []
    for r in (0, 3.5, STEEL_R):
        n = 1 if r == 0 else 10
        pts += [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    steel = len(pts)
    for r in np.linspace(*RIM, 2):
        pts += [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    return np.array(pts), steel


OFFS, N_STEEL = _offsets()


def features(lab, c, xy):
    """6 numbers per spot: mean Lab of the steel disc and of the rim. xy: (N, 2) mm."""
    xy = np.atleast_2d(np.asarray(xy, float))
    P = xy[:, None, :] + OFFS[None]                         # (N, K, 2) mm
    u, v = c.mm_to_px(P[..., 0], P[..., 1])
    u = np.clip(np.rint(np.asarray(u, float)), 0, lab.shape[1] - 1).astype(int)
    v = np.clip(np.rint(np.asarray(v, float)), 0, lab.shape[0] - 1).astype(int)
    px = lab[v, u].astype(float)                            # (N, K, 3)
    return np.concatenate([px[:, :N_STEEL].mean(1), px[:, N_STEEL:].mean(1)], axis=1)


def to_lab(img):
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2LAB)


@dataclass
class Model:
    mu: dict
    sd: dict

    def dist(self, f):
        """(N, 3) normalised squared distance to each class."""
        f = np.atleast_2d(f)
        return np.stack([(((f - self.mu[k]) / self.sd[k]) ** 2).sum(1) for k in CLASSES], 1)

    def classify(self, f):
        d = self.dist(f)
        best = d.argmin(1)
        s = np.sort(d, 1)
        return [CLASSES[i] for i in best], s[:, 1] - s[:, 0]

    def save(self, path=PATH):
        Path(path).write_text(json.dumps({k: {"mu": list(self.mu[k]), "sd": list(self.sd[k])} for k in CLASSES}))

    @classmethod
    def load(cls, path=PATH):
        d = json.loads(Path(path).read_text())
        return cls({k: np.array(d[k]["mu"]) for k in CLASSES}, {k: np.array(d[k]["sd"]) for k in CLASSES})


def learn(img, c, L, board, extra=None):
    """Learn the 3 classes from a photo of a known board (default: the opening).
    extra: more (features, class) pairs to merge, e.g. from earlier photos."""
    lab = to_lab(img)
    placed = L.place(board)
    taken = [xy for xy, _ in placed.values()]
    f_by = {k: [] for k in CLASSES}
    for xy, side in placed.values():
        f_by[side].append(features(lab, c, [xy])[0])
    # empty: slots of the layout that are at least a checker away from every checker
    for where, i, xy in L.all_slots(depth=6):
        if min(np.hypot(xy[0] - t[0], xy[1] - t[1]) for t in taken) > L.checker * 1.2:
            f_by["empty"].append(features(lab, c, [xy])[0])
    if extra:
        for f, k in extra:
            f_by[k].append(f)
    mu, sd = {}, {}
    for k in CLASSES:
        a = np.array(f_by[k])
        if len(a) == 0:
            raise ValueError(f"no samples for {k}")
        mu[k] = a.mean(0)
        sd[k] = np.maximum(a.std(0), 4.0)   # floor: a few samples must not make it over-sure
    return Model(mu, sd)


@dataclass
class Reading:
    checkers: list                 # [(x, y, side)]
    places: dict                   # {('point', 7): {'robot': 2, 'human': 0}, ('bar', ..): ..}
    board: list = None
    off: tuple = (0, 0)
    warnings: list = field(default_factory=list)


def _peaks(score, xs, min_sep):
    """Greedy peak picking: highest first, at least min_sep apart, score > 0."""
    order = np.argsort(-score)
    out = []
    for i in order:
        if score[i] <= 0:
            break
        if all(abs(xs[i] - xs[j]) >= min_sep for j in out):
            out.append(i)
    return sorted(out)


def find_scan(lab, c, L, model, step=3.0):
    found = []
    nrow = [L._row_y(k) for k in range(13)] + list(L.off_rows_y)
    mid = L.xmax / 2
    for ny in nrow:
        for half in (0, 1):
            xs = np.arange(-6, mid - 4, step) if half == 0 else np.arange(L.xmax + 6, mid + 4, -step)
            # 3 parallel lines, a person may leave a checker a few mm off the row
            score, mag = None, None
            for dy in (-4.5, 0, 4.5):
                m_ = np.array([L.to_magnet(x, ny + dy) for x in xs])
                d = model.dist(features(lab, c, m_))
                s_ = d[:, 0] - np.minimum(d[:, 1], d[:, 2])  # > 0: more like a checker than glass
                if score is None:
                    score, mag = s_, m_
                else:
                    better = s_ > score
                    score = np.where(better, s_, score)
                    mag[better] = m_[better]
            for i in _peaks(score, xs, L.checker * 0.85):
                x, y = mag[i]
                # refine across the row (a person may leave it off the line)
                best = None
                bx, by = L.from_magnet(x, y)
                for dy in (-3, -1.5, 0, 1.5, 3):
                    for dx in (-3, -1.5, 0, 1.5, 3):
                        q = L.to_magnet(bx + dx, by + dy)
                        dd = model.dist(features(lab, c, [q]))[0]
                        s = dd[0] - min(dd[1], dd[2])
                        if best is None or s > best[0]:
                            best = (s, q, "robot" if dd[1] < dd[2] else "human")
                found.append((float(best[1][0]), float(best[1][1]), best[2]))
    return found


def find_slots(lab, c, L, model, depth=8):
    found = []
    places = [p for p in range(1, 25)] + ["bar_robot", "bar_human", "off_robot", "off_human"]
    for w in places:
        n = 15 if str(w).startswith("off") else depth
        xy = [L.slot(w, i) for i in range(n)]
        cls, _ = model.classify(features(lab, c, xy))
        for p, k in zip(xy, cls):
            if k != "empty":
                found.append((float(p[0]), float(p[1]), k))
    return found


def find_hough(img, lab, c, L, model):
    from helpers import find_circles
    found = []
    for u, v, _ in find_circles(img):
        x, y = (float(t) for t in c.px_to_mm(u, v))
        if not (-15 <= x <= L.xmax + 15 and -15 <= y <= L.ymax + 15):
            continue
        if any(np.hypot(x - f[0], y - f[1]) < L.checker * 0.7 for f in found):
            continue
        k, _ = model.classify(features(lab, c, [(x, y)]))
        if k[0] != "empty":
            found.append((x, y, k[0]))
    return found


def read(img, c, L, model, method="scan"):
    lab = to_lab(img)
    if method == "slots":
        ck = find_slots(lab, c, L, model)
    elif method == "hough":
        ck = find_hough(img, lab, c, L, model)
    else:
        ck = find_scan(lab, c, L, model)
    return summarize(ck, L)


def assign(ck, L):
    """Place of each checker. Mostly L.nearest, but a long point (8+ checkers) reaches past
    the middle: a checker there belongs to the side whose line of checkers it continues."""
    keys = [L.nearest(x, y) for x, y, _ in ck]
    rows = {}
    for i, (x, y, _) in enumerate(ck):
        if keys[i] is None:
            continue
        nx, ny = L.from_magnet(x, y)
        rows.setdefault(round(ny / 5), []).append((nx, i))
    mid = L.xmax / 2
    for items in rows.values():
        items.sort()
        if len(items) < 8:
            continue
        # chain from the left edge
        left_end, prev = None, None
        for nx, i in items:
            if prev is None and nx > L.edge_x + L.pitch_x * 0.8:
                break
            if prev is not None and nx - prev > L.pitch_x * 1.5:
                break
            left_end, prev = i, nx
        right_end, prev = None, None
        for nx, i in reversed(items):
            if prev is None and nx < L.xmax - L.edge_x - L.pitch_x * 0.8:
                break
            if prev is not None and prev - nx > L.pitch_x * 1.5:
                break
            right_end, prev = i, nx
        for side, end in (("left", left_end), ("right", right_end)):
            if end is None:
                continue
            ex = L.from_magnet(*ck[end][:2])[0]
            if (side == "left" and ex > mid) or (side == "right" and ex < mid):
                anchor = items[0][1] if side == "left" else items[-1][1]
                for nx, i in items:
                    if (side == "left" and mid <= nx <= ex) or (side == "right" and ex <= nx <= mid):
                        keys[i] = keys[anchor]
    return keys


def summarize(ck, L):
    """Checker positions -> counts per place -> board (robot's side) + warnings."""
    places, lost = {}, 0
    for (x, y, s), k in zip(ck, assign(ck, L)):
        if k is None:
            lost += 1
            continue
        places.setdefault(k, {"robot": 0, "human": 0})[s] += 1
    r = Reading(ck, places)
    b = [0] * 26
    for (kind, key), n in places.items():
        if kind == "point":
            if n["robot"] and n["human"]:
                r.warnings.append(f"point {key}: both colours ({n['robot']} robot, {n['human']} human)")
            b[key] = n["robot"] - n["human"]
        elif kind == "bar":
            pass
    bar = [n for (kind, _), n in places.items() if kind == "bar"]
    b[0] = sum(n["robot"] for n in bar)
    b[25] = -sum(n["human"] for n in bar)
    off = [n for (kind, _), n in places.items() if kind == "off"]
    r.off = (sum(n["robot"] for n in off), sum(n["human"] for n in off))
    r.board = b
    if lost:
        r.warnings.append(f"{lost} checker(s) outside every point")
    from rules import off_count
    own_off, opp_off = off_count(b)
    if own_off != r.off[0]:
        r.warnings.append(f"robot: {15 - own_off} on the board + {r.off[0]} off = {15 - own_off + r.off[0]}, not 15")
    if opp_off != r.off[1]:
        r.warnings.append(f"human: {15 - opp_off} on the board + {r.off[1]} off = {15 - opp_off + r.off[1]}, not 15")
    return r


def mismatch(r, cand):
    """How many checkers differ between a reading and a candidate board."""
    b = r.board
    return int(sum(abs(b[i] - cand[i]) for i in range(26)))


def closest(r, candidates):
    """(board, cost) of the candidate that fits the reading best (cost 0 = exact)."""
    best = min(candidates, key=lambda cb: mismatch(r, cb))
    return best, mismatch(r, best)


def draw(img, c, r, L=None):
    """The photo with every checker found circled in its class colour, for the UI / log."""
    out = img.copy()
    for x, y, s in r.checkers:
        u, v = c.mm_to_px(x, y)
        col = (0, 255, 0) if s == "robot" else (0, 165, 255)
        cv2.circle(out, (int(u), int(v)), 14, col, 2, cv2.LINE_AA)
    return out


def export_crops(img, c, L, board, folder, size_mm=34, px=48):
    """Training data for a future CNN/YOLO: one small top-down crop per slot of a known
    board, named by class. folder/<class>/<n>.png  (option C, not needed to play)."""
    folder = Path(folder)
    n = 0
    taken = {xy: s for xy, s in L.place(board).values()}
    slots = [(xy, taken.get(xy, "empty")) for _, _, xy in L.all_slots(depth=6)]
    for xy, k in slots:
        h = size_mm / 2
        src = np.float32([c.mm_to_px(xy[0] + dx, xy[1] + dy) for dx, dy in ((-h, -h), (h, -h), (h, h))])
        dst = np.float32([[0, 0], [px, 0], [px, px]])
        crop = cv2.warpAffine(img, cv2.getAffineTransform(src, dst), (px, px))
        (folder / k).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / k / f"{n:05d}.png"), crop)
        n += 1
    return n
