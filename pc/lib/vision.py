"""Read the board from one photo: where the checkers are and whose they are.

    import vision
    model = vision.learn(img, c, L, rules.OPENING)    # once per game, board in a known position
    r = vision.read(img, c, L, model, "norm")          # method "norm", "scan" (default here), "slots" or "hough"
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

norm (game default since 12.11): per slot, steel disc / nearby glass says if a checker is there,
rim / nearby glass says whose; robust to the light falloff and notes on the glass.
Three more ways to find the checkers (all use the classes above):
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
from files import data

import cv2
import numpy as np

PATH = data("vision_model.json")   # <PC code>/data/
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
    norm: dict = None              # thresholds of the "norm" method (learn_norm)

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
        d = {k: {"mu": list(self.mu[k]), "sd": list(self.sd[k])} for k in CLASSES}
        if self.norm:
            d["norm"] = self.norm
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path=PATH):
        d = json.loads(Path(path).read_text())
        return cls({k: np.array(d[k]["mu"]) for k in CLASSES}, {k: np.array(d[k]["sd"]) for k in CLASSES},
                   d.get("norm"))


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
    return Model(mu, sd, learn_norm(img, c, L, board))


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


NORM_DISC_R, NORM_RIM, NORM_BG = 6.0, (9.0, 13.5), (30.0, 45.0)
NORM_DEFAULT = {"presence": 0.84, "colour": 0.62, "light": "human", "hidden": 0.6, "dark": True}


def _ring(r0, r1, nr, na):
    return np.array([(r * np.cos(a), r * np.sin(a)) for r in np.linspace(r0, r1, nr)
                     for a in np.linspace(0, 2 * np.pi, na, endpoint=False)])


NORM_OFFS = (_ring(0, NORM_DISC_R, 4, 16), _ring(*NORM_RIM, 3, 32), _ring(*NORM_BG, 3, 48))


def norm_features(gray, c, xy):
    """(N, 3): disc / glass, rim / glass, glass / whole-picture glass. xy: (N, 2) mm.
    "glass" = the bright part (90th percentile) of a ring 30-45 mm around the spot, so
    light falloff, glare and notes on the glass mostly cancel out."""
    xy = np.atleast_2d(np.asarray(xy, float))
    out = []
    for offs in NORM_OFFS:
        P = xy[:, None, :] + offs[None]
        u, v = c.mm_to_px(P[..., 0], P[..., 1])
        u = np.clip(np.rint(np.asarray(u, float)), 0, gray.shape[1] - 1).astype(int)
        v = np.clip(np.rint(np.asarray(v, float)), 0, gray.shape[0] - 1).astype(int)
        out.append(gray[v, u].astype(float))
    disc, rim, bg = out
    glass = np.maximum(np.percentile(bg, 90, axis=1), 1.0)
    whole = max(float(np.percentile(gray, 90)), 1.0)
    return np.stack([disc.mean(1) / glass, rim.mean(1) / glass, glass / whole], 1)


def _gray(img):
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2GRAY)


class _Shifted:
    """The calibration moved by (du, dv) pixels in the photo: the camera, mirror or glass
    moved a little since calibrate.py (seen on the real board 12.11: 8 px, 17 px)."""
    def __init__(self, c, du, dv):
        self.c, self.du, self.dv = c, du, dv

    def mm_to_px(self, x, y):
        u, v = self.c.mm_to_px(x, y)
        return np.asarray(u, float) + self.du, np.asarray(v, float) + self.dv


def find_shift(g, c, L, board, reach=40):
    """(du, dv) pixels that make the checkers of a known board stand out most from the glass
    under their slots (darker on the real board, lighter in the simulator)."""
    xy = np.array([q for q, _ in L.place(board).values()], float)
    cost = lambda du, dv: -abs(float(norm_features(g, _Shifted(c, du, dv), xy)[:, 0].mean()) - 1)  # noqa: E731
    best = min(((cost(du, dv), du, dv) for du in range(-reach, reach + 1, 4)
                for dv in range(-reach, reach + 1, 4)))
    _, u0, v0 = best
    best = min(((cost(du, dv), du, dv) for du in range(u0 - 3, u0 + 4) for dv in range(v0 - 3, v0 + 4)))
    return best[1], best[2]


def learn_norm(img, c, L, board):
    """Thresholds of the "norm" method from a photo of a known board. Tries no shift, the
    shift find_shift suggests and the half of it; keeps the one that reads this board best
    (on the real board the best-contrast shift was sometimes worse than none)."""
    g = _gray(img)
    du, dv = find_shift(g, c, L, board)
    best = None
    for sh in dict.fromkeys([(0, 0), (round(du / 2), round(dv / 2)), (du, dv)]):
        n = _learn_norm_at(g, c, L, board, *sh)
        ck, _ = find_norm(img, c, L, type("M", (), {"norm": n})())
        wrong = sum(abs(a - b) for a, b in zip(summarize(ck, L).board, board))
        if best is None or wrong < best[0]:
            best = (wrong, n)
    return best[1]


def _learn_norm_at(g, c, L, board, du, dv):
    c = _Shifted(c, du, dv)
    placed = L.place(board)
    taken = [xy for xy, _ in placed.values()]
    occ = {"robot": [], "human": []}
    for xy, side in placed.values():
        occ[side].append(norm_features(g, c, [xy])[0])
    empty = [norm_features(g, c, [xy])[0] for _, _, xy in L.all_slots(depth=6)
             if min(np.hypot(xy[0] - t[0], xy[1] - t[1]) for t in taken) > L.checker * 1.2]
    n = dict(NORM_DEFAULT)
    vis = lambda a: [f for f in a if f[2] > n["hidden"]]  # noqa: E731
    rob, hum, emp = (np.array(vis(a)) for a in (occ["robot"], occ["human"], empty))
    if len(rob) and len(hum) and len(emp):
        occ_disc = np.concatenate([rob[:, 0], hum[:, 0]])
        n["dark"] = bool(np.median(occ_disc) < np.median(emp[:, 0]))
        if n["dark"]:
            n["presence"] = float((np.percentile(occ_disc, 95) + np.percentile(emp[:, 0], 10)) / 2)
        else:
            n["presence"] = float((np.percentile(occ_disc, 5) + np.percentile(emp[:, 0], 90)) / 2)
        mr, mh = float(np.median(rob[:, 1])), float(np.median(hum[:, 1]))
        n["colour"] = (mr + mh) / 2
        n["light"] = "human" if mh > mr else "robot"
    n["shift"] = [du, dv]
    return n


def find_norm(img, c, L, model, depth=7):
    """Judge every slot by disc / glass (is there a checker) and rim / glass (whose).
    Slots whose glass is much darker than the picture's glass are hidden (not read)."""
    n = model.norm if model is not None and model.norm else NORM_DEFAULT
    g = _gray(img)
    if n.get("shift"):
        c = _Shifted(c, *n["shift"])
    dark = "robot" if n["light"] == "human" else "human"
    found, hidden = [], []
    places = list(range(1, 25)) + ["bar_robot", "bar_human", "off_robot", "off_human"]
    for w in places:
        k = 15 if str(w).startswith("off") else 4 if str(w).startswith("bar") else depth
        xy = [L.slot(w, i) for i in range(k)]
        f = norm_features(g, c, xy)
        here = []
        for (x, y), (disc, rim, vis) in zip(xy, f):
            if vis <= n["hidden"]:
                hidden.append(w)
            elif (disc < n["presence"]) == n.get("dark", True):
                here.append((float(x), float(y), rim))
        if not here:
            continue
        if isinstance(w, int):
            # a point holds one colour only: decide by the median rim of all its checkers
            # (the outermost checker sees more glass in its rim and looks lighter)
            side = n["light"] if float(np.median([h[2] for h in here])) > n["colour"] else dark
        else:
            side = w.split("_")[1]   # bar_robot / off_human ...: the place says whose
        found += [(x, y, side) for x, y, _ in here]
    return found, sorted(set(map(str, hidden)))


_detector = {}


def read(img, c, L, model, method="scan"):
    if method == "both":      # learned + cnn averaged (lib/ensemble.py)
        import cnn
        import ensemble
        import learned
        if "d" not in _detector:
            read(img, c, L, model, "learned")   # loads detector.json (or says it is missing)
        if "cnn" not in _detector:
            read(img, c, L, model, "cnn")
        x, e = cnn.inputs(img, c, L)
        return summarize(ensemble.find(_detector["d"], _detector["cnn"], learned.features(img, c, L), x, e, L), L)
    if method == "cnn":       # detector_cnn.pt from train_detector.py (needs PyTorch)
        import cnn
        if "cnn" not in _detector:
            try:
                _detector["cnn"] = cnn.Detector.load()
            except (OSError, ImportError):
                raise ValueError("no pixel network yet: install PyTorch, run collect.py, then train_detector.py") from None
        return summarize(_detector["cnn"].find(img, c, L), L)
    if method == "learned":   # detector.json from train_detector.py
        import learned
        if "d" not in _detector:
            try:
                _detector["d"] = learned.Detector.load()
            except OSError:
                raise ValueError("no detector.json yet: run collect.py, then train_detector.py") from None
        return summarize(_detector["d"].find(img, c, L)[0], L)
    if method == "norm":
        ck, hidden = find_norm(img, c, L, model)
        r = summarize(ck, L)
        if hidden:
            r.warnings.append("hidden from the camera (not read): " + ", ".join(hidden))
        return r
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


# -- reading by change: compare with the photo taken before the move -------------------
DELTA_R = 10.0          # mm, disc around each slot that is compared
DELTA_MIN = 0.12        # a slot changed when |after - before| / glass is above this (or 3x the typical)
_DELTA_OFFS = _ring(0, DELTA_R, 5, 24)


def _places(L):
    """{place: [slot xy, ...]} every slot the change reading looks at."""
    out = {p: [L.slot(p, i) for i in range(8)] for p in range(1, 25)}
    for w in ("bar_robot", "bar_human"):
        out[w] = [L.slot(w, i) for i in range(4)]
    for w in ("off_robot", "off_human"):
        out[w] = [L.slot(w, i) for i in range(15)]
    return out


def slot_changes(before, after, c, L):
    """{(place, i): True/False} which slots look different between two photos (no colours
    or learning needed: everything that did not move cancels out - notes on the glass,
    light falloff, the rail, the cart, a checker that is hard to see)."""
    gb, ga = _gray(before).astype(float), _gray(after).astype(float)
    keys, xy = [], []
    for w, pts in _places(L).items():
        for i, q in enumerate(pts):
            keys.append((w, i))
            xy.append(q)
    d = _change(gb, ga, c, xy)
    t = max(DELTA_MIN, 3 * float(np.median(d)))
    return dict(zip(keys, d > t)), dict(zip(keys, d))


def _change(gb, ga, c, xy):
    """How much each spot xy (mm) differs between two grey photos, relative to its glass."""
    xy = np.array(xy, float).reshape(-1, 2)
    P = xy[:, None, :] + _DELTA_OFFS[None]
    u, v = c.mm_to_px(P[..., 0], P[..., 1])
    u = np.clip(np.rint(np.asarray(u, float)), 0, gb.shape[1] - 1).astype(int)
    v = np.clip(np.rint(np.asarray(v, float)), 0, gb.shape[0] - 1).astype(int)
    a, b = ga[v, u], gb[v, u]
    glass = np.maximum(np.percentile(np.maximum(a, b), 90, axis=1), 1.0)
    return np.abs(a - b).mean(1) / glass


def changed_at(before, after, c, L, xy):
    """[True/False] did each spot xy (mm, anywhere on the glass) change between two photos
    (the same measure and threshold as slot_changes)."""
    if not len(xy):
        return []
    gb, ga = _gray(before).astype(float), _gray(after).astype(float)
    ref = _change(gb, ga, c, [q for pts in _places(L).values() for q in pts])
    t = max(DELTA_MIN, 3 * float(np.median(ref)))
    return list(_change(gb, ga, c, xy) > t)


def _expected(L, b0, b1):
    """{(place, i): True} the slots that change between two boards."""
    p0, p1 = L.place(b0), L.place(b1)
    return {k for k in set(p0) | set(p1) if (k in p0) != (k in p1)
            or (k in p0 and p0[k][1] != p1[k][1])}


def change_cost(changed, L, b0, b1):
    """Mismatch between the slots seen to change and the slots that should change, counted
    per place (a checker put a slot further along its point still counts as the right point)."""
    exp = _expected(L, b0, b1)
    seen, want = {}, {}
    for (w, i), ch in changed.items():
        seen[w] = seen.get(w, 0) + int(ch)
    for w, i in exp:
        want[w] = want.get(w, 0) + 1
    return sum(abs(seen.get(w, 0) - want.get(w, 0)) for w in set(seen) | set(want))


def closest_by_change(before, after, c, L, board, candidates):
    """(best board, cost, margin) out of candidates, judged only by what changed since the
    `before` photo of `board`. margin = cost of the second best minus the best."""
    changed, _ = slot_changes(before, after, c, L)
    uniq = []
    for cb in candidates:
        if list(cb) not in uniq:
            uniq.append(list(cb))
    candidates = uniq
    costs = sorted(((change_cost(changed, L, board, cb), i) for i, cb in enumerate(candidates)))
    best = candidates[costs[0][1]]
    margin = (costs[1][0] - costs[0][0]) if len(costs) > 1 else 99
    return best, costs[0][0], margin


def side_colours(img, c, L, board, radius_mm=6.0):
    """{'robot': '#rrggbb', 'human': '#rrggbb'}: what each side's checkers look like in a photo of
    board (normally the opening): the median over the side's checkers of the mean colour of the
    disc on top of each. Brightened a little as a pair (the mirror view is dim) so the page can
    paint its drawing in the real colours (Nadav 14.11)."""
    import calib
    top, f = calib.view(img, c, xmax=L.xmax, ymax=L.ymax, board=False)
    k = f.k
    yy, xx = np.mgrid[0:top.shape[0], 0:top.shape[1]]
    got = {"robot": [], "human": []}
    for (xy, side) in L.place(board).values():
        u, v = (float(t) for t in f.mm_to_px(*xy))
        r = radius_mm * k
        if not (r < u < top.shape[1] - r and r < v < top.shape[0] - r):
            continue
        sel = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
        got[side].append(top[sel].reshape(-1, 3).mean(0))
    if not got["robot"] or not got["human"]:
        return None
    col = {s: np.median(np.array(v), axis=0) for s, v in got.items()}      # BGR
    peak = max(float(col["robot"].max()), float(col["human"].max()), 1.0)
    gain = min(255.0 / peak, 2.0) if peak < 230 else 1.0
    return {s: "#%02x%02x%02x" % tuple(int(min(255, round(float(x) * gain))) for x in v[::-1]) for s, v in col.items()}


def draw(img, c, r, L=None):
    """The photo with every checker found circled in its class colour, for the UI / log."""
    out = img.copy()
    for x, y, s in r.checkers:
        u, v = c.mm_to_px(x, y)
        col = (0, 255, 0) if s == "robot" else (0, 165, 255)
        cv2.circle(out, (int(u), int(v)), 14, col, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(u), int(v)), col, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)   # the centre
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
        cv2.imencode(".png", crop)[1].tofile(str(folder / k / f"{n:05d}.png"))
        n += 1
    return n
