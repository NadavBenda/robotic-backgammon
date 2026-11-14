"""Checker detection learned from labelled photos (collect.py), numpy only (no installs).

Two small neural networks (one hidden layer each):
  presence: for every slot, is there a checker?  Input: what the slot looks like
            against the glass around it, whether circle detection found a round
            shape there, the same for its 4 neighbours (they hide it
            diagonally in the far rows), how far it is from the camera, and how it
            compares with the empty-board photo when there is one.
  colour:   for every point (column) that has checkers, whose are they? Input: the
            averages over its checkers + distance from the camera.
Bars and bear-off trays (15 places each): the place says whose.
Then: at most 15 per colour (the least sure ones go first).

    import learned
    det = learned.Detector.load()                 # detector.json next to the scripts
    ck, conf = det.find(img, c, L, empty=None)    # [(x, y, side)], {place: confidence}
"""
import json
from pathlib import Path
from files import data

import numpy as np

import vision

PATH = data("detector.json")   # <PC code>/data/
DEPTH = 7
NEIGH = ((27.0, 0.0), (-27.0, 0.0), (0.0, 29.0), (0.0, -29.0))
SMEAR = np.array([(6 * np.cos(a), 6 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)])
MATCH_MM = 14.0     # a labelled checker belongs to the nearest slot within this
POS_PRESENCE = True  # the presence network sees where on the board the slot is
POS_COLOUR = True    # the colour network sees where on the board the column is (may learn
                     # "this column is usually the robot's" instead of looking: diag_detector.py)


def slots(L):
    """[(place, i, (x, y))] the slots the detector judges."""
    out = [(p, i, L.slot(p, i)) for p in range(1, 25) for i in range(DEPTH)]
    out += [(w, i, L.slot(w, i)) for w in ("bar_robot", "bar_human") for i in range(4)]
    out += [(w, i, L.slot(w, i)) for w in ("off_robot", "off_human") for i in range(15)]   # bear-off trays
    return out


def _mean_gray(g, c, xy, r=6.0):
    offs = vision.NORM_OFFS[0]
    P = np.asarray(xy, float)[:, None, :] + offs[None]
    u, v = c.mm_to_px(P[..., 0], P[..., 1])
    u = np.clip(np.rint(np.asarray(u, float)), 0, g.shape[1] - 1).astype(int)
    v = np.clip(np.rint(np.asarray(v, float)), 0, g.shape[0] - 1).astype(int)
    return g[v, u].astype(float).mean(1)


HOUGH_PX_MM = 2.0


def hough_centres(img, c, L):
    """(K, 2) mm centres of the round shapes found in the top view (checker bottoms and rims)."""
    import cv2
    import calib
    k, m = HOUGH_PX_MM, 20.0
    top = calib.topdown(img, c, k, L.xmax, L.ymax, m)
    g = cv2.medianBlur(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), 5)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=18 * k, param1=100, param2=22,
                          minRadius=int(7 * k), maxRadius=int(15 * k))
    if cs is None:
        return np.zeros((0, 2))
    return cs[0][:, :2] / k - m


def features(img, c, L, empty=None):
    """(N, F) one row per slot of slots(L)."""
    g = vision._gray(img)
    S = slots(L)
    xy = np.array([q for _, _, q in S], float)
    own = vision.norm_features(g, c, xy)                       # disc, rim, visible
    cols = [own]
    for d in NEIGH:
        cols.append(vision.norm_features(g, c, xy + np.array(d))[:, :2])
    smear = np.min([vision.norm_features(g, c, xy + d)[:, 0] for d in SMEAR], axis=0)
    cols.append(smear[:, None])
    hc = hough_centres(img, c, L)            # Nadav 13.11: circle detection as one more input
    if len(hc):
        dd = np.hypot(xy[:, 0, None] - hc[None, :, 0], xy[:, 1, None] - hc[None, :, 1])
        cols.append(np.stack([np.minimum(dd.min(1), 30.0) / 30.0, (dd < 10).sum(1)], 1))
    else:
        cols.append(np.tile([1.0, 0.0], (len(xy), 1)))
    cols.append(np.stack([xy[:, 0] / L.xmax, xy[:, 1] / L.ymax], 1))
    if empty is not None:
        ge = vision._gray(empty)
        ratio = _mean_gray(g, c, xy) / np.maximum(_mean_gray(ge, c, xy), 1.0)
    else:
        ratio = np.full(len(xy), np.nan)
    cols.append(ratio[:, None])
    F = np.hstack(cols)
    F[:, -1] = np.where(np.isnan(F[:, -1]), 1.0, F[:, -1])
    F = np.hstack([F, np.isnan(ratio)[:, None].astype(float)])  # "no empty photo" flag
    return F


def labels(checkers, L):
    """Slot labels from collect.py's checkers: 0 empty, 1 robot, 2 human."""
    S = slots(L)
    xy = np.array([q for _, _, q in S], float)
    y = np.zeros(len(S), int)
    for x0, y0, side, _ in checkers:
        d = np.hypot(xy[:, 0] - x0, xy[:, 1] - y0)
        k = int(np.argmin(d))
        if d[k] <= MATCH_MM:
            y[k] = 1 if side == "robot" else 2
    return y


def colour_rows(F, present, L):
    """(points, rows): per point with checkers, the mean of its present slots' features."""
    S = slots(L)
    pts, rows = [], []
    for p in range(1, 25):
        idx = [k for k, (w, i, _) in enumerate(S) if w == p and present[k]]
        if not idx:
            continue
        sub = F[idx]
        pos = sub[:, -4:-2].mean(0) if POS_COLOUR else np.zeros(2)
        rows.append(np.concatenate([sub[:, :2].mean(0), sub[:, :2].min(0), pos, [len(idx) / DEPTH]]))
        pts.append(p)
    return pts, np.array(rows).reshape(len(rows), -1)


def _pres(F):
    """The presence network's input: F, without the position columns when POS_PRESENCE is off."""
    if POS_PRESENCE:
        return F
    G = np.array(F, float)
    G[:, -4:-2] = 0.0
    return G


def full(r):
    """A vision.Reading as 28 numbers: the board + (robot, human) in the trays."""
    return list(r.board) + list(r.off)


class MLP:
    """One hidden layer, tanh, sigmoid output. Trained full-batch with Adam."""

    def __init__(self, n_in, hidden=24, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def prob(self, X):
        Z = (X - self.mu) / self.sd
        H = np.tanh(Z @ self.W1 + self.b1)
        return 1 / (1 + np.exp(-(H @ self.W2 + self.b2)[:, 0]))

    def loss(self, X, y):
        p = np.clip(self.prob(X), 1e-6, 1 - 1e-6)
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))
        return float(np.mean(-w * (y * np.log(p) + (1 - y) * np.log(1 - p))))

    def fit(self, X, y, epochs=1500, lr=0.02, l2=1e-3, val=None, every=20, patience=8):
        """val = (Xv, yv), photos it does not learn from: every `every` steps it checks the
        loss there, keeps the best weights, and stops after `patience` checks without a
        better one (early stopping: past that point it only learns the training photos by
        heart). Without val: a fixed `epochs`."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-3), 0.5 / max(1 - y.mean(), 1e-3))   # balance classes
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        if val is None:
            epochs = min(epochs, 600)      # no photos to check on: the old fixed length
        best, keep, bad = np.inf, None, 0
        self.stopped_at = epochs
        for t in range(1, epochs + 1):
            if val is not None and t % every == 0:
                lv = self.loss(*val)
                if lv < best - 1e-4:
                    best, keep, bad = lv, [p.copy() for p in params], 0
                else:
                    bad += 1
                    if bad >= patience:
                        self.stopped_at = t
                        break
            H = np.tanh(Z @ self.W1 + self.b1)
            p = 1 / (1 + np.exp(-(H @ self.W2 + self.b2)[:, 0]))
            d = (p - y) * w / len(y)
            gW2 = H.T @ d[:, None] + l2 * self.W2
            gb2 = np.array([d.sum()])
            dH = (d[:, None] @ self.W2.T) * (1 - H ** 2)
            gW1 = Z.T @ dH + l2 * self.W1
            gb1 = dH.sum(0)
            for k, gr in enumerate((gW1, gb1, gW2, gb2)):
                m[k] = 0.9 * m[k] + 0.1 * gr
                v[k] = 0.999 * v[k] + 0.001 * gr ** 2
                params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        if keep is not None:
            for p, k in zip(params, keep):
                p[...] = k
        return self

    def dump(self):
        return {k: getattr(self, k).tolist() for k in ("W1", "b1", "W2", "b2", "mu", "sd")}

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        for k, val in d.items():
            setattr(m, k, np.asarray(val, float))
        return m


def split(samples, share=0.15):
    """(learn, check): the last share of the photos are kept apart to tell when to stop.
    The two copies of a photo (with / without the empty board) stay on the same side."""
    n = len(samples)
    if n < 8:
        return samples, []
    k = n - max(2, int(round(n * share)))
    while 0 < k < n and samples[k][1] is samples[k - 1][1]:   # same photo's labels
        k += 1
    return samples[:k], samples[k:]


class Detector:
    def __init__(self, presence=None, colour=None, light="human"):
        self.presence, self.colour, self.light = presence, colour, light

    # -- training -------------------------------------------------------------
    @classmethod
    def train(cls, samples, L, seed=0):
        """samples: [(F, y)] from features() and labels()."""
        learn, check = split(samples)

        def stack(ss):
            if not ss:
                return None
            return np.vstack([_pres(F) for F, _ in ss]), (np.concatenate([yy for _, yy in ss]) > 0).astype(float)

        def colour(ss):
            R, t = [], []
            S = slots(L)
            for F, yy in ss:
                pts, rows = colour_rows(F, yy > 0, L)
                for p, row in zip(pts, rows):
                    k = next(k for k, (w, _, _) in enumerate(S) if w == p and yy[k] > 0)
                    R.append(row)
                    t.append(1.0 if yy[k] == 1 else 0.0)
            return (np.array(R), np.array(t)) if R else None
        X, y = stack(learn)
        pres = MLP(X.shape[1], seed=seed).fit(X, y, val=stack(check))
        cl, cv = colour(learn), colour(check)
        col = MLP(cl[0].shape[1], hidden=8, seed=seed).fit(*cl, val=cv if cv and len(set(cv[1])) > 1 else None) if cl else None
        return cls(pres, col)

    # -- use ------------------------------------------------------------------
    def find(self, img, c, L, empty=None):
        F = features(img, c, L, empty)
        return self.find_f(F, L)

    def find_f(self, F, L):
        """([(x, y, side)], [(confidence, x, y, side)]) from a feature matrix."""
        S = slots(L)
        pp = self.presence.prob(_pres(F))
        present = pp > 0.5
        side_of = {}
        pts, rows = colour_rows(F, present, L)
        if len(pts) and self.colour is not None:
            pr = self.colour.prob(rows)
            side_of = {p: ("robot" if q > 0.5 else "human", abs(q - 0.5) * 2) for p, q in zip(pts, pr)}
        out = []
        for k, (w, i, (x, y)) in enumerate(S):
            if not present[k]:
                continue
            if isinstance(w, int):
                side, cc = side_of.get(w, ("robot", 0.0))
            else:
                side, cc = w.split("_")[1], 1.0
            out.append((min(abs(pp[k] - 0.5) * 2, cc), float(x), float(y), side))
        for s in ("robot", "human"):   # never more than 15 of a colour: drop the least sure
            mine = sorted([o for o in out if o[3] == s])
            while len(mine) > 15:
                out.remove(mine.pop(0))
        return [(x, y, s) for _, x, y, s in out], out

    def board(self, F, L):
        """26 numbers + (robot, human) in the bear-off trays."""
        return full(vision.summarize(self.find_f(F, L)[0], L))

    def colour_prob(self, F, present, L):
        """{point: P(robot)} for the points with present checkers."""
        pts, rows = colour_rows(F, present, L)
        if not len(pts) or self.colour is None:
            return {}
        return dict(zip(pts, self.colour.prob(rows)))

    # -- file -----------------------------------------------------------------
    def save(self, path=PATH, **extra):
        d = {"presence": self.presence.dump(), "colour": self.colour.dump() if self.colour else None}
        d.update(extra)
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path=PATH):
        d = json.loads(Path(path).read_text())
        return cls(MLP.from_dict(d["presence"]), MLP.from_dict(d["colour"]) if d.get("colour") else None)
