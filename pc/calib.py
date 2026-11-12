"""Camera calibration: pixel <-> board mm (the same mm as the G command).

    import calib
    c = calib.load()                 # reads calib.json next to this file
    x, y = c.px_to_mm(512, 300)      # pixel in the XGA photo -> mm on the board
    u, v = c.mm_to_px(200, 150)      # mm -> pixel (to look at a known spot)

Pixel -> mm: undo the barrel distortion of the 160 deg lens (radial k1, k2 around the image
centre), then a homography (perspective of the tilted view + mirror flip), then a small
polynomial correction. All fitted on points the robot itself measured (calibrate.py),
so no separate lens calibration (chessboard) is needed. mm -> pixel is fitted separately.
"""
import json
from pathlib import Path

import cv2
import numpy as np

PATH = Path(__file__).parent / "calib.json"


def _terms(a, b, deg):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.stack([a ** i * b ** j for i in range(deg + 1) for j in range(deg + 1 - i)], axis=-1)


def _homog(H, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = H[2, 0] * a + H[2, 1] * b + H[2, 2]
    return (H[0, 0] * a + H[0, 1] * b + H[0, 2]) / w, (H[1, 0] * a + H[1, 1] * b + H[1, 2]) / w


CENTER, NORM = (512.0, 384.0), 512.0  # XGA image centre, radius unit


def _undistort(a, b, k):
    a, b = np.asarray(a, float), np.asarray(b, float)
    x, y = (a - CENTER[0]) / NORM, (b - CENTER[1]) / NORM
    r2 = x * x + y * y
    f = 1 + k[0] * r2 + k[1] * r2 * r2
    return CENTER[0] + x * f * NORM, CENTER[1] + y * f * NORM


class _Map:
    """src -> dst: optional radial undistortion of src, homography, polynomial correction."""

    def __init__(self, H, corr, unit, deg, k=(0.0, 0.0)):
        self.H, self.corr, self.unit, self.deg = np.asarray(H), np.asarray(corr), unit, deg
        self.k = tuple(k)

    @classmethod
    def _fit_k(cls, src, dst, unit, deg, k):
        a, b = _undistort(src[:, 0], src[:, 1], k)
        H, _ = cv2.findHomography(np.stack([a, b], -1), np.float64(dst), 0)
        a, b = _homog(H, a, b)
        res = dst - np.stack([a, b], -1)
        corr = np.zeros((len(_terms(0, 0, deg)), 2))
        if deg > 0:
            corr = np.linalg.lstsq(_terms(a / unit, b / unit, deg), res / unit, rcond=None)[0]
        m = cls(H, corr, unit, deg, k)
        err = np.hypot(*(np.array(m(src[:, 0], src[:, 1])) - dst.T))
        return m, float(np.sqrt(np.mean(err ** 2)))

    @classmethod
    def fit(cls, src, dst, unit, deg, radial=False):
        if not radial:
            return cls._fit_k(src, dst, unit, deg, (0.0, 0.0))[0]
        # coarse-to-fine search for k1, k2 (homography-only fit), then refit with correction
        best, span, step = (0.0, 0.0), (2.0, 1.0), (0.1, 0.1)
        for _ in range(4):
            cand = [(best[0] + i * step[0], best[1] + j * step[1])
                    for i in np.arange(-span[0] / step[0], span[0] / step[0] + 1)
                    for j in np.arange(-span[1] / step[1], span[1] / step[1] + 1)]
            best = min(cand, key=lambda k: cls._fit_k(src, dst, unit, 0, k)[1])
            span, step = (step[0] * 2, step[1] * 2), (step[0] / 5, step[1] / 5)
        return cls._fit_k(src, dst, unit, deg, best)[0]

    def __call__(self, a, b):
        a, b = _undistort(a, b, self.k)
        a, b = _homog(self.H, a, b)
        d = _terms(a / self.unit, b / self.unit, self.deg) @ self.corr * self.unit
        return a + d[..., 0], b + d[..., 1]

    def dump(self):
        return {"H": self.H.tolist(), "corr": self.corr.tolist(), "unit": self.unit,
                "deg": self.deg, "k": list(self.k)}


class Calib:
    def __init__(self, to_mm, to_px, rms_mm=None, max_mm=None):
        self.to_mm, self.to_px = to_mm, to_px
        self.rms_mm, self.max_mm = rms_mm, max_mm

    def px_to_mm(self, u, v):
        return self.to_mm(u, v)

    def mm_to_px(self, x, y):
        return self.to_px(x, y)

    def save(self, path=PATH):
        Path(path).write_text(json.dumps({
            "to_mm": self.to_mm.dump(), "to_px": self.to_px.dump(),
            "rms_mm": self.rms_mm, "max_mm": self.max_mm,
        }, indent=1))


def fit(px, mm, deg=2):
    """px, mm: lists of (u, v) and (x, y) for the same points. Returns Calib with its error."""
    px, mm = np.asarray(px, float), np.asarray(mm, float)
    c = Calib(_Map.fit(px, mm, 400.0, deg, radial=True), _Map.fit(mm, px, 1024.0, 3))
    # leave-one-out error = honest estimate of the error at points not used in the fit
    errs = []
    for i in range(len(px)):
        keep = np.arange(len(px)) != i
        m = _Map._fit_k(px[keep], mm[keep], 400.0, deg, c.to_mm.k)[0]
        errs.append(float(np.hypot(*(np.array(m(px[i, 0], px[i, 1])) - mm[i]))))
    c.rms_mm = float(np.sqrt(np.mean(np.square(errs))))
    c.max_mm = float(np.max(errs))
    return c


def topdown(img, c, px_per_mm=1.0, xmax=410.0, ymax=420.0, margin=20.0):
    """The photo as if taken straight from above: lens distortion, tilt and mirror removed.
    Pixel (i, j) of the result is board mm (x0 + j / px_per_mm, y0 + i / px_per_mm) with
    x0 = y0 = -margin, so x grows to the right and y downward. Straight lines on the
    board come out straight when the calibration is good."""
    xs = np.arange(-margin, xmax + margin, 1.0 / px_per_mm)
    ys = np.arange(-margin, ymax + margin, 1.0 / px_per_mm)
    gx, gy = np.meshgrid(xs, ys)
    u, v = c.mm_to_px(gx, gy)
    out = cv2.remap(img, np.float32(u), np.float32(v), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
    # board edge (the magnet travel) as a thin frame
    k = px_per_mm
    cv2.rectangle(out, (int(margin * k), int(margin * k)),
                  (int((xmax + margin) * k), int((ymax + margin) * k)), (0, 200, 255), 1)
    return out


def load(path=PATH):
    d = json.loads(Path(path).read_text())
    mk = lambda m: _Map(m["H"], m["corr"], m["unit"], m["deg"], m.get("k", (0.0, 0.0)))
    return Calib(mk(d["to_mm"]), mk(d["to_px"]), d.get("rms_mm"), d.get("max_mm"))
