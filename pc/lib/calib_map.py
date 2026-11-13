"""Map of the calibration error: where on the board the camera <-> mm map is good or not.

    import calib_map
    img = calib_map.draw(c, background_photo)   # c = calib.load() from a run that saved its points

Each measured point: a dot coloured by its error (green < 2 mm, orange 2-3, red > 3), the
error in mm next to it, and a line from where the robot put the checker to where the map
says it is, drawn 5 times longer. The error is leave-one-out: the map fitted without that
point, so it is the error you get between the points too. Behind the dots the colour is
filled in between the points, so a bad area shows as a red patch.
"""
import cv2
import numpy as np

import calib

LIMIT_MM = 3.0
K = calib.PX_PER_MM


def _colour(e):
    if e < 2.0:
        return (60, 190, 60)
    if e < LIMIT_MM:
        return (0, 165, 255)
    return (40, 40, 230)


def draw(c, photo=None, xmax=410.0, ymax=420.0):
    flat = calib.Flat(c)
    if photo is not None:
        out = calib.view(photo, c, xmax=xmax, ymax=ymax, board=False)[0]
        out = (out * 0.6).astype(np.uint8)
    else:
        w, h = int((xmax + 2 * flat.m) * K), int((ymax + 2 * flat.m) * K)
        out = np.full((h, w, 3), 40, np.uint8)
    pts = np.array([p[:5] for p in c.points], float)
    if len(pts) == 0:
        cv2.putText(out, "no points saved in calib.json: run calibrate.py again", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    # fill between the points (inverse distance weighting), coarse grid
    h, w = out.shape[:2]
    gy, gx = np.mgrid[0:h:8, 0:w:8]
    mx, my = flat.px_to_mm(gx + 4, gy + 4)
    d = np.hypot(mx[..., None] - pts[:, 2], my[..., None] - pts[:, 3]) + 1.0
    wgt = 1 / d ** 2
    e = (wgt * pts[:, 4]).sum(-1) / wgt.sum(-1)
    inside = (mx >= 0) & (mx <= xmax) & (my >= 0) & (my <= ymax)
    fill = np.zeros_like(out)
    for i in range(gy.shape[0]):
        for j in range(gx.shape[1]):
            if inside[i, j]:
                fill[gy[i, j]:gy[i, j] + 8, gx[i, j]:gx[i, j] + 8] = _colour(e[i, j])
    out = np.where(fill.any(-1, keepdims=True), cv2.addWeighted(out, 0.55, fill, 0.45, 0), out)
    for u, v, x, y, err in pts:
        seen = np.array([float(t) for t in c.px_to_mm(u, v)])   # where the map puts the click
        a = np.array(flat.mm_to_px(x, y), float)
        z = np.array(flat.mm_to_px(*(np.array([x, y]) + (seen - [x, y]) * 5)), float)
        cv2.line(out, tuple(np.int32(a)), tuple(np.int32(z)), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, tuple(np.int32(a)), 9, _colour(err), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(np.int32(a)), 9, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"{err:.1f}", (int(a[0]) + 12, int(a[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    bad = int((pts[:, 4] > LIMIT_MM).sum())
    txt = (f"error per point (mm, leave-one-out): RMS {np.sqrt(np.mean(pts[:, 4] ** 2)):.1f}, "
           f"worst {pts[:, 4].max():.1f}, {bad} of {len(pts)} over {LIMIT_MM:.0f} mm")
    cv2.putText(out, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "green <2  orange 2-3  red >3 mm;  white line = error x5", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
