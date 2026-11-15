"""Dice: three ways to get a roll, the game picks by settings ("dice" in game settings).

    import dice
    dice.digital()                 # (5, 2)  the PC rolls (secure random), shown in the UI
    dice.parse("5 2") / parse("52") # (5, 2)  the person rolled real dice and typed them
    dice.read(before, after, c, L) # (5, 2) or None  real dice anywhere on the glass, found by
                                   #   what changed between two photos (the game uses this one;
                                   #   dice_test.py tries it on the real board)
    dice.from_photo(img, c, L)     # (5, 2) or None  the first try: white dice in a fixed dice zone

The camera looks up through the glass, so it sees the face lying on the glass: the
bottom. Opposite faces add up to 7, so top = 7 - (pips seen). Plain white dice with
black pips work best. The dice zone is a rectangle in the layout's picture frame
(layout.json "dice_zone"; default: between the two bear-off trays, just beyond them,
outside the checkers' travel). Experimental:
always shown in the UI so a person can correct it.
"""
import re
import secrets

import cv2
import numpy as np

DIE_MM = 16.0
PX_PER_MM = 4


def digital():
    return secrets.randbelow(6) + 1, secrets.randbelow(6) + 1


def opening_roll():
    """Each side rolls one die until they differ: ('robot' | 'human', (d1, d2))."""
    while True:
        a, b = digital()
        if a != b:
            return ("robot" if a > b else "human"), (a, b)


def parse(text):
    """'5 2', '5,2', '52', '5-2' -> (5, 2); None if it is not two numbers 1..6."""
    d = re.findall(r"[1-6]", text or "")
    if len(d) != 2 or len(re.findall(r"\d", text)) != 2:
        return None
    return int(d[0]), int(d[1])


_PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
         4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
         6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def _zone_warp(c, L, zone=None):
    zone = zone or L.dice_zone
    x0, y0, x1, y1 = zone
    src = np.float32([c.mm_to_px(*L.to_magnet(x, y)) for x, y in ((x0, y0), (x1, y0), (x1, y1))])
    w, h = int((x1 - x0) * PX_PER_MM), int((y1 - y0) * PX_PER_MM)
    dst = np.float32([[0, 0], [w, 0], [w, h]])
    return cv2.getAffineTransform(src, dst), (w, h)


def draw_dice(img, c, dice, L=None, zone=None, dark=False, size=DIE_MM, gap=0.46):
    """For the simulator: two dice in the zone, seen from below (7 - top pips). White with
    black pips, or dark with white pips; gap: how far apart (0.25 = touching for 16 mm in the default zone)."""
    from layout import Layout
    L = L or Layout.load()
    x0, y0, x1, y1 = zone or L.dice_zone
    for k, top in enumerate(dice):
        cx = x0 + (x1 - x0) * (0.5 - gap / 2 + gap * k)
        cy = (y0 + y1) / 2 + (3 if k else -3)
        h = size / 2
        face, pip = ((40, 30, 150), (240, 240, 240)) if dark else ((235, 235, 235), (20, 20, 20))
        u = size / 16.0
        sq = [c.mm_to_px(*L.to_magnet(cx + dx, cy + dy)) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        cv2.fillPoly(img, [np.int32(sq)], face, cv2.LINE_AA)
        for px, py in _PIPS[7 - top]:
            t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = [c.mm_to_px(*L.to_magnet(cx + u * (px * 4.5 + 1.5 * np.cos(a)), cy + u * (py * 4.5 + 1.5 * np.sin(a))))
                    for a in t]
            cv2.fillPoly(img, [np.int32(ring)], pip, cv2.LINE_AA)


def from_photo(img, c, L, zone=None, debug=None):
    """The two dice in the zone, or None when it does not see exactly two clear dice."""
    M, (w, h) = _zone_warp(c, L, zone)
    top = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white)
    side = DIE_MM * PX_PER_MM
    dice = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (0.5 * side * side < area < 2.0 * side * side):
            continue
        crop = g[y:y + bh, x:x + bw]
        mask = (lab[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, cs, -1, 1, -1)       # fill the holes (the pips)
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))  # keep the dark border out
        dark = ((crop < 0.6 * np.median(crop[mask > 0])) & (mask > 0)).astype(np.uint8)
        k, _, st, _ = cv2.connectedComponentsWithStats(dark)
        pip_area = np.pi * (1.5 * PX_PER_MM) ** 2  # pip about 3 mm
        pips = sum(1 for j in range(1, k) if 0.25 * pip_area < st[j, 4] < 5 * pip_area)
        if 1 <= pips <= 6:
            dice.append((x, 7 - pips))
    if debug is not None:
        debug.append(top)
    if len(dice) != 2:
        return None
    dice.sort()
    return dice[0][1], dice[1][1]


# -- real dice anywhere on the glass, found by what changed (Nadav 15.11) ----------------
VIEW_K = 4             # px per mm of the top view the dice are looked for in
VIEW_MARGIN = 50.0     # mm of glass around the magnet's travel that is searched too
DIE_MM_RANGE = (10.0, 30.0)   # sides of the dice that are accepted (bigger and clearer is better)
CHANGE_MIN = 18        # grey levels: less than this between the two photos is not a change


def _view(img, c, L):
    import calib
    return calib.topdown(img, c, VIEW_K, L.xmax, L.ymax, VIEW_MARGIN, frame=False)


def _pips(g, mask, die_px2):
    """[(x, y)] centres of the pips inside mask of the grey picture g: the face and the pips
    are two brightness classes (Otsu), the smaller class is the pips, whichever is darker."""
    k = max(3, int(round(np.sqrt(die_px2) * 0.10)) | 1)
    inner = cv2.erode(mask, np.ones((k, k), np.uint8))          # keep the die's edge out
    vals = g[inner > 0]
    if len(vals) < 50 or int(vals.max()) - int(vals.min()) < 25:
        return []
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = ((g > t) & (inner > 0)).astype(np.uint8)
    lo = ((g <= t) & (inner > 0)).astype(np.uint8)
    small = hi if hi.sum() < lo.sum() else lo
    n, _, st, cen = cv2.connectedComponentsWithStats(small)
    blobs = [(st[j, 2], st[j, 3], st[j, 4], cen[j]) for j in range(1, n) if st[j, 4] >= 0.004 * die_px2]
    if not blobs or sum(a for _, _, a, _ in blobs) > 0.45 * die_px2:
        return []
    # one pip: a round blob. Pips that run together (a 6 seen at low resolution) count by area.
    single = [a for w, h, a, _ in blobs if 0.6 < w / max(1, h) < 1.7 and a > 0.55 * w * h and a < 0.09 * die_px2]
    unit = float(np.median(single)) if single else 0.028 * die_px2
    out = []
    for w, h, a, cxy in blobs:
        k = max(1, int(round(a / unit)))
        if k > 3 or a < 0.35 * unit:
            return []            # not pips: a shadow, a reflection, a checker's rim
        out += [(float(cxy[0]), float(cxy[1]))] * k
    return out


def read(before, after, c, L, debug=None, checkers=()):
    """(d1, d2) the two dice a person rolled on the glass, or None when exactly two clear
    dice are not seen. No fixed place and no fixed colour: a die is a die-sized square that
    appeared between the two photos (before: the board without the dice; after: with
    them), and its pips are the smaller of the two brightness classes on its face. The
    camera sees the face lying on the glass, so each value is 7 - (pips seen).
    Two dice that touch are split along their long side. A die on top of a checker is hidden
    from the camera: None, and the person rolls again or types the roll.
    checkers: [(x, y)] magnet mm of the checkers the game knows about; nothing is looked for on them.
    debug: a list; the top view with what was found drawn on it is appended."""
    a, b = _view(after, c, L), _view(before, c, L)
    sharp = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    ga = cv2.GaussianBlur(sharp, (5, 5), 0)
    gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(ga, gb)
    ck = [((x + VIEW_MARGIN) * VIEW_K, (y + VIEW_MARGIN) * VIEW_K) for x, y in checkers]
    t = max(CHANGE_MIN, 4.0 * float(np.median(diff)) + 6)
    mask = (diff > t).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = (s * VIEW_K for s in DIE_MM_RANGE)
    found = []          # (x, value, box) per die
    view = a.copy() if debug is not None else None
    for cnt in cs:
        hull = cv2.convexHull(cnt)
        (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
        short, long_ = min(w, h), max(w, h)
        area = cv2.contourArea(hull)
        if not (lo * 0.8 < short < hi * 1.15) or area < 0.8 * w * h:
            continue
        if any(np.hypot(cx - u, cy - v) < 13 * VIEW_K for u, v in ck):
            continue             # on a checker: its rim moved a hair, or a die that the checker hides
        two = 1.6 < long_ / short < 2.4          # two dice side by side, touching
        if not two and long_ / short > 1.4:
            continue
        m = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        pips = _pips(sharp, m, short * short)
        if view is not None:
            cv2.drawContours(view, [np.int32(cv2.boxPoints(((cx, cy), (w, h), ang)))], -1, (0, 255, 0), 2)
            for px, py in pips:
                cv2.circle(view, (int(px), int(py)), 6, (0, 0, 255), 2)
        if two:
            ax = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]) if w >= h else \
                np.array([-np.sin(np.radians(ang)), np.cos(np.radians(ang))])
            side = [float(np.dot((px - cx, py - cy), ax)) for px, py in pips]
            for sign in (-1, 1):
                n = sum(1 for s in side if s * sign > 0)
                if 1 <= n <= 6:
                    found.append((cx + sign * ax[0] * long_ / 4, n))
        elif 1 <= len(pips) <= 6:
            found.append((cx, len(pips)))
    if debug is not None:
        debug.append(view)
    if len(found) != 2:
        return None
    found.sort()
    return 7 - found[0][1], 7 - found[1][1]
