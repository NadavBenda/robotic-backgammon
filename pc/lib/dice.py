"""Dice: three ways to get a roll, the game picks by settings ("dice" in game settings).

    import dice
    dice.digital()                 # (5, 2)  the PC rolls (secure random), shown in the UI
    dice.parse("5 2") / parse("52") # (5, 2)  the person rolled real dice and typed them
    dice.from_photo(img, c, L)     # (5, 2) or None  real dice in the dice zone, seen from below

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


def draw_dice(img, c, dice, L=None, zone=None):
    """For the simulator: two white dice in the zone, seen from below (7 - top pips)."""
    from layout import Layout
    L = L or Layout.load()
    x0, y0, x1, y1 = zone or L.dice_zone
    for k, top in enumerate(dice):
        cx = x0 + (x1 - x0) * (0.27 + 0.46 * k)
        cy = (y0 + y1) / 2 + (3 if k else -3)
        h = DIE_MM / 2
        sq = [c.mm_to_px(*L.to_magnet(cx + dx, cy + dy)) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        cv2.fillPoly(img, [np.int32(sq)], (235, 235, 235), cv2.LINE_AA)
        for px, py in _PIPS[7 - top]:
            t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = [c.mm_to_px(*L.to_magnet(cx + px * 4.5 + 1.5 * np.cos(a), cy + py * 4.5 + 1.5 * np.sin(a)))
                    for a in t]
            cv2.fillPoly(img, [np.int32(ring)], (20, 20, 20), cv2.LINE_AA)


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
