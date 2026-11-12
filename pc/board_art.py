"""Draw the board graphics at true size (1 unit = 1 mm) from layout.json, to print, cut as
vinyl, or paint under/over the glass. The drawing and the robot then agree by construction.

    py board_art.py        -> board_art.svg (open in a browser / Inkscape; print at 100 %)

Seen from above, near side (bear-off trays) at the bottom. Light colours on purpose: the
camera must still see the checkers' rims through it. Points are long thin triangles along
each row, the middle lane is left empty (the magnet's road), point numbers are the
human's (1 = the human's deepest home point). If it is put under the glass, mirror it.
"""
from pathlib import Path

from layout import Layout

OUT = Path(__file__).parent / "board_art.svg"
MARGIN = (25, 45, 25, 25)   # mm around the travel: left, near, right, far (lid opening 460 x 490)


def svg(L=None, mirror=False):
    L = L or Layout.load()
    x0 = -MARGIN[0]
    w, h = L.xmax + MARGIN[0] + MARGIN[2], L.ymax + MARGIN[1] + MARGIN[3]
    Y = lambda y: (L.ymax + MARGIN[3]) - y    # picture y (up, from the near side) -> svg y (down)
    r = L.checker / 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
             f'viewBox="{x0} 0 {w} {h}">',
             f'<g{" transform=" + chr(34) + f"translate({2 * x0 + w},0) scale(-1,1)" + chr(34) if mirror else ""}>',
             f'<rect x="{x0}" y="0" width="{w}" height="{h}" fill="none" stroke="#999" stroke-width="0.5"/>']
    lane0, lane1 = L.lane_x()
    for p in range(1, 25):
        x, y = L._nominal(p, 0)
        left = x < L.xmax / 2
        tip = lane0 - 4 if left else lane1 + 4
        base = x - r if left else x + r
        col = "#d9c3a0" if p % 2 else "#b99a72"
        parts.append(f'<polygon points="{base},{Y(y) - r + 1} {base},{Y(y) + r - 1} {tip},{Y(y)}" '
                     f'fill="{col}" fill-opacity="0.55"/>')
        tx = base - 7 if left else base + 7
        parts.append(f'<text x="{tx}" y="{Y(y) + 2.5}" font-size="7" text-anchor="middle" '
                     f'font-family="sans-serif" fill="#555">{25 - p}</text>')
    by = L._row_y(6)
    parts.append(f'<rect x="{x0 + 3}" y="{Y(by) - r}" width="{w - 6}" height="{2 * r}" rx="4" '
                 f'fill="#cfc6b8" fill-opacity="0.35"/>')
    parts.append(f'<text x="{L.xmax / 2}" y="{Y(by) + 2.5}" font-size="7" text-anchor="middle" '
                 f'font-family="sans-serif" fill="#777">BAR</text>')
    for side in ("off_robot", "off_human"):
        for i in range(14):
            x, y = L._nominal(side, i)
            parts.append(f'<circle cx="{x}" cy="{Y(y)}" r="{r}" fill="none" stroke="#bbb" stroke-width="0.4"/>')
    dx0, dy0, dx1, dy1 = L.dice_zone
    parts.append(f'<rect x="{dx0}" y="{Y(dy1)}" width="{dx1 - dx0}" height="{dy1 - dy0}" rx="3" '
                 f'fill="none" stroke="#999" stroke-dasharray="2 2" stroke-width="0.5"/>')
    parts.append(f'<text x="{(dx0 + dx1) / 2}" y="{Y(dy0) + 6}" font-size="5" text-anchor="middle" '
                 f'font-family="sans-serif" fill="#999">dice</text>')
    parts.append("</g></svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(svg())
    print(f"Wrote {OUT}")
