"""Small pieces shared by several scripts: photo, circle finder, click window, magnet on.

    from helpers import photo, ask, find_circles, step_aside, magnet_on
"""
import time

import cv2
import numpy as np

import ui
from board import BoardError

WIN = "photo"


def photo(b):
    jpeg = b.capture()
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), jpeg


def find_circles(img):
    s = img.shape[1] / 1024  # the sizes below were tuned on 1024 px wide photos
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20 * s,
                         param1=100, param2=28, minRadius=int(6 * s), maxRadius=int(70 * s))
    return [] if c is None else [(float(x), float(y), float(r)) for x, y, r in c[0]]


def ask(img, guess, label, maps=None, keys="Enter=OK  click=fix  s=skip  q=stop"):
    """Show the photo; return (u, v), 'skip' or 'quit'.
    maps = (to_mm, to_px): draw the expected outline of the checker bottom at the point."""
    state = {"pt": guess}
    lines = [label, keys]
    dy = ui.bar_h(lines)

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and y >= dy:
            state["pt"] = (float(x), float(y - dy))

    ui.show(WIN, ui.with_bar(img, lines))
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        view = ui.with_bar(img, lines)
        if state["pt"]:
            u, v = state["pt"]
            if maps:
                ui.draw_bottom(view, *maps, u, v, dy=dy)
            else:
                cv2.circle(view, (int(u), int(v) + dy), 14, (0, 255, 0), 2)
                cv2.drawMarker(view, (int(u), int(v) + dy), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
        ui.show(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and state["pt"]:
            return state["pt"]
        if k == ord("s"):
            return "skip"
        if k == ord("q"):
            return "quit"


def step_aside(x, y, args):
    """A spot --away mm from (x, y) in both axes: larger, or smaller if that leaves the board."""
    ax = x + args.away if x + args.away <= args.xmax else x - args.away
    ay = y + args.away if y + args.away <= args.ymax else y - args.away
    return ax, ay


def magnet_on(b):
    while True:
        try:
            b.magnet(True)
            return
        except BoardError as e:
            if "MAG_HOT" not in str(e):
                raise
            print("   magnet hot, waiting 10 s...")
            time.sleep(10)
