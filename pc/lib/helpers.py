"""Small pieces shared by several scripts: photo, circle finder, click window, magnet on.

    from helpers import photo, ask, find_circles, step_aside, magnet_on
"""
import time

import cv2
import numpy as np

import ui
from board import BoardError

WIN = "photo"


STRIPES = 2.5   # good photos score 0-1.5; a broken frame (stripes) 6 or more
COLOUR_JUMP = 30.0   # mean colour (0-255) that may change between two photos of the same board
_last = {"mean": None}


def stripes(img, n=4):
    """Broken frames from the camera turn (part of) the picture into vertical or horizontal
    stripes. Per tile of an n x n grid: the step between neighbouring columns against that
    between neighbouring rows; stripes make one much larger than the other. Only tiles with
    real contrast count (a dim, smooth tile says nothing). Returns the worst tile."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    worst = 0.0
    for i in range(n):
        for j in range(n):
            t = g[i * h // n:(i + 1) * h // n, j * w // n:(j + 1) * w // n]
            dx = float(np.abs(np.diff(t, axis=1)).mean())
            dy = float(np.abs(np.diff(t, axis=0)).mean())
            if max(dx, dy) > 4:
                worst = max(worst, max(dx, dy) / max(min(dx, dy), 0.5))
    return worst


def broken(img, jpeg):
    """Why this photo looks broken, or '' when it looks fine."""
    if img is None:
        return "not a picture"
    if not jpeg.rstrip(b"\r\n\x00").endswith(b"\xff\xd9"):
        return "cut short"
    if stripes(img) > STRIPES:
        return "stripes"
    mean = img.reshape(-1, 3).mean(0)
    if _last["mean"] is not None and np.max(np.abs(mean - _last["mean"])) > COLOUR_JUMP:
        return "colour cast"   # e.g. all blue: the board does not change colour between photos
    return ""


def photo(b, tries=4):
    """(image, jpeg). A broken frame (cut short, stripes, a sudden colour cast) is taken
    again by itself, up to `tries` times."""
    for k in range(tries):
        jpeg = b.capture()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        why = broken(img, jpeg)
        if not why:
            break
        print(f"   broken photo from the camera ({why}), taking it again")
    if img is not None:
        _last["mean"] = img.reshape(-1, 3).mean(0)   # what the board looks like now
    return img, jpeg


def find_circles(img):
    s = img.shape[1] / 1024  # the sizes below were tuned on 1024 px wide photos
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20 * s,
                         param1=100, param2=28, minRadius=int(6 * s), maxRadius=int(70 * s))
    return [] if c is None else [(float(x), float(y), float(r)) for x, y, r in c[0]]


def ask(img, guess, label, maps=None, keys="Enter=OK  click=fix  s=skip  q=stop", retake=False):
    """Show the photo; return (u, v), 'skip', 'quit', or 'retake' (key r, when retake=True).
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
        if k == ord("r") and retake:
            return "retake"
        if k == ord("q"):
            return "quit"


def step_aside(x, y, args):
    """A spot --away mm from (x, y) in both axes: larger, or smaller if that leaves the board."""
    ax = x + args.away if x + args.away <= args.xmax else x - args.away
    ay = y + args.away if y + args.away <= args.ymax else y - args.away
    return ax, ay


def magnet_on(b, need_s=0.0):
    """Magnet on. First, when what is left of the board's heat budget can't cover need_s
    seconds (the drag about to happen), wait until it can: otherwise the board cuts the
    magnet halfway (! MAG_TIMEOUT) and the checker is dropped."""
    try:
        st = b.status()
        heat, budget = st.get("HEAT", 0), st.get("BUDGET", 60)
        left = (100 - heat) / 100.0 * budget
        if left < need_s:
            wait = (need_s - left) * 2.0 + 1     # it cools at half the rate it heats
            print(f"   magnet warm ({heat}%), cooling {wait:.0f} s before the next drag...")
            time.sleep(wait)
    except Exception:
        pass
    while True:
        try:
            b.magnet(True)
            return
        except BoardError as e:
            if "MAG_HOT" not in str(e):
                raise
            print("   magnet hot, waiting 10 s...")
            time.sleep(10)
