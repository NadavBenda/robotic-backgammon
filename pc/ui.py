"""Small helpers shared by the scripts that open a photo window.

    import ui
    ui.keep_open()                                   # first line of main: window stays on double-click
    view = ui.with_bar(img, ["point 3/36", "Enter=OK  click=fix"])   # text above the photo
    ui.show(WIN, view)                               # clicks: subtract ui.BAR_H(lines) from y
    ui.draw_bottom(view, to_mm, to_px, u, v)         # expected outline of a checker bottom
"""
import atexit
import sys

import cv2
import numpy as np

CHECKER_R_MM = 13.0
FONT, SCALE, THICK, LINE_H = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2, 36


def keep_open():
    """When the script was started by a double-click (own console window on Windows),
    wait for Enter at the end, also after an error, so the window does not just vanish."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        n = ctypes.windll.kernel32.GetConsoleProcessList((ctypes.c_uint * 4)(), 4)
    except Exception:
        return
    if n <= 2:  # only py.exe / python.exe in this console: nobody else keeps it open
        atexit.register(lambda: input("\nPress Enter to close..."))


def bar_h(lines):
    return LINE_H * len(lines) + 12


def with_bar(img, lines, color=(255, 255, 255)):
    """The photo with a black strip on top holding the text, so nothing covers the photo."""
    h = bar_h(lines)
    out = np.zeros((img.shape[0] + h, img.shape[1], 3), np.uint8)
    out[h:] = img
    for i, t in enumerate(lines):
        cv2.putText(out, t, (12, LINE_H * (i + 1)), FONT, SCALE, color, THICK, cv2.LINE_AA)
    return out


def show(win, view):
    try:
        visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        visible = False
    if not visible:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(win, view.shape[1], view.shape[0])
    cv2.imshow(win, view)


def bottom_outline(to_mm, to_px, u, v, r_mm=CHECKER_R_MM):
    """Where the round bottom (radius r_mm) of a checker centred at pixel (u, v) should
    appear: the circle on the glass mapped through the calibration -> an ellipse in the photo."""
    x, y = (float(t) for t in to_mm(u, v))
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    a, b = to_px(x + r_mm * np.cos(t), y + r_mm * np.sin(t))
    return np.int32(np.stack([np.asarray(a, float), np.asarray(b, float)], -1)).reshape(-1, 1, 2)


def draw_bottom(view, to_mm, to_px, u, v, color=(0, 255, 0), dy=0, thick=2):
    """Draw that ellipse on view (dy = height of the text bar, if the view has one)."""
    pts = bottom_outline(to_mm, to_px, u, v)
    pts[..., 1] += dy
    cv2.polylines(view, [pts], True, color, thick, cv2.LINE_AA)
    cv2.drawMarker(view, (int(u), int(v) + dy), color, cv2.MARKER_CROSS, 16, 1)
