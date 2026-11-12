"""Paths for dragging a checker without touching the others.

    from paths import plan, drag
    pts = plan((100, 50), (300, 200), others=[(150, 60), (160, 90)])
    # [(100, 50), (128, 112), (300, 200)]   corners only, in mm
    drag(b, pts)                             # magnet on, through the corners, overshoot, off

The dragged checker's centre keeps CLEAR mm from every other checker's centre
(26 = touching; 30 leaves 4 mm air, so the magnet does not pull the neighbour too).
A neighbour that already stands closer than that to the start or the goal (a checker
in the same point) is allowed to stay that close, but the path never gets closer to it.
Beyond that hard limit the path keeps as much room as it can (SOFT, 50 mm centre to
centre = 24 mm air), so a checker that lags, a sloppy hand placement or a camera error of
a few mm does not turn into a knock: a straight line is used only when it already keeps
that room; otherwise A* on a 5 mm grid with a strong penalty for passing near checkers,
then the corners that are not needed are cut, but never in a way that passes closer to
a checker than the A* path did. Most moves are still 1-3 straight lines, and each corner
between them that turns more than 30 deg is cut by one short line (round_corners).
"""
import heapq
import time

import numpy as np

CHECKER_MM = 26.0
CLEAR = 30.0      # min centre distance to other checkers while dragging
SOFT = 50.0       # prefer staying at least this far (24 mm air), worth a detour
NEAR_W = 3.0      # how much a mm next to a checker costs over a mm in the open (x (1 + NEAR_W))
GRID = 5.0        # mm
XMAX, YMAX = 410.0, 420.0
OVERSHOOT = 3.0   # mm past the target before letting go (the checker lags)
GRAB_S = 0.3      # s after the magnet turns on
ROUND_MM = 10.0   # a sharp corner is cut by one short line from this far before it to this far after
ROUND_N = 0       # extra points between those two (0 = a plain cut; each point is a brief stop)
ROUND_MIN_DEG = 30.0  # corners that turn less than this are left alone


class NoPath(RuntimeError):
    pass


def _seg_dist(p, a, b):
    """Distance from points p (N,2) to segment a-b."""
    p, a, b = np.atleast_2d(p).astype(float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def _needs(start, goal, others, clear):
    """Per obstacle: how close the path may get (less than clear only for close neighbours
    of the start or goal, and never closer than they already are)."""
    if len(others) == 0:
        return np.zeros(0)
    ds = np.hypot(*(others - start).T)
    dg = np.hypot(*(others - goal).T)
    return np.minimum(clear, np.minimum(ds, dg)) - 0.5


def _seg_ok(a, b, others, need):
    return len(others) == 0 or bool(np.all(_seg_dist(others, a, b) >= need))


def _short(a, b, others, soft):
    """How many mm the segment a-b comes closer than wanted (soft) to its worst checker."""
    if len(others) == 0:
        return 0.0
    return float(max(0.0, np.max(soft - _seg_dist(others, a, b))))


def clearance(path, others):
    """Smallest centre distance from the path to any of the others (mm)."""
    o = np.asarray(others, float).reshape(-1, 2)
    if len(o) == 0:
        return float("inf")
    return float(min(np.min(_seg_dist(o, a, b)) for a, b in zip(path, path[1:])))


def plan(start, goal, others=(), xmax=XMAX, ymax=YMAX, clear=CLEAR):
    """Corners of a path from start to goal (mm), first = start, last = goal.
    Raises NoPath when the checkers block every way (move one of them first)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    others = np.asarray(others, float).reshape(-1, 2)
    # the checker being moved, if it was passed among the others, is not an obstacle
    others = others[np.hypot(*(others - start).T) > 3] if len(others) else others
    if len(others) and np.min(np.hypot(*(others - goal).T)) < CHECKER_MM - 1:
        raise NoPath(f"goal ({goal[0]:.0f},{goal[1]:.0f}) is taken")
    need = _needs(start, goal, others, clear)
    soft = _needs(start, goal, others, max(SOFT, clear))   # same idea, the distance we like
    if _seg_ok(start, goal, others, need) and _short(start, goal, others, soft) == 0:
        return [tuple(start), tuple(goal)]

    nx, ny = int(xmax // GRID) + 1, int(ymax // GRID) + 1
    gx, gy = np.meshgrid(np.arange(nx) * GRID, np.arange(ny) * GRID, indexing="ij")
    pts = np.stack([gx, gy], -1)
    d = np.hypot(pts[..., None, 0] - others[:, 0], pts[..., None, 1] - others[:, 1])
    free = np.all(d >= need, axis=-1)
    # 0 far from all; 1 per checker at the hard limit; grows fast closer in
    near = (np.clip((SOFT - d) / (SOFT - clear), 0, 1) ** 2).sum(-1)

    # start / goal connect to grid nodes around them by a clear straight line
    def links(p):
        i0, j0 = int(round(p[0] / GRID)), int(round(p[1] / GRID))
        out = []
        for i in range(i0 - 2, i0 + 3):
            for j in range(j0 - 2, j0 + 3):
                if 0 <= i < nx and 0 <= j < ny and free[i, j] and \
                        _seg_ok(p, pts[i, j], others, need):
                    out.append((i, j))
        return out

    S, G = "S", "G"
    goal_links = set(links(goal))
    if not goal_links:
        raise NoPath("no room around the goal")
    h = lambda n: 0.0 if n == G else float(np.hypot(*(pts[n] - goal)))
    best = {S: 0.0}
    prev = {}
    heap = [(float(np.hypot(*(start - goal))), 0, S)]
    k = 1
    while heap:
        _, _, n = heapq.heappop(heap)
        if n == G:
            break
        here = start if n == S else pts[n]
        if n == S:
            nbrs = [(m, float(np.hypot(*(pts[m] - start)))) for m in links(start)]
        else:
            i, j = n
            nbrs = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    m = (i + di, j + dj)
                    if (di or dj) and 0 <= m[0] < nx and 0 <= m[1] < ny and free[m]:
                        nbrs.append((m, GRID * np.hypot(di, dj) * (1 + NEAR_W * near[m])))
            if n in goal_links:
                nbrs.append((G, float(np.hypot(*(goal - here)))))
        for m, c in nbrs:
            nc = best[n] + c
            if nc < best.get(m, np.inf):
                best[m], prev[m] = nc, n
                heapq.heappush(heap, (nc + h(m), k, m))
                k += 1
    if G not in prev:
        raise NoPath(f"checkers block every way from ({start[0]:.0f},{start[1]:.0f}) "
                     f"to ({goal[0]:.0f},{goal[1]:.0f})")
    chain, n = [goal], prev[G]
    while n != S:
        chain.append(pts[n])
        n = prev[n]
    chain.append(start)
    chain = chain[::-1]
    # cut corners: from each corner jump to the farthest point still reachable in a line
    # that keeps at least the room the A* path kept there
    short = [_short(a, b, others, soft) for a, b in zip(chain, chain[1:])]
    out, i = [chain[0]], 0
    while i < len(chain) - 1:
        j = len(chain) - 1
        while j > i + 1 and not (_seg_ok(chain[i], chain[j], others, need) and
                                 _short(chain[i], chain[j], others, soft) <= max(short[i:j]) + 0.5):
            j -= 1
        out.append(chain[j])
        i = j
    return round_corners([(float(p[0]), float(p[1])) for p in out], others, need)


def round_corners(path, others=(), need=None, r=ROUND_MM, n=ROUND_N, min_deg=ROUND_MIN_DEG):
    """Cut each sharp corner (turn over min_deg) with one short line from r mm before it to
    r mm after it (n > 0: a curve through n more points), so the checker turns in two
    smaller steps. The firmware stops at every point, so this is kept to one extra point.
    A cut that would come closer to a checker than the path may (need) is not made."""
    if len(path) < 3:
        return list(path)
    o = np.asarray(others, float).reshape(-1, 2)
    P = [np.asarray(p, float) for p in path]
    out = [P[0]]
    for k in range(1, len(P) - 1):
        a, c, b = out[-1], P[k], P[k + 1]
        la, lb = np.hypot(*(c - a)), np.hypot(*(b - c))
        rr = min(r, la / 2, lb / 2)
        cosang = float(np.dot(c - a, b - c) / max(la * lb, 1e-9))
        if rr < 4 or np.degrees(np.arccos(np.clip(cosang, -1, 1))) < min_deg:
            out.append(c)
            continue
        p0 = c + (a - c) / la * rr
        p2 = c + (b - c) / lb * rr
        t = np.linspace(0, 1, n + 2)[:, None]
        arc = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
        ok = need is None or len(o) == 0 or all(
            np.all(_seg_dist(o, u, v) >= need) for u, v in zip(arc, arc[1:]))
        out += list(arc) if ok else [c]
    out.append(P[-1])
    return [(float(p[0]), float(p[1])) for p in out]


def length(path):
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def drag(b, path, overshoot=OVERSHOOT, grab_s=GRAB_S, xmax=XMAX, ymax=YMAX):
    """Drive the magnet along a planned path with the checker on it."""
    from helpers import magnet_on  # waits if the magnet is hot
    b.move(*path[0])
    magnet_on(b)
    time.sleep(grab_s)
    for p in path[1:-1]:
        b.move(*p)
    a, z = np.asarray(path[-2], float), np.asarray(path[-1], float)
    d = z - a
    n = float(np.hypot(*d))
    end = np.clip(z + (d / n * overshoot if n > 1 else 0), 0, [xmax, ymax])
    b.move(float(end[0]), float(end[1]))
    b.magnet(False)  # the board waits 250 ms before it answers
