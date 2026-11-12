"""Paths for dragging a checker without touching the others.

    from paths import plan, drag
    pts = plan((100, 50), (300, 200), others=[(150, 60), (160, 90)])
    # [(100, 50), (128, 112), (300, 200)]   corners only, in mm
    drag(b, pts)                             # magnet on, through the corners, overshoot, off

The dragged checker's centre keeps CLEAR mm from every other checker's centre
(26 = touching; 30 leaves 4 mm air, so the magnet does not pull the neighbour too).
A neighbour that already stands closer than that to the start or the goal (a checker
in the same point) is allowed to stay that close, but the path never gets closer to it.
How: A* on a 5 mm grid with a small penalty for passing near checkers, then the corners
that are not needed are cut, so most moves are 1-3 straight lines.
"""
import heapq
import time

import numpy as np

CHECKER_MM = 26.0
CLEAR = 30.0      # min centre distance to other checkers while dragging
SOFT = 45.0       # prefer staying at least this far, when it costs little
GRID = 5.0        # mm
XMAX, YMAX = 410.0, 420.0
OVERSHOOT = 3.0   # mm past the target before letting go (the checker lags)
GRAB_S = 0.3      # s after the magnet turns on


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
    if _seg_ok(start, goal, others, need):
        return [tuple(start), tuple(goal)]

    nx, ny = int(xmax // GRID) + 1, int(ymax // GRID) + 1
    gx, gy = np.meshgrid(np.arange(nx) * GRID, np.arange(ny) * GRID, indexing="ij")
    pts = np.stack([gx, gy], -1)
    d = np.hypot(pts[..., None, 0] - others[:, 0], pts[..., None, 1] - others[:, 1])
    free = np.all(d >= need, axis=-1)
    near = np.clip((SOFT - d) / SOFT, 0, None).sum(-1)  # 0 far from all, grows near checkers

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
                        nbrs.append((m, GRID * np.hypot(di, dj) * (1 + 0.5 * near[m])))
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
    out, i = [chain[0]], 0
    while i < len(chain) - 1:
        j = len(chain) - 1
        while j > i + 1 and not _seg_ok(chain[i], chain[j], others, need):
            j -= 1
        out.append(chain[j])
        i = j
    return [(float(p[0]), float(p[1])) for p in out]


def length(path):
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def drag(b, path, overshoot=OVERSHOOT, grab_s=GRAB_S, xmax=XMAX, ymax=YMAX):
    """Drive the magnet along a planned path with the checker on it."""
    from calibrate import magnet_on  # waits if the magnet is hot
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
    b.magnet(False)
    time.sleep(0.2)
