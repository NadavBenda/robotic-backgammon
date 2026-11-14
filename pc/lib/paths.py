"""Paths for dragging a checker without touching the others.

    from paths import plan, drag
    pts = plan((100, 50), (300, 200), others=[(150, 60), (160, 90)])
    # [(100, 50), (128, 112), (300, 200)]   corners only, in mm
    drag(b, pts)                             # magnet on, corners, overshoot, back, wait, off

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
SEG_MM = 70.0     # what one more line in a path costs, as mm of travel (Nadav 13.11): the cart stops at
                  # every corner, and a stop + restart take about v*v/a = 180*180/800 mm worth of time
XMAX, YMAX = 410.0, 420.0
OVERSHOOT = 5.0   # mm past the target along the last line (the checker lags), then back
SETTLE_S = 0.4    # s on the target with the magnet still on, so the checker centres over it
GRAB_S = 0.3      # s after the magnet turns on
EDGE_MM = 5.0     # paths keep this far from x = 0 and y = 0: a move toward 0 that touches a limit switch
                  # stops the board with ERR LIMIT_X / LIMIT_Y and it must home again (Nadav 14.11)
LEAD_MM = 35.0    # a drag starts with one straight run of at least this (Nadav 14.11): the checker is
                  # picked up there at full acceleration, a turn straight away would let it slip off
ROUND_MM = 10.0   # a sharp corner is cut by one short line from this far before it to this far after
ROUND_N = 0       # extra points between those two (0 = a plain cut; each point is a brief stop)
ROUND_MIN_DEG = 30.0  # corners that turn less than this are left alone
DRAG_V, DRAG_A = 180.0, 600.0   # mm/s, mm/s2 with a checker (Nadav 14.11: softer than 800, checkers fell at corners)
CORNER_S = (0.1, 0.3)  # s the magnet waits on at a corner while dragging, slight -> sharp turn (Nadav 14.11),
CORNER_DEG = (30.0, 120.0)  # so the lagging checker catches up before the cart sets off the new way


def corner_wait(deg):
    """Seconds to wait at a corner that turns deg degrees (0.1 s up to 30 deg, 0.3 s from 120)."""
    (a, b), (d0, d1) = CORNER_S, CORNER_DEG
    return a + (b - a) * float(np.clip((deg - d0) / (d1 - d0), 0, 1))


def _turn_deg(a, p, b):
    u, v = np.subtract(p, a), np.subtract(b, p)
    nu, nv = np.hypot(*u), np.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def drag_seconds(path, grab_s=GRAB_S, settle_s=SETTLE_S, v=DRAG_V, a=DRAG_A):
    """How long the magnet stays on for this drag (for the heat budget)."""
    t = grab_s + settle_s + 0.5            # + the board's release wait
    for p, q in zip(path, path[1:]):
        L = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        t += L / v + v / a if L > v * v / a else 2 * np.sqrt(L / a)
    t += sum(corner_wait(_turn_deg(*path[i - 1:i + 2])) for i in range(1, len(path) - 1))
    return t + 2 * np.sqrt(OVERSHOOT / a) * 2


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


def _fewest_lines(chain, others, need, clear, soft_mm, seg_mm, corner_mm=0.0):
    """The corners to keep out of an A* chain: the cheapest way through its points, where a
    line costs its length (more next to checkers, as in the A*) + seg_mm for being one more
    line, + corner_mm per second the magnet waits at the corner (corner_wait: more for a
sharper turn, Nadav 14.11; 0 for magnet-off moves). A line may skip points only when it keeps the hard distance to every checker."""
    P = np.asarray(chain, float)
    n = len(P)
    if n <= 2:
        return list(chain)
    K = 24
    t = (np.arange(K) + 0.5) / K
    C = np.full((n, n), np.inf)          # C[i, j]: the line i -> j
    for i in range(n - 1):
        ab = P[i + 1:] - P[i]
        L = np.hypot(ab[:, 0], ab[:, 1])
        cost = L.copy()
        ok = np.ones(len(ab), bool)
        if len(others):
            S = P[i] + t[None, :, None] * ab[:, None, :]                          # (m, K, 2) points on each line
            d = np.hypot(S[:, :, None, 0] - others[:, 0], S[:, :, None, 1] - others[:, 1])   # (m, K, M)
            if soft_mm > clear:
                cost = L * (1 + NEAR_W * (np.clip((soft_mm - d) / (soft_mm - clear), 0, 1) ** 2).sum(-1).mean(-1))
            # exact distance checker - line, for the hard limit
            tt = np.clip(((others - P[i]) @ ab.T).T / np.maximum(L * L, 1e-9)[:, None], 0, 1)   # (m, M)
            Q = P[i] + tt[:, :, None] * ab[:, None, :]
            ok = np.all(np.hypot(Q[..., 0] - others[:, 0], Q[..., 1] - others[:, 1]) >= need, axis=1)
            ok[0] = True      # the next point of the chain: the A* already took this step
        C[i, i + 1:] = np.where(ok, cost + seg_mm, np.inf)
    best = np.full((n, n), np.inf)       # best[i, j]: the cheapest path from the start that ends with line i -> j
    back = np.zeros((n, n), int)
    best[0] = C[0]
    for j in range(1, n - 1):
        din = P[j] - P[:j]
        dout = P[j + 1:] - P[j]
        din = din / np.maximum(np.hypot(din[:, 0], din[:, 1]), 1e-9)[:, None]
        dout = dout / np.maximum(np.hypot(dout[:, 0], dout[:, 1]), 1e-9)[:, None]
        cand = best[:j, j][:, None]                                # (incoming, outgoing)
        if corner_mm:
            deg = np.degrees(np.arccos(np.clip(din @ dout.T, -1, 1)))
            (a, b), (d0, d1) = CORNER_S, CORNER_DEG
            cand = cand + corner_mm * (a + (b - a) * np.clip((deg - d0) / (d1 - d0), 0, 1))
        k = np.argmin(cand, axis=0)
        best[j, j + 1:] = cand[k, np.arange(cand.shape[1])] + C[j, j + 1:]
        back[j, j + 1:] = k
    j = n - 1
    i = int(np.argmin(best[:, j]))
    out = [j, i]
    while i:
        i, j = int(back[i, j]), i
        out.append(i)
    return [chain[k] for k in out[::-1]]


def _long_start(path, others, need, lead=None):
    """Make the first line at least LEAD_MM + ROUND_MM long (the corner is cut ROUND_MM before it):
    when it is shorter, drop the first corner if the line to the next one keeps clear, or else
    carry the first line straight on to that length and turn there, if that keeps clear."""
    lead = (LEAD_MM + ROUND_MM) if lead is None else lead
    path = list(path)
    A = lambda p: np.asarray(p, float)  # noqa: E731
    while len(path) > 2 and np.hypot(path[1][0] - path[0][0], path[1][1] - path[0][1]) < lead:
        if _seg_ok(A(path[0]), A(path[2]), others, need):
            del path[1]
            continue
        d = A(path[1]) - A(path[0])
        q = A(path[0]) + d / max(float(np.hypot(*d)), 1e-9) * lead      # or go on straight to `lead` and turn there
        if _seg_ok(A(path[0]), q, others, need) and _seg_ok(q, A(path[2]), others, need) \
                and EDGE_MM <= q[0] <= XMAX and EDGE_MM <= q[1] <= YMAX:
            path[1] = (float(q[0]), float(q[1]))
        break
    return path


def plan(start, goal, others=(), xmax=XMAX, ymax=YMAX, clear=CLEAR, soft_mm=SOFT, seg_mm=SEG_MM, drag=True):
    """Corners of a path from start to goal (mm), first = start, last = goal.
    seg_mm: what one more line costs (as mm), so a slightly longer path with fewer stops wins.
    Raises NoPath when the checkers block every way (move one of them first)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    others = np.asarray(others, float).reshape(-1, 2)
    # the checker being moved, if it was passed among the others, is not an obstacle
    others = others[np.hypot(*(others - start).T) > 3] if len(others) else others
    if len(others) and np.min(np.hypot(*(others - goal).T)) < CHECKER_MM - 1:
        raise NoPath(f"goal ({goal[0]:.0f},{goal[1]:.0f}) is taken")
    need = _needs(start, goal, others, clear)
    soft = _needs(start, goal, others, max(soft_mm, clear))   # same idea, the distance we like
    if _seg_ok(start, goal, others, need) and _short(start, goal, others, soft) == 0:
        return [tuple(start), tuple(goal)]

    nx, ny = int(xmax // GRID) + 1, int(ymax // GRID) + 1
    gx, gy = np.meshgrid(np.arange(nx) * GRID, np.arange(ny) * GRID, indexing="ij")
    pts = np.stack([gx, gy], -1)
    d = np.hypot(pts[..., None, 0] - others[:, 0], pts[..., None, 1] - others[:, 1])
    free = np.all(d >= need, axis=-1)
    free &= (gx >= EDGE_MM) & (gy >= EDGE_MM)     # the limit switches sit at x = 0 and y = 0
    # 0 far from all; 1 per checker at the hard limit; grows fast closer in
    near = (np.clip((soft_mm - d) / (soft_mm - clear), 0, 1) ** 2).sum(-1) if soft_mm > clear \
        else np.zeros(d.shape[:-1])

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
    # cut corners: the cheapest set of straight lines through the A* points, each line paying
    # its length (more next to checkers) + seg_mm for the stop it adds
    out = _fewest_lines(chain, others, need, clear, soft_mm, seg_mm, DRAG_V if drag else 0.0)
    out = [(float(p[0]), float(p[1])) for p in out]
    if drag:
        out = _long_start(out, others, need)
    return round_corners(out, others, need)


FREE_CLEARS = (40.0, 32.0, 25.0, 18.0, 14.0)  # magnet off: the first of these (mm from checker centres) that has a way
FREE_SOFT = 45.0    # and within it, prefer staying this far (Nadav 13.11: the core still pulls checkers it passes near)


def travel(b, frm, to, others=(), xmax=XMAX, ymax=YMAX):
    """Move the cart with the magnet OFF from frm to `to` without passing right under other
    checkers: the steel in a checker is pulled a little by the magnet's iron core even when
    it is off, so a checker the cart passes under can shift. Checkers at frm or `to` (the one
    just let go, the one about to be picked up) do not count. When no way around exists
    (crowded board) it tries a tighter clearance, then goes straight. Returns the corners."""
    o = np.asarray(others, float).reshape(-1, 2)
    if frm is None:
        b.move(*to)
        return [tuple(to)]
    f, t = np.asarray(frm, float), np.asarray(to, float)
    if len(o):
        o = o[(np.hypot(*(o - f).T) > 13) & (np.hypot(*(o - t).T) > 13)]
    path = [tuple(f), tuple(t)]
    for clear in FREE_CLEARS:
        try:
            path = plan(f, t, o, xmax, ymax, clear=clear, soft_mm=max(FREE_SOFT, clear), drag=False)
            break
        except NoPath:
            continue
    for p in path[1:]:
        b.move(*p)
    return path


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


def drag(b, path, overshoot=OVERSHOOT, grab_s=GRAB_S, xmax=XMAX, ymax=YMAX, settle_s=SETTLE_S, corners=True):
    """Drive the magnet along a planned path with the checker on it. At the end: on past the
    target by `overshoot` mm (pulls the lagging checker up), back onto the target, wait
    `settle_s` with the magnet on so the checker centres over it, then let go. At each corner
    the magnet waits corner_wait(turn) so the lagging checker catches up (corners=False: no wait)."""
    from helpers import magnet_on  # waits if the heat budget can't cover this drag
    b.move(*path[0])
    magnet_on(b, need_s=drag_seconds(path, grab_s, settle_s) + 2.0)
    time.sleep(grab_s)
    for i, p in enumerate(path[1:-1], 1):
        b.move(*p)
        if corners:
            time.sleep(corner_wait(_turn_deg(path[i - 1], p, path[i + 1])))
    a, z = np.asarray(path[-2], float), np.asarray(path[-1], float)
    d = z - a
    n = float(np.hypot(*d))
    end = np.clip(z + (d / n * overshoot if n > 1 else 0), 0, [xmax, ymax])
    b.move(float(end[0]), float(end[1]))
    if overshoot > 0:
        b.move(float(z[0]), float(z[1]))
    time.sleep(settle_s)
    b.magnet(False)  # the board waits 500 ms before it answers
