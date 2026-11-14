"""Turn a move (list of Step) into magnet drags on the real board, and check the result.

    from executor import Executor
    ex = Executor(b, L)                       # b = board.Board or sim.SimBoard
    ex.set_from_board(rules.OPENING)          # where the checkers are: nominal slots ...
    ex.set_from_reading(reading)              # ... or measured by vision (a person's sloppy placement)
    ex.play(steps, side="robot")              # drags; a hit goes to the bar first
    ex.arrange(rules.OPENING)                 # set up a position from whatever is on the board

For each Step:
  1. hit: the single opposing checker on `to` goes to its bar first.
  2. the top checker of `frm` (the one nearest the lane) goes to the next free slot of `to`
     (a bear-off tray when to == OFF).
Path, first that works ("route" setting, the rest are the fallbacks, in this order):
  astar  - paths.plan: A* around every other checker, 30 mm clear (default)
  tight  - the same with 27 mm (rows are only 29 mm apart, so a long row is tight)
  lane   - out along its own row to the middle lane, along the lane, into the target row
           (the layout keeps the lane free, so this almost never fails)
  straight - a straight line, may push a neighbour (logged as a warning)
"""
import time

import numpy as np

import paths
from gnubg import BAR, OFF

DRAG = (paths.DRAG_V, paths.DRAG_A)   # mm/s, mm/s2 with a checker; drag_test 12.11 at Vref 0.86: 250/800 failed,
                                      # 200/800 held; 180 for margin; accel 600 since 14.11 (checkers fell at corners)
ROUTES = ("astar", "tight", "lane", "straight")


class Executor:
    def __init__(self, b, L, route="astar", drag=DRAG, log=print, grab_s=paths.GRAB_S,
                 overshoot=paths.OVERSHOOT):
        self.b, self.L, self.log = b, L, log
        self.route = route
        self.drag = drag
        self.grab_s, self.overshoot = grab_s, overshoot
        self.settle_s = paths.SETTLE_S if grab_s else 0.0   # no waiting in the simulator
        self.places = {}          # key -> [[x, y], ...] edge first; keys: 1..24, bar_*, off_*
        self.side_of = {}         # point -> 'robot' / 'human' (who holds it)
        self.warnings = []
        self.dragged_mm = 0.0
        self.cart = None          # where the magnet is (None = unknown: go straight)
        if drag and hasattr(b, "drag_profile"):
            b.drag_profile(*drag)

    # -- where the checkers are --------------------------------------------
    def set_from_board(self, board):
        self.places, self.side_of = {}, {}
        for (where, i), (xy, side) in sorted(self.L.place(board).items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            self.places.setdefault(where, []).append(list(xy))
            if isinstance(where, int):
                self.side_of[where] = side

    def set_from_reading(self, r):
        """Use measured positions (vision.Reading.checkers)."""
        self.places, self.side_of = {}, {}
        for x, y, side in r.checkers:
            k = self.L.nearest(x, y)
            if k is None:
                self.warnings.append(f"checker at ({x:.0f},{y:.0f}) is not on any point, ignored")
                continue
            kind, key = k
            where = key if kind == "point" else f"{kind}_{side}" if kind != "point" else key
            if kind == "point":
                self.side_of[key] = side
            self.places.setdefault(where, []).append([x, y])
        for where, lst in self.places.items():   # edge first
            e = self.L.slot(where, 0)
            lst.sort(key=lambda p: np.hypot(p[0] - e[0], p[1] - e[1]))

    def all_xy(self):
        return [tuple(p) for lst in self.places.values() for p in lst]

    # -- moves --------------------------------------------------------------
    def _key(self, p, side):
        """Step point (that side's numbering) -> place key."""
        if p == BAR:
            return f"bar_{side}"
        if p == OFF:
            return f"off_{side}"
        return p if side == "robot" else 25 - p

    def _free_slot(self, where):
        lst = self.places.get(where, [])
        others = self.all_xy()
        for i in range(len(lst), len(lst) + 8):
            xy = self.L.slot(where, i)
            if all(np.hypot(xy[0] - o[0], xy[1] - o[1]) >= self.L.checker - 2 for o in others):
                return xy
        return self.L.slot(where, len(lst))

    def _next_to(self, where, frm_key, bar_key):
        """Where the hitter waits next to the blot: a free spot 28-45 mm from it from which
        the blot still has a clean way to the bar and the hitter a clean way in.
        None if there is no such spot (then the old order: blot out first)."""
        lst, src = self.places.get(where) or [], self.places.get(frm_key) or []
        if not lst or not src:
            return None
        mid = self.L.xmax / 2
        start = tuple(src[min(range(len(src)), key=lambda i: abs(self.L.from_magnet(*src[i])[0] - mid))])
        blot = np.asarray(lst[0], float)
        bar = self._free_slot(bar_key)
        rest = [p for p in self.all_xy() if np.hypot(p[0] - start[0], p[1] - start[1]) > 1]
        others_blot = [p for p in rest if np.hypot(p[0] - blot[0], p[1] - blot[1]) > 1]
        best = None
        for r in (28.0, 36.0, 45.0):
            for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                q = (float(blot[0] + r * np.cos(a)), float(blot[1] + r * np.sin(a)))
                if not (8 <= q[0] <= self.L.xmax - 8 and 8 <= q[1] <= self.L.ymax - 8):
                    continue
                if any(np.hypot(q[0] - o[0], q[1] - o[1]) < self.L.checker + 1 for o in others_blot):
                    continue
                try:
                    p1 = paths.plan(start, q, rest, self.L.xmax, self.L.ymax)
                    p2 = paths.plan(tuple(blot), bar, others_blot + [q], self.L.xmax, self.L.ymax)
                except paths.NoPath:
                    continue
                cost = paths.length(p1) + paths.length(p2) + 2 * r
                if best is None or cost < best[0]:
                    best = (cost, q)
            if best is not None:
                return best[1]
        return None

    def move_checker(self, frm_key, to_key, side=None, goal=None, pick=None):
        lst = self.places.get(frm_key)
        if not lst:
            raise RuntimeError(f"no checker on {frm_key}")
        # the top checker: the one nearest the lane (a person may have shuffled the order)
        mid = self.L.xmax / 2
        top = min(range(len(lst)), key=(lambda i: pick(lst[i])) if pick else
                  (lambda i: abs(self.L.from_magnet(*lst[i])[0] - mid)))
        lst.append(lst.pop(top))
        start = tuple(lst[-1])
        if goal is None:
            goal = self._free_slot(to_key)
        others = [p for p in self.all_xy() if np.hypot(p[0] - start[0], p[1] - start[1]) > 1]
        path, how = self.plan(start, goal, others)
        self.log(f"   {frm_key} -> {to_key}: {len(path) - 1} line(s), {paths.length(path):.0f} mm ({how})")
        self.goto(start, others)
        paths.drag(self.b, path, overshoot=self._overshoot(path, others), grab_s=self.grab_s,
                   xmax=self.L.xmax, ymax=self.L.ymax, settle_s=self.settle_s,
                   corners=bool(self.grab_s))
        self.dragged_mm += paths.length(path)
        self.cart = tuple(goal)
        lst.pop()
        self.places.setdefault(to_key, []).append(list(goal))
        if isinstance(to_key, int) and side:
            self.side_of[to_key] = side
        if isinstance(frm_key, int) and not lst:
            self.side_of.pop(frm_key, None)
        if not lst and str(frm_key).startswith("_"):
            self.places.pop(frm_key, None)

    def goto(self, xy, others=None):
        """Magnet off: drive to xy around the checkers (paths.travel)."""
        if others is None:
            others = self.all_xy()
        paths.travel(self.b, self.cart, xy, others, self.L.xmax, self.L.ymax)
        self.cart = tuple(xy)

    def park(self, xy=None):
        """To the photo spot, around the checkers."""
        self.goto(xy or self.L.park())

    def _overshoot(self, path, others):
        """No overshoot when it would push the checker into a neighbour ahead of it."""
        a, z = np.asarray(path[-2], float), np.asarray(path[-1], float)
        n = np.hypot(*(z - a))
        if n < 1 or not len(others):
            return self.overshoot
        end = z + (z - a) / n * self.overshoot
        o = np.asarray(others, float).reshape(-1, 2)
        if np.min(np.hypot(*(o - end).T)) < self.L.checker + 0.5:
            return 0.0
        return self.overshoot

    def plan(self, start, goal, others):
        order = ROUTES[ROUTES.index(self.route):] if self.route in ROUTES else ROUTES
        for how in order:
            try:
                if how == "astar":
                    return paths.plan(start, goal, others, self.L.xmax, self.L.ymax), how
                if how == "tight":
                    return paths.plan(start, goal, others, self.L.xmax, self.L.ymax, clear=27.0), how
                if how == "lane":
                    o = np.asarray(others, float).reshape(-1, 2)
                    need = paths._needs(np.asarray(start, float), np.asarray(goal, float), o,
                                        self.L.checker - 1.5)
                    return paths.round_corners(self.lane_path(start, goal, others), o, need), how
            except paths.NoPath as e:
                self.last_nopath = str(e)
                continue
        self.warnings.append(f"straight drag ({start[0]:.0f},{start[1]:.0f}) -> ({goal[0]:.0f},{goal[1]:.0f}) "
                             f"may push a neighbour ({getattr(self, 'last_nopath', '')})")
        return [start, goal], "straight"

    def lane_path(self, start, goal, others):
        """Row -> middle lane -> row (picture frame), each leg checked for clearance.
        The lane x is chosen where it has the most room (a point with 8 checkers
        reaches into the lane)."""
        L = self.L
        (sx, sy), (gx, gy) = L.from_magnet(*start), L.from_magnet(*goal)
        o = np.asarray(others, float).reshape(-1, 2)
        # a touch (1.5 mm) is accepted here: rows are only 29 mm apart
        need = paths._needs(np.asarray(start, float), np.asarray(goal, float), o, L.checker - 1.5)
        lo, hi = L.lane_x()
        mid = (lo + hi) / 2
        best, why = None, "lane blocked"
        for lane in sorted(np.arange(lo - 15, hi + 15.1, 3.0), key=lambda x: abs(x - mid)):
            pts = [(sx, sy), (lane, sy), (lane, gy), (gx, gy)]
            mag = [L.to_magnet(*p) for p in pts]
            out = [mag[0]]
            for p in mag[1:]:
                if np.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 0.5:
                    out.append(p)
            if len(out) < 2:
                out.append(mag[-1])
            if len(o) == 0:
                return out
            margin = min(float(np.min(paths._seg_dist(o, a, z) - need)) for a, z in zip(out, out[1:]))
            if margin >= 0:
                return out
            if best is None or margin > best:
                best = margin
                d = [paths._seg_dist(o, a, z) - need for a, z in zip(out, out[1:])]
                leg = int(np.argmin([x.min() for x in d]))
                k = int(np.argmin(d[leg]))
                a, z = out[leg], out[leg + 1]
                why = (f"lane blocked by ({o[k][0]:.0f},{o[k][1]:.0f}) on "
                       f"({a[0]:.0f},{a[1]:.0f})->({z[0]:.0f},{z[1]:.0f})")
        raise paths.NoPath(why)

    def play(self, steps, side="robot"):
        """Carry out one move. side: whose move it is (its numbering in the steps)."""
        opp = "human" if side == "robot" else "robot"
        t0 = time.time()
        for st in steps:
            to_key = self._key(st.to, side)
            frm_key = self._key(st.frm, side)
            if st.hit:
                near = self._next_to(to_key, frm_key, f"bar_{opp}")
                if near is not None:
                    # Nadav 12.11: the hitter first stops next to the blot, the blot goes to
                    # the middle of the bar, then the hitter takes its place
                    self.move_checker(frm_key, "_hitter", side, goal=near)
                    self.move_checker(to_key, f"bar_{opp}", opp)
                    self.move_checker("_hitter", to_key, side)
                    continue
                self.move_checker(to_key, f"bar_{opp}", opp)
            self.move_checker(frm_key, to_key, side)
        return time.time() - t0

    # -- set up a position -------------------------------------------------
    def arrange(self, board, park=None, max_moves=80):
        """Move the checkers that are not where `board` wants them. Uses the current
        positions (set_from_reading first). Greedy: each wrong checker goes to the
        nearest free target of its colour; a target taken by a wrong checker is freed
        first. Returns the number of drags."""
        targets = {"robot": [], "human": []}
        slot_of = {}                                  # target xy -> (place, index in it)
        for (where, i), (xy, side) in self.L.place(board).items():
            targets[side].append(tuple(xy))
            slot_of[tuple(xy)] = (where, i)
        current = []   # (x, y, side, place) from places
        for where, lst in self.places.items():
            side = (self.side_of.get(where) if isinstance(where, int) else where.split("_")[1])
            current += [(p[0], p[1], side, where) for p in lst]
        if sum(1 for c in current if c[2] == "robot") < 15 or sum(1 for c in current if c[2] == "human") < 15:
            self.warnings.append("fewer than 15 checkers of a side seen; arranging what is there")
        moves = 0
        cur = [list(c) for c in current]
        ok = lambda c, t: np.hypot(c[0] - t[0], c[1] - t[1]) < 6
        for _ in range(max_moves):
            # which targets are already done
            done, todo = set(), []
            for c in cur:
                hit = next((t for t in targets[c[2]] if t not in done and ok(c, t)), None)
                if hit:
                    done.add(hit)
                else:
                    todo.append(c)
            if not todo:
                break
            free = {s: [t for t in targets[s] if t not in done] for s in targets}

            def top_of_column(c):
                """No checker stands farther from the board edge in the same place: the board
                keeps looking valid (no gaps) when it is lifted (Nadav 14.11)."""
                e = self.L.slot(c[3], 0)
                d = np.hypot(c[0] - e[0], c[1] - e[1])
                return all(np.hypot(o[0] - e[0], o[1] - e[1]) <= d + 1 for o in cur if o is not c and o[3] == c[3])

            def fills_from_the_edge(t):
                """Every lower place of the same column is already done (or nobody needs it)."""
                where, idx = slot_of[t]
                return all((tuple(self.L.slot(where, k)) in done) or (tuple(self.L.slot(where, k)) not in slot_of)
                           for k in range(idx))
            best = None
            for rule in (3, 2, 1, 0):   # both rules; then only the lifting rule; only the filling rule; neither
                for c in todo:
                    if rule & 2 and not top_of_column(c):
                        continue
                    for t in free[c[2]]:
                        if rule & 1 and not fills_from_the_edge(t):
                            continue
                        if any(np.hypot(o[0] - t[0], o[1] - t[1]) < self.L.checker - 2 for o in cur if o is not c):
                            continue
                        d = np.hypot(c[0] - t[0], c[1] - t[1])
                        if best is None or d < best[0]:
                            best = (d, c, t)
                if best is not None:
                    break
            if best is None:   # every free target is covered by a wrong checker: move one to the lane
                c = todo[0]
                lane = sum(self.L.lane_x()) / 2
                t = self.L.to_magnet(lane, self.L._row_y(3 + moves % 7))
                best = (0, c, t)
            _, c, t = best
            others = [tuple(o[:2]) for o in cur if o is not c]
            path, how = self.plan(tuple(c[:2]), tuple(t), others)
            self.goto(tuple(c[:2]), others)
            paths.drag(self.b, path, overshoot=self._overshoot(path, others), grab_s=self.grab_s,
                       xmax=self.L.xmax, ymax=self.L.ymax, settle_s=self.settle_s,
                   corners=bool(self.grab_s))
            c[0], c[1] = t
            c[3] = slot_of[tuple(t)][0] if tuple(t) in slot_of else c[3]
            self.cart = tuple(t)
            moves += 1
        if park:
            self.set_from_board(board)
            self.park(park)
        self.set_from_board(board)
        return moves


def expected_after(board, steps):
    """Board after the robot plays steps (robot's side), from rules."""
    from rules import apply_step
    b = list(board)
    for st in steps:
        frm = BAR if st.frm == BAR else st.frm
        die = (25 if frm == BAR else frm) - (st.to if st.to != OFF else 0)
        r = apply_step(b, frm, die)
        if r is None and st.to == OFF:   # bear off with a higher die
            for d in range(die + 1, 7):
                r = apply_step(b, frm, d)
                if r:
                    break
        if r is None:
            raise ValueError(f"step {st} is not legal on this board")
        b = r[0]
    return b
