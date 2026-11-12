"""Backgammon rules in plain Python: legal moves, a fallback engine, and finding the human's move.

Board format = gnubg.py ("set board simple"), always from the side on roll:
    b[0]      checkers of the side on roll on the bar   (>= 0)
    b[1..24]  points; > 0 own checkers, < 0 opponent    (own side moves 24 -> 1)
    b[25]     opponent checkers on the bar              (<= 0)
Checkers borne off are not on the board: off = 15 - (on board + bar).

    from rules import legal_moves, best_move_simple, find_move, OPENING
    for steps, after in legal_moves(OPENING, (3, 1)): ...
    steps = best_move_simple(OPENING, (3, 1))        # no gnubg needed
    steps = find_move(before, after, (3, 1))          # which legal move turns before into after

Why here and not only gnubg: (1) a fallback when gnubg is missing or slow, (2) the
simulator and tests run anywhere, (3) the human's move can be checked and named.
Legality checks are optional in the game (Nadav, 7.11); they only warn.
"""
from gnubg import BAR, OFF, OPENING, Step, flip  # noqa: F401  (re-exported)

__all__ = ["legal_moves", "apply_step", "best_move_simple", "find_move", "off_count",
           "pips", "winner", "OPENING", "BAR", "OFF", "Step", "flip"]


def off_count(b):
    """(own borne off, opponent borne off)."""
    own = b[0] + sum(n for n in b[1:25] if n > 0)
    opp = -b[25] - sum(n for n in b[1:25] if n < 0)
    return 15 - own, 15 - opp


def pips(b):
    """(own pip count, opponent pip count)."""
    own = 25 * b[0] + sum(i * n for i, n in enumerate(b[1:25], 1) if n > 0)
    opp = 25 * -b[25] + sum((25 - i) * -n for i, n in enumerate(b[1:25], 1) if n < 0)
    return own, opp


def winner(b):
    """'own', 'opp' or None (from the side on roll)."""
    own_off, opp_off = off_count(b)
    return "own" if own_off == 15 else "opp" if opp_off == 15 else None


def _all_home(b):
    return b[0] == 0 and all(n <= 0 for n in b[7:25])


def apply_step(b, frm, die):
    """New board after moving one checker from `frm` (BAR=25 or 1..24) by `die`,
    and the Step, or None if that is not legal."""
    if frm == BAR:
        if b[0] <= 0:
            return None
        to = 25 - die
    else:
        if b[0] > 0 or b[frm] <= 0:
            return None
        to = frm - die
    if to <= 0:  # bearing off
        if not _all_home(b):
            return None
        if to < 0 and any(n > 0 for n in b[frm + 1:7]):
            return None  # a higher die may only bear off from the highest point
        nb = list(b)
        nb[frm] -= 1
        return nb, Step(frm, OFF, False)
    if b[to] <= -2:
        return None
    nb = list(b)
    if frm == BAR:
        nb[0] -= 1
    else:
        nb[frm] -= 1
    hit = nb[to] == -1
    if hit:
        nb[to] = 0
        nb[25] -= 1
    nb[to] += 1
    return nb, Step(frm, to, hit)


def _sequences(b, dice):
    """All (steps, dice used, board) using dice in this order, as far as possible."""
    if not dice:
        return [([], [], b)]
    out = []
    d, rest = dice[0], dice[1:]
    for frm in ([BAR] if b[0] > 0 else range(24, 0, -1)):
        r = apply_step(b, frm, d)
        if r:
            nb, st = r
            out += [([st] + s, [d] + u, bb) for s, u, bb in _sequences(nb, rest)]
    return out or [([], [], b)]


def legal_moves(b, dice):
    """[(steps, board after)] for every legal play, one per distinct result.
    Applies the rules: use as many dice as possible; if only one of two can be used,
    the larger must be used when possible. [([], b)] when nothing can move."""
    d1, d2 = dice
    orders = [[d1] * 4] if d1 == d2 else [[d1, d2], [d2, d1]]
    plays = [p for o in orders for p in _sequences(list(b), o)]
    most = max(len(s) for s, _, _ in plays)
    plays = [p for p in plays if len(p[0]) == most]
    if most == 1 and d1 != d2:
        plays = [p for p in plays if p[1][0] == max(d1, d2)] or plays
    seen, out = set(), []
    for s, _, nb in plays:
        k = tuple(nb)
        if k not in seen:
            seen.add(k)
            out.append((s, nb))
    return out


def _score(b):
    """Higher = better for the side that just moved (b is from its point of view)."""
    own, opp = pips(b)
    s = (opp - own) * 1.0
    for i in range(1, 25):
        n = b[i]
        if n == 1:  # blot: worse the more opponent checkers are behind it (they move 1 -> 24)
            s -= 6 + 0.5 * sum(-m for m in b[1:i] if m < 0) + (4 if i <= 6 else 0)
        elif n >= 2:
            s += 4 + (3 if i <= 7 else 0)
            s -= 3 * max(0, n - 5)   # tall stacks: bad play, and 8+ do not fit on a point
    s += 12 * -b[25]          # opponent checkers on the bar
    s += 8 * off_count(b)[0]
    return s


def best_move_simple(b, dice):
    """A quick, weak but sane move (pip race + safety). For when gnubg is not there."""
    plays = legal_moves(b, dice)
    return max(plays, key=lambda p: _score(p[1]))[0]


def find_move(before, after, dice):
    """The legal steps that turn `before` into `after` (both from the mover's side),
    or None if no legal play gives exactly that board."""
    for s, nb in legal_moves(before, dice):
        if list(nb) == list(after):
            return s
    return None


def move_text(steps):
    """[Step(24,18,False), Step(13,off,...)] -> '24/18 13/off' (gnubg style)."""
    name = lambda p: "bar" if p == BAR else "off" if p == OFF else str(p)
    return " ".join(f"{name(s.frm)}/{name(s.to)}{'*' if s.hit else ''}" for s in steps)
