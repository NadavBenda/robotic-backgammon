"""Whole games in the simulator: rules + executor (paths) + vision (no hardware)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lib/

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import dice  # noqa: E402
import paths  # noqa: E402
import rules  # noqa: E402
import vision  # noqa: E402
from executor import Executor  # noqa: E402
from layout import Layout  # noqa: E402
from sim import SimBoard  # noqa: E402

paths.GRAB_S = 0


def photo(b, L, ex=None):
    if ex is not None:
        ex.park()          # like the game: magnet off, around the checkers
    else:
        b.move(*L.park())
    return cv2.imdecode(np.frombuffer(b.capture(), np.uint8), 1)


def test_dice_from_photo():
    L = Layout()
    for d in [(1, 6), (5, 2), (3, 3), (6, 4)]:
        b = SimBoard.with_board(rules.OPENING, L, dice=d)
        assert dice.from_photo(photo(b, L), b.calib, L) == d


def test_dice_read_by_change():
    """Real dice anywhere on the glass, white or dark, found by what changed between two photos."""
    L = Layout()
    ck = [xy for xy, _ in L.place(rules.OPENING).values()]
    for look in ({}, {"dark": True}, {"zone": (160, 150, 250, 190)}, {"size": 22.0, "zone": (20, -10, 120, 40), "dark": True}):
        for d in [(1, 6), (5, 2), (3, 3), (6, 4), (6, 6), (2, 1)]:
            b = SimBoard.with_board(rules.OPENING, L, noise_mm=0.5, seed=3)
            before = photo(b, L)
            b.dice, b.dice_look = d, look
            assert dice.read(before, photo(b, L), b.calib, L, checkers=ck) == d, (look, d)
    assert dice.read(before, before, b.calib, L, checkers=ck) is None      # no dice: no reading


def test_vision_reads_sloppy_human_move():
    L = Layout()
    b = SimBoard.with_board(rules.OPENING, L, noise_mm=1.0, seed=5)
    m = vision.learn(photo(b, L), b.calib, L, rules.OPENING)
    b.human_move(L.slot(1, 1), L.slot(7, 0), sloppy=4)
    r = vision.read(photo(b, L), b.calib, L, m)
    exp = list(rules.OPENING)
    exp[1], exp[7] = -1, -1
    assert r.board == exp and not r.warnings



def test_read_by_change():
    """The human's move found from what changed between two photos, among the legal moves."""
    L = Layout()
    b = SimBoard.with_board(rules.OPENING, L, noise_mm=0.5, seed=3)
    before = photo(b, L)
    b.human_move(L.slot(1, 1), L.slot(7, 0), sloppy=4)   # human 6-1: robot view 1 -> 7 and 17 -> 18
    b.human_move(L.slot(17, 2), L.slot(18, 0), sloppy=4)
    after = photo(b, L)
    hv = rules.flip(rules.OPENING)
    cands = [rules.flip(nb) for _, nb in rules.legal_moves(hv, (6, 1))]
    exp = list(rules.OPENING)
    exp[1], exp[7], exp[17], exp[18] = -1, -1, -2, -1
    best, cost, margin = vision.closest_by_change(before, after, b.calib, L, rules.OPENING, cands)
    assert best == exp and cost == 0 and margin >= 1, (best, cost, margin)

def test_bear_off_right_off_the_glass():
    """A person bears off and takes the checkers away altogether (not into the tray): the
    reading by change does not judge the trays, so the move still fits (Nadav 15.11)."""
    L = Layout()
    hv = [0] * 26                      # the person's view: 6 checkers left at home, the robot's all at its home
    hv[6], hv[5], hv[3] = 2, 2, 2
    hv[19], hv[20] = -8, -7
    board = rules.flip(hv)
    b = SimBoard.with_board(board, L, noise_mm=0.5, seed=4)
    before = photo(b, L)
    for p in (6, 5):                   # 6-5: 6/off 5/off, and both checkers leave the glass
        xy = L.slot(25 - p, 1)
        i = min(range(len(b.ck)), key=lambda i: np.hypot(b.ck[i][0] - xy[0], b.ck[i][1] - xy[1]))
        del b.ck[i]
    after = photo(b, L)
    cands = [rules.flip(nb) for _, nb in rules.legal_moves(hv, (6, 5))]
    trays = ("off_robot", "off_human")
    cost = vision.rank_by_change(before, after, b.calib, L, board, cands, trays)[0][0]
    strict = vision.rank_by_change(before, after, b.calib, L, board, cands)[0][0]
    assert cost < 1.0 <= strict, (cost, strict)


def play_game(seed):
    L = Layout()
    b = SimBoard.with_board(rules.OPENING, L, seed=seed, noise_mm=0.5)
    m = vision.learn(photo(b, L), b.calib, L, rules.OPENING)
    ex = Executor(b, L, grab_s=0, log=lambda *a: None)
    ex.set_from_board(rules.OPENING)
    board, side, bad = list(rules.OPENING), "robot", 0
    random.seed(seed)
    for _ in range(300):
        d = (random.randint(1, 6), random.randint(1, 6))
        steps = rules.best_move_simple(board, d)
        ex.play(steps, side)
        nb = [nb for s, nb in rules.legal_moves(board, d) if s == steps][0]
        r = vision.read(photo(b, L, ex), b.calib, L, m)
        exp = nb if side == "robot" else rules.flip(nb)
        if r.board != exp:
            bad += 1
            ex.set_from_board(exp)
        else:
            ex.set_from_reading(r)
        if rules.winner(nb):
            break
        board, side = rules.flip(nb), ("human" if side == "robot" else "robot")
    return bad, b.bumps, ex.warnings


def test_full_game_in_simulator():
    bad, bumps, warnings = play_game(0)
    assert bad == 0 and bumps == 0, (bad, bumps, warnings)


def test_arrange_opening_from_the_trays():
    """arrange_opening.py: all 30 checkers from the bear-off trays to the opening."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import arrange_opening
    from place_opening import ALL_OUT
    L = Layout.load()
    b = SimBoard.with_board(ALL_OUT, L, seed=1)

    def placed(board):
        truth = b.checkers()
        return all(any(np.hypot(c[0] - xy[0], c[1] - xy[1]) < 6 and c[2] == s for c in truth)
                   for xy, s in L.place(board).values())
    assert placed(ALL_OUT)
    ex, n = arrange_opening.arrange(b, L, ALL_OUT, rules.OPENING, log=lambda *a: None)
    assert n == 30 and b.bumps == 0 and placed(rules.OPENING), (n, b.bumps, ex.warnings)


def test_arrange_takes_the_outer_tray_row_first():
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import arrange_opening
    L = Layout.load()
    for w in ("off_robot", "off_human"):
        pts = [L.slot(w, i) for i in range(14)]
        order = sorted(range(14), key=lambda i: arrange_opening.outer_first(L, pts[i]))
        assert sorted(order[:7]) == list(range(7)), (w, order)   # row 0 (outer) is emptied first


def test_assign_finds_the_smallest_total():
    import itertools
    from executor import assign
    rng = random.Random(1)
    for _ in range(150):
        n, m = rng.randint(1, 5), rng.randint(1, 5)
        c = [[rng.random() for _ in range(m)] for _ in range(n)]
        got = assign(c)
        k = min(n, m)
        best = min(sum(c[i][j] for i, j in zip(rows, cols)) for rows in itertools.combinations(range(n), k)
                   for cols in itertools.permutations(range(m), k))
        assert len({j for j in got if j is not None}) == k
        assert abs(sum(c[i][j] for i, j in enumerate(got) if j is not None) - best) < 1e-9


def test_arrange_from_the_middle_of_a_game():
    """executor.arrange: from a position in the middle of a game back to the opening, nothing pushed."""
    L = Layout()
    frm = [0, 0, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, -1, 0, 0]
    b = SimBoard.with_board(frm, L, seed=1)
    ex = Executor(b, L, grab_s=0, log=lambda *a: None)
    ex.set_from_board(frm)
    n = ex.arrange(rules.OPENING)
    truth = b.checkers()
    assert n <= 31 and b.bumps == 0 and not ex.warnings, (n, b.bumps, ex.warnings)
    assert all(any(np.hypot(c[0] - xy[0], c[1] - xy[1]) < 6 and c[2] == s for c in truth)
               for xy, s in L.place(rules.OPENING).values())


def test_second_checker_does_not_hit_its_own_first():
    """3-3: 13/10*(2) onto a human blot - the second checker landing there is a plain move."""
    import gnubg
    L = Layout()
    board = list(rules.OPENING)
    board[10] = -1                      # a human blot on the robot's point 10
    board[5] -= 0
    b = SimBoard.with_board(board, L, seed=2, noise_mm=0.3)
    ex = Executor(b, L, grab_s=0, log=lambda *a: None)
    ex.set_from_board(board)
    ex.play(gnubg.parse_move("13/10*(2)"), "robot")
    assert len(ex.places[10]) == 2 and ex.side_of[10] == "robot", ex.places[10]
    assert len(ex.places.get("bar_human", [])) == 1 and not ex.places.get("bar_robot"), ex.places
    assert b.bumps == 0, b.bumps


def test_anchor_keeps_what_did_not_change():
    """The detector misses the person's checkers on point 2 (robot's 23) every time; after the
    board was fixed once, places that did not change in the photo keep the known count."""
    L = Layout()
    b = SimBoard.with_board(rules.OPENING, L, noise_mm=0.5, seed=4)
    before = photo(b, L)
    b.human_move(L.slot(1, 1), L.slot(7, 0), sloppy=3)       # person: robot view 1 -> 7
    after = photo(b, L)
    exp = list(rules.OPENING)
    exp[1], exp[7] = -1, -1
    seen = list(exp)
    seen[12] = 0                                              # the detector misses point 12 (5 checkers)
    r = vision.Reading([], {}, seen, (0, 0), ["human: 10 on the board + 0 off = 10, not 15"])
    a = vision.anchor(r, rules.OPENING, vision.slot_changes(before, after, b.calib, L)[0])
    assert a.board == exp and not a.warnings and a.anchored == [12], (a.board, a.warnings)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles (cp1255) and Hebrew paths
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
