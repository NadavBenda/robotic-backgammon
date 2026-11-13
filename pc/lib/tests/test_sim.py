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


def test_collect_random_moves():
    """collect.py: random moves keep their labels, a knocked checker is noticed."""
    import json
    import tempfile
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import collect
    L = Layout.load()
    b = SimBoard.with_board(rules.OPENING, L)
    d = tempfile.mkdtemp()
    collect.STATE = Path(d) / "state.json"
    col = collect.Collector(b, b.calib, L, d, auto=True, seed=3, log=lambda *a: None)
    costs = []
    real = collect.vision.change_cost
    collect.vision.change_cost = lambda *a: costs.append(real(*a)) or costs[-1]
    try:
        assert col.start()
        for _ in range(30):
            col.step()
        assert b.bumps == 0 and max(costs) < collect.STOP_COST, (b.bumps, costs)
        last = json.loads(Path(d, "labels.jsonl").read_text().splitlines()[-1])
        truth = b.checkers()
        assert all(min(np.hypot(x - t[0], y - t[1]) for t in truth if t[2] == s) < 4
                   for x, y, s, _ in last["checkers"])
        b.ck[5][1] += 29          # a checker knocked out of its place
        col.step()
        assert costs[-1] >= collect.STOP_COST, costs[-1]
        col.back_to_opening()
        assert col.board() == rules.OPENING
    finally:
        collect.vision.change_cost = real


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
