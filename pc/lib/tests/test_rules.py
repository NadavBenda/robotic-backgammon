"""py -m pytest tests   (or: py tests/test_rules.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lib/

import rules  # noqa: E402
from rules import OPENING, legal_moves, move_text  # noqa: E402


def empty():
    return [0] * 26


def test_opening_31():
    plays = legal_moves(OPENING, (3, 1))
    assert len(plays) == 16
    assert "8/5 6/5" in [move_text(s) for s, _ in plays]


def test_larger_die_rule():
    b = empty()
    b[13] = 1
    b[7], b[2], b[1] = -2, -2, -2      # 6 blocked; 5 then 6 blocked
    assert [move_text(s) for s, _ in legal_moves(b, (6, 5))] == ["13/8"]


def test_bar_must_enter_first():
    b = list(OPENING)
    b[0], b[6] = 1, 4
    for s, _ in legal_moves(b, (6, 1)):
        assert s[0].frm == rules.BAR


def test_bear_off_higher_die():
    b = empty()
    b[5], b[2] = 1, 1
    texts = {move_text(s) for s, _ in legal_moves(b, (6, 1))}
    assert "5/off 2/1" in texts


def test_no_move():
    b = empty()
    b[0] = 1
    for p in range(19, 25):
        b[p] = -2
    assert legal_moves(b, (3, 4)) == [([], b)]


def test_find_move_and_flip():
    steps, after = legal_moves(OPENING, (6, 5))[0]
    assert rules.find_move(OPENING, after, (6, 5)) is not None
    assert rules.flip(rules.flip(after)) == list(after)


def test_simple_engine_plays_a_game():
    import random
    random.seed(0)
    b, n = list(OPENING), 0
    while not rules.winner(b) and n < 400:
        d = (random.randint(1, 6), random.randint(1, 6))
        steps = rules.best_move_simple(b, d)
        b = [nb for s, nb in legal_moves(b, d) if s == steps][0]
        if not rules.winner(b):
            b = rules.flip(b)
        n += 1
    assert rules.winner(b) and n < 400


def test_doubled_hit_hits_once():
    """gnubg's '13/10*(2)' (3-3) is two checkers to the point, only the first one hits."""
    import gnubg
    steps = gnubg.parse_move("24/21(2) 13/10*(2)")
    assert [(s.frm, s.to, s.hit) for s in steps] == [
        (24, 21, False), (24, 21, False), (13, 10, True), (13, 10, False)], steps


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
