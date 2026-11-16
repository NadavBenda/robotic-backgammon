"""Your move = auto: the camera sees when the person has finished, nobody presses I'm done
(simulator: the page's picker moves the checkers "by hand"). Run: py tests/test_auto_done.py"""
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lib/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pc/ (game.py)

import game as G  # noqa: E402


def test_auto_done_in_simulator():
    G.RECORD = False
    G.AUTO_POLL_S = 0.2
    G.GAMES = Path(__file__).resolve().parent / "_games_tmp_auto"
    g = G.Game({"mode": "robot_vs_human", "engine": "simple", "board": "sim", "dice": "digital",
                "human_move": "auto"}, log=lambda *a: None)
    threading.Thread(target=g.run_forever, daemon=True).start()

    def until(cond, t=60):
        end = time.time() + t
        while time.time() < end:
            if cond():
                return
            time.sleep(0.02)
        raise AssertionError(f"stuck: {g.phase} {g.asks} {g.prompt}")

    g.post({"name": "new"})
    until(lambda: "start" in g.asks)
    g.post({"name": "start"})
    random.seed(2)
    until(lambda: g.phase == "human_move" and "move_pick" in g.asks and g.legal)
    time.sleep(1.5)            # nothing moved yet: the turn must not end by itself
    assert g.phase == "human_move" and not g.history, "an untouched board was taken as a move"
    turns = 0
    while turns < 6 and not g.winner:
        until(lambda: g.winner or (g.phase == "human_move" and "move_pick" in g.asks and g.legal))
        if g.winner:
            break
        n = len(g.history)
        g.post({"name": "move", "value": random.choice(g.legal)})      # moved "by hand", no I'm done
        until(lambda: len(g.history) > n or g.phase == "check", t=30)
        assert g.phase != "check", "auto stopped to ask"
        assert "read by change" in g.history[n]["note"], g.history[n]
        turns += 1
    assert turns >= 6
    import shutil
    shutil.rmtree(G.GAMES, ignore_errors=True)


if __name__ == "__main__":
    test_auto_done_in_simulator()
    print("ok")
