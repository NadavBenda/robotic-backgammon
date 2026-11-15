"""Remote mode: a whole game in the simulator, the far player's side played over its phone
page's HTTP endpoints (no tunnel). Run: py tests/test_remote.py"""
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lib/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pc/ (game.py)

import game as G  # noqa: E402
import remote  # noqa: E402

PORT = 8797


def test_remote_game_in_simulator():
    G.RECORD = False
    g = G.Game({"mode": "remote", "engine": "simple", "board": "sim", "dice": "digital"}, log=lambda *a: None)
    r = remote.Remote(g, port=PORT, log=lambda *a: None)
    g.remote = r
    G.GAMES = Path(__file__).resolve().parent / "_games_tmp"
    threading.Thread(target=g.run_forever, daemon=True).start()
    base = f"http://127.0.0.1:{PORT}"
    get = lambda p: json.loads(urllib.request.urlopen(f"{base}{p}?k={r.key}").read())  # noqa: E731
    post = lambda a: urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        f"{base}/a?k={r.key}", data=json.dumps(a).encode(), method="POST"))
    try:
        urllib.request.urlopen(f"{base}/s?k=wrong")
        raise AssertionError("a wrong key must get nothing")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    post({"name": "name", "value": "Noa<b>"})
    assert g.remote_name == "Noab"

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
    random.seed(1)
    for _ in range(400):
        until(lambda: g.winner or (g.phase == "human_move" and g.asks) or g.phase == "check"
              or get("/s")["what"] in ("roll", "move"))
        if g.winner:
            break
        s = get("/s")
        if s["what"] == "roll":
            post({"name": "roll"})
        elif s["what"] == "move":
            post({"name": "move", "value": s["hint"] or s["plays"][0]["t"]})
        elif "done" in g.asks:
            g.post({"name": "done"})
        else:
            g.post({"name": "move", "value": random.choice(g.legal)})
        time.sleep(0.05)
    assert g.winner, "the game did not end"
    assert any("from the phone" in h["note"] for h in g.history)
    r.srv.shutdown()
    import shutil
    shutil.rmtree(G.GAMES, ignore_errors=True)


if __name__ == "__main__":
    test_remote_game_in_simulator()
    print("ok")
