"""GNU Backgammon as a move engine.

    from gnubg import best_move, OPENING
    steps = best_move(OPENING, (3, 1))
    # [Step(frm=8, to=5, hit=False), Step(frm=6, to=5, hit=False)]

Board format (same as gnubg "set board simple"), always from the side that is on roll
(the robot, when we ask for its move):
    board[0]      robot checkers on the bar          (>= 0)
    board[1..24]  points 1..24 of the robot (1 = deepest in its home, it moves 24 -> 1)
                  > 0 robot checkers, < 0 opponent checkers
    board[25]     opponent checkers on the bar        (<= 0)

Step numbering: frm/to are the robot's point numbers, BAR = 25, OFF = 0.
hit=True means an opponent checker stands on `to`: move it to the bar first.
"""
import re
import subprocess
from dataclasses import dataclass

GNUBG = r"C:\Users\nadavben\AppData\Local\gnubg\gnubg-cli.exe"
PLIES = 2  # 0 = fast and weak, 2 = strong; each hint still takes about a second

BAR, OFF = 25, 0

OPENING = [0,
           -2, 0, 0, 0, 0, 5,  0, 3, 0, 0, 0, -5,
           5, 0, 0, 0, -3, 0,  -5, 0, 0, 0, 0, 2,
           0]


@dataclass(frozen=True)
class Step:
    frm: int
    to: int
    hit: bool


class GnubgError(RuntimeError):
    pass


_HINT = re.compile(r"^\s*(\d+)\.\s+Cubeful\s+(\d+)-ply\s+(.+?)\s+Eq\.:\s*([+-]?\d+\.\d+)", re.M)


def check_board(board):
    if len(board) != 26:
        raise GnubgError(f"board must have 26 numbers, got {len(board)}")
    if board[0] < 0 or board[25] > 0:
        raise GnubgError("board[0] (robot bar) must be >= 0 and board[25] (opponent bar) <= 0")
    mine = board[0] + sum(n for n in board[1:25] if n > 0)
    theirs = -board[25] - sum(n for n in board[1:25] if n < 0)
    # gnubg silently ignores a bad board, so catch it here
    if mine > 15 or theirs > 15:
        raise GnubgError(f"too many checkers: robot {mine}, opponent {theirs} (max 15)")


def _point(s):
    s = s.lower()
    if s == "bar":
        return BAR
    if s == "off":
        return OFF
    return int(s)


def parse_move(text):
    """'24/18*/13 6/off(2)' -> list of Step, in the order gnubg wrote them."""
    steps = []
    for token in text.split():
        m = re.fullmatch(r"(.+?)(?:\((\d)\))?", token)
        chain, count = m.group(1), int(m.group(2) or 1)
        parts = chain.split("/")
        single = []
        for a, b in zip(parts, parts[1:]):
            single.append(Step(_point(a.rstrip("*")), _point(b.rstrip("*")), b.endswith("*")))
        steps += single * count
    return steps


def parse_hints(output):
    """All candidate moves from gnubg's hint text: [(rank, plies, move_text, equity)]."""
    return [(int(r), int(p), mv.strip(), float(eq)) for r, p, mv, eq in _HINT.findall(output)]


def run(commands, timeout=30):
    """Run gnubg-cli with these commands on stdin, return its whole text output."""
    script = "\n".join(commands + ["quit", "y", ""])
    try:
        res = subprocess.run([GNUBG, "-q", "-t"], input=script, capture_output=True,
                             text=True, timeout=timeout)
    except FileNotFoundError:
        raise GnubgError(f"gnubg not found at {GNUBG}")
    return res.stdout


def hints(board, dice, plies=PLIES):
    check_board(board)
    d1, d2 = dice
    out = run([
        "set player 0 human",
        "set player 1 human",
        f"set evaluation chequerplay evaluation plies {plies}",
        "new game",
        "set board simple " + " ".join(str(n) for n in board),
        f"set dice {d1} {d2}",
        "hint",
    ])
    return parse_hints(out), out


def best_move(board, dice, plies=PLIES):
    """The best move as a list of Step. [] when there is no legal move (e.g. stuck on the bar)."""
    found, out = hints(board, dice, plies)
    if not found:
        if re.search(r"no legal moves|cannot move|no moves", out, re.I):
            return []
        raise GnubgError("gnubg gave no hint. Output:\n" + out[-1500:])
    return parse_move(found[0][2])


def flip(board):
    """The same board seen from the other side (to ask gnubg about the human's move)."""
    return [-board[25]] + [-n for n in reversed(board[1:25])] + [-board[0]]


if __name__ == "__main__":
    print(best_move(OPENING, (3, 1)))
