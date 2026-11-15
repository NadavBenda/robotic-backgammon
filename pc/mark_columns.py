"""Mark the columns on the glass: the robot puts one checker on every column, the colours
taking turns (Nadav 15.11), you draw your marks beside them, then it puts them back.

    py mark_columns.py     (double-click)

  1. All 30 checkers in the bear-off trays (place_trays.py shows you where).
  2. Enter: the robot takes 12 of each colour from the trays and puts one on the first place
     (at the board edge) of each of the 24 columns, black and white taking turns.
  3. Draw your marks in the strip outside the board: the sides of each checker are where its
     column starts and ends. The bear-off line goes in the middle of the gap between the
     checkers left in the trays and the first column.
  4. Enter: the robot puts the 24 checkers back in the trays and parks (q = leave them there).
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import ui
from arrange_opening import arrange
from place_opening import ALL_OUT

# one checker on every point, the colours taking turns: + = the robot's colour, - = the other
ALTERNATE = [0] + [1 if p % 2 else -1 for p in range(1, 25)] + [0]


def main():
    from board import Board
    from layout import Layout
    L = Layout.load()
    print(__doc__.split("\n\n")[1])
    if input("All 30 checkers in the trays?\n   Enter = start\n   q = stop\n> ").strip().lower() == "q":
        return
    with Board() as b:
        b.home_if_needed()
        print("Putting one checker on every column (24 drags, a few minutes)...")
        _, n = arrange(b, L, ALL_OUT, ALTERNATE)
        print(f"   {n} drags done")
        ans = input("\nDraw your marks. Then:\n   Enter = back to the trays\n   q = leave them\n> ")
        if ans.strip().lower() != "q":
            print("Back to the trays...")
            _, n = arrange(b, L, ALTERNATE, ALL_OUT)
            print(f"   {n} drags done")
    print("Done.")


if __name__ == "__main__":
    ui.keep_open()
    main()
