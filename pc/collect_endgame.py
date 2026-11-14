"""Like collect.py, but the robot fills the home boards: most moves take a checker from
outside into its own home (columns 1-6 of each side), so the photos look like the end of
a game, with many checkers in the homes (Nadav 14.11). Same keys and same steps as
collect.py; the photos go to dataset/<date-time>-endgame/ and train_detector.py uses them
like any other run.

    py collect_endgame.py     (double-click)
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import collect

collect.HOME_P = 0.85      # 85% of the column moves go into the mover's home
collect.TRAY_P = 0.2       # more bear-offs, as at the end of a game
collect.BAR_P = 0.05       # few hits
collect.RUN_TAG = "-endgame"

if __name__ == "__main__":
    import ui
    ui.keep_open()
    collect.main()
