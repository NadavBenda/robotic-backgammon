"""Put all 30 checkers in the bear-off trays, the robot shows you where (like place_opening.py).

    py place_trays.py     (double-click)

The magnet cart drives under each place of the two bear-off trays, one after the other.
Put a checker of the right colour right above it and press Enter; the magnet turns on
for a moment so the checker slides to the exact centre.
First the robot's 15 (its tray), then your 15 (your tray).
  Enter = placed    s = skip this one    b = back one    q = stop
Then arrange_opening.py: the robot sets up the opening from the trays by itself.
"""
import sys  # noqa: E402
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "lib"))  # the internal code lives in lib/

import place_opening
import ui

if __name__ == "__main__":
    ui.keep_open()
    place_opening.main(place_opening.ALL_OUT, __doc__,
                       "Done. Now: py arrange_opening.py, the robot sets up the opening from the trays.")
