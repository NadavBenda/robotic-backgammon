"""Grab photos from the board over USB.

    python capture.py            # Enter = take a photo, q + Enter = quit
    python capture.py COM5       # pick the port yourself

Photos are saved in captures/ with a timestamp and shown in a window.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from board import Board

out = Path(__file__).parent / "captures"
out.mkdir(exist_ok=True)

with Board(sys.argv[1] if len(sys.argv) > 1 else None) as b:
    print("Connected:", b.status())
    while input("Enter = photo, q = quit: ").strip().lower() != "q":
        t0 = time.time()
        jpeg = b.capture()
        path = out / time.strftime("%Y%m%d_%H%M%S.jpg")
        path.write_bytes(jpeg)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        print(f"{path.name}: {img.shape[1]}x{img.shape[0]}, {len(jpeg) // 1024} KB, {time.time() - t0:.2f} s")
        cv2.imshow("board", img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
