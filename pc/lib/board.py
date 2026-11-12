"""PC side of the AI-bg board protocol (see firmware/board/board.ino).

    from board import Board
    with Board("COM5") as b:
        b.home()
        b.move(100, 200)       # mm
        b.magnet(True)
        jpeg = b.capture()     # bytes

Every call sends one command and waits for DONE (returns) or ERR (raises BoardError),
so two commands can never overlap.
"""
import time

import serial  # pip install pyserial
from serial.tools import list_ports


class BoardError(RuntimeError):
    pass


def find_port():
    """First port that looks like an ESP32-S3 (native USB, CH343 or CP210x)."""
    for p in list_ports.comports():
        if p.vid in (0x303A, 0x1A86, 0x10C4):
            return p.device
    raise BoardError("ESP32 not found. Is the USB cable connected?")


class Board:
    def __init__(self, port=None, baud=921600, on_event=print, on_log=None):
        self.on_event = on_event  # called with "! ..." lines (e.g. MAG_TIMEOUT)
        self.on_log = on_log      # called with every other line ("# ...", DONE, ERR, ...) if set
        self.ser = serial.Serial()
        self.ser.port = port or find_port()
        self.ser.baudrate = baud
        self.ser.timeout = 0.2
        # Keep DTR/RTS low so opening the port does not reset the board
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self._sync()

    # -- context manager -------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.command("M 0", timeout=2)
        except Exception:
            pass
        self.ser.close()

    # -- low level -------------------------------------------------------
    def _readline(self, deadline):
        buf = b""
        while time.time() < deadline:
            buf += self.ser.readline()
            if buf.endswith(b"\n"):
                return buf.decode(errors="replace").strip()
        raise BoardError("timeout")

    def _sync(self):
        """Wait for READY after a reset, or check that a running board answers."""
        for attempt in range(4):
            time.sleep(0.3)
            self.ser.reset_input_buffer()
            try:
                self.command("?", timeout=3)
                return
            except BoardError:
                pass
        raise BoardError(f"no answer from the board on {self.ser.port}")

    def command(self, line, timeout=30):
        """Send one command, return the info lines before DONE. Raises on ERR / timeout."""
        self.ser.write((line.strip() + "\n").encode())
        deadline = time.time() + timeout
        info = []
        while True:
            reply = self._readline(deadline)
            if self.on_log and not reply.startswith("!"):
                self.on_log(reply)
            if reply == "DONE":
                return info
            if reply.startswith("ERR"):
                raise BoardError(f"{line!r}: {reply}")
            if reply.startswith("!"):
                self.on_event(reply)
            elif reply.startswith("IMG "):
                info.append(self._read_exact(int(reply[4:]), deadline))
            elif reply and not reply.startswith("#") and reply != "READY":
                info.append(reply)

    def _read_exact(self, n, deadline):
        data = bytearray()
        while len(data) < n:
            if time.time() > deadline:
                raise BoardError("timeout reading image")
            data += self.ser.read(n - len(data))
        return bytes(data)

    # -- commands --------------------------------------------------------
    def home(self):
        self.command("H", timeout=60)

    def home_if_needed(self, force=False):
        """H only when the board does not know where it is (after a reset / power-up)."""
        if force or not self.status().get("HOMED"):
            print("Homing...")
            self.home()

    def move(self, x_mm, y_mm):
        self.command(f"G {x_mm:.2f} {y_mm:.2f}", timeout=30)

    def move_by(self, dx_mm, dy_mm):
        self.command(f"R {dx_mm:.2f} {dy_mm:.2f}", timeout=30)

    def magnet(self, on):
        self.command("M 1" if on else "M 0", timeout=2)

    def drag_profile(self, speed_mm_s, accel_mm_s2):
        """Speed and acceleration while the magnet is on (firmware default 150 / 300, until reset)."""
        self.command(f"P {speed_mm_s:.0f} {accel_mm_s2:.0f}", timeout=2)

    def free_profile(self, speed_mm_s, accel_mm_s2):
        """Speed and acceleration while the magnet is off (firmware default 150 / 1200, until reset)."""
        self.command(f"F {speed_mm_s:.0f} {accel_mm_s2:.0f}", timeout=2)

    def capture(self):
        """One JPEG (UXGA 1600x1200, set in board.ino) as bytes."""
        info = self.command("C", timeout=10)
        return next(x for x in info if isinstance(x, bytes))

    def status(self):
        """{'POS': (x, y), 'HOMED': 1, 'MAG': 0, 'HEAT': 3, 'CAM': 1, 'DRAG_V': 150, 'DRAG_A': 300, ...}"""
        line = next(x for x in self.command("?", timeout=5) if x.startswith("POS"))
        t = line.split()
        out = {"POS": (float(t[1]), float(t[2]))}
        for k, v in zip(t[3::2], t[4::2]):
            out[k] = int(v)
        return out
