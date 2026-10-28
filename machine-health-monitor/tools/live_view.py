"""Live vibration spectrum from the board, and recording for training.

Upload the firmware, close the PlatformIO serial monitor, then:

    python tools/live_view.py                                # live plot
    python tools/live_view.py --record healthy --seconds 120 # plot and record

Recordings go to data/<label>.csv, one row per spectrum (every 0.5 s):
t, rms_x, rms_y, rms_z, peak_hz, peak_mg, then 256 spectrum bins in milli-g
(bin i covers i*1.95 Hz to (i+1)*1.95 Hz).
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_RATE_HZ = 1000.0
N_BINS = 256
BIN_HZ = SAMPLE_RATE_HZ / 2 / N_BINS
N_FIELDS = 6 + N_BINS
HEADER = ["t", "rms_x", "rms_y", "rms_z", "peak_hz", "peak_mg"] + [f"b{i}" for i in range(N_BINS)]
HISTORY = 120  # spectra kept for the RMS plot (60 s)

# USB vendor IDs of the serial chips found on ESP32 boards:
# Silicon Labs CP210x, WCH CH340/CH9102, FTDI, Espressif.
ESP32_USB_VENDORS = {0x10C4, 0x1A86, 0x0403, 0x303A}


def pick_port(requested):
    if requested:
        return requested
    ports = sorted(list_ports.comports(), key=lambda p: p.device)
    if not ports:
        sys.exit("No serial port found. Is the board plugged in?")
    if len(ports) == 1:
        return ports[0].device
    usb = [p for p in ports if getattr(p, "vid", None) is not None]
    known = [p for p in usb if p.vid in ESP32_USB_VENDORS]
    for candidates in (known, usb):
        if len(candidates) == 1:
            print(f"Using {candidates[0].device} ({candidates[0].description}).")
            return candidates[0].device
    print("Several serial ports found:")
    for i, p in enumerate(ports, 1):
        print(f"  {i}. {p.device}  {p.description}")
    while True:
        answer = input(f"Which one is the ESP32? (1-{len(ports)}): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(ports):
            return ports[int(answer) - 1].device


class Board(threading.Thread):
    """Reads the board in the background and keeps the latest spectrum."""

    def __init__(self, port, baud, csv_file=None, max_rows=None):
        super().__init__(daemon=True)
        self.ser = serial.Serial(port, baud, timeout=0.5)
        self.csv_file = csv_file
        self.max_rows = max_rows
        self.rows = 0
        self.latest = None
        self.history = deque(maxlen=HISTORY)
        self.health = None  # (state, score, smoothed, health %, band low Hz, band high Hz)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None

    def run(self):
        try:
            self._wait_until_ready()
            self.ser.write(b"s\n")
            last_spectrum = time.time()
            while not self.stop.is_set():
                line = self.ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    if time.time() - last_spectrum > 3:  # board missed the command
                        self.ser.write(b"s\n")
                        last_spectrum = time.time()
                    continue
                if line.startswith("H,"):
                    parts = line.split(",")
                    if len(parts) == 7:
                        try:
                            with self.lock:
                                self.health = (parts[1],) + tuple(float(v) for v in parts[2:])
                        except ValueError:
                            pass
                elif line.startswith("S,"):
                    if self._handle_spectrum(line):
                        last_spectrum = time.time()
                elif line.startswith("#"):
                    print("  board:", line.lstrip("# "))
        except serial.SerialException as err:
            self.error = f"Lost the connection to the board: {err}"
        finally:
            self.stop.set()

    def _wait_until_ready(self):
        # Opening the port usually resets the ESP32; wait for its start-up.
        deadline = time.time() + 4
        while time.time() < deadline and not self.stop.is_set():
            line = self.ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("#"):
                print("  board:", line.lstrip("# "))
            if "Ready" in line:
                return

    def _handle_spectrum(self, line):
        parts = line[2:].split(",")
        if len(parts) != N_FIELDS:
            return False
        try:
            values = np.array([float(v) for v in parts])
        except ValueError:
            return False
        with self.lock:
            self.latest = values
            self.history.append(values[:6])
        if self.csv_file is not None:
            self.csv_file.write(",".join(parts) + "\n")
            self.csv_file.flush()
            self.rows += 1
            if self.max_rows and self.rows >= self.max_rows:
                print(f"Recorded {self.rows} spectra.")
                self.stop.set()
        return True

    def snapshot(self):
        with self.lock:
            latest = None if self.latest is None else self.latest.copy()
            return latest, np.array(self.history)

    def health_text(self):
        with self.lock:
            h = self.health
        if h is None:
            return ""
        state, _, smoothed, health, lo, hi = h
        if health < 0:
            return state
        text = f"{state}  health {health:.0f}%"
        if state in ("WARNING", "FAULT"):
            text += f"  (biggest change {lo:.0f}-{hi:.0f} Hz)"
        return text

    def close(self):
        self.stop.set()
        self.join(timeout=2)
        try:
            self.ser.write(b"q\n")
            self.ser.close()
        except serial.SerialException:
            pass


def open_recording(label):
    label = label.strip().lower()
    if not label.replace("_", "").isalnum():
        sys.exit("Use only letters, digits and underscores in the label.")
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{label}.csv"
    mode = "w"
    if path.exists() and path.stat().st_size > 0:
        rows = sum(1 for _ in path.open()) - 1
        answer = ""
        while answer not in ("a", "s"):
            answer = input(f"{path.name} already has {rows} spectra. Add to them or start over? (a/s): ").strip().lower()
        mode = "a" if answer == "a" else "w"
    f = path.open(mode)
    if mode == "w":
        f.write(",".join(HEADER) + "\n")
    print(f"Recording into {path.relative_to(ROOT)}.")
    return f


def choose_backend():
    """A desktop window if this Python has one, otherwise a browser tab."""
    try:
        import tkinter  # noqa: F401
        tkinter.Tk().destroy()
        return "TkAgg"
    except Exception:
        pass
    try:
        import tornado  # noqa: F401
        return "WebAgg"
    except ImportError:
        return None


def make_plot(board, record_label, browser=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    freqs = (np.arange(N_BINS) + 0.5) * BIN_HZ
    fig, (ax_spec, ax_rms) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})
    fig.canvas.manager.set_window_title("Machine health monitor")
    (spec_line,) = ax_spec.plot(freqs, np.zeros(N_BINS), color="#1f6fb2", lw=1.2)
    peak_mark = ax_spec.axvline(0, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax_spec.set_xlim(0, SAMPLE_RATE_HZ / 2)
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Amplitude (mg)")
    ax_spec.grid(alpha=0.3)
    title = ax_spec.set_title("Waiting for the board...")

    rms_lines = [ax_rms.plot([], [], label=a, color=c)[0]
                 for a, c in (("x", "#d62728"), ("y", "#2ca02c"), ("z", "#1f6fb2"))]
    ax_rms.set_xlabel("Time (s)")
    ax_rms.set_ylabel("RMS vibration (mg)")
    ax_rms.legend(loc="upper left")
    ax_rms.grid(alpha=0.3)
    fig.tight_layout()

    def update(_frame):
        latest, history = board.snapshot()
        if latest is None:
            if board.stop.is_set():
                finish()
            return []
        bins = latest[6:]
        spec_line.set_ydata(bins)
        ax_spec.set_ylim(0, max(5.0, bins.max() * 1.15))
        peak_mark.set_xdata([latest[4], latest[4]])
        status = board.health_text()
        text = (status + "   |   " if status else "") + f"Peak {latest[4]:.1f} Hz ({latest[4] * 60:.0f} rpm)"
        colour = {"WARNING": "#b36b00", "FAULT": "#c62828"}.get(status.split(" ")[0] if status else "", "black")
        title.set_color(colour)
        if record_label:
            text += f"   |   recording '{record_label}': {board.rows} spectra"
        title.set_text(text)
        if len(history):
            t = history[:, 0]
            for i, line in enumerate(rms_lines):
                line.set_data(t, history[:, 1 + i] * 1000)
            ax_rms.set_xlim(t[0], max(t[-1], t[0] + 1))
            ax_rms.set_ylim(0, max(5.0, history[:, 1:4].max() * 1000 * 1.2))
        if board.stop.is_set():
            finish()
        return []

    def finish():
        plt.close(fig)
        if browser:  # end the web server so the script can exit
            import tornado.ioloop
            tornado.ioloop.IOLoop.current().stop()

    anim = FuncAnimation(fig, update, interval=250, cache_frame_data=False)
    return fig, anim, update


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", help="serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--record", metavar="LABEL", help="save spectra to data/LABEL.csv")
    parser.add_argument("--seconds", type=float, help="stop recording after this many seconds")
    parser.add_argument("--no-plot", action="store_true", help="print only, no window")
    args = parser.parse_args()

    csv_file = open_recording(args.record) if args.record else None
    max_rows = int(args.seconds * 2) if args.seconds else None

    port = pick_port(args.port)
    try:
        board = Board(port, args.baud, csv_file, max_rows)
    except serial.SerialException as err:
        sys.exit(f"Could not open {port}: {err}\nClose the PlatformIO serial monitor and try again.")
    print(f"Connected to {port}. Close the window or press Ctrl+C to stop.")
    board.start()

    backend = None if args.no_plot else choose_backend()
    if not args.no_plot and backend is None:
        print("No way to show a plot on this Python (run setup.bat again to install it). Text only.")

    try:
        if backend is None:
            shown = -1
            while not board.stop.is_set():
                time.sleep(0.5)
                latest, _ = board.snapshot()
                if latest is not None and board.rows != shown:
                    shown = board.rows
                    print(f"t={latest[0]:7.1f}s  {board.health_text():<20} peak {latest[4]:6.1f} Hz  {latest[5]:7.1f} mg  "
                          f"rms x {latest[1]*1000:6.1f}  y {latest[2]*1000:6.1f}  z {latest[3]*1000:6.1f} mg"
                          + (f"  [{board.rows} recorded]" if csv_file else ""))
        else:
            import matplotlib
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            if backend == "WebAgg":
                matplotlib.rcParams["webagg.open_in_browser"] = True
                print("The live plot opens in your web browser. Press Ctrl+C here to stop.")
            fig, anim, _ = make_plot(board, args.record, browser=(backend == "WebAgg"))
            plt.show()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        board.close()
        if csv_file is not None:
            csv_file.close()
            print(f"Saved {board.rows} spectra.")
        if board.error:
            print(board.error)


if __name__ == "__main__":
    main()
