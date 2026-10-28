"""Train the anomaly detector on healthy data and export it as include/model.h.

    python tools/train.py

Uses only recordings of normal running: every data/healthy*.csv file. Other
CSV files in data/ (for example imbalance.csv) are only used for testing.

How it works
  1. Each spectrum is reduced to 32 frequency bands (15.6 Hz wide).
  2. An autoencoder (a small neural network) learns to reproduce the band
     pattern of healthy running through a narrow bottleneck.
  3. On new data, the score is the reconstruction error of the worst band:
     a large error means "this band does not look like anything seen while
     healthy". Using the worst band catches a change in one place (for
     example the shaft-speed peak growing) that an average would dilute.
  4. The alarm level is set from 20 s blocks of healthy data the model did
     not train on, so the false-alarm figures reported here are honest.
  5. The first and last seconds of each session are dropped: that is when
     the machine is being touched to start or stop the recording.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
N_BINS = 256
BIN_HZ = 500.0 / N_BINS
# Bands of 8 bins (15.6 Hz); the first starts at bin 3 to skip slow drift.
BAND_EDGES = [3] + list(range(8, N_BINS + 1, 8))
N_BANDS = len(BAND_EDGES) - 1
FLOOR_MG = 0.1          # keeps log() finite in silent bands
HIDDEN = (16, 4, 16)    # autoencoder layout; 4 is the bottleneck
THRESHOLD_MARGIN = 1.25
SESSION_GAP_S = 1.5     # a time jump larger than this starts a new session
BLOCK = 40              # spectra per held-out block (20 s)
N_FOLDS = 5
TRIM_START = 6          # spectra dropped at the start of a session (3 s)
TRIM_END = 30           # and at the end (15 s)
ALARM_RUN = 3           # spectra in a row above the level before a warning


# ---------------------------------------------------------------- data

def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)
    if data.size == 0:
        return np.empty((0, 6 + N_BINS)), []
    if data.shape[1] != 6 + N_BINS:
        sys.exit(f"{path.name}: expected {6 + N_BINS} columns, found {data.shape[1]}.")
    t = data[:, 0]
    breaks = np.where((np.diff(t) > SESSION_GAP_S) | (np.diff(t) < 0))[0] + 1
    return data, np.split(np.arange(len(data)), breaks)


def features(bins_mg):
    """Same maths as Detector::features in src/anomaly.cpp."""
    bins_mg = np.atleast_2d(bins_mg)
    out = np.empty((len(bins_mg), N_BANDS))
    for k in range(N_BANDS):
        band = bins_mg[:, BAND_EDGES[k]:BAND_EDGES[k + 1]]
        out[:, k] = np.log10(np.sqrt((band ** 2).sum(axis=1)) + FLOOR_MG)
    return out


def total_rms_mg(rows):
    return np.sqrt((rows[:, 1:4] ** 2).sum(axis=1)) * 1000


def augment(bins_mg, copies, rng):
    """Slightly louder or quieter copies (mounting and speed differ a little)."""
    out = [bins_mg]
    for _ in range(copies):
        gain = rng.uniform(0.85, 1.15, size=(len(bins_mg), 1))
        jitter = rng.uniform(0.9, 1.1, size=bins_mg.shape)
        out.append(bins_mg * gain * jitter)
    return np.vstack(out)


# ---------------------------------------------------------------- model

class Detector:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, bins_mg):
        rng = np.random.default_rng(self.seed)
        x = features(augment(bins_mg, 4, rng))
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 0.05)
        z = (x - self.mean) / self.std
        self.net = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="relu", alpha=1e-3,
                                learning_rate_init=3e-3, max_iter=3000, random_state=self.seed)
        self.net.fit(z, z)
        return self

    def errors(self, bins_mg):
        """Anomaly error per spectrum (worst band), and the error of every band."""
        z = (features(bins_mg) - self.mean) / self.std
        per_band = (self.net.predict(z) - z) ** 2
        return per_band.max(axis=1), per_band


def smooth(scores, alpha=0.3):
    """Exponential moving average, same as the firmware."""
    out, s = np.empty_like(scores), None
    for i, v in enumerate(scores):
        s = v if s is None else alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def sustained_level(errors):
    """Highest level the smoothed error stayed above for ALARM_RUN spectra."""
    s = smooth(errors)
    if len(s) < ALARM_RUN:
        return float(s.max())
    return float(max(s[i - ALARM_RUN + 1:i + 1].min() for i in range(ALARM_RUN - 1, len(s))))


def alarms(smoothed, level, need=ALARM_RUN):
    """True where the smoothed score has been above level for `need` spectra."""
    run, out = 0, np.zeros(len(smoothed), bool)
    for i, v in enumerate(smoothed):
        run = run + 1 if v > level else 0
        out[i] = run >= need
    return out


# ---------------------------------------------------------------- fault simulation

def band_of(hz):
    b = int(hz / BIN_HZ)
    return next(k for k in range(N_BANDS) if BAND_EDGES[k] <= b < BAND_EDGES[k + 1])


def simulate_faults(bins_mg, shaft_hz):
    """Turn healthy spectra into the classic fault patterns."""
    hz = (np.arange(N_BINS) + 0.5) * BIN_HZ
    near = lambda f, w=2.0: np.abs(hz - f) <= w
    faults = {}

    for factor in (1.5, 2.0, 3.0):  # imbalance: the 1x shaft peak grows
        b = bins_mg.copy()
        b[:, near(shaft_hz)] *= factor
        faults[f"imbalance x{factor:g}"] = b

    b = bins_mg.copy()  # looseness: many shaft harmonics appear
    for h in range(2, 9):
        b[:, near(h * shaft_hz)] += 4.0
    faults["looseness (harmonics)"] = b

    b = bins_mg.copy()  # bearing wear: a new high-frequency peak
    b[:, near(287.0, 4.0)] += 5.0
    faults["bearing tone 287 Hz"] = b

    b = bins_mg * 1.0  # rubbing: broadband noise floor rises
    b[:, hz > 100] += 2.0
    faults["rubbing (noise floor)"] = b
    return faults


# ---------------------------------------------------------------- export

def c_array(values, per_line=8):
    flat = np.asarray(values, dtype=np.float32).ravel()
    return "\n".join("  " + ", ".join(f"{v:.8e}f" for v in flat[i:i + per_line]) + ","
                     for i in range(0, len(flat), per_line))


def write_header(path, det, threshold, stop_rms_mg, summary):
    layers = [(w.T, b) for w, b in zip(det.net.coefs_, det.net.intercepts_)]
    sizes = [N_BANDS] + [w.shape[0] for w, _ in layers]
    text = f"""// Generated by tools/train.py. Do not edit by hand.
// {summary}
#pragma once

#define MODEL_IS_PLACEHOLDER 0
#define MODEL_SUMMARY "{summary}"

constexpr int MODEL_INPUT_BINS = {N_BINS};
constexpr int MODEL_N_BANDS = {N_BANDS};
static const int MODEL_BAND_EDGES[] = {{{", ".join(map(str, BAND_EDGES))}}};
constexpr float MODEL_FLOOR_MG = {FLOOR_MG}f;

static const float MODEL_FEATURE_MEAN[] = {{
{c_array(det.mean)}
}};
static const float MODEL_FEATURE_STD[] = {{
{c_array(det.std)}
}};

// Autoencoder: ReLU hidden layers, linear output. Each layer is [output][input].
constexpr int MODEL_N_LAYERS = {len(layers)};
constexpr int MODEL_MAX_WIDTH = {max(sizes)};
static const int MODEL_LAYER_SIZES[] = {{{", ".join(map(str, sizes))}}};
static const float MODEL_WEIGHTS[] = {{
{chr(10).join(c_array(w) for w, _ in layers)}
}};
static const float MODEL_BIASES[] = {{
{chr(10).join(c_array(b) for _, b in layers)}
}};

// Worst-band reconstruction error at which the score reaches 1.0 (warning).
constexpr float MODEL_THRESHOLD = {threshold:.8e}f;
// Below this total vibration (mg) the machine is considered stopped.
constexpr float MODEL_STOP_RMS_MG = {stop_rms_mg:.3f}f;
"""
    path.write_text(text)


def save_plot(path, panels, threshold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x0 = 0
    for name, scores, colour in panels:
        x = np.arange(x0, x0 + len(scores)) * 0.5
        ax.plot(x, smooth(scores / threshold), color=colour, lw=1.2)
        ax.text(x[len(x) // 2], 0.02 + 0.06 * (len(ax.texts) % 2), name, ha="center", va="bottom", fontsize=8,
                transform=ax.get_xaxis_transform(), color=colour)
        x0 += len(scores) + 10
    ax.axhline(1, color="#e6a700", ls="--", lw=1, label="warning (1.0)")
    ax.axhline(2, color="#d62728", ls="--", lw=1, label="fault (2.0)")
    ax.set_yscale("log")
    ax.set_xlabel("time (s), recordings placed one after another")
    ax.set_ylabel("anomaly score (smoothed)")
    ax.set_title("Anomaly score: held-out healthy sessions vs. simulated faults")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"Saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "include" / "model.h")
    args = parser.parse_args()

    healthy_files = sorted(args.data.glob("healthy*.csv"))
    if not healthy_files:
        sys.exit("No data/healthy*.csv found. Record normal running with record.bat first.")

    sessions, rows_all = [], []
    print("Healthy recordings:")
    for f in healthy_files:
        rows, parts = load_csv(f)
        for p in parts:
            if len(p) >= 20:
                sessions.append(rows[p])
        rows_all.append(rows)
        print(f"  {f.name}: {len(rows)} spectra in {len(parts)} session(s)")
    rows_all = np.vstack(rows_all)
    if len(rows_all) < 100:
        sys.exit("Need at least 100 healthy spectra (about a minute). Record more.")

    # Machines that are switched off must not count as healthy data.
    rms = total_rms_mg(rows_all)
    running = rms > 0.5 * np.percentile(rms, 50)
    stop_rms = float(max(0.5 * np.percentile(rms[running], 1), 1.0))
    cleaned, dropped = [], 0
    for s in sessions:
        keep = s[TRIM_START:len(s) - TRIM_END] if len(s) > TRIM_START + TRIM_END + 20 else s
        r = total_rms_mg(keep)
        keep = keep[(r > stop_rms) & (r < 2.0 * np.median(r))]  # off, or bumped
        dropped += len(s) - len(keep)
        if len(keep) >= 20:
            cleaned.append(keep)
    sessions = cleaned
    if not sessions:
        sys.exit("No usable healthy data: was the machine running during the recording?")
    print(f"  {dropped} spectra dropped (start and end of sessions, bumps, machine off)")
    print(f"  machine counted as stopped below {stop_rms:.1f} mg total vibration")

    # Threshold from data the model did not see: every session is cut into
    # 20 s blocks, and each block is scored by a model trained without it.
    blocks = []
    for si, s in enumerate(sessions):
        for start in range(0, len(s), BLOCK):
            blocks.append((si, s[start:start + BLOCK]))
    folds = min(N_FOLDS, len(blocks))
    held_out = [[] for _ in sessions]
    print(f"\nChecking on held-out data ({folds}-fold, 20 s blocks):")
    for f in range(folds):
        test = [(si, b) for k, (si, b) in enumerate(blocks) if k % folds == f]
        train = np.vstack([b for k, (_, b) in enumerate(blocks) if k % folds != f])
        model = Detector().fit(train[:, 6:])
        for si, b in test:
            held_out[si].append(model.errors(b[:, 6:])[0])
    held_out = [np.concatenate(h) for h in held_out if h]
    pooled = np.concatenate(held_out)
    for i, e in enumerate(held_out):
        print(f"  session {i + 1}: median error {np.median(e):.3f}, worst {e.max():.3f}")

    # Would a completely new session look normal? If not, normal running
    # varies more than the recordings show: more sessions would help.
    if len(sessions) >= 2:
        unseen = []
        for i, test in enumerate(sessions):
            train = np.vstack([s for j, s in enumerate(sessions) if j != i])
            unseen.append(np.median(Detector().fit(train[:, 6:]).errors(test[:, 6:])[0]))
        typical = np.median(pooled)
        odd = [i + 1 for i, u in enumerate(unseen) if u > 3 * typical]
        if odd:
            print(f"  Note: session(s) {odd} look different from the others. The model knows them now,"
                  " but record a few more healthy sessions so it learns all normal conditions.")
    threshold = max(sustained_level(e) for e in held_out) * THRESHOLD_MARGIN

    healthy_scores = pooled / threshold
    healthy_alarms = sum(alarms(smooth(e / threshold), 1.0).sum() for e in held_out)
    print(f"\nAlarm threshold: {threshold:.4f}")
    print(f"Healthy data the model never saw ({len(pooled) / 120:.1f} min): {healthy_alarms} warnings,"
          f" {100 * (healthy_scores > 1).mean():.1f}% of single spectra above the level")

    # Final model on all healthy data.
    train_all = np.vstack(sessions)
    det = Detector().fit(train_all[:, 6:])

    hz = (np.arange(N_BINS) + 0.5) * BIN_HZ
    mean_spec = train_all[:, 6:].mean(axis=0)
    low = (hz > 8) & (hz < 60)
    shaft_hz = float(hz[low][np.argmax(mean_spec[low])])
    print(f"Shaft speed estimate: {shaft_hz:.1f} Hz ({shaft_hz * 60:.0f} rpm)")

    # Simulated faults on held-out data, scored by a model that never saw it.
    test = np.vstack([b for k, (_, b) in enumerate(blocks) if k % folds == 0])
    train = np.vstack([b for k, (_, b) in enumerate(blocks) if k % folds != 0])
    probe = Detector().fit(train[:, 6:])
    print("\nSimulated faults (on a held-out session):")
    panels = [(f"healthy {i + 1}", e, "#2ca02c") for i, e in enumerate(held_out)]
    colours = ["#e6a700", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for (name, bins), colour in zip(simulate_faults(test[:, 6:], shaft_hz).items(), colours):
        err, per_band = probe.errors(bins)
        s = err / threshold
        warned = alarms(smooth(s), 1.0)
        worst = int(np.bincount(per_band.argmax(axis=1), minlength=N_BANDS).argmax())
        lo, hi = BAND_EDGES[worst] * BIN_HZ, BAND_EDGES[worst + 1] * BIN_HZ
        print(f"  {name:<24} warning {100 * warned.mean():5.1f}% of the time,  median score {np.median(s):6.2f},"
              f"  biggest change {lo:.0f}-{hi:.0f} Hz")
        panels.append((name, err, colour))

    # Other recordings in data/ are test data.
    others = [f for f in sorted(args.data.glob("*.csv")) if not f.name.startswith("healthy")]
    if others:
        print("\nOther recordings (test only):")
    for f in others:
        rows, _ = load_csv(f)
        if not len(rows):
            continue
        stopped = total_rms_mg(rows) <= stop_rms
        line = f"  {f.name:<22} {len(rows)} spectra: "
        if stopped.mean() > 0.9:
            line += "machine STOPPED (no running vibration). Was it switched on, is the sensor attached?"
        else:
            s = det.errors(rows[~stopped, 6:])[0] / threshold
            line += (f"{100 * (s > 1).mean():.1f}% above threshold, median score {np.median(s):.2f}"
                     + (f", {100 * stopped.mean():.0f}% stopped" if stopped.any() else ""))
        print(line)

    save_plot(ROOT / "docs" / "anomaly_scores.png", panels, threshold)
    import datetime
    summary = (f"{len(train_all)} healthy spectra, {len(sessions)} sessions, "
               f"threshold {threshold:.4f}, shaft {shaft_hz:.1f} Hz, "
               f"trained {datetime.datetime.now():%Y-%m-%d %H:%M}")
    # The exported weights must give the same answers as scikit-learn.
    z = (features(train_all[:, 6:]) - det.mean) / det.std
    a = z
    layers = list(zip(det.net.coefs_, det.net.intercepts_))
    for i, (w, bias) in enumerate(layers):
        a = a @ w + bias
        if i < len(layers) - 1:
            a = np.maximum(a, 0)
    if np.abs(a - det.net.predict(z)).max() > 1e-4:
        sys.exit("Export check failed: model.h was not written.")
    write_header(args.out, det, threshold, stop_rms, summary)
    n_params = sum(w.size + b.size for w, b in zip(det.net.coefs_, det.net.intercepts_))
    print(f"\nWrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}"
          f" ({n_params} parameters). Upload the firmware again.")


if __name__ == "__main__":
    main()
