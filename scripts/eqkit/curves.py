# -*- coding: utf-8 -*-
"""Curves as (freq_hz, level_db) pairs.

Every function here takes and returns plain numpy arrays; nothing knows about
files or sources.  The rule the whole toolkit obeys: a curve is only ever
compared after being put on the canonical grid and normalised to the same
reference, because raw SPL from a fixture is offset by tens of dB and absolute
level carries no tonal information.
"""
import numpy as np
import io
import csv

from .grid import GRID

# The reference frequency for tonal normalisation.  1 kHz is the anchor the
# evaluation site itself uses ("all curves aligned at 1000 Hz"), so matching it
# keeps our residuals comparable with the numbers a user can read there.
ANCHOR_HZ = 1000.0


def read_csv(path, freq_col=0, level_col=1, delimiter=None):
    """Read a two-column CSV/TSV with a header row.  Returns (freq, level)."""
    f, v = [], []
    with io.open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
        fh.seek(0)
        delim = delimiter or ("\t" if "\t" in first else ",")
        reader = csv.reader(fh, delimiter=delim)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) <= max(freq_col, level_col):
                continue
            try:
                freq = float(row[freq_col])
                level = float(row[level_col])
            except ValueError:
                continue
            f.append(freq)
            v.append(level)
    if not f:
        raise ValueError("no numeric rows in %s" % path)
    f = np.asarray(f, float)
    v = np.asarray(v, float)
    order = np.argsort(f)
    f, v = f[order], v[order]
    keep = np.concatenate([[True], np.diff(f) > 0])
    return f[keep], v[keep]


def write_csv(path, freq, level, header=("frequency", "db")):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for a, b in zip(freq, level):
            w.writerow(["%.4f" % a, "%.4f" % b])


def from_xy(pairs):
    """Build sorted, de-duplicated arrays from [(freq, level), ...]."""
    arr = np.asarray([(float(a), float(b)) for a, b in pairs if a and b not in (None, "")], float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 0])
    return arr[keep, 0], arr[keep, 1]


def resample(freq, level, grid=None):
    """Log-log linear interpolation onto the canonical grid.

    Log interpolation is the right model: a headphone response between two
    measured points is close to linear in log frequency, and a linear-in-Hz
    interpolation up at 10 kHz would wipe out a whole octave of detail.
    Points outside the measured range are clamped, not extrapolated -- and the
    caller is told the covered range so it can refuse to claim anything there.
    """
    g = GRID if grid is None else grid
    return np.interp(np.log10(g), np.log10(freq), level)


def normalise(level, grid=None, at_hz=ANCHOR_HZ):
    """Shift a curve so it reads 0 dB at the anchor frequency."""
    g = GRID if grid is None else grid
    ref = np.interp(np.log10(at_hz), np.log10(g), level)
    return level - ref


def smooth(level, octaves, grid=None):
    """Fractional-octave moving average on the log grid (edge-padded)."""
    g = GRID if grid is None else grid
    oct_per_point = (np.log2(g[-1]) - np.log2(g[0])) / (len(g) - 1)
    width = max(1, int(round(octaves / oct_per_point)))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = np.concatenate([np.full(pad, level[0]), level, np.full(pad, level[-1])])
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")


def smooth_var(level, grid=None, below_hz=7000.0, above_hz=9000.0, oct_low=1 / 6, oct_high=0.75):
    """Width-ramped smoothing: fine where the measurement is trustworthy, broad above.

    The knee is where fixtures stop being reliable (see docs: comb filtering and
    ear-canal cancellation make 1/12-octave detail above ~7 kHz meaningless), so
    the width is ramped between two frequencies with a smoothstep instead of
    switching abruptly.
    """
    g = GRID if grid is None else grid
    a = smooth(level, oct_low, g)
    b = smooth(level, oct_high, g)
    t = np.clip((np.log10(g) - np.log10(below_hz)) / (np.log10(above_hz) - np.log10(below_hz)), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return a * (1.0 - t) + b * t


def value_at(level, freq, grid=None):
    g = GRID if grid is None else grid
    return float(np.interp(np.log10(freq), np.log10(g), level))


def deviation(a, b, mask=None):
    """Band-limited, level-removed difference a - b.

    The mean over the evaluation band is removed before scoring: absolute level
    is a free parameter the user sets with the volume knob, so only the *shape*
    of the error is a tuning problem.
    """
    r = np.asarray(a) - np.asarray(b)
    m = np.ones_like(r, bool) if mask is None else mask
    return r - np.mean(r[m])
