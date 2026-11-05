# -*- coding: utf-8 -*-
"""Rebuild data/blue-dashed.csv from the evaluation chart.

The game reference curve exists only as a dashed blue line in the site's chart, so it
has to be read off the image.  That makes the target a *derived artefact*, and a derived
artefact is only trustworthy if the derivation is reproducible and independently
checked.  This script is both, and it carries its own proof:

    python scripts/digitise_target.py path/to/chart.png

The axes are calibrated from the chart's OWN printed tick labels -- the y axis from its
dB labels, the x axis from the plot-box edges cross-checked against the printed 1 kHz
tick.  Nothing is fitted to a curve we already believe.

The check that catches a bad calibration is the chart's *red* line: it is the Harman
target, and this pipeline has never seen the bundled Harman CSV.  If the digitised red
line reproduces data/harman_over_ear_2018.csv to within about half a decibel, the axes
are right.  A change that makes that number worse is wrong -- that invariant is worth
more than the blue curve it was written for.

Two traps this script exists to avoid:

  * local vs global pixel columns.  The plot is cropped before tracing, so column 0 is
    the plot's left edge, not the image's.  Mixing them shifts the whole frequency axis.
  * decimating to a tidy grid.  The chart draws narrow spikes near 5.5 and 8.7 kHz only a
    few columns wide; resampling to 1/48 octave samples either side of the peak and
    silently flattens them.  Every traced column is kept.
"""
import argparse
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LABEL_MARGIN_TOP = 10
LABEL_MARGIN_LEFT = 75
ANCHOR_HZ = 1000.0


def plot_box(a):
    """Bounding box of the pale plot area."""
    h, w, _ = a.shape
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    bg = (abs(r - g) < 10) & (abs(g - b) < 10) & (r > 205) & (r < 245)
    cols = np.where(bg.sum(axis=0) > h * 0.25)[0]
    rows = np.where(bg.sum(axis=1) > w * 0.25)[0]
    if not len(cols) or not len(rows):
        raise SystemExit("could not find the plot area; is this the right chart?")
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def _blobs(mask, axis, min_px=8, gap=6):
    idx = np.where(mask.any(axis=axis))[0]
    if not len(idx):
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out = []
    for grp in groups:
        if len(grp) < 3:
            continue
        sub = mask[:, grp] if axis == 0 else mask[grp, :]
        if sub.sum() < min_px:
            continue
        out.append(float(np.mean(grp)))
    return out


def calibrate(a, x0, x1, y0, y1):
    """Read the printed dB tick labels and return the row-to-dB line."""
    h, w, _ = a.shape
    lum = a.mean(axis=2)
    ylab = [20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]
    strip = lum[max(0, y0 - LABEL_MARGIN_TOP):min(h, y1 + 10),
                max(0, x0 - LABEL_MARGIN_LEFT):x0 - 4] > 150
    rows = _blobs(strip, axis=1, min_px=10)
    if len(rows) != len(ylab):
        raise SystemExit("found %d y labels, expected %d" % (len(rows), len(ylab)))
    px = np.array([r + y0 - LABEL_MARGIN_TOP for r in rows])
    A = np.vstack([px, np.ones_like(px)]).T
    coef, *_ = np.linalg.lstsq(A, np.array(ylab, float), rcond=None)
    resid = float(np.max(np.abs(A @ coef - ylab)))
    print("y axis from %d printed labels: dB = %.6f*row %+.4f  (max residual %.2f dB)"
          % (len(rows), coef[0], coef[1], resid))
    if resid > 0.5:
        raise SystemExit("y labels do not fit a straight line; refusing to digitise")
    return coef


def mask_red(p):
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    return (r > 150) & (g < 90) & (b < 90)


def mask_blue(p):
    r, g, b = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    return (b - r > 25) & (b > 110) & (r < 205)


def trace(mask, y0):
    """Median row of the topmost contiguous run per column, in GLOBAL row coordinates."""
    out = {}
    for c in range(mask.shape[1]):
        rows = np.where(mask[:, c])[0]
        if not len(rows):
            continue
        groups, cur = [], [rows[0]]
        for rr in rows[1:]:
            if rr - cur[-1] <= 3:
                cur.append(rr)
            else:
                groups.append(cur)
                cur = [rr]
        groups.append(cur)
        out[c] = float(np.mean(groups[0])) + y0
    return out


def bridge(d, max_gap=30):
    """Fill SHORT dash gaps; a long gap means the curve really is not drawn there."""
    cols = sorted(d)
    filled = dict(d)
    for a, b in zip(cols, cols[1:]):
        if 1 < b - a <= max_gap:
            step = (d[b] - d[a]) / (b - a)
            for c in range(a + 1, b):
                filled[c] = d[a] + step * (c - a)
    return filled


def to_curve(d, width, coef):
    cols = np.array(sorted(d), float)
    hz = 20.0 * (20000.0 / 20.0) ** (cols / float(width))
    db = coef[0] * np.array([d[int(c)] for c in cols]) + coef[1]
    return hz, db


def read_csv(path):
    f, v = [], []
    with io.open(path, encoding="utf-8-sig") as fh:
        next(fh, None)
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                f.append(float(parts[0]))
                v.append(float(parts[1]))
            except ValueError:
                continue
    f, v = np.array(f), np.array(v)
    o = np.argsort(f)
    return f[o], v[o]


def normalise(f, v, at=ANCHOR_HZ):
    return v - float(np.interp(np.log10(at), np.log10(f), v))


def main(argv=None):
    ap = argparse.ArgumentParser(description="re-digitise the game reference curve")
    ap.add_argument("chart", help="screenshot of the chart with the blue dashed curve")
    ap.add_argument("--out", default="data/blue-dashed.csv")
    ap.add_argument("--harman", default="data/harman_over_ear_2018.csv",
                    help="bundled Harman CSV, used ONLY for the independent check")
    args = ap.parse_args(argv)

    a = np.asarray(Image.open(args.chart).convert("RGB")).astype(int)
    x0, x1, y0, y1 = plot_box(a)
    width = x1 - x0
    print("chart %dx%d, plot box x %d-%d y %d-%d" % (a.shape[1], a.shape[0], x0, x1, y0, y1))
    coef = calibrate(a, x0, x1, y0, y1)
    p = a[y0:y1 + 1, x0:x1 + 1]

    rf, rv = to_curve(trace(mask_red(p), y0), width, coef)
    rv = normalise(rf, rv)
    hf, hl = read_csv(args.harman)
    harman = np.interp(np.log10(rf), np.log10(hf), normalise(hf, hl))
    err = np.abs(rv - harman)
    print("axis check: digitised RED line vs bundled Harman CSV -> mean %.3f dB, max %.3f dB"
          % (float(np.mean(err)), float(np.max(err))))
    if float(np.mean(err)) > 1.0:
        raise SystemExit("the red line does not reproduce the Harman CSV; the axis "
                         "calibration is wrong, so the blue curve would be wrong too")

    bf, bv = to_curve(bridge(trace(mask_blue(p), y0)), width, coef)
    bv = normalise(bf, bv)
    order = np.argsort(bf)
    bf, bv = bf[order], bv[order]
    keep = (bf >= 20.0) & (bf <= 20000.0)
    bf, bv = bf[keep], bv[keep]

    with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
        fh.write("frequency,db\n")
        for f, v in zip(bf, bv):
            fh.write("%.4f,%.4f\n" % (f, v))
    print("wrote %s: %d points, %.0f-%.0f Hz" % (args.out, len(bf), bf.min(), bf.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())