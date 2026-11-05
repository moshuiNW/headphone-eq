# -*- coding: utf-8 -*-
"""The one analysis grid every curve is compared on.

1/48 octave from 20 Hz to 20 kHz.  Frequencies are spaced uniformly in octaves,
so a uniform window in *points* is a uniform window in *octaves* -- that identity
is what makes the smoothing code correct and cheap.
"""
import numpy as np

F_MIN = 20.0
F_MAX = 20000.0
POINTS_PER_OCT = 48
FINE_F_MIN = 5.0
FINE_F_MAX = 24000.0

_n_oct = np.log2(F_MAX / F_MIN)
GRID = F_MIN * 2.0 ** (np.arange(int(round(_n_oct * POINTS_PER_OCT)) + 1) / POINTS_PER_OCT)
GRID_LOG = np.log10(GRID)
# A fine grid for peak hunting: 5 Hz .. 24 kHz, dense enough for narrow high-Q shapes.
FINE = np.geomspace(FINE_F_MIN, FINE_F_MAX, 40000)

OCT_PER_POINT = (np.log2(GRID[-1]) - np.log2(GRID[0])) / (len(GRID) - 1)
POINTS_PER_OCTAVE = 1.0 / OCT_PER_POINT


def index_of(freq, grid=None):
    """Index of the sample nearest to *freq* on a log axis."""
    g = GRID if grid is None else grid
    return int(np.argmin(np.abs(np.log10(g) - np.log10(freq))))


def band_mask(lo, hi, grid=None):
    """Boolean mask of grid points in [lo, hi)."""
    g = GRID if grid is None else grid
    return (g >= lo) & (g < hi)
