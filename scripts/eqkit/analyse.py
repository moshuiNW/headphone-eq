# -*- coding: utf-8 -*-
"""From a measurement to a correction curve -- trusting only what the data supports.

The single most important idea in this file: a narrow spike in a headphone
measurement may be the headphone, or it may be the fixture.  Correcting a fixture
artefact means injecting a narrow boost into someone's ear.  So before fitting,
we ask each sharp feature whether it *survives* being looked at through a wider
window.  A real housing/driver resonance barely moves; comb filtering collapses
toward the local trend.

The verdict is data, not opinion: report it, and the fit below then only smooths
as finely as the verdict allows.
"""
import numpy as np

from .grid import GRID, index_of, band_mask
from . import curves

# Peak/valley candidates worth interrogating, by band.  These are *probes*, not
# assumptions: features that fail the stability test are simply not fitted.
PROBE_FREQS = (60, 100, 150, 220, 300, 450, 700, 1000, 1500, 2200, 2800, 3400,
               4200, 5000, 6000, 7000, 8000, 9000, 10500, 12000, 15000)

STABILITY_WIDTHS = (0.15, 0.3, 0.5, 0.75, 1.0)
# A feature counts as real when it loses less than this much amplitude across
# the widest smoothing still allowed to count as "the same feature".
STABLE_DROP_DB = 3.0


def stability_scan(stable_curve, grid=GRID, probes=PROBE_FREQS):
    """How each probe frequency behaves under widening smoothing."""
    rows = []
    for fc in probes:
        i = index_of(fc, grid)
        vals = [float(curves.smooth(stable_curve, w, grid)[i]) for w in STABILITY_WIDTHS]
        drop = vals[0] - vals[-1]
        broad = float(curves.smooth(stable_curve, 1.0, grid)[i])
        local = float(curves.smooth(stable_curve, 0.15, grid)[i])
        rows.append({
            "freq": float(grid[i]) if abs(grid[i] - fc) / fc < 0.02 else float(fc),
            "levels": vals,
            "drop": drop,
            "excess": local - broad,      # amplitude of the feature over the broad trend
            "real": bool(abs(drop) <= STABLE_DROP_DB),
        })
    return rows


def confidence_map(grid=GRID, measurement_freq=None, smoothing_oct=1 / 6):
    """Per-grid-point data confidence in {0,1,2}: 2 measured, 1 trend, 0 unsupported."""
    conf = np.full(len(grid), 2, dtype=int)
    # Above 5 kHz the fixture, not the headphone, dominates: use wide smoothing.
    conf[grid >= 10000.0] = 1
    conf[grid >= 16000.0] = 0
    # Below 20 Hz there is no data worth trusting either.
    conf[grid < 20.0] = 0
    if measurement_freq is not None and len(measurement_freq):
        lo, hi = float(np.min(measurement_freq)), float(np.max(measurement_freq))
        outside = (grid < max(lo, 20.0) * 1.0001) | (grid > min(hi, 20000.0))
        conf[outside] = np.minimum(conf[outside], 1)
        conf[grid > min(hi, 20000.0)] = 0
    return conf


def analysis_width(grid=GRID, fine_below=7000.0, broad_above=9000.0,
                   narrow=1 / 6, broad=1.0):
    """Smoothing width used for the *correction target*, as a function of frequency.

    Below 7 kHz the measurement is trustworthy to ~1/6 octave and the headphone's
    real resonances live there.  Above it, coupler and ear-canal cancellation make
    narrow detail meaningless, so the width ramps to a full octave: what survives
    is the broad trend, which is the only thing worth correcting up there.
    """
    t = np.clip((np.log10(grid) - np.log10(fine_below)) / (np.log10(broad_above) - np.log10(fine_below)), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return narrow + (broad - narrow) * t


def band_edge_fade(correction, grid=GRID, edge_hz=16000.0, fade_oct=0.5):
    """Force the correction to 0 dB above the trustworthy band edge.

    A correction that still has value at 16 kHz forces the optimiser to place
    shapes right at Nyquist, where a biquad's own response is climbing and any
    filter costs a decibel of preamp for nothing audible.  Beyond the data the
    honest target is "no change", so the correction is faded to zero rather than
    extrapolated.
    """
    out = np.array(correction, float, copy=True)
    lo = edge_hz * (2.0 ** -fade_oct)
    t = np.clip((np.log10(grid) - np.log10(lo)) / (np.log10(edge_hz) - np.log10(lo)), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    out[(grid >= edge_hz)] = 0.0
    band = (grid >= lo) & (grid < edge_hz)
    out[band] = out[band] * (1.0 - t[band])
    return out


def build_correction(measured, target, grid=GRID, measured_freq=None, **kw):
    """correction[f] = target - smoothed(measured), plus the pieces the caller reports."""
    w = analysis_width(grid, **kw)
    stable = np.empty_like(measured)
    # Ramp the width in whole-point windows: cheap, and identical to a per-point
    # convolution because each point's window is centred on itself.
    oct_per_point = (np.log2(grid[-1]) - np.log2(grid[0])) / (len(grid) - 1)
    widths_pts = np.maximum(3, np.round(w / oct_per_point).astype(int))
    widths_pts += (1 - widths_pts % 2)
    for n in np.unique(widths_pts):
        sel = widths_pts == n
        stable[sel] = curves.smooth(measured, n * oct_per_point, grid)[sel]
    correction = band_edge_fade(target - stable, grid)
    return {
        "measured_smoothed": stable,
        "correction": correction,
        "widths": w,
        "confidence": confidence_map(grid, measured_freq),
    }


def tilt_target(base, grid=GRID, pivot_hz=1000.0, bass_db=0.0, treble_db=0.0,
                bass_freq=200.0, treble_freq=4000.0):
    """A target with a deliberate tonal tilt -- the gameplay-intent knob.

    A plain target asks for "flat to the reference".  Competitive play wants a
    different compromise (less low-end masking, more positional cues), and that
    is a *decision*, so it lives here as an explicit, documented offset rather
    than as unexplained extra filters.
    """
    out = np.array(base, float)
    log_g = np.log10(grid)
    if bass_db:
        # smooth low shelf, centred on bass_freq, one octave wide
        t = 1.0 / (1.0 + np.exp((log_g - np.log10(bass_freq)) * 6.0))
        out += bass_db * t
    if treble_db:
        t = 1.0 / (1.0 + np.exp((np.log10(treble_freq) - log_g) * 6.0))
        out += treble_db * t
    return out - curves.value_at(out, pivot_hz, grid)
