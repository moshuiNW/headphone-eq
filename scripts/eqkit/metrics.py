# -*- coding: utf-8 -*-
"""Residual metrics and safety checks.

A single "RMS error" number is easy to game: pile narrow high-Q notches onto a
comb-filtered measurement and it drops while the headphone sounds worse.  So the
score here is *banded*, weights differ per band, and every report carries the
worst point and the local deviation around it.  The bands and the acceptance
thresholds are declared once, in BANDS and GATES, and every caller reads them
from here -- change the judgement in one place only.
"""
import numpy as np

from .grid import GRID, index_of

# (name, lo_hz, hi_hz, weight, confidence)
#   weight     -- how much this band counts towards the headline RMS
#   confidence -- how much the *data* can support ("measured" / "trend" / "blind")
BANDS = (
    ("sub",        20.0,    60.0, 0.7, "measured"),
    ("bass",       60.0,   250.0, 1.0, "measured"),
    ("low-mid",   250.0,  1000.0, 1.0, "measured"),
    ("mid",      1000.0,  3000.0, 1.0, "measured"),
    ("presence", 3000.0,  5000.0, 1.0, "measured"),
    ("treble",   5000.0, 10000.0, 0.5, "trend"),
    ("air",     10000.0, 16000.0, 0.25, "trend"),
)

# The band the headline number is quoted on.
SCORE_LO = 20.0
SCORE_HI = 16000.0

# Acceptance gates.  These are the *judgement* of the skill, not a knob to tune
# per headphone: if a fit cannot meet them, report why instead of relaxing them.
GATES = {
    "rms_total": 1.0,          # dB, weighted, 20 Hz - 16 kHz
    "rms_core": 0.8,           # dB, weighted, 20 Hz - 5 kHz (the reliable part)
    "worst_core": 2.5,         # dB, absolute, 20 Hz - 5 kHz
    "worst_treble": 5.0,       # dB, absolute, 5 - 10 kHz
    "max_boost": 6.0,          # dB, any single filter's positive gain
    "max_boost_treble": 3.0,   # dB, positive gain above 5 kHz
    "max_cut": 20.0,           # dB, any single filter's negative gain
    "max_q_below_5k": 4.5,     # no needle shapes where the data is good
    "max_q_above_5k": 3.0,
    "min_preamp_headroom": 0.5,  # dB kept below 0 dBFS for inter-sample peaks
}

# Which gates the *fitter* is responsible for.
#
# The two residual gates (worst_core, worst_treble) are capability questions, not
# effort questions.  A +10 dB narrow dip at 3.8 kHz needs a +10 dB boost to fix,
# and boosting 3.8 kHz by 10 dB into someone's ear to chase one measurement is
# exactly the trade the boost cap exists to forbid.  So when the caps and the
# residual gate disagree, the caps win and the residual gate is reported as
# "not reachable within the safety constraints" rather than silently relaxed.
FIT_GATES = ("rms_total", "rms_core", "preamp", "filter_shape")


def band_weights(grid=GRID):
    w = np.zeros(len(grid), dtype=float)
    for _, lo, hi, weight, _ in BANDS:
        w[(grid >= lo) & (grid < hi)] = weight
    return w


def score_band(grid=GRID):
    return (grid >= SCORE_LO) & (grid < SCORE_HI)


def core_band(grid=GRID):
    return (grid >= SCORE_LO) & (grid < 5000.0)


def residual_stats(predicted, target, grid=GRID, mask=None):
    """Weighted RMS, worst point and per-band breakdown of predicted - target.

    The mean offset over the score band is removed first: level is free.
    """
    g = grid
    r = np.asarray(predicted, float) - np.asarray(target, float)
    m = score_band(g) if mask is None else mask
    r = r - float(np.mean(r[m]))

    w = band_weights(g)
    sel = m
    wsum = float(np.sum(w[sel]))
    rms_total = float(np.sqrt(np.sum(w[sel] * r[sel] ** 2) / wsum)) if wsum > 0 else float("nan")

    core = core_band(g) & m
    rms_core = float(np.sqrt(np.mean(r[core] ** 2))) if core.any() else float("nan")
    wc_i = int(np.argmax(np.abs(r[core]))) if core.any() else 0
    worst_core = float(r[core][wc_i]) if core.any() else float("nan")
    worst_core_hz = float(g[core][wc_i]) if core.any() else float("nan")

    tre = (g >= 5000.0) & (g < 10000.0) & m
    worst_treble = float(np.max(np.abs(r[tre]))) if tre.any() else float("nan")
    tre_i = int(np.argmax(np.abs(r[tre]))) if tre.any() else 0
    worst_treble_hz = float(g[tre][tre_i]) if tre.any() else float("nan")

    bands = []
    for name, lo, hi, weight, conf in BANDS:
        sel_b = (g >= lo) & (g < hi) & m
        if not sel_b.any():
            continue
        rb = r[sel_b]
        j = int(np.argmax(np.abs(rb)))
        bands.append({
            "name": name, "lo": lo, "hi": hi, "weight": weight, "confidence": conf,
            "rms": float(np.sqrt(np.mean(rb ** 2))),
            "mean": float(np.mean(rb)),
            "worst": float(rb[j]),
            "worst_hz": float(g[sel_b][j]),
        })
    return {
        "offset_db": float(np.mean(np.asarray(predicted, float)[m] - np.asarray(target, float)[m])),
        "rms_weighted": rms_total,
        "rms_core": rms_core,
        "worst_core": worst_core,
        "worst_core_hz": worst_core_hz,
        "worst_treble": worst_treble,
        "worst_treble_hz": worst_treble_hz,
        "bands": bands,
        "residual": r,
    }


def chain_peak(filters, fs=48000.0, audible_hz=20000.0):
    """Peak of the IIR cascade over 5 Hz .. Nyquist, and where it sits.

    Searching only 20 Hz - 16 kHz is the classic mistake: low shelves accumulate
    below 20 Hz and the chain can boost down there, which a real signal with
    sub-bass content will clip on.  So the *magnitude* covers 5 Hz - Nyquist --
    that is what the DAC sees.

    The reported *location* is clamped to the audible range instead: a chain that
    is attenuation-only reaches its maximum at Nyquist purely because everything
    below it is attenuated more, and printing "peak at 23995 Hz" invites the
    reader to hunt for a filter that does not exist.
    """
    from .apo import chain_db
    from .grid import FINE
    g = FINE[FINE < fs / 2.0]
    r = chain_db(filters, g, fs)
    peak = float(np.max(r))
    i = int(np.argmax(r))
    audible = g <= audible_hz
    if audible.any() and r[i] > r[audible].max() + 1e-6:
        i = int(np.argmax(np.where(audible, r, -np.inf)))
    return peak, float(g[i])


def required_preamp(filters, fs=48000.0, headroom=None):
    """Preamp that keeps the whole chain at or below 0 dBFS minus headroom."""
    headroom = GATES["min_preamp_headroom"] if headroom is None else headroom
    peak, at = chain_peak(filters, fs)
    # The no-preamp case is decided by the peak alone: a cascade that never rises
    # above 0 dB cannot clip, so it needs nothing added.  (Testing peak + headroom
    # here asks for half a decibel on a chain that is already attenuation-only,
    # because the headroom term is positive even when the peak is negative.)
    if peak > 1e-6:
        # Round *down* to the 0.1 dB the config file can express, so rounding can
        # never eat the headroom it was computed to protect.
        import math
        return math.floor(-(peak + headroom) * 10.0) / 10.0, peak, at
    return 0.0, peak, at


def unreachable_notes(stats):
    """Where a residual gate is missed, say what it would have cost to meet it.

    This is the difference between a report that says "failed" and one that says
    "the 3.8 kHz dip needs +10.7 dB of boost, the cap is +6 dB, so it was left
    alone on purpose".
    """
    notes = []
    if abs(stats["worst_core"]) > GATES["worst_core"]:
        want = stats["worst_core"]
        if want > 0:      # the headphone is louder than target: capped by max_cut
            need = "a cut deeper than the %.0f dB cap" % GATES["max_cut"]
        else:
            need = "a boost of %.1f dB against a %.1f dB cap" % (abs(want), GATES["max_boost"])
        notes.append("worst 20 Hz-5 kHz residual %+.2f dB at %.0f Hz needs %s"
                     % (want, stats["worst_core_hz"], need))
    if abs(stats["worst_treble"]) > GATES["worst_treble"]:
        notes.append("worst 5-10 kHz residual %+.2f dB at %.0f Hz; beyond the data confidence there"
                     % (stats["worst_treble"], stats["worst_treble_hz"]))
    return notes


def filter_violations(filters):
    """Human-readable list of the constraint violations in a filter list."""
    out = []
    for f in filters:
        if f["kind"] == "highpass":
            continue
        above5k = f["f"] > 5000.0
        if f["gain"] > (GATES["max_boost_treble"] if above5k else GATES["max_boost"]) + 1e-9:
            out.append("%.0f Hz boost %+.1f dB exceeds the %.1f dB cap%s"
                       % (f["f"], f["gain"], GATES["max_boost_treble"] if above5k else GATES["max_boost"],
                          " above 5 kHz" if above5k else ""))
        if -f["gain"] > GATES["max_cut"]:
            out.append("%.0f Hz cut %+.1f dB exceeds the %.1f dB cap" % (f["f"], f["gain"], GATES["max_cut"]))
        if f["kind"] in ("lowshelf", "highshelf") and f["q"] < 0.6 - 1e-9:
            out.append("%.0f Hz shelf Q %.2f is below the 0.6 floor: the shape is not the shelf it claims to be"
                       % (f["f"], f["q"]))
        qcap = GATES["max_q_above_5k"] if above5k else GATES["max_q_below_5k"]
        if f["q"] > qcap + 1e-9:
            out.append("%.0f Hz Q %.2f exceeds the %.1f cap%s"
                       % (f["f"], f["q"], qcap, " above 5 kHz" if above5k else ""))
    return out


def gate_report(stats, filters, preamp, fs=48000.0):
    """Evaluate every GATE; returns (passed, rows, violations).

    passed means every gate in FIT_GATES holds, and every capability gate is
    either satisfied or explicitly accounted for in the returned rows -- callers
    print the rows, so a miss is visible instead of being absorbed.
    """
    peak, _at = chain_peak(filters, fs)
    combined = peak + preamp
    rows = []
    def add(name, ok, detail):
        rows.append({"gate": name, "ok": bool(ok), "detail": detail})

    add("rms_total", stats["rms_weighted"] <= GATES["rms_total"],
        "%.2f dB (limit %.2f)" % (stats["rms_weighted"], GATES["rms_total"]))
    add("rms_core", stats["rms_core"] <= GATES["rms_core"],
        "%.2f dB (limit %.2f)" % (stats["rms_core"], GATES["rms_core"]))
    add("worst_core", abs(stats["worst_core"]) <= GATES["worst_core"],
        "%+.2f dB @ %.0f Hz (limit %.2f)" % (stats["worst_core"], stats["worst_core_hz"], GATES["worst_core"]))
    add("worst_treble", abs(stats["worst_treble"]) <= GATES["worst_treble"],
        "%+.2f dB @ %.0f Hz (limit %.2f)" % (stats["worst_treble"], stats["worst_treble_hz"], GATES["worst_treble"]))
    # Two ways to be safe: the chain never rises above 0 dB at all, or the preamp
    # brings the combination under full scale with the inter-sample margin left over.
    never_boosts = peak <= 1e-6
    add("preamp", never_boosts or combined <= -GATES["min_preamp_headroom"] + 1e-9,
        "chain peak %+.2f dB, preamp %+.1f dB, combined %+.2f dB" % (peak, preamp, combined))
    viol = filter_violations(filters)
    add("filter_shape", not viol, "ok" if not viol else "%d violation(s)" % len(viol))
    passed = all(r["ok"] for r in rows if r["gate"] in FIT_GATES)
    return (passed, rows, viol)
