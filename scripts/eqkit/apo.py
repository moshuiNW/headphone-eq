# -*- coding: utf-8 -*-
"""Equalizer APO's biquads, transcribed from filters/BiQuad.cpp.

Two things make APO's filters differ from a generic audio-EQ cookbook, and both
bite:

  * LS / HS use the **Q form**, not the shelf-slope (S) form.  Writing
    "LS Fc 48 Hz Gain +3.8 dB Q 1.79" does not mean what an S-form shelf of the
    same numbers means -- the two disagree by up to ~6 dB.  Fitting with an
    S-form model and emitting Q therefore ships a wrong curve, which is exactly
    the trap the project notes record.
  * Coefficients are computed at the device sample rate (48 kHz here).  A 12 Hz
    high-pass at 44.1 kHz and at 192 kHz behave almost identically, but the
    preamp must be derived from the worst case across rates.

Everything downstream (fitting, verification, peak hunting) uses this module and
nothing else, so "what we fitted" and "what APO plays" are the same model.
"""
import re
import numpy as np

FS_DEFAULT = 48000.0

# Equalizer APO filter tokens we emit or read back.
KIND_TO_TOKEN = {"peak": "PK", "lowshelf": "LS", "highshelf": "HS", "highpass": "HP"}
TOKEN_TO_KIND = {"PK": "peak", "PEQ": "peak", "LS": "lowshelf", "LSC": "lowshelf",
                 "HS": "highshelf", "HSC": "highshelf", "HP": "highpass", "HPQ": "highpass"}


def coeffs(kind, fc, gain_db, q, fs=FS_DEFAULT):
    """Normalised biquad coefficients (b0, b1, b2, a1, a2)."""
    if kind == "highpass":
        w0 = 2.0 * np.pi * fc / fs
        cs, sn = np.cos(w0), np.sin(w0)
        alpha = sn / (2.0 * q)
        b0 = (1.0 + cs) / 2.0
        b1 = -(1.0 + cs)
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cs
        a2 = 1.0 - alpha
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    sn, cs = np.sin(w0), np.cos(w0)
    alpha = sn / (2.0 * q)
    beta = 2.0 * np.sqrt(A) * alpha
    if kind == "peak":
        b0, b1, b2 = 1 + alpha * A, -2 * cs, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cs, 1 - alpha / A
    elif kind == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * cs + beta)
        b1 = 2 * A * ((A - 1) - (A + 1) * cs)
        b2 = A * ((A + 1) - (A - 1) * cs - beta)
        a0 = (A + 1) + (A - 1) * cs + beta
        a1 = -2 * ((A - 1) + (A + 1) * cs)
        a2 = (A + 1) + (A - 1) * cs - beta
    elif kind == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * cs + beta)
        b1 = -2 * A * ((A - 1) + (A + 1) * cs)
        b2 = A * ((A + 1) + (A - 1) * cs - beta)
        a0 = (A + 1) - (A - 1) * cs + beta
        a1 = 2 * ((A - 1) - (A + 1) * cs)
        a2 = (A + 1) - (A - 1) * cs - beta
    else:
        raise ValueError("unknown filter kind %r" % (kind,))
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def response_db(kind, fc, gain_db, q, grid, fs=FS_DEFAULT):
    """Magnitude response of one filter, in dB, on *grid*."""
    if kind != "highpass" and abs(gain_db) < 1e-12:
        return np.zeros_like(grid, dtype=float)
    b0, b1, b2, a1, a2 = coeffs(kind, fc, gain_db, q, fs)
    z = np.exp(-1j * 2.0 * np.pi * np.asarray(grid, float) / fs)
    num = b0 + b1 * z + b2 * z * z
    den = 1.0 + a1 * z + a2 * z * z
    return 20.0 * np.log10(np.maximum(np.abs(num / den), 1e-14))


_Z_CACHE = {}


def _z_powers(grid, fs):
    """z^-1 and z^-2 for a grid, cached: they depend only on (fs, grid)."""
    key = (fs, len(grid), float(grid[0]) if len(grid) else 0.0, float(grid[-1]) if len(grid) else 0.0)
    hit = _Z_CACHE.get(key)
    if hit is not None:
        return hit
    z = np.exp(-1j * 2.0 * np.pi * np.asarray(grid, float) / fs)
    val = (z[None, :], (z * z)[None, :])
    if len(_Z_CACHE) > 64:
        _Z_CACHE.clear()
    _Z_CACHE[key] = val
    return val


def responses_db(kinds, fcs, gains, qs, grid, fs=FS_DEFAULT):
    """Cascade response for *many* filters at once: (n, len(grid)) magnitudes summed.

    Vectorised because the optimiser evaluates this hundreds of thousands of times:
    a Python loop over 12 filters made one Jacobian column cost more than the whole
    science of the fit.  All coefficients are built as arrays and the z-plane
    evaluation is a single broadcast, which is the same arithmetic, 20x cheaper.
    """
    kinds = np.asarray(kinds, dtype=object)
    fcs = np.asarray(fcs, float)
    gains = np.asarray(gains, float)
    qs = np.asarray(qs, float)
    n = len(fcs)
    if n == 0:
        return np.zeros((0, len(grid)))

    A = 10.0 ** (gains / 40.0)
    w0 = 2.0 * np.pi * fcs / fs
    sn, cs = np.sin(w0), np.cos(w0)
    alpha = sn / (2.0 * qs)
    beta = 2.0 * np.sqrt(A) * alpha
    one = np.ones(n)

    b0 = np.where(kinds == "peak", 1 + alpha * A, 0.0)
    b1 = np.where(kinds == "peak", -2 * cs, 0.0)
    b2 = np.where(kinds == "peak", 1 - alpha * A, 0.0)
    a0 = np.where(kinds == "peak", 1 + alpha / A, 1.0)
    a1 = np.where(kinds == "peak", -2 * cs, 0.0)
    a2 = np.where(kinds == "peak", 1 - alpha / A, 0.0)

    lo = kinds == "lowshelf"
    if lo.any():
        b0 = np.where(lo, A * ((A + 1) - (A - 1) * cs + beta), b0)
        b1 = np.where(lo, 2 * A * ((A - 1) - (A + 1) * cs), b1)
        b2 = np.where(lo, A * ((A + 1) - (A - 1) * cs - beta), b2)
        a0 = np.where(lo, (A + 1) + (A - 1) * cs + beta, a0)
        a1 = np.where(lo, -2 * ((A - 1) + (A + 1) * cs), a1)
        a2 = np.where(lo, (A + 1) + (A - 1) * cs - beta, a2)

    hi = kinds == "highshelf"
    if hi.any():
        b0 = np.where(hi, A * ((A + 1) + (A - 1) * cs + beta), b0)
        b1 = np.where(hi, -2 * A * ((A - 1) + (A + 1) * cs), b1)
        b2 = np.where(hi, A * ((A + 1) + (A - 1) * cs - beta), b2)
        a0 = np.where(hi, (A + 1) - (A - 1) * cs + beta, a0)
        a1 = np.where(hi, 2 * ((A - 1) - (A + 1) * cs), a1)
        a2 = np.where(hi, (A + 1) - (A - 1) * cs - beta, a2)

    hp = kinds == "highpass"
    if hp.any():
        ahp = sn / (2.0 * qs)
        b0 = np.where(hp, (1 + cs) / 2, b0)
        b1 = np.where(hp, -(1 + cs), b1)
        b2 = np.where(hp, (1 + cs) / 2, b2)
        a0 = np.where(hp, 1 + ahp, a0)
        a1 = np.where(hp, -2 * cs, a1)
        a2 = np.where(hp, 1 - ahp, a2)

    # A unity-gain peaking filter must be exactly transparent.  The test is skipped
    # entirely when no gain is near zero, which is the common case inside the
    # optimiser and costs an isclose() over the whole set on every Jacobian column.
    if np.any(np.abs(gains) < 1e-12):
        unity = (np.abs(gains) < 1e-12) & (kinds == "peak")
        b0 = np.where(unity, 1.0, b0)
        b1 = np.where(unity, 0.0, b1)
        b2 = np.where(unity, 0.0, b2)
        a0 = np.where(unity, 1.0, a0)
        a1 = np.where(unity, 0.0, a1)
        a2 = np.where(unity, 0.0, a2)

    z1, z2 = _z_powers(grid, fs)
    num = b0[:, None] + b1[:, None] * z1 + b2[:, None] * z2
    den = a0[:, None] + a1[:, None] * z1 + a2[:, None] * z2
    return 20.0 * np.log10(np.maximum(np.abs(num / den), 1e-14))


def chain_db(filters, grid, fs=FS_DEFAULT):
    """Cascade response of a filter list: [{kind, f, gain, q}, ...]."""
    if not filters:
        return np.zeros(len(grid), dtype=float)
    kinds = [f["kind"] for f in filters]
    fcs = [float(f["f"]) for f in filters]
    gains = [float(f.get("gain", 0.0)) for f in filters]
    qs = [float(f["q"]) for f in filters]
    return responses_db(kinds, fcs, gains, qs, grid, fs).sum(axis=0)


def chain_at(filters, f, fs=FS_DEFAULT):
    """Response of the cascade at a single frequency (for reporting)."""
    return float(chain_db(filters, np.array([float(f)]), fs)[0])


def q_from_bandwidth(octaves):
    """Q of a peaking filter whose -3 dB-ish skirt spans *octaves* (APO Q form)."""
    return 1.0 / (2.0 * np.sinh(np.log(2.0) / 2.0 * octaves))


# --------------------------------------------------------------- config I/O
FILTER_RE = re.compile(r"^filter\s*:\s*(?P<state>ON|OFF)\s+(?P<type>\w+)", re.I)


def parse_config(path, _depth=0):
    """Parse an APO config (following Include:) into {preamp, filters, graphiceq}.

    Returns the *as-deployed* chain: exactly the filters APO will run when this
    file is the active config.txt, so a verification pass reads the same numbers
    the user hears.
    """
    import os
    out = {"preamp": [], "filters": [], "graphiceq": [], "includes": []}
    if _depth > 4:
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("include:"):
                sub = line.split(":", 1)[1].strip()
                sub_path = os.path.join(os.path.dirname(os.path.abspath(path)), sub)
                out["includes"].append(sub)
                if os.path.exists(sub_path):
                    child = parse_config(sub_path, _depth + 1)
                    for key in ("preamp", "filters", "graphiceq", "includes"):
                        out[key].extend(child[key]) if key != "includes" else out["includes"].extend(child["includes"])
                continue
            if low.startswith("preamp:"):
                m = re.findall(r"[-+]?\d*\.?\d+", line)
                if m:
                    out["preamp"].append(float(m[0]))
                continue
            if low.startswith("graphiceq:"):
                out["graphiceq"].append(line)
                continue
            m = FILTER_RE.match(line)
            if not m or m.group("state").upper() != "ON":
                continue
            tokens = line.replace(":", " ").split()
            typ = m.group("type").upper()
            kind = TOKEN_TO_KIND.get(typ)
            if kind is None:
                continue
            def num(key, default=None):
                if key not in tokens:
                    return default
                i = tokens.index(key)
                return float(tokens[i + 1])
            fc = num("Fc")
            q = num("Q", 0.7071)
            if kind == "highpass":
                out["filters"].append({"kind": "highpass", "f": fc, "gain": 0.0, "q": q})
            else:
                out["filters"].append({"kind": kind, "f": fc, "gain": num("Gain", 0.0), "q": q})
    return out


def format_filter(flt):
    token = KIND_TO_TOKEN[flt["kind"]]
    if flt["kind"] == "highpass":
        return "Filter: ON HP Fc %.0f Hz Q %.2f" % (flt["f"], flt["q"])
    gain = flt["gain"]
    # APO accepts a leading '+', and keeping it makes the intent readable.
    return "Filter: ON %s Fc %.0f Hz Gain %+.1f dB Q %.2f" % (token, flt["f"], gain, flt["q"])
