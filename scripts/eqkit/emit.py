# -*- coding: utf-8 -*-
"""Turning a fitted filter list into files a human can read, trust and edit.

A generated config is a *document*, not a data dump.  The header states what
the file was fitted to, from which measurement, on which date, with which
caveats -- because six months later the only thing telling two similarly named
files apart is that header.  Filters are ordered by frequency, low to high, and
each carries a one-line comment saying which band it is fixing, so a user can
find "the 2.8 kHz resonance" without decoding numbers.
"""
import io
import time

import numpy as np

from .grid import GRID
from . import apo


def band_label(freq):
    if freq < 60:
        return "sub-bass"
    if freq < 250:
        return "bass"
    if freq < 1000:
        return "low-mid"
    if freq < 3000:
        return "mid"
    if freq < 5000:
        return "presence"
    if freq < 10000:
        return "treble"
    return "air"


def order_filters(filters):
    """High-pass first (it belongs to the safety chain), then ascending frequency."""
    hp = [f for f in filters if f["kind"] == "highpass"]
    rest = [f for f in filters if f["kind"] != "highpass"]
    return hp + sorted(rest, key=lambda f: f["f"])


def intent(filt, before, after, grid=GRID):
    """One short sentence: what this filter changes, at its own centre."""
    i = int(np.argmin(np.abs(np.log10(grid) - np.log10(filt["f"]))))
    delta = float(after[i] - before[i])
    direction = "cuts" if delta < 0 else "raises"
    return "%s %s by %.1f dB" % (direction, band_label(filt["f"]), abs(delta))


def render_config(filters, preamp, header):
    """Config text: header comments, Preamp, then filters with intent comments."""
    lines = ["# " + h if h else "#" for h in header]
    lines.append("Preamp: %+.1f dB" % preamp)
    lines.append("")
    for f in order_filters(filters):
        if f["kind"] == "highpass":
            note = "rumble filter: nothing below this is reproduced, removing it buys headroom"
        else:
            # The marginal effect of this filter alone: cumulative chains make
            # every later comment describe the sum of everything before it.
            note = intent(f, np.zeros_like(GRID), apo.chain_db([f], GRID))
        lines.append("# %-8s %s" % (band_label(f["f"]), note))
        lines.append(apo.format_filter(f))
    return "\n".join(lines) + "\n"


def write_config(path, filters, preamp, header):
    text = render_config(filters, preamp, header)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def plot(path, measured, target, predicted, stats, title, measured_label="measured (smoothed)"):
    """Three-line overlay plus the residual panel.

    The residual panel is the point: it turns "looks close" into a number a
    reader can disagree with, band by band, and it shows the two gate lines so
    a failing band is obvious without reading the table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Auto-generated titles carry Chinese headphone names; without a CJK face
    # matplotlib silently draws boxes instead of the model name.
    wanted = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
              "PingFang SC", "WenQuanYi Zen Hei")
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [w for w in wanted if w in available]
    if picked:
        plt.rcParams["font.sans-serif"] = picked + list(plt.rcParams["font.sans-serif"])
        plt.rcParams["axes.unicode_minus"] = False

    m = (GRID >= 20) & (GRID <= 16000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1.0]})

    def shape(y):
        y = np.asarray(y, float)
        return (y - float(np.mean(y[m])))[m]

    ax1.semilogx(GRID[m], shape(measured), color="#9aa0a6", lw=1.1, label=measured_label)
    ax1.semilogx(GRID[m], shape(target), color="#111", lw=1.6, ls="--", label="target")
    ax1.semilogx(GRID[m], shape(predicted), color="#0a7d32", lw=2.2, label="predicted after EQ")
    ax1.set_ylabel("dB (shape-normalised)")
    ax1.set_title(title)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower left", fontsize=9)

    resid = stats["residual"]
    ax2.semilogx(GRID[m], resid[m], color="#c0392b", lw=1.3, label="residual (after EQ)")
    ax2.axhline(0, color="#555", lw=0.8)
    for gate, color in ((2.5, "#e67e22"), (5.0, "#c0392b")):
        ax2.axhline(gate, color=color, lw=0.7, ls=":")
        ax2.axhline(-gate, color=color, lw=0.7, ls=":")
    ax2.set_ylabel("residual (dB)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_xlim(20, 16000)

    txt = "weighted RMS %.2f dB | core %.2f dB | worst core %+.2f dB @ %.0f Hz" % (
        stats["rms_weighted"], stats["rms_core"], stats["worst_core"], stats["worst_core_hz"])
    fig.text(0.01, 0.005, txt, fontsize=9, color="#333")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def report_markdown(path, ctx):
    """The written half of the deliverable: numbers, provenance and caveats."""
    s = ctx["stats"]
    out = []
    out.append("# %s" % ctx["title"])
    out.append("")
    out.append("Generated %s by headphone-eq." % time.strftime("%Y-%m-%d %H:%M"))
    out.append("")
    out.append("## Fit")
    out.append("")
    out.append("| item | value |")
    out.append("|---|---|")
    out.append("| target | %s |" % ctx["target_name"])
    out.append("| measurement | %s |" % ctx["measurement_note"])
    out.append("| filters | %d |" % len(ctx["filters"]))
    out.append("| weighted RMS 20 Hz-16 kHz | **%.2f dB** |" % s["rms_weighted"])
    out.append("| RMS 20 Hz-5 kHz | %.2f dB |" % s["rms_core"])
    out.append("| worst 20 Hz-5 kHz | %+.2f dB @ %.0f Hz |" % (s["worst_core"], s["worst_core_hz"]))
    out.append("| worst 5-10 kHz | %+.2f dB @ %.0f Hz |" % (s["worst_treble"], s["worst_treble_hz"]))
    out.append("| chain peak | %+.2f dB |" % ctx["peak"])
    out.append("| preamp | %+.1f dB |" % ctx["preamp"])
    out.append("")
    out.append("## Per band")
    out.append("")
    out.append("| band | range | weight | confidence | RMS | worst |")
    out.append("|---|---|---|---|---|---|")
    for b in s["bands"]:
        out.append("| %s | %.0f-%.0f Hz | %.1f | %s | %.2f dB | %+.2f dB @ %.0f Hz |"
                   % (b["name"], b["lo"], b["hi"], b["weight"], b["confidence"],
                      b["rms"], b["worst"], b["worst_hz"]))
    out.append("")
    if ctx.get("caveats"):
        out.append("## What this fit cannot fix")
        out.append("")
        for c in ctx["caveats"]:
            out.append("- %s" % c)
        out.append("")
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return path
