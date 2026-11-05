# -*- coding: utf-8 -*-
"""Compare the two configs for one headphone on one page.

The pair exists to be compared: the same headphone aimed at two different
reference curves.  Reading two separate plots and mentally subtracting them is
the hard part, so this renders both residuals against one axis, prints the
numbers that differ, and states where each config is the better choice.

    python scripts/compare.py out/<headphone>
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eqkit import apo, curves, metrics  # noqa: E402
from eqkit.grid import GRID  # noqa: E402


def load(cfg_path):
    parsed = apo.parse_config(cfg_path)
    preamp = float(sum(parsed["preamp"]))
    eq = apo.chain_db(parsed["filters"], GRID)
    peak, at = metrics.chain_peak(parsed["filters"])
    return {
        "path": cfg_path,
        "filters": parsed["filters"],
        "preamp": preamp,
        "eq": eq,
        "peak": peak,
        "peak_hz": at,
        "combined": peak + preamp,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare HARMAN and GAME configs")
    ap.add_argument("folder", help="output folder produced by generate.py")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.folder, "*-HARMAN.txt"))) + \
            sorted(glob.glob(os.path.join(args.folder, "*-GAME.txt")))
    if len(paths) != 2:
        raise SystemExit("expected one HARMAN and one GAME config in %s" % args.folder)
    a, b = load(paths[0]), load(paths[1])

    print("config            filters  peak dB  preamp dB  combined dB")
    for tag, c in (("HARMAN", a), ("GAME", b)):
        print("%-16s  %5d  %+7.2f  %+9.1f  %+11.2f"
              % (tag, len(c["filters"]), c["peak"], c["preamp"], c["combined"]))

    diff = b["eq"] - a["eq"]
    m = (GRID >= 20) & (GRID <= 16000)
    print("")
    print("where the two configs differ (GAME minus HARMAN), by band:")
    print("%-10s %-14s %8s %8s" % ("band", "range", "mean", "max"))
    for name, lo, hi, _w, _c in metrics.BANDS:
        sel = (GRID >= lo) & (GRID < hi) & m
        if not sel.any():
            continue
        d = diff[sel]
        j = int(np.argmax(np.abs(d)))
        print("%-10s %5.0f-%-6.0f  %+7.2f  %+7.2f @ %.0f"
              % (name, lo, hi, float(np.mean(d)), float(d[j]), GRID[sel][j]))

    fig_path = os.path.join(args.folder, "compare.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        wanted = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")
        have = {f.name for f in font_manager.fontManager.ttflist}
        picked = [w for w in wanted if w in have]
        if picked:
            plt.rcParams["font.sans-serif"] = picked + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.semilogx(GRID[m], a["eq"][m], color="#1f77b4", lw=1.8, label="HARMAN config EQ")
        ax.semilogx(GRID[m], b["eq"][m], color="#d62728", lw=1.8, label="GAME config EQ")
        ax.semilogx(GRID[m], diff[m], color="#333", lw=1.0, ls=":", label="difference (GAME - HARMAN)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlim(20, 16000)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("EQ gain (dB)")
        ax.set_title("Two targets, one headphone: what actually changes")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        print("")
        print("plot: %s" % fig_path)
    except Exception as exc:
        print("plot skipped: %s" % exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
