# -*- coding: utf-8 -*-
"""Verify a delivered config against its target, starting from the file on disk.

Everything here re-reads the .txt as Equalizer APO would: parse the Preamp,
parse each Filter line, rebuild the cascade with the APO biquad model, and
score it.  Nothing is reused from the generation run, so a config that was
hand-edited after generation still gets verified as edited -- which is the
point, because hand-editing is the supported workflow.

    python scripts/verify.py out/<headphone>/<name>-HARMAN.txt --target harman \
        --uuid <evaluation> --variant "频响"
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eqkit import analyse, apo, curves, fetch, metrics, repo  # noqa: E402
from eqkit.grid import GRID  # noqa: E402

TARGETS = {"harman": "harman_over_ear_2018.csv", "game": "blue-dashed.csv"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="verify a delivered APO config")
    ap.add_argument("config", help="the .txt to verify")
    ap.add_argument("--target", choices=sorted(TARGETS), default="harman")
    ap.add_argument("--uuid", help="huihifi evaluation the config was fitted to")
    ap.add_argument("--variant", default="频响", help="which response curve of that evaluation")
    ap.add_argument("--curve", help="or your own measurement CSV")
    ap.add_argument("--proxy")
    args = ap.parse_args(argv)

    parsed = apo.parse_config(args.config)
    preamp = float(sum(parsed["preamp"]))
    filters = parsed["filters"]
    print("config     : %s" % args.config)
    print("includes   : %s" % (", ".join(parsed["includes"]) or "-"))
    print("filters    : %d" % len(filters))
    print("preamp     : %+.1f dB" % preamp)

    peak, at = metrics.chain_peak(filters)
    print("chain peak : %+.2f dB at %.0f Hz" % (peak, at))
    print("combined   : %+.2f dB  (headroom %.2f dB)" % (peak + preamp, -(peak + preamp)))
    if peak + preamp > 0:
        print("  PROBLEM: the chain can exceed 0 dBFS and will clip")

    # measurement
    if args.uuid:
        rec = fetch.huihifi_record(args.uuid, proxy=args.proxy)
        variant = None
        for v in rec["variants"]:
            if args.variant in v["name"] or args.variant in v.get("label", ""):
                variant = v
                break
        if variant is None:
            raise SystemExit("variant %r not found; available: %s"
                             % (args.variant, "; ".join(v.get("label", v["name"]) for v in rec["variants"])))
        freq, level = variant["freq"], variant["level"]
    elif args.curve:
        freq, level = curves.read_csv(args.curve)
    else:
        raise SystemExit("give --uuid or --curve so the residual can be computed")

    meas = curves.normalise(curves.resample(freq, level))
    tf, tl = curves.read_csv(os.path.join(repo.DATA, TARGETS[args.target]))
    target = curves.normalise(curves.resample(tf, tl))
    built = analyse.build_correction(meas, target, measured_freq=freq)
    target = analyse.band_edge_fade(target - built["measured_smoothed"], GRID) + built["measured_smoothed"]
    predicted = built["measured_smoothed"] + apo.chain_db(filters, GRID)
    stats = metrics.residual_stats(predicted, target)

    print("target     : %s" % TARGETS[args.target])
    print("")
    print("weighted RMS 20 Hz-16 kHz : %.2f dB" % stats["rms_weighted"])
    print("RMS 20 Hz-5 kHz          : %.2f dB" % stats["rms_core"])
    print("worst 20 Hz-5 kHz        : %+.2f dB at %.0f Hz" % (stats["worst_core"], stats["worst_core_hz"]))
    print("worst 5-10 kHz           : %+.2f dB at %.0f Hz" % (stats["worst_treble"], stats["worst_treble_hz"]))
    print("")
    print("%-10s %-14s %-8s %-8s" % ("band", "range", "RMS", "worst"))
    for b in stats["bands"]:
        print("%-10s %5.0f-%-6.0f  %6.2f   %+6.2f @ %.0f"
              % (b["name"], b["lo"], b["hi"], b["rms"], b["worst"], b["worst_hz"]))

    passed, rows, viol = metrics.gate_report(stats, filters, preamp)
    print("")
    for row in rows:
        print("  [%s] %-12s %s" % ("ok" if row["ok"] else "!!", row["gate"], row["detail"]))
    for note in metrics.unreachable_notes(stats):
        print("  -> %s" % note)
    print("")
    print("verdict: %s" % ("PASS" if passed and peak + preamp <= 0 else "CHECK"))
    return 0 if passed and peak + preamp <= 0 else 2


if __name__ == "__main__":
    sys.exit(main())
