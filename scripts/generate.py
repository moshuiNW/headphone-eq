
# -*- coding: utf-8 -*-
"""The one command line that produces both deliverables.

    python scripts/generate.py --headphone "HIFIMAN Sundara"
    python scripts/generate.py --curve data/headphones/x/curve.csv --label "My HD600"

Both targets are always produced, because the point of the pair is comparison:
the Harman config is the reference tuning, the game config is the same
headphone aimed at the reference curve the evaluation site draws as a blue
dashed line.  They are fitted independently -- same measurement, different
target, different weights -- so each keeps its own intent.
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eqkit import analyse, apo, curves, emit, fetch, fit, judge, metrics, repo  # noqa: E402
from eqkit.grid import GRID  # noqa: E402

# Target curves by role.  "reference" is the preference target everybody knows;
# "game" is the blue dashed curve from the evaluation site, which is what the
# user's second config has to follow.
TARGET_FILES = {
    "harman": "harman_over_ear_2018.csv",
    "game": "blue-dashed.csv",
}

# Weight profile per role.  This is the "same measurement, different intent"
# switch: the reference fit chases fidelity across the whole reliable band,
# the game fit deliberately spends less accuracy in the sub-bass and more in
# the 1-4 kHz band where positional cues live.
TARGET_LABELS = {
    "harman": "Harman over-ear 2018",
    "game": "game reference curve (blue dashed)",
}

WEIGHTS = {
    "harman": {"sub": 0.7, "bass": 1.0, "low-mid": 1.0, "mid": 1.0, "presence": 1.0, "treble": 0.5, "air": 0.25},
    "game":   {"sub": 0.5, "bass": 0.9, "low-mid": 0.8, "mid": 1.4, "presence": 1.3, "treble": 0.5, "air": 0.2},
}


def load_target(role, explicit=None, proxy=None):
    """Read a target curve from the skill data dir (or an explicit path)."""
    if explicit:
        freq, level = curves.read_csv(explicit)
        return explicit, freq, level
    name = TARGET_FILES[role]
    path = os.path.join(repo.DATA, name)
    if not os.path.exists(path):
        raise SystemExit("missing target curve %s (see data/README.md)" % path)
    freq, level = curves.read_csv(path)
    return name, freq, level


def weights_for(role):
    w = np.zeros(len(GRID))
    for b in metrics.BANDS:
        name, lo, hi = b[0], b[1], b[2]
        w[(GRID >= lo) & (GRID < hi)] = WEIGHTS[role][name]
    return w


def resolve_source(args):
    """Return a record dict {"title","source","variants",...} plus a note string."""
    if args.curve:
        rec = fetch.from_file(args.curve)
        if args.label:
            rec["title"] = args.label
        return rec, None
    if args.uuid:
        return fetch.huihifi_record(args.uuid, proxy=args.proxy), None
    if not args.headphone:
        raise SystemExit("give --headphone, --uuid or --curve")
    hits = fetch.huihifi_search(args.headphone, proxy=args.proxy)
    if not hits:
        raise SystemExit(
            "no measurement found for %r on huihifi.\n"
            "Refusing to fit: a config built from a different headphone is worse than none.\n"
            "Supply your own curve with --curve, or a known evaluation with --uuid." % args.headphone
        )
    ranked = judge.rank_for_query(hits, args.headphone)
    for entry in ranked:
        print("  candidate: %s | %s | modes=%s | penalty=%d %s"
              % (entry["record"]["brand"], entry["record"]["title"], entry["modes"],
                 entry["penalty"], "; ".join(entry["reasons"])))
    print("")
    chosen = ranked[0]
    # A fuzzy search returns siblings and unrelated products; fitting the best
    # -ranked *anything* would silently model a different headphone.  So the pick
    # must actually contain what the user typed.
    if not judge.name_matches(args.headphone, chosen["record"]["title"]):
        lines = ["%r is not in the title of the best match (%r)."
                 % (args.headphone, chosen["record"]["title"]), ""]
        lines.append("candidates found:")
        for entry in ranked[:8]:
            lines.append('  --uuid %s    %s | %s'
                         % (entry["record"]["uuid"], entry["record"]["brand"], entry["record"]["title"]))
        lines.append("")
        lines.append("Ask the user which one they mean, or pass --curve with their own measurement.")
        raise SystemExit("\n".join(lines))
    rec = fetch.huihifi_record(chosen["record"]["uuid"], proxy=args.proxy)
    return rec, chosen


def build_role(role, rec, variant, target_name, target_curve, out_dir, args):
    """Everything a role needs *before* the fit: curves, weights, the shaped target.

    Split from `finish_role` so both roles can hand their (target, weights) to one
    shared pool.  Nothing here depends on a filter, so it is cheap and stays serial.
    """
    meas = curves.normalise(curves.resample(variant["freq"], variant["level"]))
    target = curves.normalise(curves.resample(*target_curve))

    built = analyse.build_correction(meas, target, measured_freq=variant["freq"])
    correction = built["correction"]
    stable = built["measured_smoothed"]
    # Beyond the band edge the honest statement is "unchanged", so the achieved
    # curve is scored there too.  Otherwise the headphone's natural 16 kHz
    # roll-off shows up as a residual the EQ could only "fix" by boosting into a
    # region no measurement supports.
    target_eff = analyse.band_edge_fade(target - stable, GRID) + stable
    mask = metrics.score_band()
    # The chain should only ever attenuate: slide the correction down so its
    # peak sits at 0 dB.  This is why most configs need no preamp at all.
    offset = float(np.max(correction[mask]))
    return {
        "role": role,
        "target_name": "%s [%s]" % (TARGET_LABELS[role], target_name),
        "target": target_eff,
        "measured": stable,
        "correction": correction,
        "offset": offset,
        "weights": weights_for(role),
        "shaped": correction - offset,
    }


def finish_role(prep, best, all_counts, args):
    """Score the chosen chain and build the record the reports are written from."""
    role = prep["role"]
    target_eff = prep["target"]
    stable = prep["measured"]
    print("    filter-count scan (%s):" % role)
    for c in all_counts:
        print("      n=%2d -> %2d filters  RMS %.2f dB  worst-core %+.2f dB  maxcut %+.1f dB"
              % (c["n"], len(c["filters"]), c["stats"]["rms_weighted"],
                 c["stats"]["worst_core"],
                 min([f["gain"] for f in c["filters"]] or [0.0])))
    filters = list(best["filters"])
    if args.highpass:
        filters = [{"kind": "highpass", "f": 12.0, "gain": 0.0, "q": 0.71}] + filters

    preamp, peak, peak_at = metrics.required_preamp(filters)
    pred = apo.chain_db(filters, GRID)
    # Predict on the smoothed measurement: that is the curve the fit was computed
    # against, and the fine structure above 5 kHz it removes is fixture noise, not
    # something an EQ could or should chase.
    predicted = stable + pred
    stats = metrics.residual_stats(predicted, target_eff)
    passed, gate_rows, violations = metrics.gate_report(stats, filters, preamp)
    notes = metrics.unreachable_notes(stats)

    out = dict(prep)
    out.update({
        "filters": filters,
        "preamp": preamp,
        "peak": peak,
        "peak_at": peak_at,
        "predicted": predicted,
        "stats": stats,
        "gates": gate_rows,
        "violations": violations,
        "passed": passed,
        "unreachable": notes,
        "candidates": all_counts,
    })
    return out


def fit_finish(prep, args):
    """Fit every role through ONE pool, then score each result.

    Both roles share the pool, so a two-role run finishes in the time of its slowest
    candidate rather than the sum of two independent scans.
    """
    specs = [(role, {"target": p["shaped"], "weights": p["weights"],
                     "counts": tuple(args.counts), "role": role})
             for role, p in prep.items()]
    done = fit.build_many(specs, jobs=args.jobs)
    return {role: finish_role(prep[role], *done[role], args) for role in prep}


def slug_for(rec, args):
    base = args.label or rec.get("title") or "headphone"
    if rec.get("brand"):
        base = "%s %s" % (rec["brand"], base)
    return repo.slugify(base)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit Equalizer APO configs to a headphone measurement.")
    ap.add_argument("--headphone", help="model name to search on huihifi")
    ap.add_argument("--uuid", help="a specific huihifi evaluation uuid")
    ap.add_argument("--curve", help="path to your own frequency-response CSV")
    ap.add_argument("--label", help="name to use for the output folder")
    ap.add_argument("--mode", help="tuning mode / variant name to fit, e.g. \"7.1\"")
    ap.add_argument("--counts", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18])
    ap.add_argument("--highpass", action="store_true", default=True)
    ap.add_argument("--no-highpass", dest="highpass", action="store_false")
    ap.add_argument("--proxy", help="http proxy for fetching")
    ap.add_argument("--out", help="output directory (default out/<slug>)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes per fit (default: cores, capped to the work); 1 = serial")
    ap.add_argument("--list-curves", action="store_true",
                    help="list the response curves of the selected evaluation and stop")
    args = ap.parse_args(argv)

    repo.ensure_dirs()
    rec, chosen = resolve_source(args)

    if args.list_curves:
        print("evaluation : %s" % rec.get("title"))
        print("brand      : %s" % rec.get("brand"))
        print("url        : %s" % rec.get("url"))
        print("modes      : %s" % ", ".join(rec.get("modes") or []) or "-")
        print("curves     :")
        for i, v in enumerate(rec.get("variants") or []):
            print("  [%d] %s   (%d points, %.0f-%.0f Hz)"
                  % (i, v.get("label") or v["name"], len(v["freq"]), v["freq"].min(), v["freq"].max()))
        return 0

    variant, why = judge.pick_variant(rec, args.mode)
    if variant is None:
        lines = ["refusing to fit: %s" % why]
        variants = rec.get("variants") or []
        if variants:
            lines.append("")
            lines.append("this evaluation publishes %d curves; fit the one the user means:" % len(variants))
            for v in variants:
                lines.append("  --mode \"%s\"    %s" % (v.get("label") or v["name"], v.get("note", "")))
            lines.append("")
            lines.append("ask the user which curve to use, then re-run with --mode.")
        else:
            lines.append("")
            lines.append("supply a curve with --curve, or a different evaluation with --uuid.")
        raise SystemExit("\n".join(lines))
    print("measurement: %s (%s)" % (rec.get("title"), why))

    harman_name, harman_freq, harman_level = load_target("harman")
    game_name, game_freq, game_level = load_target("game")

    meas = curves.normalise(curves.resample(variant["freq"], variant["level"]))
    ok, reasons, gap = judge.eligibility(meas, curves.normalise(curves.resample(harman_freq, harman_level)),
                                         variant["freq"])
    if not ok:
        for r in reasons:
            print("  ! %s" % r)
        raise SystemExit("refusing to fit: the measurement does not support a config")

    slug = slug_for(rec, args)
    out_dir = args.out or repo.out_dir(slug)
    os.makedirs(out_dir, exist_ok=True)

    # The two roles are independent fits from the same measurement, and within each
    # role every filter-count candidate is an independent optimisation.  Both facts
    # together mean the unit of work is the *candidate*, not the role: flattening
    # (role, count) into one queue keeps every core busy from the first second to the
    # last.  Running the roles one after the other instead leaves the machine half
    # idle for the second half of the wall clock.
    jobs = [("harman", harman_name, harman_freq, harman_level),
            ("game", game_name, game_freq, game_level)]
    args.jobs = getattr(args, "jobs", 0) or 0
    prep = {}
    for role, name, tf, tl in jobs:
        prep[role] = build_role(role, rec, variant, name, (tf, tl), out_dir, args)
    results = fit_finish(prep, args)
    for role, r in results.items():
        print("  %-6s %2d filters  RMS %.2f dB  worst-core %+.2f dB  peak %+.2f dB  preamp %+.1f dB  %s"
              % (role, len(r["filters"]), r["stats"]["rms_weighted"], r["stats"]["worst_core"],
                 r["peak"], r["preamp"], "PASS" if r["passed"] else "CHECK"))
        for gate in r["gates"]:
            if not gate["ok"]:
                print("         ! %-12s %s" % (gate["gate"], gate["detail"]))
        for n in r["unreachable"]:
            print("         - %s" % n)

    # persist the raw data + provenance so the run can be replayed offline
    repo.write_meta(slug, {
        "title": rec.get("title"),
        "brand": rec.get("brand"),
        "source": rec.get("source"),
        "url": rec.get("url"),
        "uuid": rec.get("uuid"),
        "fetched_at": rec.get("fetched_at"),
        "variant": variant["name"],
        "variant_reason": why,
        "specifications": rec.get("specifications"),
        "modes": rec.get("modes"),
        "targets": {"harman": harman_name, "game": game_name},
    })
    curves.write_csv(repo.variant_path(slug, variant["name"]), variant["freq"], variant["level"],
                     ("freq_hz", "spl_db"))

    fit.shutdown_pool()
    written = emit_all(slug, out_dir, rec, variant, results, args)
    print("")
    print("wrote:");
    for w_ in written:
        print("  %s" % w_)
    return 0 if all(r["passed"] for r in results.values()) else 2


def emit_all(slug, out_dir, rec, variant, results, args):
    written = []
    base = slug.replace("/", "-")
    for role, r in results.items():
        tag = "HARMAN" if role == "harman" else "GAME"
        cfg_path = os.path.join(out_dir, "%s-%s.txt" % (base, tag))
        header = [
            "%s -> %s" % (rec.get("title") or base, TARGET_LABELS[role]),
            "measurement : %s %s" % (rec.get("source"), rec.get("url") or ""),
            "variant    : %s" % variant["name"],
            "fetched    : %s" % rec.get("fetched_at", "unknown"),
            "target     : %s" % r["target_name"],
            "fitted     : weighted RMS %.2f dB over 20 Hz-16 kHz, worst %.2f dB"
                          % (r["stats"]["rms_weighted"], abs(r["stats"]["worst_core"])),
            "peak       : %s -> Preamp %+.1f dB keeps the chain at or below 0 dBFS"
                          % (("-%.2f dB (attenuation only, %+.2f dB at %.0f Hz)"
                              % (abs(r["peak"]), r["peak"], r["peak_at"])) if r["peak"] < 0.05
                             else "+%.2f dB at %.0f Hz" % (r["peak"], r["peak_at"]),
                             r["preamp"]),
            "verified   : Equalizer APO biquad model (filters/BiQuad.cpp), 48 kHz",
            "caveat     : above 5 kHz this is a trend correction, not a fit",
        ]
        emit.write_config(cfg_path, r["filters"], r["preamp"], header)
        written.append(cfg_path)

        png = os.path.join(out_dir, "%s-%s.png" % (base, tag))
        emit.plot(png, r["measured"], r["target"], r["predicted"], r["stats"],
                  "%s -> %s" % (rec.get("title") or base, TARGET_LABELS[role]))
        written.append(png)

        md = os.path.join(out_dir, "%s-%s.md" % (base, tag))
        emit.report_markdown(md, {
            "title": "%s -> %s" % (rec.get("title") or base, TARGET_LABELS[role]),
            "target_name": r["target_name"],
            "measurement_note": "%s, variant %r" % (rec.get("url") or rec.get("source"), variant["name"]),
            "filters": r["filters"],
            "stats": r["stats"],
            "peak": r["peak"],
            "preamp": r["preamp"],
            "caveats": judge.caveats(rec, variant, variant["freq"], r["stats"]) + r["unreachable"],
        })
        written.append(md)

    summary = os.path.join(out_dir, "summary.json")
    payload = {
        "headphone": rec.get("title"),
        "source": rec.get("source"),
        "url": rec.get("url"),
        "variant": variant["name"],
        "targets": {role: r["target_name"] for role, r in results.items()},
        "roles": {
            role: {
                "filters": r["filters"],
                "preamp": r["preamp"],
                "rms_weighted": r["stats"]["rms_weighted"],
                "rms_core": r["stats"]["rms_core"],
                "worst_core": r["stats"]["worst_core"],
                "gates": r["gates"],
                "violations": r["violations"],
                "passed": r["passed"],
            } for role, r in results.items()
        },
    }
    with io.open(summary, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    written.append(summary)
    return written


if __name__ == "__main__":
    sys.exit(main())
