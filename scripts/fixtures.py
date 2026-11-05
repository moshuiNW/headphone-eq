# -*- coding: utf-8 -*-
"""Regression fixtures: the fixed set of headphones every change is re-scored on.

The failure mode this exists for: a tweak that makes one model fit beautifully
while quietly making three others worse.  A frozen expectation file catches it,
because a "better" number on the model being stared at hides the damage done to
the others.

    python scripts/fixtures.py --record     # re-fit every fixture, write expectations
    python scripts/fixtures.py              # re-fit and compare against expectations

Each row stores the weighted RMS, the filter count and the gate verdict, so drift
shows up as a delta rather than as a pass/fail you have to interpret.
"""
import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eqkit import analyse, apo, curves, fetch, fit, metrics, repo  # noqa: E402
from eqkit.grid import GRID  # noqa: E402

FIXTURE_FILE = os.path.join(repo.FIXTURES, "cases.json")
EXPECT_FILE = os.path.join(repo.FIXTURES, "expected.json")

# The filter-count candidates every fixture is scored on.  Frozen with the
# expectations: a run with different counts is not comparable with the recorded
# numbers, so the suite pins them instead of exposing them as a flag.
COUNTS = (10, 12)

DEFAULT_CASES = [
    {"name": "edifier-g2pro", "uuid": "f7e7a082-1244-4894-829d-b7f7fb8b9720", "variant": "频响"},
    {"name": "hifiman-sundara", "uuid": "d738a18d-3d81-49af-9a25-985fa76ecc0f", "variant": "频响"},
    {"name": "hyperx-cloud2-71", "uuid": "4ec95765-455e-431d-ad84-9b4194ba5610", "variant": "7.1模式频响"},
    {"name": "hyperx-cloud2-stock", "uuid": "4ec95765-455e-431d-ad84-9b4194ba5610", "variant": "默认模式频响"},
    {"name": "moondrop-blessing3l", "uuid": "3f5848c8-e0a0-4b57-a62a-3954b17cfbff", "variant": "频响"},
]


def load_cases():
    if os.path.exists(FIXTURE_FILE):
        with io.open(FIXTURE_FILE, encoding="utf-8") as fh:
            return json.load(fh)["cases"]
    return DEFAULT_CASES


def target_curve(role):
    name = "harman_over_ear_2018.csv" if role == "harman" else "blue-dashed.csv"
    freq, level = curves.read_csv(os.path.join(repo.DATA, name))
    return name, curves.normalise(curves.resample(freq, level))


def _case_specs(case, proxy, counts, roles=("harman", "game")):
    """Fetch one fixture once and build the fit spec for every role it is scored on.

    Fetching per role would double the network round trips for the same measurement,
    and a case is the natural unit of parallelism: one process owns one headphone and
    fits both of its roles through a single fit.build_many call, so the processes the
    suite starts stay busy instead of each blocking on a nested pool of its own."""
    rec = fetch.huihifi_record(case["uuid"], proxy=proxy)
    variant = None
    for v in rec["variants"]:
        if v["name"] == case["variant"] or case["variant"] in v.get("label", ""):
            variant = v
            break
    if variant is None:
        raise RuntimeError("variant %r missing" % case["variant"])
    meas = curves.normalise(curves.resample(variant["freq"], variant["level"]))
    specs, built = [], {}
    for role in roles:
        _name, target = target_curve(role)
        b = analyse.build_correction(meas, target, measured_freq=variant["freq"])
        target = analyse.band_edge_fade(target - b["measured_smoothed"], GRID) + b["measured_smoothed"]
        correction = b["correction"]
        offset = float(np.max(correction[metrics.score_band()]))
        specs.append((role, {"target": correction - offset, "counts": tuple(counts), "role": role}))
        built[role] = {"smoothed": b["measured_smoothed"], "target": target}
    return variant, specs, built


def _row(case, role, filters, built):
    preamp, peak, _at = metrics.required_preamp(filters)
    pred = apo.chain_db(filters, GRID)
    stats = metrics.residual_stats(built["smoothed"] + pred, built["target"])
    passed, _rows, viol = metrics.gate_report(stats, filters, preamp)
    return {
        "name": case["name"],
        "role": role,
        "filters": len(filters),
        "rms": round(stats["rms_weighted"], 3),
        "rms_core": round(stats["rms_core"], 3),
        "worst_core": round(stats["worst_core"], 2),
        "peak": round(peak, 2),
        "preamp": preamp,
        "passed": bool(passed),
        "violations": viol,
    }


def run_case(case, role="harman", proxy=None, counts=(10, 12)):
    """One (fixture, role) fit.  Kept for interactive use; the suite runs _run_case."""
    _variant, specs, built = _case_specs(case, proxy, counts, roles=(role,))
    best, _all = fit.build_many(specs)[role]
    return _row(case, role, list(best["filters"]), built[role])


def _run_case(item):
    """One fixture, every role: both roles on the same measurement, one process."""
    case, proxy, counts = item
    roles = ("harman", "game")
    try:
        _variant, specs, built = _case_specs(case, proxy, counts)
        out = fit.build_many(specs)
        return [_row(case, role, list(out[role][0]["filters"]), built[role]) for role, _spec in specs]
    except Exception as exc:
        return [{"name": case["name"], "role": role, "error": str(exc)} for role in roles]



def main(argv=None):
    ap = argparse.ArgumentParser(description="fixture regression run")
    ap.add_argument("--record", action="store_true", help="write the expectation file")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (default: cores, capped)")
    ap.add_argument("--proxy")
    ap.add_argument("--tolerance", type=float, default=0.15, help="allowed RMS drift in dB")
    args = ap.parse_args(argv)
    repo.ensure_dirs()

    # One process per fixture, each fitting both of its roles: the fixtures are
    # independent, so this is the whole parallelism the suite needs.  It stays one
    # level deep on purpose -- a worker that opens a pool of its own deadlocks the
    # moment its children want the same cores.
    cases = load_cases()
    work = [(case, args.proxy, COUNTS) for case in cases]
    jobs = max(1, min(args.jobs or (os.cpu_count() or 1), len(work)))
    if jobs > 1:
        try:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                rows = [row for batch in pool.map(_run_case, work) for row in batch]
        except Exception:
            rows = [row for w in work for row in _run_case(w)]
    else:
        rows = [row for w in work for row in _run_case(w)]
    fit.shutdown_pool()

    if args.record:
        repo.write_json(EXPECT_FILE, {"tolerance": args.tolerance, "rows": rows})
        print("recorded %d rows -> %s" % (len(rows), EXPECT_FILE))
        return 0

    if not os.path.exists(EXPECT_FILE):
        print("no expectations recorded yet; run with --record first")
        return 1

    with io.open(EXPECT_FILE, encoding="utf-8") as fh:
        expected = {(r["name"], r["role"]): r for r in json.load(fh)["rows"]}

    bad = 0
    print("%-24s %-7s %8s %8s %6s %s" % ("fixture", "role", "rms", "delta", "filt", "verdict"))
    for row in rows:
        if "error" in row:
            print("%-24s %-7s  ERROR %s" % (row["name"], row["role"], row["error"]))
            bad += 1
            continue
        exp = expected.get((row["name"], row["role"]))
        if exp is None:
            print("%-24s %-7s %8.2f %8s %6d %s" % (row["name"], row["role"], row["rms"], "-", row["filters"], "new"))
            continue
        delta = row["rms"] - exp["rms"]
        verdict = "ok"
        if abs(delta) > args.tolerance:
            verdict = "DRIFT"
            bad += 1
        if not row["passed"]:
            verdict += " GATE-FAIL"
            bad += 1
        print("%-24s %-7s %8.2f %+8.2f %6d %s" % (row["name"], row["role"], row["rms"], delta, row["filters"], verdict))
    print("")
    print("%d regression(s)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
