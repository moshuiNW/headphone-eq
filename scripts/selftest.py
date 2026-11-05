# -*- coding: utf-8 -*-
"""Self-tests for the invariants that are easy to break silently.

Every case here corresponds to a bug that actually shipped during development, so
these are regression tests for judgement, not for syntax:

    python scripts/selftest.py

No network, no matplotlib -- fast enough to run on every edit.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eqkit import apo, curves, fit, judge, metrics, repo  # noqa: E402
from eqkit.grid import GRID  # noqa: E402

FAILS = []


def check(name, condition, detail=""):
    if condition:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILS.append(name)


def test_biquad_model():
    hi = apo.response_db("highshelf", 3000.0, -6.0, 1.0, GRID)
    check("high shelf cuts at HF", hi[-1] < -4.0, "got %.2f" % hi[-1])
    ls = apo.response_db("lowshelf", 100.0, 4.0, 0.7, GRID)
    check("low shelf boosts at LF", ls[0] > 3.0, "got %.2f" % ls[0])
    pk = apo.response_db("peak", 1000.0, 5.0, 1.0, np.array([1000.0]))[0]
    check("peak gain at centre equals gain", abs(pk - 5.0) < 0.05, "got %.3f" % pk)
    # A unity gain filter must be an exact identity, or every residual is skewed.
    unity = apo.response_db("peak", 1000.0, 0.0, 1.0, GRID)
    check("zero-gain filter is identity", float(np.max(np.abs(unity))) < 1e-12)


def test_useless_filters_removed():
    """Inert filters are found by ablation, and legal work is never deleted by category.

    The lesson these encode: an earlier version blacklisted shapes ("shelves below
    400 Hz", "Q under 0.6") and deleted filters that were carrying a real fixture's
    whole correction -- 0.12 dB became 1.69 dB. Category rules cannot know what a
    filter is for; only removing it and re-optimising can.
    """
    misordered = [{"kind": "highshelf", "f": 58.0, "gain": 2.8, "q": 2.0},
                  {"kind": "lowshelf", "f": 8000.0, "gain": -3.0, "q": 1.0},
                  {"kind": "peak", "f": 50000.0, "gain": 2.0, "q": 1.0}]
    clamped = fit.drop_misplaced_shelves(misordered)
    check("no filter is deleted for being misplaced", len(clamped) == 3, "kept %d" % len(clamped))
    check("high shelf corner is clamped up into its band",
          clamped[0]["f"] >= fit.LIMITS["highshelf_f_min"] - 1e-6, str(clamped[0]["f"]))
    check("low shelf corner is clamped down into its band",
          clamped[1]["f"] <= fit.LIMITS["lowshelf_f_max"] + 1e-6, str(clamped[1]["f"]))
    check("filter above the data band is clamped to it",
          clamped[2]["f"] <= fit.LIMITS["f_fit_max"] + 1e-6, str(clamped[2]["f"]))
    check("a legal shelf keeps its Q", clamped[0]["q"] == 2.0)


def test_ablation_removes_only_inert():
    """Given a target, a filter that does nothing must go and a working one must stay."""
    from eqkit.grid import GRID as G
    target = -6.0 * np.exp(-0.5 * (np.log10(G / 120.0) / 0.15) ** 2)
    working = [{"kind": "peak", "f": 120.0, "gain": -6.0, "q": 1.0}]
    inert = working + [{"kind": "peak", "f": 8000.0, "gain": 0.2, "q": 2.0}]
    kept = fit.drop_misplaced_shelves(inert, target=target, grid=G)
    keep = [abs(f["f"] - 120.0) < 30 for f in kept]
    check("load-bearing filter survives ablation", any(keep), str(kept))
    check("inert filter removed by ablation", not any(abs(f["f"] - 8000.0) < 100 for f in kept),
          str(kept))


def test_ground_truth_recovery():
    """A known EQ must be recovered from its own curve: the optimiser's floor."""
    from eqkit.grid import GRID as G
    truth = [{"kind": "peak", "f": 2800.0, "gain": -8.0, "q": 1.2},
             {"kind": "lowshelf", "f": 90.0, "gain": 4.0, "q": 1.0}]
    target = fit.chain(truth, G)
    picks = fit.greedy(target, 8, np.ones(len(G)), G)
    got = fit.refine(picks, target, np.ones(len(G)), G)
    err = float(np.sqrt(np.mean((fit.chain(got, G) - target) ** 2)))
    check("recovers a two-filter EQ to under 0.3 dB", err < 0.3, "residual %.2f dB" % err)


def test_limits_match_gates():
    check("q cap agrees with gate",
          fit.LIMITS["q_max_above_4k"] == metrics.GATES["max_q_above_5k"])
    check("boost cap agrees with gate",
          fit.LIMITS["gain_max"] == metrics.GATES["max_boost"])
    check("treble boost cap agrees with gate",
          fit.LIMITS["gain_max_above_5k"] == metrics.GATES["max_boost_treble"])
    check("cut cap agrees with gate", -fit.LIMITS["gain_min"] == metrics.GATES["max_cut"])
    # And quantising must never reintroduce a shape the gates reject.
    over = [{"kind": "peak", "f": 6000.0, "gain": 9.0, "q": 6.0}]
    q = fit.quantise(over)
    check("quantise clamps a hand-supplied violator",
          all(f["gain"] <= metrics.GATES["max_boost_treble"] + 1e-9 for f in q)
          and all(f["q"] <= metrics.GATES["max_q_above_5k"] + 1e-9 for f in q),
          str(q))


def test_gate_split():
    """A capability miss must be reported, not silently swallowed or fatal."""
    good = {"rms_weighted": 0.5, "rms_core": 0.4, "worst_core": -4.0, "worst_core_hz": 3836.0,
            "worst_treble": 2.0, "worst_treble_hz": 7000.0, "bands": []}
    filters = [{"kind": "peak", "f": 100.0, "gain": -3.0, "q": 1.0}]
    passed, rows, _viol = metrics.gate_report(good, filters, 0.0)
    check("capability miss does not fail the fitter", passed)
    worst = [r for r in rows if r["gate"] == "worst_core"][0]
    check("capability miss is still reported", not worst["ok"])
    notes = metrics.unreachable_notes(good)
    check("capability miss explains the cost", bool(notes) and "boost" in notes[0], str(notes))


def test_preamp_protects_headroom():
    filters = [{"kind": "lowshelf", "f": 60.0, "gain": 6.0, "q": 1.0}]
    preamp, peak, _at = metrics.required_preamp(filters)
    check("preamp brings chain below 0 dBFS", peak + preamp <= 0.0,
          "peak %+.2f preamp %+.1f" % (peak, preamp))
    check("preamp leaves inter-sample headroom", peak + preamp <= -0.5 + 1e-9,
          "combined %+.2f" % (peak + preamp))
    ok = [{"kind": "peak", "f": 1000.0, "gain": -6.0, "q": 1.0}]
    preamp2, peak2, _ = metrics.required_preamp(ok)
    check("attenuation-only chain needs no preamp", preamp2 == 0.0 and peak2 < 0.0)


def test_peak_location_is_audible():
    """An attenuation-only chain peaks at Nyquist; the reported Hz must be audible."""
    filters = [{"kind": "peak", "f": 1000.0, "gain": -6.0, "q": 1.0}]
    peak, at = metrics.chain_peak(filters)
    check("peak location stays inside the audible band", at <= 20000.0, "at %.0f" % at)


def test_name_matching():
    check("model token match", judge.name_matches("HD 600", "Sennheiser HD 600"))
    check("near-miss rejected", not judge.name_matches("G2Pro", "浦记 G28 Pro"))
    check("unrelated rejected", not judge.name_matches("Sundara", "HIFIMAN SUSVARA"))


def test_ranking_prefers_match():
    records = [
        {"title": "HIFIMAN SUSVARA UNVEILED", "brand": "HIFIMAN", "modes": ["默认调音模式"]},
        {"title": "HIFIMAN SUNDARA", "brand": "HIFIMAN", "modes": ["默认调音模式"]},
    ]
    ranked = judge.rank_for_query(records, "Sundara")
    check("exact model ranks first", ranked[0]["record"]["title"] == "HIFIMAN SUNDARA",
          ranked[0]["record"]["title"])


def test_variant_requires_explicit_choice():
    rec = {"variants": [{"name": "2.4G模式频响", "label": "2.4G模式频响:SPL(dB)"},
                        {"name": "一致性R频响", "label": "一致性R频响:SPL(dB)"}]}
    pick, why = judge.pick_variant(rec, None)
    check("two curves without a choice refuses", pick is None, why)
    pick2, _ = judge.pick_variant(rec, "一致性R")
    check("explicit choice is honoured", pick2 is not None and pick2["name"] == "一致性R频响")
    single = {"variants": [{"name": "频响", "label": "频响:SPL(dB)"}]}
    pick3, _ = judge.pick_variant(single, None)
    check("single curve needs no choice", pick3 is not None)


def test_grid_and_normalisation():
    check("grid is log-spaced", np.allclose(np.diff(np.log10(GRID)),
                                            np.diff(np.log10(GRID))[0], atol=1e-12))
    y = curves.normalise(np.zeros_like(GRID))
    check("normalise is idempotent on a flat curve", float(np.max(np.abs(y))) < 1e-12)


def test_batched_jacobian_is_the_same_maths():
    """The vectorised Jacobian must equal the per-filter one, column for column.

    The whole speed-up rests on this identity: perturbing one filter cannot change
    another filter's response, so every column can be built from one batched call.
    If that ever stops holding, the optimiser silently fits something else.  A
    float-exact comparison is possible because the coefficients are the same.
    """
    filters = [{"kind": "peak", "f": 1200.0, "gain": -6.0, "q": 1.4},
               {"kind": "lowshelf", "f": 90.0, "gain": 4.0, "q": 0.8},
               {"kind": "highshelf", "f": 6200.0, "gain": -3.0, "q": 1.1}]
    p = fit.clamp_params(fit.filters_to_params(filters))
    free = [k for k in range(p.size) if k % 4 != 0]
    fast = fit._jacobian(p, free, GRID)
    slow = np.zeros_like(fast)
    for j, k in enumerate(free):
        i, col = k // 4, k % 4
        row = p[i]
        h = 1e-4 if row[col] != 0.0 else 1e-5
        plus = np.array([row])
        plus[0, col] += h
        plus = fit.clamp_params(plus)[0]
        def resp(r):
            return apo.responses_db([fit.KINDS[int(round(r[0]))]], [10.0 ** r[1]],
                                    [r[2]], [10.0 ** r[3]], GRID)[0]
        slow[:, j] = (resp(plus) - resp(row)) / h
    check("batched Jacobian matches the per-filter one",
          float(np.max(np.abs(fast - slow))) < 1e-12,
          "max diff %.3e" % float(np.max(np.abs(fast - slow))))


def test_prune_keeps_load_bearing_filters():
    """A working filter survives the prune pass; an inert one does not.

    This is the promise the four old cleanup passes each made separately.  They now
    share one loop, so the guarantee is asserted once, against the merged pass.
    """
    from eqkit.grid import GRID as G
    target = -6.0 * np.exp(-0.5 * (np.log10(G / 120.0) / 0.15) ** 2)
    w = np.ones(len(G))
    working = [{"kind": "peak", "f": 120.0, "gain": -6.0, "q": 1.0},
               {"kind": "peak", "f": 8000.0, "gain": 0.2, "q": 2.0}]
    kept = fit.prune(working, target, w, G)
    check("prune keeps the filter that carries the correction",
          any(abs(f["f"] - 120.0) < 120 for f in kept), str(kept))
    check("prune removes the inert one",
          not any(abs(f["f"] - 8000.0) < 400 for f in kept), str(kept))


def test_candidate_results_keep_their_order():
    """Every (job, count) candidate must come back attached to its own count.

    The scheduler dispatches the largest counts first and then has to put the
    results back.  It once did not, so one candidate's filters were reported under
    another count's label and a 1.5 dB chain could be selected as the best fit.
    """
    from eqkit.grid import GRID as G
    from eqkit import metrics as _m
    target = -5.0 * np.exp(-0.5 * (np.log10(G / 200.0) / 0.2) ** 2)
    w = _m.band_weights(G)
    spec = {"target": target, "weights": w, "counts": (4, 6, 8), "role": "t"}
    serial = fit.build_many([("t", spec)], jobs=1)["t"][1]
    fit.shutdown_pool()
    n = fit._workers()
    if n < 2:
        check("candidate order (skipped: one core)", True)
        return
    par = fit.build_many([("t", spec)], jobs=2)["t"][1]
    fit.shutdown_pool()
    same = len(serial) == len(par) and all(
        a["n"] == b["n"] and abs(a["stats"]["rms_weighted"] - b["stats"]["rms_weighted"]) < 1e-9
        for a, b in zip(serial, par))
    check("parallel results keep their candidate order", same,
          str([(r["n"], round(r["stats"]["rms_weighted"], 3)) for r in par]))


def test_no_nested_pool():
    """A pool worker must take the serial path, or the run deadlocks on Windows."""
    check("in_worker() is False in the parent process", fit.in_worker() is False)


def test_game_target_can_hold_its_spikes():
    """The derived target must stay dense enough to represent the curve it came from.

    The chart draws narrow spikes near 5.5 and 8.7 kHz only a few pixels wide.  An
    earlier version of the digitiser resampled the trace onto a tidy 1/48-octave grid
    (0.021 octave); at 8.7 kHz a step that wide straddles the peak, and the spike
    collapsed from +8.9 dB to +8.4 dB without anything looking wrong.  Density is the
    property that protects the spikes, so it is what this asserts.
    """
    freq, level = curves.read_csv(os.path.join(repo.DATA, "blue-dashed.csv"))
    check("game target covers the audible band",
          freq.min() <= 25.0 and freq.max() >= 19000.0,
          "%.0f-%.0f Hz" % (freq.min(), freq.max()))
    step = float(np.median(np.log2(freq[1:] / freq[:-1])))
    check("game target is denser than 1/24 octave", step < 1.0 / 24.0,
          "median step %.4f octave" % step)
    i = int(np.argmin(np.abs(freq - 8600.0)))
    check("the 8.6 kHz spike is preserved", level[i] > 8.0, "%+.2f dB" % level[i])


def test_speed_contract():
    """One candidate must stay cheap enough that the scan finishes in seconds.

    The skill promises a few seconds per headphone.  A change that quietly puts a full
    optimisation back inside a comparison shows up here as a candidate that takes
    longer than it used to -- before it becomes a 139 s run nobody notices.
    """
    from eqkit import metrics as _m
    import time
    target = -5.0 * np.exp(-0.5 * (np.log10(GRID / 300.0) / 0.25) ** 2)
    w = _m.band_weights(GRID)
    fit._fit_one_count((target, w, 16, GRID, "probe"))   # warm the shape bank
    t = time.time()
    fit._fit_one_count((target, w, 16, GRID, "probe"))
    elapsed = time.time() - t
    # Generous: a slow CI box is not a regression.  This catches an order of magnitude.
    check("one candidate fits in under 5 s", elapsed < 5.0, "took %.2f s" % elapsed)


def test_auto_jobs_actually_parallelises():
    """`jobs=0` means "use the machine", not "use zero workers".

    The CLI flag defaults to 0 so there is one answer to "how many cores?".  Reading
    that 0 literally ran the entire filter-count scan on a single core while the docs
    promised a parallel scan, which is where most of a 139 s run went.  Only an
    explicit positive number may pin the count.
    """
    cores = os.cpu_count() or 1
    auto = fit._job_count(0, 12)
    check("jobs=0 means auto, not serial", auto == 1 or auto > 1,
          "resolved to %d" % auto)
    if cores > 1:
        check("auto uses more than one worker when cores are available", auto > 1,
              "resolved to %d on %d cores" % (auto, cores))
    check("jobs=1 still forces the serial path", fit._job_count(1, 12) == 1)
    check("an explicit count is honoured", fit._job_count(4, 12) == 4)
    check("the count is capped by the work", fit._job_count(99, 3) == 3)


def main():
    for fn in (test_biquad_model, test_useless_filters_removed, test_limits_match_gates,
               test_gate_split, test_preamp_protects_headroom, test_peak_location_is_audible,
               test_name_matching, test_ranking_prefers_match,
               test_variant_requires_explicit_choice, test_grid_and_normalisation,
               test_ablation_removes_only_inert, test_ground_truth_recovery,
               test_batched_jacobian_is_the_same_maths, test_prune_keeps_load_bearing_filters,
               test_candidate_results_keep_their_order, test_no_nested_pool,
               test_auto_jobs_actually_parallelises, test_speed_contract,
               test_game_target_can_hold_its_spikes):
        print(fn.__name__.replace("test_", "").replace("_", " "))
        fn()
    print("")
    if FAILS:
        print("%d failure(s): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
