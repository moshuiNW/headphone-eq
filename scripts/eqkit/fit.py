# -*- coding: utf-8 -*-
"""Fitting a PEQ cascade to a correction curve.

Two stages, because a single optimiser cannot do both jobs well:

  1. greedy structure: pick filter centres/Qs from a shape bank by orthogonal
     matching pursuit, with the gain solved in closed form at each step. This
     finds *where* the problems are and how many filters each one deserves.
  2. joint Levenberg-Marquardt: with the structure fixed, optimise every
     frequency / gain / Q together, which the greedy pass cannot do (a greedy
     pick is optimal given all earlier picks, so two filters that should have
     moved toward each other never do).

After that, one **prune** pass decides which filters were never worth their
slot, and **quantise** rounds the survivors to what the config file can express.

The optimiser is *constrained*, and the constraints are not cosmetic: they are
what keeps a numerically good fit from being unlistenable. A +10 dB boost at
9 kHz fits a measurement artefact beautifully and sounds like a razor blade, so
boosts get capped, Qs get capped above 5 kHz, and the band weights from
metrics.BANDS decide where the remaining error is allowed to live.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .grid import GRID, FINE
from . import apo
from .metrics import GATES, band_weights, score_band

KINDS = ("peak", "lowshelf", "highshelf")

# Per-filter hard limits, *derived* from metrics.GATES rather than restated.
# A second copy of the same numbers drifts, and the way it drifts is silent: the
# optimiser happily produces a Q 5.8 filter that the acceptance check then rejects,
# so every config needs a human to reconcile two tables that should have agreed.
LIMITS = {
    "f_min": 18.0,
    "f_max": 19000.0,
    "q_min": 0.30,
    "q_max": max(GATES["max_q_below_5k"], GATES["max_q_above_5k"]),
    "q_max_above_4k": GATES["max_q_above_5k"],
    "q_max_above_5k": GATES["max_q_above_5k"],
    "gain_min": -GATES["max_cut"],
    "gain_max": GATES["max_boost"],
    "gain_max_above_5k": GATES["max_boost_treble"],
    "min_effect": 0.6,
    # Above 5 kHz the measurement cannot justify a dense correction.  Three
    # filters cover a shelf plus two broad shapes; beyond that the fit is
    # sculpting the fixture's comb filtering into the listener's ear.
    "max_filters_above_5k": 3,
    # Nothing above 16 kHz.  There is no data there (every fixture stops at 19-20k
    # and the ear is done above 16k), so a filter placed there can only be shaping
    # the model's own overshoot near Nyquist -- exactly the kind of ghost filter
    # that costs preamp and does nothing audible.
    "f_fit_max": 16000.0,
    # Shelves are the shape most prone to interaction: two shelves overlapping with
    # a deep cut between them can overshoot by several dB.  Two of each kind is
    # enough for any real headphone.
    "max_shelves": 2,
    # A shelf whose corner is far below the band it is meant to shape is not a
    # shelf in APO's Q form: it degenerates into a broad bump that also lifts
    # Nyquist, which then costs preamp for nothing audible.  Keep shelf corners
    # where the shape still behaves like the name says.
    "lowshelf_f_max": 400.0,
    "highshelf_f_min": 3000.0,
    # APO's Q-form shelf at very low Q approaches a step, not a slope; at Q 0.3 it
    # is within 0.01 dB of a constant offset across the whole band, which costs
    # preamp and does nothing.  Require a shelf narrow enough to be a shelf.
    "shelf_q_min": 0.6,
    # A filter whose total swing over 20 Hz-16 kHz is smaller than this is a level
    # shift in disguise, and the offset already covers level.
    "min_band_effect": 1.0,
}

SHELF_FREQS_LOW = np.geomspace(90, LIMITS["lowshelf_f_max"], 18)
SHELF_FREQS_HIGH = np.geomspace(LIMITS["highshelf_f_min"], 14000, 18)
PEAK_FREQS = tuple(np.geomspace(20, 16000, 110))
PEAK_QS = (0.35, 0.5, 0.7, 0.95, 1.25, 1.7, 2.3, 3.0, 4.0)

# Iteration budget for every *comparison* refine (prune, polish).  The rule the
# module is built around: two filter sets are only comparable at the same budget,
# so this number is a constant, never a per-call argument chosen by the caller.
COMPARE_ITERS = 90
# How many candidate removals the prune pass verifies per round.  The screen that
# orders them is cheap and accurate; verifying all of them is what made a fit take
# two minutes.
PRUNE_TRIALS = 3
# A removal is kept when the weighted RMS does not get worse than this.
PRUNE_TOL = 0.02
# Give up after this many non-improving LM steps: the tail of the curve is the
# optimiser re-deriving the same point in smaller steps.
STALL_STEPS = 18


def clamp_params(p):
    """Apply LIMITS to a (n,4) parameter array [kind_id, logf, gain, logq]."""
    p = np.array(p, float, copy=True)
    p[:, 1] = np.clip(p[:, 1], np.log10(LIMITS["f_min"]), np.log10(min(LIMITS["f_max"], LIMITS["f_fit_max"])))
    p[:, 2] = np.clip(p[:, 2], LIMITS["gain_min"], LIMITS["gain_max"])
    p[:, 3] = np.clip(p[:, 3], np.log10(LIMITS["q_min"]), np.log10(LIMITS["q_max"]))
    fc = 10.0 ** p[:, 1]
    p[fc <= 5000.0, 3] = np.minimum(p[fc <= 5000.0, 3], np.log10(GATES["max_q_below_5k"]))
    p[fc > 4000.0, 3] = np.minimum(p[fc > 4000.0, 3], np.log10(LIMITS["q_max_above_4k"]))
    p[fc > 5000.0, 3] = np.minimum(p[fc > 5000.0, 3], np.log10(LIMITS["q_max_above_5k"]))
    p[fc > 5000.0, 2] = np.minimum(p[fc > 5000.0, 2], LIMITS["gain_max_above_5k"])
    ls = p[:, 0].round() == KINDS.index("lowshelf")
    hs = p[:, 0].round() == KINDS.index("highshelf")
    p[ls, 1] = np.minimum(p[ls, 1], np.log10(LIMITS["lowshelf_f_max"]))
    p[hs, 1] = np.maximum(p[hs, 1], np.log10(LIMITS["highshelf_f_min"]))
    # Shelves are constrained at the source.  A Q-form shelf at Q < 0.6 is within a
    # hair of a straight step across the audible band, i.e. a level change, and it
    # also made the optimiser wander into a shape that the gate then rejects.  Keeping
    # the optimiser inside the legal set is cheaper than policing its output.
    shelf = ls | hs
    p[shelf, 3] = np.maximum(p[shelf, 3], np.log10(LIMITS["shelf_q_min"]))
    return p


def params_to_filters(p):
    out = []
    for row in np.atleast_2d(p):
        if not np.all(np.isfinite(row)):
            continue
        out.append({
            "kind": KINDS[int(np.clip(round(row[0]), 0, len(KINDS) - 1))],
            "f": float(10.0 ** row[1]),
            "gain": float(row[2]),
            "q": float(10.0 ** row[3]),
        })
    return out


def filters_to_params(filters):
    return np.array([[KINDS.index(f["kind"]), np.log10(f["f"]), f["gain"], np.log10(f["q"])]
                     for f in filters], float)


def chain(filters, grid=GRID, fs=apo.FS_DEFAULT):
    return apo.chain_db(filters, grid, fs)


def _shape_bank(grid=GRID):
    specs, shapes = [], []
    fit_max = LIMITS["f_fit_max"]
    for fc in PEAK_FREQS:
        if fc > fit_max:
            continue
        for q in PEAK_QS:
            specs.append(("peak", fc, q))
            shapes.append(apo.response_db("peak", fc, 1.0, q, grid))
    for fc in SHELF_FREQS_LOW:
        for q in (0.7, 1.0, 1.4):
            specs.append(("lowshelf", fc, q))
            shapes.append(apo.response_db("lowshelf", fc, 1.0, q, grid))
    for fc in SHELF_FREQS_HIGH:
        for q in (0.7, 1.0, 1.4):
            specs.append(("highshelf", fc, q))
            shapes.append(apo.response_db("highshelf", fc, 1.0, q, grid))
    return specs, np.asarray(shapes)


_BANK = None


def bank(grid=GRID):
    global _BANK
    if _BANK is None:
        _BANK = _shape_bank(grid)
    return _BANK


def greedy(target, n_filters, weights, grid=GRID):
    """Greedy structure search: each step picks the shape that removes most error."""
    specs, shapes = bank(grid)
    sw = shapes * weights[None, :]
    den = np.sum(sw * shapes, axis=1)
    resid = np.array(target, float)
    picks = []
    allowed = np.ones(len(specs), bool)
    for _ in range(n_filters):
        num = sw @ resid
        gains = np.clip(num / np.maximum(den, 1e-12), LIMITS["gain_min"], LIMITS["gain_max"])
        sse = np.sum(weights * resid ** 2) - 2.0 * gains * num + gains ** 2 * den
        sse = np.where(allowed, sse, np.inf)
        i = int(np.argmin(sse))
        if not np.isfinite(sse[i]):
            break
        k, fc, q = specs[i]
        resid = resid - gains[i] * shapes[i]
        picks.append({"kind": k, "f": float(fc), "gain": float(gains[i]), "q": float(q)})
        same = np.array([s[0] == k and abs(np.log2(s[1] / fc)) < 0.002 and abs(np.log10(s[2] / q)) < 0.002
                         for s in specs])
        allowed = allowed & ~same
    return picks


def _jacobian(p, free, grid, fs=apo.FS_DEFAULT):
    """One filter perturbation per column, all filters in a single vectorised call.

    Perturbing filter i changes only filter i's response, so the column is that
    filter's own difference -- not the cascade's.  Building every column from one
    (2m, grid) magnitude call instead of 2m single-filter calls is the same
    arithmetic on the same coefficients at a third of the cost, and the Jacobian is
    where the optimiser spends its time.
    """
    m = len(free)
    idx_i = np.array([k // 4 for k in free], dtype=int)
    idx_c = np.array([k % 4 for k in free], dtype=int)
    rows = p[idx_i]
    h = np.where(rows[np.arange(m), idx_c] != 0.0, 1e-4, 1e-5)
    plus = rows.copy()
    plus[np.arange(m), idx_c] += h
    plus = clamp_params(plus)
    both = np.vstack([rows, plus])
    kinds = [KINDS[int(np.clip(round(r[0]), 0, len(KINDS) - 1))] for r in both]
    resp = apo.responses_db(kinds, 10.0 ** both[:, 1], both[:, 2], 10.0 ** both[:, 3], grid, fs)
    return ((resp[m:] - resp[:m]) / h[:, None]).T


def refine(filters, target, weights, grid=GRID, iters=140, tol=1e-10):
    """Joint Levenberg-Marquardt on frequency/gain/Q of every filter.

    The Jacobian is assembled one filter at a time: perturbing filter i changes only
    filter i's response, so the residual difference for that column is the *filter's
    own* change, not the chain's.  Finite-differencing the whole chain instead costs
    one cascade evaluation per parameter, which is where whole seconds per fit went.

    Two stops keep the tail cheap: a converged step (the original exit), and a run of
    non-improving steps -- past that the optimiser is re-deriving the same point in
    smaller and smaller steps, and the last forty iterations buy nothing.
    """
    if not filters:
        return []
    p = clamp_params(filters_to_params(filters))
    free = [k for k in range(p.size) if k % 4 != 0]
    lam = 1e-2
    best = p.copy()
    best_e = float(np.sum(weights * (chain(params_to_filters(p), grid) - target) ** 2))
    stall = 0
    for _ in range(iters):
        resid = chain(params_to_filters(p), grid) - target
        J = _jacobian(p, free, grid)
        if not np.all(np.isfinite(J)):
            break
        jw = J * weights[:, None]
        A = jw.T @ jw
        A = A + lam * np.diag(np.diag(A) + 1e-9)
        try:
            dx = np.linalg.solve(A, -jw.T @ (resid * weights))
        except np.linalg.LinAlgError:
            break
        nrm = float(np.max(np.abs(dx))) if dx.size else 0.0
        if nrm > 1.0:
            dx = dx * (1.0 / nrm)
        cand = p.copy().ravel()
        for j, k in enumerate(free):
            cand[k] += dx[j]
        cand = clamp_params(cand.reshape(p.shape))
        err = float(np.sum(weights * (chain(params_to_filters(cand), grid) - target) ** 2))
        if err < best_e - tol * max(1.0, abs(best_e)):
            p, best_e, lam = cand, err, max(lam / 2.5, 1e-7)
            best = cand.copy()
            stall = 0
            if nrm < 1e-6:
                break            # converged: further iterations only re-derive the same point
        else:
            lam *= 3.0
            stall += 1
            if lam > 1e5 or stall >= STALL_STEPS:
                break
    return params_to_filters(best)


# ------------------------------------------------------------------- scoring
def _wrms(filters, target, weights, grid=GRID, mask=None):
    """Weighted RMS of (chain - target) over the score band, level removed.

    One definition, used by every decision in this module: the prune screen, the
    prune verification and the count selection all compare the same number.
    """
    m = score_band(grid) if mask is None else mask
    if not filters:
        return float("inf")
    r = chain(filters, grid) - target
    r = r - float(np.mean(r[m]))
    wsum = float(np.sum(weights[m]))
    return float(np.sqrt(np.sum(weights[m] * r[m] ** 2) / wsum)) if wsum > 0 else float("inf")


def _contribution_matrix(filters, grid=GRID):
    """(n, len(grid)) matrix: each filter's own response in dB.

    The cascade sums these, so `mat.sum(0)` is the chain and `mat.sum(0) - mat[i]`
    is the chain without filter i -- the first-order prediction of what a removal
    costs, for every filter at once.
    """
    if not filters:
        return np.zeros((0, len(grid)))
    kinds = [f["kind"] for f in filters]
    fcs = [float(f["f"]) for f in filters]
    gains = [float(f.get("gain", 0.0)) for f in filters]
    qs = [float(f["q"]) for f in filters]
    return apo.responses_db(kinds, fcs, gains, qs, grid)


def _screen_drops(filters, target, weights, grid=GRID, mask=None):
    """Rank every filter by what dropping it (and re-levelling) would cost.

    Closed form, no re-optimisation: the removal is assumed to cost exactly its own
    contribution, then the level is re-centred.  That is a *first-order* estimate and
    it is only used to decide which removals are worth a full verification -- never to
    accept one.  Ordering by it means the prune pass verifies about three candidates
    per round instead of every filter.
    """
    m = score_band(grid) if mask is None else mask
    mat = _contribution_matrix(filters, grid)
    if not len(mat):
        return []
    base = mat.sum(axis=0)
    wsum = float(np.sum(weights[m]))
    if wsum <= 0:
        return []
    out = []
    for i in range(len(mat)):
        r = (base - mat[i]) - target
        r = r - float(np.mean(r[m]))
        rms = float(np.sqrt(np.sum(weights[m] * r[m] ** 2) / wsum))
        out.append((rms, i))
    out.sort()
    return out


def _structural_offender(filters, order):
    """Index of a filter a hard cap says must go, or None.

    These are caps, not preferences: more than `max_filters_above_5k` filters above
    5 kHz, more than `max_shelves` shelves of one kind, or any filter above the data
    band cannot be shipped whatever it does for the number.  Which one goes is still
    decided by measured cost -- the screen supplies the ranking -- so the rule is a
    cap on the count and never a blacklist of shapes.
    """
    by_i = {i: loss for loss, i in order}
    work = list(filters)
    over = [i for i, f in enumerate(work)
            if f["kind"] != "highpass" and f["f"] > LIMITS["f_fit_max"]]
    if over:
        return min(over, key=lambda i: by_i.get(i, float("inf")))
    treble = [i for i, f in enumerate(work) if f["f"] > 5000.0]
    if len(treble) > LIMITS["max_filters_above_5k"]:
        return min(treble, key=lambda i: by_i.get(i, float("inf")))
    for kind in ("lowshelf", "highshelf"):
        idxs = [i for i, f in enumerate(work) if f["kind"] == kind]
        if len(idxs) > LIMITS["max_shelves"]:
            return min(idxs, key=lambda i: by_i.get(i, float("inf")))
    return None


def _clamp_positions(filters):
    """Move corners into the band where their name still means what it says.

    Clamps, never deletions: a shelf placed outside its meaningful band is moved to
    the edge of it, which changes the shape slightly and keeps the work.
    """
    work = []
    for g in filters:
        g = dict(g)
        if g["kind"] == "lowshelf":
            g["f"] = min(max(g["f"], LIMITS["f_min"]), LIMITS["lowshelf_f_max"])
        elif g["kind"] == "highshelf":
            g["f"] = max(min(g["f"], LIMITS["f_max"]), LIMITS["highshelf_f_min"])
        elif g["kind"] != "highpass":
            g["f"] = min(max(g["f"], LIMITS["f_min"]), LIMITS["f_fit_max"])
        work.append(g)
    return work


def prune(filters, target, weights, grid=GRID, tol=PRUNE_TOL, iters=COMPARE_ITERS):
    """Remove filters that are not worth their slot -- verified, never assumed.

    Earlier versions blacklisted shapes by category (shelves below 400 Hz, shelves at
    Q < 0.6, small local swing).  Every one of those rules deleted load-bearing filters
    on real fixtures: one fixture went from 0.12 dB to 1.69 dB when four legal shelves
    were removed for being "misplaced" while they were doing most of the work.  Category
    rules cannot know what a filter is for.

    So the only test kept is the honest one: remove the filter, re-optimise the rest at
    the *same* iteration budget, and keep the removal when the error does not get worse.
    Two economies make that affordable without weakening it:

      * the closed-form screen orders the candidates by predicted cost, so only the
        cheapest few are ever verified;
      * the four passes this replaces (misplaced-shelf ablation, interaction cap, treble
        cap, compaction) each ran their own ablation loop over the same filter list.
        One loop, one budget, one accepted answer.

    Position rules survive as clamps, applied before the first optimisation.
    """
    work = _clamp_positions(filters)
    if target is None or not work:
        return work

    mask = score_band(grid)
    cur = refine(work, target, weights, grid, iters=iters)
    cur_rms = _wrms(cur, target, weights, grid, mask)
    while len(cur) > 3:
        order = _screen_drops(cur, target, weights, grid, mask)
        forced = _structural_offender(cur, order)
        tried = []
        if forced is not None:
            tried.append(forced)
        for _loss, i in order:
            if i in tried:
                continue
            tried.append(i)
            if len(tried) >= PRUNE_TRIALS + (1 if forced is not None else 0):
                break
        improved = False
        for i in tried:
            if i >= len(cur):
                continue
            trial = cur[:i] + cur[i + 1:]
            cand = refine(trial, target, weights, grid, iters=iters)
            rms = _wrms(cand, target, weights, grid, mask)
            if rms <= cur_rms + tol or i == forced:
                cur, cur_rms, improved = cand, rms, True
                break
        if not improved:
            break
    return cur


def quantise(filters, merge_octaves=0.06, target=None, weights=None, grid=GRID):
    """Round to what the file can express, then drop what stopped mattering.

    Rounding happens *after* optimisation and the residual is re-measured on the
    quantised chain: quoting the unrounded number is a silent lie that shows up as a
    systematic error in the delivered curve.

    Every lossy step here is *verified* against the target when one is supplied.
    Merging two filters sums their gains, and a sum past the cap is then clipped -- two
    +6 dB filters half a comma apart merged to +12 and clipped back to +6, which deleted
    half the correction (0.33 dB -> 1.07 dB on a real fixture) while the report claimed
    the unrounded number.  A merge is kept only if it does not measurably cost accuracy.

    Pruning is *not* repeated here: `prune` already ran on the unrounded chain, and
    running the same ablation again after rounding was a second full optimisation pass
    that changed nothing but the clock.
    """
    mask = score_band(grid)
    w = band_weights(grid) if weights is None else weights

    def err_of(fs_):
        if target is None or not fs_:
            return float("inf")
        r = chain(fs_, grid) - target
        r = r - float(np.mean(r[mask]))
        return float(np.sqrt(np.sum(w[mask] * r[mask] ** 2) / np.sum(w[mask])))

    baseline = err_of(filters)

    out = []
    for f in filters:
        gain = round(float(f["gain"]), 1)
        if abs(gain) < LIMITS["min_effect"]:
            continue
        out.append({
            "kind": f["kind"],
            "f": float(round(float(np.clip(f["f"], LIMITS["f_min"], LIMITS["f_max"])))),
            "gain": gain,
            "q": round(float(np.clip(f["q"], LIMITS["q_min"], LIMITS["q_max"])), 2),
        })

    merged = []
    for f in sorted(out, key=lambda x: -abs(x["gain"])):
        for m in merged:
            if m["kind"] != f["kind"] or abs(np.log2(f["f"] / m["f"])) >= merge_octaves:
                continue
            summed = m["gain"] + f["gain"]
            if abs(summed) > -LIMITS["gain_min"]:
                continue
            before_gain, before_q = m["gain"], m["q"]
            m["gain"] = summed
            m["q"] = min(m["q"], f["q"]) * 0.85
            if target is not None:
                trial = [dict(x) for x in merged] + [dict(f)]
                if err_of(trial) > baseline + 0.02:
                    m["gain"], m["q"] = before_gain, before_q
                    continue
            break
        else:
            merged.append(dict(f))

    fixed = []
    for m in merged:
        g = float(np.clip(round(m["gain"], 1), LIMITS["gain_min"], LIMITS["gain_max"]))
        q = float(np.clip(m["q"], LIMITS["q_min"], LIMITS["q_max"]))
        if m["f"] > 4000.0:
            q = min(q, LIMITS["q_max_above_4k"])
        if m["f"] > 5000.0:
            q = min(q, LIMITS["q_max_above_5k"])
            g = min(g, LIMITS["gain_max_above_5k"])
        if abs(g) < LIMITS["min_effect"]:
            continue
        fixed.append({"kind": m["kind"], "f": m["f"], "gain": g, "q": round(q, 2)})
    return fixed


# The three passes below are the pre-prune vocabulary.  They stay as thin wrappers
# because the selftest asserts their promises ("no filter is deleted for being
# misplaced") and because a reader arriving from the old code should find the rule
# it knew, now with one implementation and one budget behind it.
def drop_misplaced_shelves(filters, grid=GRID, target=None, weights=None, tol=PRUNE_TOL):
    """Clamp misplaced corners; with a target, also prune.  See `prune`."""
    if target is None:
        return _clamp_positions(filters)
    w = band_weights(grid) if weights is None else weights
    return prune(filters, target, w, grid, tol=tol)


def limit_interaction(filters, target, weights, grid=GRID, iters=COMPARE_ITERS):
    """Compatibility alias for the structural caps inside `prune`."""
    return prune(filters, target, weights, grid, iters=iters)


def compact(filters, target, weights, grid=GRID, iters=COMPARE_ITERS, tolerance=PRUNE_TOL):
    """Compatibility alias for the elimination loop inside `prune`."""
    return prune(filters, target, weights, grid, tol=tolerance, iters=iters)


def limit_treble_filters(filters, target, weights, grid=GRID, iters=COMPARE_ITERS):
    """Compatibility alias for the treble cap inside `prune`."""
    return prune(filters, target, weights, grid, iters=iters)


def polish(filters, target, weights, grid=GRID, iters=COMPARE_ITERS):
    """Final joint refinement of the surviving structure: one call, not a loop."""
    if not filters:
        return filters
    return refine(filters, target, weights, grid, iters=iters)


def overshoot(filters):
    """How far the cascade rises above 0 dB, and at which frequency.

    A fit whose chain overshoots by 5 dB is telling you something: the correction
    needed alternating boosts and cuts that partially cancel, and the sum of the
    biquads lands higher than the target curve did.  It is not a bug, but it *is*
    a cost (5 dB of preamp), so it is reported rather than hidden.
    """
    peak, at = chain_peak_db(filters)
    return {"peak_db": round(peak, 2), "at_hz": round(at, 1), "preamp_needed": round(max(0.0, peak + 0.5), 1)}


def trim_overshoot(filters, headroom=0.0):
    """Remove the part of the cascade peak that the fit did not need.

    A biquad cascade does not track its own target curve exactly: where a broad cut
    and a narrow boost overlap, the sum can sit above 0 dB even though the correction
    curve peaks at 0 dB.  Left alone, that overshoot needs extra negative preamp, and
    every decibel of preamp is a decibel of loudness spent for nothing -- the shape is
    identical, only the level changed.

    The fix is a uniform gain shift, which is exactly what a preamp already does,
    *minus* what the preamp already accounts for.  So this only reports the shift;
    the caller folds it into the preamp and adds it back as a constant offset when
    scoring, keeping the residual honest.  Scaling every gain instead would change
    the shape where filters overlap, which is not a uniform shift at all.

    Returns (shift_db, filters_unchanged): shift_db > 0 means the chain peaks
    above 0 dB and the preamp must cover it.
    """
    peak, _at = chain_peak_db(filters)
    return peak + headroom, list(filters)


def chain_peak_db(filters, fs=apo.FS_DEFAULT):
    """Peak of the cascade over 5 Hz .. Nyquist, and where it is."""
    g = FINE[FINE < fs / 2.0]
    r = chain(filters, g)
    i = int(np.argmax(r))
    return float(r[i]), float(g[i])


def residual_score(filters, target, grid=GRID, mask=None):
    from .metrics import residual_stats
    pred = chain(filters, grid) if filters else np.zeros_like(grid)
    return residual_stats(pred, target, grid, mask)


# ------------------------------------------------------------------- pipeline
def _fit_one_count(payload):
    """One (target, filter-count) candidate, top level so a worker process can run it.

    Structure -> joint refine -> prune -> round.  Each candidate is an independent
    optimisation from the same target, which is what makes the whole scan a flat,
    perfectly parallel work queue: one candidate per process, no shared state.
    """
    target, weights, n, grid, role = payload
    picks = greedy(target, n, weights, grid)
    ref = refine(picks, target, weights, grid)
    kept = prune(ref, target, weights, grid)
    q = quantise(kept, target=target, weights=weights, grid=grid)
    stats = residual_score(q, target, grid, score_band(grid))
    return {"role": role, "n": n, "filters": q, "all": list(q), "stats": stats}


_worker = _fit_one_count


def _workers():
    """How many processes to use: every logical core this machine has.

    The candidates are independent, each is bound to numpy inside C loops, and the
    work is floating-point arithmetic -- so one process per logical core is the right
    answer, and an arbitrary cap only leaves cores idle while the user waits.  `EQ_JOBS`
    overrides it; `--jobs 1` forces the serial path for debugging.
    """
    try:
        n = int(os.environ.get("EQ_JOBS", "0"))
        if n > 0:
            return n
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _job_count(jobs, n_tasks):
    """Resolve a caller's `jobs` request to a worker count.

    `None` and any non-positive number mean auto -- the flag defaults to 0 so "how
    many cores should I use" has one answer, and reading that 0 literally is how this
    tool spent its life running the whole scan on a single core while claiming to be
    parallel.  Only an explicit positive number pins it.
    """
    if jobs is None:
        return max(1, min(_workers(), n_tasks))
    try:
        n = int(jobs)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return max(1, min(_workers(), n_tasks))
    return max(1, min(n, n_tasks))


_POOL = None
_POOL_SIZE = 0


def _pool(n_jobs):
    """One process pool for the whole run.

    Spawning a pool per call paid the interpreter start-up on every filter-count scan,
    which is most of the wall clock on a fast machine.  One pool, created lazily and
    reused, amortises that to nothing.  It is grown only if a later call asks for more
    workers than it has.
    """
    global _POOL, _POOL_SIZE
    if _POOL is None or n_jobs > _POOL_SIZE:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
        _POOL = ProcessPoolExecutor(max_workers=n_jobs)
        _POOL_SIZE = n_jobs
    return _POOL


def shutdown_pool():
    """Close the shared pool (called at the end of a CLI run, not per fit)."""
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
        _POOL_SIZE = 0


def in_worker():
    """True when this process is itself a pool worker.

    A worker must never open a pool of its own: a process that both runs tasks and
    supervises other processes deadlocks the moment its children want the same cores,
    and on Windows the nested pool simply never receives a result.  Callers that
    parallelise at an outer level (fixtures.py) therefore get the serial path here,
    which computes exactly the same thing.
    """
    try:
        return multiprocessing.parent_process() is not None
    except Exception:
        return False


def _run_tasks(payload, jobs=None):
    """Run every (target, count) payload, using the whole machine when allowed.

    Results come back in *payload order*, whatever order the work actually ran in:
    callers pair them with their own index, and a schedule that reorders the queue
    without unsorting the results silently hands one candidate's filters to another.
    Largest counts are dispatched first because they dominate the wall clock, so the
    work is submitted in that order and then put back.
    """
    if not payload:
        return []
    n_jobs = _job_count(jobs, len(payload))
    if n_jobs > 1 and not in_worker():
        try:
            order = sorted(range(len(payload)), key=lambda i: -payload[i][2])
            out = list(_pool(n_jobs).map(_fit_one_count, [payload[i] for i in order]))
            settled = [None] * len(payload)
            for slot, row in zip(order, out):
                settled[slot] = row
            return settled
        except Exception:
            # A sandbox that forbids subprocesses must not become a failure: fall back
            # to the serial path, which computes exactly the same thing.
            shutdown_pool()
    return [_fit_one_count(p) for p in payload]


def _admissible(stats):
    """Inside the core-band gates, i.e. a candidate that actually did the job."""
    return (stats["rms_weighted"] <= 1.0
            and stats["rms_core"] <= 0.8
            and abs(stats["worst_core"]) <= 2.5)


def _select(rows, target, grid, mask, extra_filters=()):
    """Pick the count at the knee of the error curve among *admissible* candidates."""
    for r in rows:
        r["all"] = list(extra_filters) + list(r["filters"])
        r["stats"] = residual_score(r["all"], target, grid, mask)
    rows = sorted(rows, key=lambda r: r["n"])
    # The scan is not monotonic: greedy structure search can land in a worse local
    # minimum when it is given one more pick, so "more filters" sometimes fits worse.
    #
    # Picking the smallest weighted RMS alone is not safe.  A chain that has collapsed
    # to one filter can score *better* on the weighted metric than a faithful chain,
    # because the weights let it abandon the bands it stopped modelling -- it wins the
    # number by refusing the job.  So candidates must first be admissible: inside the
    # core-band gates.  Among admissible counts, take the best score, with a 0.03 dB
    # preference for more filters (the point where another filter pays for itself).
    good = [r for r in rows if _admissible(r["stats"])]
    pool = good or rows
    best = pool[0]
    for r in pool[1:]:
        if r["stats"]["rms_weighted"] < best["stats"]["rms_weighted"] - 0.03:
            best = r
    return best, rows


def build_many(specs, jobs=None, grid=GRID):
    """Fit several (name -> target) jobs in ONE flat pool across every core.

    The old shape ran each job's counts in its own pool and the jobs one after another,
    so a two-role run left part of the machine idle for the second half of the wall
    clock.  Flattening (job, count) into a single work queue keeps every core busy from
    the first second to the last, and the wall clock becomes one candidate's time plus
    queueing rather than the sum over jobs.

    `specs` is a list of (key, {"target", "weights", "counts", "role", "extra_filters"}).
    Returns {key: (best, all_counts)}.
    """
    payload, index = [], []
    for key, spec in specs:
        weights = spec.get("weights")
        weights = band_weights(grid) if weights is None else weights
        role = spec.get("role", key)
        for n in spec["counts"]:
            index.append((key, n))
            payload.append((spec["target"], weights, n, grid, role))
    raw = _run_tasks(payload, jobs)
    mask = score_band(grid)
    by_key = {}
    for (key, _n), row in zip(index, raw):
        by_key.setdefault(key, []).append(row)
    out = {}
    for key, spec in specs:
        rows = by_key.get(key, [])
        best, all_counts = _select(rows, spec["target"], grid, mask, spec.get("extra_filters", ()))
        out[key] = (best, all_counts)
    return out


def build(target, weights=None, counts=(8, 10, 12, 14, 16, 18), grid=GRID,
          candidates=False, extra_filters=(), jobs=None):
    """Fit the correction, choosing the filter count at the knee of the error curve.

    More filters always lower RMS a little; the knee is where another pair stops
    paying for itself, and past it the extra filters spend dynamic range chasing
    measurement noise.  `candidates=True` returns every count so the caller can
    show the trade-off instead of asserting it.

    A single-target wrapper around `build_many`: a caller with more than one target
    should call `build_many` directly so both targets share one pool.
    """
    spec = {"target": target, "weights": weights, "counts": tuple(counts),
            "extra_filters": extra_filters}
    best, rows = build_many([("only", spec)], jobs=jobs, grid=grid)["only"]
    return (rows if candidates else best), rows
