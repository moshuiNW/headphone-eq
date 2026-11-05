---
name: headphone-eq
description: Fit two Equalizer APO parametric-EQ configs to one headphone measurement -- one to Harman over-ear 2018, one to the blue dashed game reference curve -- and verify both against the APO biquad model. Use when the user names a headphone model or supplies a frequency-response curve and wants an EQ config, an AutoEq-style preset, a Harman target fit, or a gaming EQ; or asks to re-check, compare or re-fit an existing config.
---

# Headphone EQ

Two configs, one measurement. The **Harman** config is the reference tuning: this
is the curve that says "accurate". The **GAME** config aims at the blue dashed
reference curve the evaluation site draws, which keeps more low end and more
3 kHz energy, and is what people mean when they want a headphone tuned "for
gaming". They are fitted independently, from the same measurement, with
different band weights, because the second is a different claim about
preference -- not the same claim with a different line drawn on it.

## What "fitted" means here

A config ships only when it satisfies all of these. They are the judgement of
this skill, not preferences to negotiate per headphone:

| gate | threshold | why this number |
|---|---|---|
| weighted RMS, 20 Hz-16 kHz | **<= 1.0 dB** | one dB is where a broad residual starts reading as a tonal cast |
| RMS, 20 Hz-5 kHz | **<= 0.8 dB** | below 5 kHz the measurement is trustworthy, so this band must be right |
| worst point, 20 Hz-5 kHz | **<= 2.5 dB** | one such error over a wide span is a colouration; several in a row would be a tuning choice |
| worst point, 5-10 kHz | **<= 5.0 dB** | the data cannot support better; claiming it would be false precision |
| any single boost | **<= +6 dB**, and **<= +3 dB above 5 kHz** | a large narrow boost drives a driver where it is least linear, and above 5 kHz it usually fits a fixture artefact |
| any single cut | **<= -20 dB** | deeper than this means the model is modelling the measurement, not the headphone |
| Q, below / above 5 kHz | **<= 4.5 / <= 3.0** | high-Q shapes create local holes no measurement confirms |
| filters above 5 kHz | **<= 3** | one shelf plus two broad shapes is the most the data can justify |
| chain peak + preamp | **<= -0.5 dBFS** | inter-sample peaks on real masters |

The RMS gates measure the fitter's work. The worst-point gates are capability
questions: a +10 dB dip at 3.8 kHz needs a +10 dB boost to close, and boosting
3.8 kHz by 10 dB to chase one measurement is exactly what the boost cap
forbids. When they disagree the caps win, and the miss is reported with the
number it would have cost -- never silently relaxed, never hidden.

## Refuse, do not guess

A config built on the wrong curve is worse than no config: it costs the user
their listening time and teaches them the EQ does not work. Stop and ask, or
stop and say so, when:

| situation | what to do |
|---|---|
| the model search returns nothing | say so and offer `--curve` with the user's own measurement; do not substitute a similar model |
| the best hit does not contain the model name the user typed | list the hits and ask which one |
| the evaluation publishes more than one response curve | **ask the user which curve**, by its full site label (`2.4G模式频响:SPL(dB)`, `一致性R频响:SPL(dB)`); two curves of one model can differ by more than the correction itself |
| the measurement stops below 10 kHz, or has fewer than 60 points | refuse: the treble cannot be fitted and the rest cannot be resolved |
| the headphone is already within 0.8 dB of the target | refuse: the EQ would be all cost and no benefit |

When the multi-curve row fires, the run stops and prints every curve with its point
count and range. Ask in the user's own terms, quoting the labels verbatim so the
answer maps straight onto `--mode`: "this evaluation publishes 2 curves --
`2.4G模式频响:SPL(dB)` (201 points, 20-20000 Hz) and `一致性R频响:SPL(dB)`. Which one
should the EQ follow?" Then never pick the first one yourself. A different tuning
mode or pad set is a different measurement: fit it from its own curve.


## Running it

```bash
python scripts/generate.py --headphone "HIFIMAN Sundara"
python scripts/generate.py --uuid <evaluation-uuid> --mode "2.4G模式"
python scripts/generate.py --curve my-measurement.csv --label "HD600 mine"
```

On a Windows console set `$env:PYTHONIOENCODING='utf-8'` first, or the Chinese model
names come back as mojibake (purely cosmetic, but it makes the output unreadable).

Outputs land in `out/<slug>/`: two `.txt` configs, one plot each, one `.md`
report each, and a `summary.json` holding the filters, gates and provenance.
Exit code is 0
when both configs pass, 2 when either is short.

When the search finds several curves, list them before fitting:

```bash
python scripts/generate.py --uuid <uuid> --list-curves
```

Verify what you shipped, from the file itself (this catches hand edits):

```bash
python scripts/verify.py "out/<slug>/<name>-GAME.txt" --target game --uuid <uuid>
python scripts/compare.py out/<slug>          # the two configs side by side
```

## Reading the two configs

The GAME config is not the Harman config with a different title. What changes,
and why:

| band | difference | reason |
|---|---|---|
| 60-250 Hz | GAME is **about 2 dB higher**, peaking at +4.1 dB at 226 Hz | the reference carries a bass shelf the Harman target does not; this is the largest real difference between the two curves |
| 20-60 Hz | GAME about **0.4 dB lower**, not higher | the reference shelf sits above the sub-bass, so the deepest octave is actually a little flatter than Harman |
| 2.5-3.5 kHz | GAME up to **+3 dB higher** | the reference has a taller 3 kHz hill than Harman does |
| 5-10 kHz | **not identical**: GAME runs up to +6.9 dB above Harman, with a narrow spike at 8.7 kHz | the reference draws a jagged treble; Harman keeps falling smoothly |

The band weights in `scripts/generate.py` (`WEIGHTS`) encode this: GAME spends
accuracy at 1-4 kHz, where positional cues live, and buys leeway in the
sub-bass. If the intent changes, change the weights -- not the filter list.

These numbers come from `data/blue-dashed.csv` minus
`data/harman_over_ear_2018.csv`, both 1 kHz-normalised.

`blue-dashed.csv` is a **derived artefact**: the curve exists only as a dashed
line in the evaluation site's chart, so it is read off that image by
`scripts/digitise_target.py` from `data/source/game-reference-chart.png` (rerun it
with a new screenshot; the output reproduces byte-for-byte). Two properties keep it
honest:

- the axes come from the chart's OWN printed tick labels -- the y axis from its
  eleven dB labels (0.19 dB residual), the x axis from the plot-box edges
  cross-checked against the printed 1 kHz tick. Nothing is fitted to a curve
  already believed.
- the derivation *checks itself* on the chart's **red** line, which is Harman and
  which the pipeline never sees as a CSV. Digitised red vs the bundled
  `harman_over_ear_2018.csv`: **0.55 dB mean**, and 0.3-0.6 dB per octave up to
  14 kHz, growing only in the steep 14-20 kHz tail where the log axis compresses
  the curve into a near-vertical stroke. A systematic axis error would show up
  there first, so that number is the test -- if a change to the digitiser makes it
  worse, the change is wrong.

Anything that quotes a difference between the two targets and disagrees with
those two files is wrong -- the files are the specification.

## Knobs, and what they are not for

| flag | effect | when it is right |
|---|---|---|
| `--counts` | filter-count search | leave alone; the scan picks the count where another filter stops paying |
| `--no-highpass` | drops the 12 Hz rumble filter | only when the chain has no sub-bass content that could clip |
| `--mode` | chooses the response curve | whenever the evaluation publishes more than one |

Nothing here tunes the caps, weights or gates per headphone. If a fit fails
them, the answer is to report which gate failed and why, not to move it.

## Speed

A single model takes about **3 s** end to end: fetch, analyse, fit both roles, plot,
report. The user is waiting, so the shape of the work is part of the design:

- **The unit of work is the (role, filter-count) candidate, not the role.** Both roles
  and all their counts go into ONE flat queue over every logical core. Running one
  role's counts and then the other's leaves half the machine idle for half the wall
  clock.
- **`--jobs 0` means "use the machine".** The flag defaults to 0 so "how many cores?"
  has one answer -- and reading that 0 literally is how the run spent its life on a
  single core while the docs promised a parallel scan. `_job_count` resolves 0, None
  and negatives to the core count; only an explicit positive number pins it. The
  selftest asserts the auto path is not the serial one.
- **One prune pass, not four.** Each cleanup stage used to carry its own ablation loop
  over the same filter list; they now share one loop and one iteration budget.
- **The screen is closed-form.** Dropping a filter is *predicted* in one vectorised
  call for every filter at once, and only the cheapest few candidates get the full
  re-optimisation that decides. The prediction never accepts a drop.
- **The Jacobian is one batched call.** Perturbing filter *i* cannot move filter *j*,
  so all columns are built together from one magnitude evaluation instead of 2 per
  filter.
- `fixtures.py` needs **one process per fixture** (both roles inside each), never a
  pool inside a worker: a worker that opens its own pool on Windows never returns.
  `fit.in_worker()` enforces this, and the selftest asserts the parallel and serial
  runs produce identical filters.
- `EQ_JOBS` overrides the worker count; `--jobs 1` forces the serial path for debugging.
  Scaling saturates around 5x on 24 cores because one candidate is the critical path --
  past that, make the slowest candidate cheaper rather than adding workers.

Three rules keep it fast *and* correct. Breaking any of them is how this skill got
slow, or wrong, in the past:

- **Compare every candidate at the same iteration budget.** Refining a trial more
  cheaply than the baseline makes every filter look droppable, and the elimination
  then eats working filters one at a time.
- **Predict cheaply, decide expensively.** A closed-form estimate may order the work;
  only a full re-optimisation may accept a removal.
- **Never decide a filter is useless by category.** Blacklisting shapes ("shelves
  below 400 Hz", "Q under 0.6") deleted filters that carried a whole correction on a
  real fixture (0.12 dB became 1.69 dB). Removing the filter and re-optimising the
  rest is the only test that knows what a filter is for.

## Structure

Five layers, so a new target curve or a new fitter does not mean a rewrite:

- `data/` -- target curves and cached measurements. Values, not code.
  `blue-dashed.csv` is derived, and `scripts/digitise_target.py` regenerates it from
  the source chart -- keep them together, or the target silently becomes folklore.
- `scripts/eqkit/` -- the library: `grid` (one analysis grid), `curves`
  (normalise/resample/smooth), `apo` (the biquad model, transcribed from
  `BiQuad.cpp`), `metrics` (bands, gates), `analyse` (smoothing width and
  correction), `fit` (structure, optimiser, prune), `judge` (which curve, when to
  refuse), `emit` (files a human can edit), `fetch` (huihifi and AutoEq).

The fit itself is four stages, in `scripts/eqkit/fit.py`:

| stage | what it does | why there |
|---|---|---|
| `greedy` | picks filter centres and Qs from a shape bank, gain solved in closed form | finds *where* the problems are before anything is optimised |
| `refine` | joint Levenberg-Marquardt on every frequency, gain and Q | two filters that should have moved toward each other never do under a greedy pass |
| `prune` | removes filters that are not worth their slot, verified by full re-optimisation | the only test that knows what a filter is for |
| `quantise` | rounds to what the config file can express | quoting an unrounded number ships a curve nobody hears |

Then the rest:

- `scripts/generate.py` -- the single command that drives both roles.
- `scripts/verify.py`, `compare.py`, `fixtures.py` -- independent checks.
- `fixtures/` -- the frozen regression set; `expected.json` is the contract.

To add a target curve: drop a CSV in `data/`, add it to `TARGET_FILES` and
`WEIGHTS` in `generate.py`. Nothing else changes.

## Why the model matters

`eqkit/apo.py` is a transcription of Equalizer APO's own biquads, not a generic
cookbook. `LS` and `HS` use the Q form: writing `Q 1.8` on a low shelf is not
the shelf-slope form of the same numbers, and the two disagree by up to 6 dB.
Fitting with a generic model and shipping APO Q values is the standard way to
deliver a curve nobody measured. Every number in the reports comes from this
model, and `verify.py` re-derives them from the shipped file.

## Honest limitations, which belong in the report

- **One measurement.** Pad seal and placement move 1-4 kHz by up to 3 dB on the
  same unit.
- **Above 5 kHz it is a trend, not a fit.** Coupler and ear-canal cancellation
  make narrow detail meaningless; the fit uses a full-octave window up there
  and says so.
- **A prediction, not a guarantee.** Unit-to-unit spread is 1-2 dB, and the
  residual assumes the EQ is exactly the model in the file.
- **Tuning modes differ.** A 7.1 or ANC mode needs its own config, from its own
  curve.

## Regression

Two suites, both run on every change to the fitter:

```bash
python scripts/selftest.py            # invariants, no network, ~3 s
python scripts/fixtures.py            # compares against fixtures/expected.json, ~8 s
python scripts/fixtures.py --record   # intentional change: update the contract
```

The selftest guards the invariants that have each already failed once: the batched
Jacobian equals the per-filter one, the prune pass keeps load-bearing filters, the
candidate order survives the parallel scheduler, a pool worker never opens a pool of
its own, `--jobs 0` really does mean auto, and one candidate still fits in seconds.
That last check is the speed contract: it fails before a slow path becomes a slow run
nobody notices.

A change that improves one model and degrades another shows up as a positive
delta on the others. Treat a delta above 0.15 dB as a regression to explain,
not as noise -- and when a delta is a large *improvement*, prove it before
re-recording: ablate every filter in the winning chain and check they are all
load-bearing. That is how the `hyperx-cloud2-stock` row was cleared (0.78 -> 0.31
dB, 10 of 10 filters load-bearing) instead of being waved through.
