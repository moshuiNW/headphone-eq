# headphone-eq

Fit two Equalizer APO configs (Harman over-ear 2018 and the blue dashed game
reference curve) to one headphone measurement. See [SKILL.md](SKILL.md) for the
judgement, gates and refusal rules; this file covers the package itself.

## Requirements

- Python 3.9+, `numpy`, `matplotlib` (plots), no other dependencies.
- Network access for fetching measurements from huihifi.com. Behind a proxy, set
  `HTTPS_PROXY` or pass `--proxy http://127.0.0.1:7897`.
- On a Windows console set `$env:PYTHONIOENCODING='utf-8'` before running anything, or
  the Chinese model names print as mojibake.

## Quick start

```bash
cd skills/headphone-eq
python scripts/generate.py --headphone "HIFIMAN Sundara"
python scripts/compare.py out/<slug>
```

## Layout

```
SKILL.md              judgement: what counts as fitted, when to refuse
data/                 target curves (harman_over_ear_2018.csv, blue-dashed.csv [derived])
data/headphones/      cached measurements + meta.json per headphone
scripts/eqkit/        the library (see SKILL.md "Structure")
scripts/generate.py   both configs in one run
scripts/verify.py     re-derive a shipped config from the file
scripts/compare.py    the two configs side by side
scripts/fixtures.py   regression set against fixtures/expected.json
scripts/digitise_target.py  rebuild data/blue-dashed.csv from the source chart
data/source/          the source screenshot for the derived target
out/                  generated configs, plots, reports
fixtures/             frozen regression cases and expectations
```

## Data provenance

| file | origin |
|---|---|
| `data/harman_over_ear_2018.csv` | AutoEq `targets/Harman over-ear 2018.csv`, verified identical to the project copy |
| `data/blue-dashed.csv` | the game reference curve, digitised from the evaluation site's chart by `scripts/digitise_target.py`. Axes calibrated from the chart's own printed tick labels; the derivation is self-checked by digitising the chart's *red* Harman line and comparing it with the bundled Harman CSV (0.55 dB mean, 0.3-0.6 dB per octave to 14 kHz). Reproduces byte-for-byte from `data/source/game-reference-chart.png`; rerun the script to refresh it from a new screenshot |

Both are shape-normalised to 1 kHz before use: absolute SPL carries no tonal
information and differs by tens of dB between fixtures.

## Measurement sources

1. **huihifi** (default) -- client-rendered pages, so the curve only exists at
   the JSON API. Endpoints used: `POST /api/v1/evaluations` (search),
   `GET /api/v1/evaluations/<uuid>` (record, one RAW_FR child per tuning mode),
   `GET /api/v1/global-configs/unique` (the site's own targets).
2. **AutoEq** -- a second independent measurement, and its target curves.
3. **file** -- your own CSV via `--curve`.

## Tests

```bash
python scripts/selftest.py            # invariants, no network, ~3 s
python scripts/fixtures.py            # regression run (needs network, ~8 s, parallel)
python scripts/fixtures.py --record   # re-freeze after an intentional change
```

Both suites exist because every defect below shipped silently, and a smoke test
would have missed it:

| defect | symptom | test that now catches it |
|---|---|---|
| filter deleted for sitting on a boundary (400.0000000000001 > 400.0) | fit "mysteriously collapsed" to one filter | `no filter is deleted for being misplaced` |
| `quantise` merged neighbours and clipped the sum at the -20 dB cap | residual jumped 0.65 → 2.62 dB *after* the good number was reported | `quantise` fidelity + fixtures |
| shelf Q clamped upward during cleanup | a -20 dB low shelf silently became a different shape | `shelf Q` violation gate |
| the scheduler sorted the work queue but not the results | one candidate's filters were reported under another count, and a 1.5 dB chain could win | `parallel results keep their candidate order` |
| `fixtures.py` opened a pool inside its pool workers | the run hung until killed and never produced a row (Windows) | `in_worker()` + one process per fixture in `fixtures.py` |
| `--jobs 0` (the default) was read as "zero workers" | every run silently fitted on one core while the docs promised a parallel scan | `jobs=0 means auto, not serial` |

## Performance

One headphone, both targets: **~3 s** wall clock on a 24-core machine (i9-13980HX),
against 139 s when the same model was last handed over. Both roles and all of their
filter-count candidates run as one flat queue across every logical core.

| task | time |
|---|---|
| `generate.py` one model, both targets | 3.3 s |
| `fixtures.py` (10 fixtures, 20 fits, parallel) | 9.2 s |
| `selftest.py` | 2.9 s |
| `verify.py` one shipped config | 0.7 s |

Where the time goes now, per run: fetch 0.1 s, prepare 0.5 s, fit 1.9 s (the slowest
single candidate is ~1.2 s and is the critical path), plots and files 0.8 s.

Where the 139 s went, and what replaced it:

| cost | why it was there | what it became |
|---|---|---|
| `--jobs 0` read literally as zero workers | the flag defaults to 0 to mean "auto" | `_job_count` resolves 0/None/negative to the core count; this alone was most of the 139 s |
| four cleanup passes, each with its own ablation loop (`refine` called 235 times per scan) | every pass wanted its own "is this filter useless?" answer | one `prune` pass, one budget, one loop |
| full `refine` for *every* candidate removal | the only test that knows what a filter is for | same test, but a closed-form screen picks the 3 cheapest candidates per round first |
| 2 magnitude calls per Jacobian column | one column at a time | 1 batched call for all columns |
| LM running to 140 iterations regardless | a fixed budget | early stop after 18 non-improving steps (the tail buys nothing) |
| the two roles fitted one after the other | each had its own pool | one queue for both |

The speed-ups are only safe because they do not change the answer: the selftest
asserts the batched Jacobian equals the per-filter one, that the parallel and serial
runs return identical filters, and that a single candidate still fits in under 5 s
(so a regression surfaces as a failing test rather than as a slow run).