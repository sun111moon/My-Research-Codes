# Test cases — DeltScan

Input and expected output for every workflow script, with numeric pass
criteria. This is the document that answers the CPCiP requirement that "a test
run, with its input and its output" be shipped with the program.

Reference for the physics and the meaning of the numbers:
`docs/SUPPLEMENTARY.md`. Installation and the command reference:
`docs/USER_MANUAL.md`.

---

## 1. The two tiers

**Tier A — cheap tests (the test run of the submission).** Four commands, none
of which integrates the equation; together they take a couple of minutes on the
bundled data. They are automated by `tests/run_all.sh`. These are TC-01 to
TC-04 below, and they are the ones to quote in the submission.

**Tier B — code-path tests.** Short integrations (`T = 2`) that exercise the
integrator without producing publishable statistics. TC-05 to TC-08.

**Reference runs (not part of the test suite).** The production runs behind
Tables 2–4 take from twenty minutes to several hours; they are documented in
§6 of `docs/SUPPLEMENTARY.md` and are not re-run as a test.

---

## 2. Capturing the expected output

Each test writes its output to `tests/out/TC-xx.log` when run through
`tests/run_all.sh`. To snapshot the current, accepted output as the reference:

```
mkdir -p tests/expected
python 04_same_field_control.py --selftest                     2>&1 | tee tests/expected/TC-01.txt
python 06_analyze_ensemble.py --npz ensemble_results_1000.npz  2>&1 | tee tests/expected/TC-02.txt
python 07_bootstrap_statistics.py                              2>&1 | tee tests/expected/TC-03.txt
python 08_bootstrap_ci.py                                      2>&1 | tee tests/expected/TC-04.txt
```

The values quoted in the sections below are those of the release `v1.0.0`
bundled data; they were produced on Python 3.14 with numpy 2.5.5, scipy 1.18.1,
matplotlib 3.11.2 and pyFFTW 0.15.1.

---

## 3. Tier A — cheap tests

### TC-01 — band-limited resampling self-test

```
python 04_same_field_control.py --selftest
```

- **Input**: none (fields and reference grid generated in memory).
- **Runtime**: seconds.
- **Expected output**:

```
selftest: N_ref = 16384, max|psi| = 2.916333, <|psi|^2> = 4.257301
  N=4096 idempotence err = 1.530e-15  discarded amplitude = 4.072e-03  mean-intensity rel.err = 2.476e-06
  N=8192 idempotence err = 1.527e-15  discarded amplitude = 3.605e-03  mean-intensity rel.err = 1.682e-06
selftest: PASS
```

- **Pass criteria**:
  - last line contains `selftest: PASS`;
  - both idempotence errors < `1e-12`;
  - both mean-intensity relative errors < `1e-4`;
  - `<|psi|^2>` = `4.257301` at `N_ref = 16384`, i.e. the `4.26` quoted in the
    manuscript (the `N = 4096` grid in the production convention gives `4.22`).

### TC-02 — ensemble summary (Table 4, Fig. 2)

```
python 06_analyze_ensemble.py --npz ensemble_results_1000.npz
```

- **Input**: `ensemble_results_1000.npz` (bundled).
- **Runtime**: seconds.
- **Expected output** (table reproduced verbatim):

```
 N     n    r_max med      IQR          kappa med    IQR        P(k>3.9)  skew(k)
 4096 1000  80.28 [ 67.97, 97.97]     23.308 [21.271,24.976]      99.5%    -2.02
 8192 1000  16.96 [ 13.74, 58.80]      2.901 [ 2.679, 3.752]      24.1%    +2.89
16384 1000  14.43 [ 12.51, 16.78]      2.645 [ 2.513, 2.815]       1.2%    +3.74
```

- **Side effects**: `figures/fig2_ensemble.pdf`, `figures/fig2_ensemble.png`.
- **Pass criteria**:
  - median kurtosis within 1 % of `23.308` / `2.901` / `2.645`;
  - `P(kappa > 3.9)` within ±0.005 of `0.995` / `0.241` / `0.012`;
  - median `r_max` within 1 % of `80.28` / `16.96` / `14.43`;
  - the three medians are **monotonically decreasing** with `N` — this
    monotonicity, not any single number, is the physical claim of the paper.

### TC-03 — bootstrap sampling variability and apparent gaps

```
python 07_bootstrap_statistics.py
```

- **Input**: `ensemble_results_1000.npz` (bundled).
- **Runtime**: minutes.
- **Expected output**, key lines:

```
[N=4096]  n=1000, kappa median=23.308, r_max median=80.28
[N=8192]  n=1000, kappa median=2.901,  r_max median=16.96
[N=16384] n=1000, kappa median=2.645,  r_max median=14.43

N=4096:  m=20: kappa median 5-95% = [22.087, 24.428] (SE ~ 0.712)
N=8192:  m=20: kappa median 5-95% = [2.749, 3.248]   (SE ~ 0.152)
N=16384: m=20: kappa median 5-95% = [2.545, 2.743]   (SE ~ 0.060)

N=8192: apparent-gap fraction at m=20:   96.25%
N=8192: apparent-gap fraction at m=50:   64.55%
N=8192: apparent-gap fraction at m=100:  21.95%
N=8192: apparent-gap fraction at m=200:   1.25%
N=8192: apparent-gap fraction at m=500:   0.00%
N=8192: apparent-gap fraction at m=1000:  0.00%
```

- **Pass criteria**:
  - the three medians equal those of TC-02;
  - the `m = 20` intervals **widen** as the tail gets heavier
    (`0.712 → 0.152 → 0.060` for the kurtosis SE);
  - the apparent-gap fraction decreases monotonically in `m` and reaches
    `0.00%` at `m = 1000`. This monotone decay is the evidence behind the
    claim that a small sample can hide the tail altogether.

### TC-04 — percentile-bootstrap confidence intervals

```
python 08_bootstrap_ci.py
```

- **Input**: `ensemble_results_1000.npz` (bundled).
- **Runtime**: minutes.
- **Expected output**:

```
percentile bootstrap, B = 20000, seed = 0
    grid    n   median kappa [95% CI]      P(kappa>3.9) [95% CI]   median rmax [95% CI]
    4096 1000   23.3076 [23.0343,23.5472]  0.9950 [0.9900,0.9990]   80.277 [78.599,82.106]
    8192 1000    2.9012 [ 2.8695, 2.9433]  0.2410 [0.2150,0.2680]   16.962 [16.229,17.538]
   16384 1000    2.6453 [ 2.6265, 2.6641]  0.0120 [0.0060,0.0190]   14.427 [14.187,14.715]
```

- **Side effects**: `figures/bootstrap_ci_summary.txt`.
- **Pass criteria**:
  - `B = 20000`, `seed = 0`, so the intervals are reproducible exactly;
  - the point estimates match TC-02;
  - each interval brackets its point estimate, and the interval width
    **shrinks** from `N = 4096` to `N = 16384`;
  - the `N = 16384` kurtosis interval lies **above** the Gaussian value `2`.

---

## 4. Tier B — code-path tests

These need the snapshots or a short integration. They are marked *pending* in
the release until their expected output has been captured once on the authors'
machine.

### TC-05 — protocol checks, short mode  *(pending)*

```
python 01_run_convergence.py --quick
```

- **Input**: none; writes `results/psi_N8192_t60.npy` and
  `results/psi_N16384_t60.npy` (at `t = 60` in the full run, shorter in `--quick`).
- **Runtime**: minutes.
- **Expected**: the same-field refinement, production, dealiasing and
  time-step blocks all print, ending without a traceback.
- **Pass criteria**: exit status 0; both snapshot files written; the same-field
  `r_max` sequence is non-increasing in `N` (`90.1 → 30.0 → 27.5` in the full
  run — for `--quick` only the presence and ordering are checked).
- **To capture**: `python 01_run_convergence.py --quick 2>&1 | tee tests/expected/TC-05.txt`

### TC-06 — delta fingerprint test  *(pending; requires a completed TC-05-equivalent full run)*

```
python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy results/psi_N16384_t60.npy
```

- **Input**: the two `t = 60` snapshots written by `01`.
- **Runtime**: seconds.
- **Expected**: the `N = 8192` peak falls from `233.2` on the first cutoff,
  tracking the linear baseline (single-grid-cell delta fingerprint, `R^2 = 0.992`);
  the `N = 16384` peak is flat to within `6 %` until the cutoff approaches the
  spectral width of the physical core.
- **Pass criteria**: `R^2 >= 0.98` for the `N = 8192` linear tracking;
  the `N = 16384` plateau deviates by less than `6 %` over the first several
  cutoffs.
- **To capture**: `python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy results/psi_N16384_t60.npy 2>&1 | tee tests/expected/TC-06.txt`

**Note.** `02` prints text only — its imports are `numpy` and `fnls_core`, with no
plotting back end and no `savefig`. It therefore produces **Table 1** and does
**not** draw Fig. 1. Fig. 1 and Fig. 3 are drawn by `09_make_fig1_fig3.py`
(TC-09) from the same two snapshots. A package-wide `grep -n savefig *.py` finds
only `06` (Fig. 2), `07` (bootstrap diagnostics) and `09` (Fig. 1, Fig. 3).

### TC-07 — reproduction of the published single-realization numbers  *(pending)*

```
python 03_reproduce_published_tables.py --all
```

- **Input**: none (self-contained; fixed seed `300876`).
- **Runtime**: seconds to minutes.
- **Expected**: the global maxima of **Table 2**, `r_max = 90.4`, `257.3` and
  `27.5` on `N = 4096, 8192, 16384`.
- **Pass criteria**: all three values reproduced to within 1 %; the values are
  **not** monotone in `N` — that non-monotonicity is the point of Table 2.
- **To capture**: `python 03_reproduce_published_tables.py --all 2>&1 | tee tests/expected/TC-07.txt`

### TC-08 — ensemble driver, smoke mode  *(pending)*

```
python 05_run_ensemble.py --smoke
```

- **Input**: none; writes a small `.npz`.
- **Runtime**: minutes.
- **Expected**: a few short runs complete, the `.npz` is written, and re-running
  the same command skips the records already present (checkpointing).
- **Pass criteria**: exit status 0; the archive is created; the second
  invocation adds no new records and prints that it skipped existing ones.
- **To capture**: `python 05_run_ensemble.py --smoke 2>&1 | tee tests/expected/TC-08.txt`

### TC-09 — Figures 1 and 3, drawn from the snapshots  *(pending; requires the snapshots of `01`)*

```
python 09_make_fig1_fig3.py
```

- **Input**: the two `t = 60` snapshots written by `01` — the same files as
  TC-06; it cannot be run before `01`.
- **Runtime**: seconds.
- **Output**: `figures/fig1_lowpass_test.pdf/.png` (Fig. 1) and
  `figures/fig3_resolution_snapshot.pdf/.png` (Fig. 3).
- **Expected**: the `N = 8192` points of Fig. 1 reproduce the seven Table 1
  values (`233.2 → 33.2`) along the linear baseline; the spike of Fig. 3 is
  annotated with `r ≈ 233` and stands about `14.7×` above its immediate
  neighbours.
- **Pass criteria**: the four figure files are written; the printed `r` values
  agree with TC-06 to the printed digits; the printed neighbour factor lies in
  `[13, 16]`; and the `N = 16384` sequence is non-increasing across the seven
  cutoffs.
- **To capture**: `python 09_make_fig1_fig3.py 2>&1 | tee tests/expected/TC-09.txt`

---

## 5. Coverage of the manuscript objects

| Manuscript object | Test case | Status |
|---|---|---|
| Table 1 (low-pass fingerprint) | TC-06 | pending capture |
| Table 2 (independent realizations) | TC-07, TC-05 | pending capture |
| Table 3 (same physical field) | TC-01 (self-test), full `04` for the table | TC-01 automated |
| Table 4 (ensemble statistics) | TC-02 | automated |
| Fig. 1 (low-pass test) | TC-09 (drawn), numbers also from TC-06 | pending; numbers to reconcile |
| Fig. 2 (ensemble) | TC-02 | automated |
| Fig. 3 (resolution snapshot) | TC-09 (drawn) | pending; numbers to reconcile |
| Confidence intervals quoted in the text | TC-04 | automated |
| Sampling variability, apparent gaps | TC-03 | automated |
| Dealiasing and time-step checks | TC-05 | pending capture |

---

**Figure provenance.** `grep -n savefig *.py` on the authors' machine returns
`06_analyze_ensemble.py:98` (Fig. 2) and three calls in
`07_bootstrap_statistics.py` (lines 182, 199, 219 — the bootstrap diagnostics,
one of them `figures/small_sample_gaps.pdf`). Fig. 1 and Fig. 3 used to be drawn
by one-off scripts kept outside `supplementary/`; `09_make_fig1_fig3.py` now
draws both from the two `t = 60` snapshots that `01` writes, computing every
annotation from the data instead of hard-coding it.

---

## 6. Acceptance checklist

- [ ] `bash tests/run_all.sh` ends with `ALL AUTOMATED CHECKS PASSED`.
- [ ] TC-05 to TC-09 captured once and stored under `tests/expected/`.
- [ ] `python 09_make_fig1_fig3.py` reproduced Fig. 1 and Fig. 3 and wrote all
      four figure files.
- [ ] The runtime column of §6 of `docs/SUPPLEMENTARY.md` agrees with what the
      test runs actually take on the submitting machine.
