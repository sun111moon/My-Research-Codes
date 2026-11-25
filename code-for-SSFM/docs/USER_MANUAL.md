# DeltScan — User Manual

Version 1.0.0 · companion software for

> **Fidelity bounds and a single-grid convergence protocol for extreme-event
> statistics in integrable turbulence**
> Hongcai Ma, Aiping Deng — College of Mathematics and Statistics, Donghua University
> Target journal: *Computer Physics Communications* (CP / CPiP)

This manual covers installation, the command-line reference, output formats,
the mapping from the manuscript objects to the scripts, and troubleshooting.
`docs/SUPPLEMENTARY.md` holds the physics (initial-condition conventions,
parameters, expected numbers); `docs/TEST_CASES.md` holds the test runs with
their inputs, expected outputs and pass criteria.

---

## 1. What the package does

The package integrates the focusing one-dimensional nonlinear Schrödinger
equation with random initial data and diagnoses whether a large peak in the
solution is a *physical* extreme event or a *numerical* artefact. Two tools do
the work:

- a **low-pass ("delta") fingerprint test**, which distinguishes a
  single-grid-cell spike from a resolved peak by watching how the peak
  amplitude responds to lowering a spectral cutoff;
- a **three-step convergence protocol**, which fixes the grid at which the
  extreme-event statistics stop moving.

Everything runs in double precision on a single node, in NumPy, with no
compilation step and no GPU.

---

## 2. Installation

Requirements: 64-bit Linux or macOS, Python ≥ 3.10, NumPy / SciPy /
Matplotlib. No compiler, no MPI, no GPU.

```
git clone https://github.com/sun111moon/My-Research-Codes.git
cd My-Research-Codes
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the validated combination (numpy 2.5.5, scipy 1.18.1,
matplotlib 3.11.2, pyFFTW 0.15.1 on Python 3.14).

The FFT backend is optional:

- if **pyFFTW** is installed it is used automatically and is appreciably faster
  at `N = 16384`;
- if it is absent (or fails to import) the code falls back to `numpy.fft` and
  produces identical numbers.

There is no package to install: the scripts are run from the repository root as
plain files, not as an installed library.

---

## 3. Package layout

| Path | Contents |
|---|---|
| `01_run_convergence.py` … `09_make_fig1_fig3.py` | The workflow drivers; run them in numeric order. |
| `fnls_core.py` | Shared core: grid construction, production initial condition, Fourier resampling, diagnostics (`r_max`, kurtosis, low-pass filter) and the symmetrized split-step Fourier integrator. Dealiasing is off by default. |
| `ensemble_results_1000.npz` | Bundled ensembles, `n = 1000` at each of `N = 4096, 8192, 16384` (seeds `300000`–`300999`), 3000 records; ~0.1 MB. |
| `ensemble_results.npz` | Small `n = 20` companion (seeds `100`–`119`) for the runtime examples. |
| `figures/` | Figures as used in the manuscript, plus `bootstrap_ci_summary.txt` written by `08`. |
| `docs/` | This manual, `TEST_CASES.md` and `SUPPLEMENTARY.md`. |
| `tests/run_all.sh` | Runs the four cheap test cases and prints PASS/FAIL. |
| `README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, `CITATION.cff` | Orientation, GPLv3 licence, pinned dependencies, ignore rules, citation metadata. |

---

## 4. Five-minute check, no integration

Run from the repository root. None of these integrates the equation; they read
the bundled data and exercise the analysis path.

```
python 04_same_field_control.py --selftest
python 06_analyze_ensemble.py --npz ensemble_results_1000.npz
python 07_bootstrap_statistics.py
python 08_bootstrap_ci.py
```

Or simply `bash tests/run_all.sh`, which runs exactly these four and reports
PASS/FAIL against the numbers in `docs/TEST_CASES.md`.

---

## 5. Physical and numerical parameters

Defaults, as used for every number in the manuscript. The authoritative table
with definitions and provenance is §2 of `docs/SUPPLEMENTARY.md`.

| Quantity | Value |
|---|---|
| Model | `i psi_t + 0.5 psi_xx + abs(psi)^2 psi = 0` |
| Domain | periodic, `x in [-250, 250]` (`L = 500`) |
| Amplitude `A0` | `2` |
| Correlation lengths `ell_A = ell_S` | `50` |
| Semiclassical parameter `eps` | `0.1` |
| Phase-noise amplitude | `1e-3` |
| Time step `dt` | `2e-4` |
| Final time `T` | `100` (delta-test snapshot at `t = 60`) |
| Grids `N` | `4096`, `8192`, `16384` |
| Reference grid `N_ref` | `16384` |
| Dealiasing | 2/3 Orszag rule, **off** unless requested |
| Default seed | `300876`; ensembles use `300000`–`300999`, small companion `100`–`119` |

Kurtosis is `kappa = <I^2> / <I>^2` with `I = abs(psi)^2`; the Gaussian
reference value for circular complex fields is `kappa = 2`.

---

## 6. Command reference

Every integrating script accepts `--quick`, which shortens the integration to
`T = 2` for a fast check of the code path, and `--help`. The flags listed per
script below are those exercised for the manuscript; run `python <script> --help`
for the complete list (see Appendix A for a one-liner that dumps them all).
`09_make_fig1_fig3.py` integrates nothing and takes only the two snapshot paths.

### 6.1 `01_run_convergence.py`

Runs every protocol check: the same-field spatial refinement producing
**Table 3**, the production runs of **Table 2**, dealiasing at fixed `N`, and
the halving of the time step. Also writes the two `t = 60` snapshots consumed by
`02`.

```
python 01_run_convergence.py            # full, ~1.5 h
python 01_run_convergence.py --quick    # T = 2, minutes
```

Outputs: `results/psi_N8192_t60.npy`, `results/psi_N16384_t60.npy`, and the
printed protocol tables.

### 6.2 `02_lowpass_fingerprint.py`

The delta fingerprint test at `t = 60` (**Table 1**). Prints the numbers only;
Fig. 1 is drawn by `09`. Takes the snapshots as positional arguments.

```
python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy results/psi_N16384_t60.npy
```

**Requires** the snapshots from a completed `01`; it cannot be run first.

### 6.3 `03_reproduce_published_tables.py`

Self-contained reproduction of the published single-realization numbers with
the exact parameters and seed of that data set.

```
python 03_reproduce_published_tables.py --all
```

### 6.4 `04_same_field_control.py`

Isolated, minimal reproduction of **Table 3**, with a self-test of the
band-limited resampling.

```
python 04_same_field_control.py --selftest   # seconds
python 04_same_field_control.py              # three integrations, ~20 min
```

### 6.5 `05_run_ensemble.py`

Ensemble driver: many independent realizations (seeds) at several grids, in
parallel, with checkpointing. Produces the `.npz` files.

```
python 05_run_ensemble.py --n-seeds 1000 --seed-start 300000 --T 100
python 05_run_ensemble.py --smoke            # a few short runs
```

Restartable: any `(seed, N, A0, dt, T, dealias)` tuple already present in the
output `.npz` is skipped, so an interrupted run resumes where it stopped.

### 6.6 `06_analyze_ensemble.py`

Median / IQR summary (**Table 4**) and the ensemble figure (Fig. 2).

```
python 06_analyze_ensemble.py --npz ensemble_results_1000.npz
```

Writes `figures/fig2_ensemble.pdf` and `figures/fig2_ensemble.png`.

### 6.7 `07_bootstrap_statistics.py`

Bootstrap analysis of the `N = 4096`, `8192`, `16384` thousand-realization
ensembles: how the median and the tail probabilities converge with sample
size, and the *apparent-gap* fraction (the chance that a too-small sample hides
the tail entirely).

```
python 07_bootstrap_statistics.py
```

### 6.8 `08_bootstrap_ci.py`

Percentile-bootstrap 95% confidence intervals of the **Table 4** statistics at
`n = 1000` (`B = 20000`, `seed = 0`) — the intervals quoted in the text.

```
python 08_bootstrap_ci.py
```

Writes `figures/bootstrap_ci_summary.txt`.

### 6.9 `09_make_fig1_fig3.py`

Draws **Fig. 1** (the low-pass delta fingerprint) and **Fig. 3** (the resolution
snapshot) from the two `t = 60` snapshots. It reads the same input as `02`, and
its retained-peak values agree with Table 1 by construction. Every annotation
(`r`, and the factor by which the spike exceeds its immediate neighbours) is
computed from the snapshots, never hard-coded.

```
python 09_make_fig1_fig3.py
python 09_make_fig1_fig3.py results/psi_N8192_t60.npy results/psi_N16384_t60.npy
```

**Requires** the snapshots from a completed `01`, like `02`. Writes
`figures/fig1_lowpass_test.pdf/.png` and `figures/fig3_resolution_snapshot.pdf/.png`.

---

## 7. Output formats

**Ensemble archives (`.npz`).** One record per `(seed, N)` pair, holding scalar
per-realization diagnostics (`r_max`, kurtosis, mean intensity, …) rather than
the fields themselves, which is why the thousand-realization file is only
~0.1 MB. The keys read by `06_analyze_ensemble.py` are `Ns` (the grid size of
each record), `rmax_final` and `kurt_final` (final `r_max` and final kurtosis of
one realization). List the exact keys of any archive with

```
python - <<'PY'
import numpy as np
d = np.load("ensemble_results_1000.npz", allow_pickle=True)
print(d.files)
for k in d.files:
    a = d[k]
    print("  ", k, a.shape, a.dtype)
PY
```

**Snapshots (`.npy`).** Complex `psi` on the grid at the snapshot time; written
by `01`, read by `02` and `09`.

**Figures (`figures/`).** Vector PDF plus raster PNG where the script emits
both. The manuscript uses the vector versions.

**Text tables.** Printed to standard output, and for the confidence intervals
also to `figures/bootstrap_ci_summary.txt`.

---

## 8. Reproducing each object of the manuscript

| Manuscript object | Script | Command | Runtime |
|---|---|---|---|
| Table 1 | `02` | `python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy results/psi_N16384_t60.npy` | seconds, after `01` |
| Fig. 1 (low-pass test) | `09` | `python 09_make_fig1_fig3.py` | seconds, after `01` |
| Fig. 3 (resolution snapshot) | `09` | `python 09_make_fig1_fig3.py` | seconds, after `01` |
| Table 2 | `01` (part B), `03` | `python 01_run_convergence.py` / `python 03_reproduce_published_tables.py --all` | minutes / seconds |
| Table 3 | `01` (part A), `04` | `python 04_same_field_control.py` | ~20 min |
| Table 4, Fig. 2 | `06` | `python 06_analyze_ensemble.py --npz ensemble_results_1000.npz` | seconds |
| Bootstrap diagnostics (not in the manuscript) | `07` | `python 07_bootstrap_statistics.py` | minutes |
| Bootstrap intervals in the text | `08` | `python 08_bootstrap_ci.py` | minutes |
| Sampling variability, apparent gaps | `07` | `python 07_bootstrap_statistics.py` | minutes |

Without re-running any simulation, Tables 4 and the bootstrap results come
straight from the bundled `.npz` files.

---

## 9. Parallelism, restarts and cost

- `05_run_ensemble.py` parallelises over realizations; the number of workers is
  set on the command line. Twelve processes are comfortable on a modern laptop.
- Full ensemble: 60 integrations (1000 seeds × 3 grids is one long run; the
  bundled file was produced with the `--n-seeds 1000` sweep). Several hours on
  four cores, about half an hour with a dozen processes.
- Interrupting `05` is safe; re-running the same command continues.
- Peak memory is set by the largest grid; `N = 16384` with a handful of
  work arrays needs well under 1 GB per process.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: module 'numpy' has no attribute 'float'` (or `int`, `complex`) | removed NumPy 2 aliases | replace with `float` / `int` / `complex`; see `requirements.txt` for the pinned versions |
| `AttributeError: module 'numpy' has no attribute 'trapz'` | renamed in NumPy 2 | use `numpy.trapezoid` |
| `AttributeError: 'matplotlib.cm' has no attribute 'get_cmap'` | removed in modern Matplotlib | use `matplotlib.colormaps[...]` |
| Numbers differ slightly from the manuscript | a different FFT backend or BLAS | expected at round-off level; install `pyFFTW` and re-check |
| `FileNotFoundError: results/psi_N8192_t60.npy` | `02` or `09` run before a completed `01` | run `01` first; it writes both snapshots at `t = 60` |
| `git push` rejected, `non-fast-forward` | the remote already has a commit on that branch | clone and commit on top, do not `git init` in a fresh directory |
| A run is slower than documented | `--quick`/`--smoke` not used, or too few workers for `05` | use the short modes for checks; raise the worker count for `05` |

---

## 11. Licence, citation and program library

- **Licence**: GPLv3 (`LICENSE`). Because the optional `pyFFTW` backend is GPL,
  the package is distributed under GPL. Switching to MIT would require dropping
  `pyFFTW` in favour of `numpy.fft`.
- **Citation**: cite the software paper (see `CITATION.cff`); the Zenodo DOI of
  release `v1.0.0` is filled in once the release is archived.
- **Program library**: for a CPCiP submission, archive the release as a single
  zip together with this manual and `docs/TEST_CASES.md`, and point the Data
  availability statement at the repository and the DOI.

---

## Appendix A. Completing this command reference

Section 6 documents the flags used for the manuscript. To regenerate the
complete list of options of every driver:

```
for f in 0*.py fnls_core.py; do
  echo "### $f"; python "$f" --help; echo
done > docs/cli_dump.txt
```

Paste that file's contents under the matching subsection of §6 whenever the
scripts change.
