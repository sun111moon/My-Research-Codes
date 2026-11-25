# DeltScan — reproducibility package

Companion code for

> **Fidelity bounds and a single-grid convergence protocol for extreme-event statistics in integrable turbulence**
> Hongcai Ma, Aiping Deng — College of Mathematics and Statistics, Donghua University
> Target journal: *Computer Physics Communications* (CP / CPiP)

The package separates a *genuine* extreme event from a *numerical* one through a
low-pass ("delta") fingerprint test, and it fixes the grid at which the
extreme-event statistics stop moving (the three-step convergence protocol). It
reproduces every numerical table of the manuscript and implements the protocol
recommended there.

## Contents

Workflow scripts — run them in the order of their number.

| # | File | Purpose |
|---|---|---|
| 01 | `01_run_convergence.py` | All protocol checks: same-field spatial refinement (**Table 3**), the production runs of **Table 2**, dealiasing at fixed `N`, and halving of the time step. Also writes the two `t = 60` snapshots used by `02`. |
| 02 | `02_lowpass_fingerprint.py` | The low-pass (delta) fingerprint test at `t = 60`: **Table 1** and the data behind Fig. 1. |
| 03 | `03_reproduce_published_tables.py` | Self-contained reproduction of the published single-realization numbers, with the exact parameters and seed of the data set. |
| 04 | `04_same_field_control.py` | Isolated, minimal reproduction of **Table 3**, with a self-test of the resampling. |
| 05 | `05_run_ensemble.py` | Ensemble driver: many independent realizations (seeds) at multiple grids, in parallel, with checkpointing. Produces `ensemble_results.npz`. |
| 06 | `06_analyze_ensemble.py` | Median/IQR summary (**Table 4**) and the ensemble figure (Fig. 2). |
| 07 | `07_bootstrap_statistics.py` | Bootstrap analysis of the `N = 4096`, `8192` and `16384` thousand-realization ensembles: quantifies how the median and tail probabilities converge with sample size. |
| 08 | `08_bootstrap_ci.py` | Percentile-bootstrap 95% confidence intervals of the **Table 4** statistics at the full sample size `n = 1000` (the intervals quoted in the text). Writes `figures/bootstrap_ci_summary.txt`. |

Supporting files

- `fnls_core.py` — shared core module: grid, production initial condition,
  diagnostics (`r_max`, kurtosis, low-pass filter) and the symmetrized
  split-step Fourier integrator. Dealiasing is off by default.
- `ensemble_results_1000.npz` — bundled `n = 1000` ensembles at all three grids
  (`N = 4096, 8192, 16384`; seeds `300000`–`300999`; 3000 records in total).
  These are the ensembles behind **Table 4** and Fig. 2. Small file (~0.1 MB),
  shipped inside the repository — no Git LFS or external download needed.
- `ensemble_results.npz` — small `n = 20` companion (seeds `100`–`119`) kept for
  the runtime examples.
- `figures/` — figures as they appear in the manuscript (plus
  `bootstrap_ci_summary.txt` from `08`).
- `docs/SUPPLEMENTARY.md` — conventions, physical and numerical parameters, and
  the outputs every script is expected to reproduce.
- `CITATION.cff` — citation metadata (add the Zenodo DOI once release `v1.0.0`
  is archived).
- `README.md`, `LICENSE`, `requirements.txt`.

Both `.npz` files can be regenerated with `05_run_ensemble.py` (see the examples
in the script header).

## Requirements

- Python >= 3.10 on a 64-bit Linux or macOS machine
- Dependencies: see `requirements.txt`
  (`numpy`, `scipy`, `matplotlib`; `pyFFTW` is used as the fast FFT backend and
  the code falls back to `numpy.fft` when it is not installed)

## Installation

```
git clone https://github.com/<account>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```
pip install numpy matplotlib   # pyfftw optional, used if present
cd supplementary

# (a) every protocol check, and the snapshots for the delta test   (~1.5 h)
python 01_run_convergence.py

# (b) the delta fingerprint test, from the snapshots of (a)
python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy \
                                 results/psi_N16384_t60.npy

# (c) the published single-realization numbers
python 03_reproduce_published_tables.py --all

# (d) the same-field control on its own, with a resampling self-test
python 04_same_field_control.py --selftest
python 04_same_field_control.py

# (e) the ensemble of Table 4: 1000 seeds x 3 grids, 12 processes
python 05_run_ensemble.py --n-seeds 1000 --seed-start 300000 --T 100
python 06_analyze_ensemble.py --npz ensemble_results_1000.npz

# (f) bootstrap statistics and confidence intervals
python 07_bootstrap_statistics.py
python 08_bootstrap_ci.py
```

Every script accepts `--quick`, which shortens the integration to `T = 2` for a
fast check of the code path. `05_run_ensemble.py` also accepts `--smoke` for a
few short runs. Both `05` and `06` are restartable: `05` skips any
`(seed, N, A0, dt, T, dealias)` tuple already present in `ensemble_results.npz`,
so an interrupted run resumes where it stopped.

## Documentation of the conventions and expected numbers

`docs/SUPPLEMENTARY.md` records the three initial-condition conventions, the
physical and numerical parameters, and the values each script should reproduce
(**Tables 1–4**, the dealiasing and time-step checks, runtimes). It is the
reference for interpreting the outputs; this README covers installation, the
file list and the run order only.

## Citation

If you use this code, please cite

> H. Ma, A. Deng, *Fidelity bounds and a single-grid convergence protocol for
> extreme-event statistics in integrable turbulence*, Comput. Phys. Commun. (submitted).

Citation metadata is provided in `CITATION.cff`; the Zenodo DOI of release
`v1.0.0` is added once the release is archived.

## License

GPLv3 — see `LICENSE`. The optional `pyFFTW` backend is GPL-licensed; keeping
it keeps the whole package under GPL, which is the license used here. To switch
to MIT, replace `pyFFTW` with `numpy.fft` and drop the GPL dependency.
