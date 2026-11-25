# Supplementary Material — DeltScan

Companion documentation for

> **Fidelity bounds and a single-grid convergence protocol for extreme-event
> statistics in integrable turbulence**
> Hongcai Ma, Aiping Deng — College of Mathematics and Statistics, Donghua University
> Target journal: *Computer Physics Communications* (CP / CPiP)

This note is the reference for *what the computed results mean*: the
conventions adopted for the initial conditions, the physical and numerical
parameters, and the numbers each workflow script is expected to reproduce.
Installation, the file-by-file contents and the run order are in `README.md`.

Table and figure numbers below are those of the compiled manuscript, which
follows the order in which the objects appear in the source; they are cited as
`\ref` there and are consistent with it.

---

## 1. Three initial-condition conventions

The paper uses three conventions, and they answer different questions.
Confusing them is precisely the error that the paper is about.

- **Distinct realizations, one seed** (`03_reproduce_published_tables.py`,
  part (B) of `01_run_convergence.py`, and the published **Table 2**). The
  random fields are drawn with length `N`, so the three grids carry three
  independent realizations from `t = 0`. This convention reproduces the
  published values. It must *not* be read as a resolution study: because the
  runs decorrelate, their spread mixes the resolution effect with the
  realization effect.

- **One physical field** (part (A) of `01_run_convergence.py`,
  `04_same_field_control.py`, **Table 3**). A single field is generated on a
  reference grid and mapped onto the others by band-limited Fourier resampling
  (`fnls_core.fourier_resample`), so that all grids represent the same physical
  state. This removes the realization ambiguity and is the only legitimate way
  to measure a resolution error for an individual field.

- **An ensemble of distinct realizations** (`05_run_ensemble.py`,
  `06_analyze_ensemble.py`, **Table 4**). At all three grids
  (`N = 4096, 8192, 16384`) one thousand independent fields are used (seeds
  `300000`–`300999`). The resolution dependence is read from the distribution
  rather than from one run. Unlike the second convention this does not certify
  any individual field, but unlike the first it exposes the systematic trend.
  Bootstrap analysis (`07_bootstrap_statistics.py`) confirms that the median is
  reliably estimated with twenty draws, whereas tail probabilities (e.g. the
  fraction of runs with `kappa > 3.9`) require hundreds.

Running the *same* grid with the *same* parameters but a different initial
field is what separates the two effects. It is done in
`04_same_field_control.py`, and the outcome is the reason **Table 3** exists:
at `N = 8192` the band-limited field gives `r_max = 30.0`, against `257.3` for
the independently drawn field of **Table 2**. The spike is a property of the
field, not of the grid. The thousand-realization ensemble of **Table 4**
quantifies the same point statistically: at `N = 8192` the median kurtosis is
`2.90` and only one quarter of the runs have `kappa > 3.9`, but a long upper
tail extends to `kappa = 35.5`; the median `r_max` is `17.0`, not the
single-run value `257.3`.

---

## 2. Physical and numerical parameters

| Quantity | Value |
|---|---|
| Model | `i psi_t + 0.5 psi_xx + abs(psi)^2 psi = 0` |
| Domain | periodic, `x in [-250, 250]` (L = 500) |
| Amplitude | `A0 = 2` |
| Correlation lengths | `ell_A = ell_S = 50` |
| Semiclassical parameter | `eps = 0.1` |
| Phase-noise amplitude | `noise_amp = 1e-3` |
| Random seed | `300876` (`numpy.random.default_rng`); 20-realization seeds `100`–`119`, thousand-realization seeds `300000`–`300999` |
| Time step | `dt = 2e-4` |
| Final time | `T = 100` (snapshot for the delta test at `t = 60`) |
| Grids | `N = 4096, 8192, 16384` |
| Reference grid for the common field | `N_ref = 16384` |
| Dealiasing | 2/3 Orszag rule, **off** unless stated |
| Mean intensity of the initial state | `4.22` at `N = 4096`, `4.26` at `N = 8192` and `16384` |

The kurtosis is defined throughout as `kappa = <I^2> / <I>^2` with
`I = abs(psi)^2`. For Gaussian circular statistics `I` is exponential and
`kappa = 2`; this is the reference line drawn in the kurtosis panel. It is a
Gaussian reference, not a soliton-gas prediction: the kinetic theory of soliton
gases gives a kurtosis that depends on the spectral density of states and hence
on the initial data (Congy et al., PRL **132**, 207201, 2024), so no single
constant asymptote is imposed here.

---

## 3. What each table should show

**Table 1** (low-pass fingerprint). At `t = 60`, the `N = 8192` production peak
tracks the linear baseline as the cutoff is lowered, the fingerprint of a
single-grid-cell delta function (`R^2 = 0.992`); it drops from `233.2` on the
first cutoff. The `N = 16384` peak is flat to within `6%` until the cutoff is
pushed close to the spectral width of the physical core.

**Table 2** (production runs, independent realizations). The global maximum of
`r_max` is `90.4`, `257.3` and `27.5` on `N = 4096, 8192, 16384`. The
non-monotonic ordering is not a scaling law: the three runs are three different
realizations.

**Table 3** (same physical field on three grids). `90.1`, `30.0` and `27.5`.
Because the three runs now represent one field, the comparison isolates the
resolution error: the coarsest grid inflates the maximum by a factor `3.3`,
whereas `N = 8192` and `N = 16384` agree to `9%`. The `N = 8192` spike of
Table 2 does not survive the change of the initial field.

**Table 4** (ensemble, 1000 realizations per grid). Median and interquartile
range of `r_max`: `80.28 [67.97, 97.97]`, `16.96 [13.74, 58.80]` and
`14.43 [12.51, 16.78]` on `N = 4096, 8192, 16384`. The median therefore falls
by a factor `5.6` from the coarsest to the finest grid, so that resolution does
inflate the tail systematically; the long upper tail at `N = 8192` (individual
realizations reach `r_max = 253`) is wide enough to overlap both neighbours, so
that no single run at that grid is representative. The kurtosis collapses much
more sharply, from `23.31 [21.27, 24.98]` through `2.90 [2.68, 3.75]` to
`2.65 [2.51, 2.81]`, the finest-grid interval being the tightest of the three
and lying just above the Gaussian value `2`. The tail fraction
`P(kappa > 3.9)` drops from `99.5%` (`995/1000`) through `24.1%` (`241/1000`)
to `1.2%` (`12/1000`) across the three grids.

**Dealiasing and time step.** On the production `N = 8192` field, switching on
the 2/3 Orszag rule reduces the maximum from `257.3` to `42.7`, and halving
`dt` changes it from `257.3` to `236.8`. Both checks are performed on one fixed
realization, so both are free of the realization ambiguity.

---

## 4. Runtime and post-processing

With `dt = 2e-4` and `T = 100`, `01_run_convergence.py` runs seven
integrations, the two `N = 16384` ones dominating; expect roughly one and a
half hours on four cores. `04_same_field_control.py` runs three and takes about
twenty minutes. The full ensemble of `05_run_ensemble.py` is 60 integrations;
on four cores it takes several hours, on a modern laptop about half an hour
with a dozen processes, and it may be interrupted and resumed freely.

If you only want to inspect the published numbers without re-running the
simulations, `06_analyze_ensemble.py`, `07_bootstrap_statistics.py` and
`08_bootstrap_ci.py` read `ensemble_results_1000.npz` directly.
