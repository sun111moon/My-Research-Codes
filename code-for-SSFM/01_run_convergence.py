# DeltScan: Single-grid convergence diagnostics for fNLS integrable turbulence
# Copyright (C) 2026  Hongcai Ma, Aiping Deng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>{target="_blank"}.
"""Protocol checks: spatial refinement, dealiasing and temporal refinement.

This script collects, in one place, every numerical check that the three-step
convergence protocol of the manuscript calls for.  It uses two different
initial-condition conventions, and it is important not to confuse them.

  * A *common physical field* is generated once on the reference grid
    N_ref = 16384 and mapped onto N = 4096, 8192, 16384 by band-limited Fourier
    resampling.  This is the only legitimate way to measure a resolution
    error, and it produces the numbers of **Table II** (the same-field
    control).  It is reproduced in isolation by
    `04_same_field_control.py`.

  * The *production field* is the initial condition of the published run: the
    random fields are drawn directly with length N, so the grids carry
    independent realizations.  Dealiasing and time-step checks are performed
    on this field, because that is the realization that produced the spurious
    spike of the manuscript (r_max = 257.3 at N = 8192).  These runs also
    reproduce the values of **Table I**.

Outputs (in ./results):

    convergence_summary.json            all numbers below
    psi_N8192_t60.npy                   production N = 8192 field at t = 60
    psi_N16384_t60.npy                  production N = 16384 field at t = 60
    psi_samefield_N8192_t60.npy         common field, N = 8192, at t = 60

The two psi_N*_t60.npy files are the inputs of `02_lowpass_fingerprint.py`.

Usage
-----
    python 01_run_convergence.py            # everything (~1.5 h)
    python 01_run_convergence.py --quick    # T = 2, smoke test
"""

import argparse
import json
import os

import numpy as np

from fnls_core import (make_grid, semiclassical_ic, evolve, rmax, kurtosis,
                       fourier_resample)

L = 500.0
DT = 2.0e-4
T_FINAL = 100.0
T_DUMP = 60.0
A0 = 2.0
SEED = 300876

N_REF = 16384              # reference grid for the common physical field
N_PRODUCTION = 8192        # the published production grid
GRIDS = [4096, 8192, 16384]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short run (T=2) to check the pipeline")
    args = ap.parse_args()

    T = 2.0 if args.quick else T_FINAL
    dump_at = None if args.quick else T_DUMP
    os.makedirs(OUT, exist_ok=True)
    summary = {}

    # ------------------------------------------------------------------ (A)
    # Table II: one physical field on three grids.
    print("[A] same physical field, N_ref = %d (Table II), T = %.1f" % (N_REF, T))
    x_ref, k_ref, _ = make_grid(L, N_REF)
    psi0_ref = semiclassical_ic(x_ref, A0=A0, seed=SEED)

    samefield = {}
    for N in GRIDS:
        x, k, dx = make_grid(L, N)
        psi0 = fourier_resample(psi0_ref, N_REF, N)
        psi, t, hist, dump = evolve(psi0, k, DT, T, dealias=False)
        rec = {"N": N, "dx": dx,
               "rmax_max_over_run": float(np.max(hist)),
               "rmax_final": float(hist[-1]),
               "kurtosis_final": kurtosis(psi)}
        samefield["N%d" % N] = rec
        if dump is not None and N == 8192:
            np.save(os.path.join(OUT, "psi_samefield_N8192_t60.npy"), dump)
        print("    N=%6d  dx=%.4f  r_max(global)=%8.3f  kappa=%.3f"
              % (N, dx, rec["rmax_max_over_run"], rec["kurtosis_final"]))
    summary["same_field_control"] = samefield

    # ------------------------------------------------------------------ (B)
    # Table I: the published production runs, one realization per grid.
    print("[B] production field (independent realizations), Table I")
    production = {}
    for N in [N_PRODUCTION, 16384]:
        x, k, dx = make_grid(L, N)
        psi0 = semiclassical_ic(x, A0=A0, seed=SEED)
        psi, t, hist, dump = evolve(psi0, k, DT, T, dealias=False,
                                    dump_at=dump_at)
        rec = {"N": N, "dx": dx,
               "rmax_max_over_run": float(np.max(hist)),
               "kurtosis_final": kurtosis(psi)}
        production["N%d" % N] = rec
        if dump is not None:
            np.save(os.path.join(OUT, "psi_N%d_t60.npy" % N), dump)
        print("    N=%6d  dx=%.4f  r_max(global)=%8.3f  kappa=%.3f"
              % (N, dx, rec["rmax_max_over_run"], rec["kurtosis_final"]))
    summary["production"] = production

    # the production N = 8192 field is the one used for (C) and (D)
    x8, k8, dx8 = make_grid(L, N_PRODUCTION)
    psi0_8 = semiclassical_ic(x8, A0=A0, seed=SEED)

    # ------------------------------------------------------------------ (C)
    print("[C] 2/3 Orszag dealiasing on the production N = %d field"
          % N_PRODUCTION)
    psi_d, t_d, hist_d, _ = evolve(psi0_8, k8, DT, T, dealias=True)
    summary["dealiasing"] = {"N": N_PRODUCTION,
                             "rmax_max_over_run": float(np.max(hist_d)),
                             "kurtosis_final": kurtosis(psi_d)}
    print("    with dealiasing: r_max(global) = %.3f" % np.max(hist_d))

    # ------------------------------------------------------------------ (D)
    print("[D] dt halved on the production N = %d field" % N_PRODUCTION)
    psi_h, t_h, hist_h, _ = evolve(psi0_8, k8, 0.5 * DT, T, dealias=False)
    summary["temporal"] = {"N": N_PRODUCTION, "dt": 0.5 * DT,
                           "rmax_max_over_run": float(np.max(hist_h))}
    print("    dt/2: r_max(global) = %.3f" % np.max(hist_h))

    with open(os.path.join(OUT, "convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("written:", os.path.join(OUT, "convergence_summary.json"))


if __name__ == "__main__":
    main()
