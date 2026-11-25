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
#!/usr/bin/env python3
"""
04 -- Same-field control experiment (reproduces Table II of the manuscript).

The published Table I compares three grids launched from initial conditions
whose random fields are drawn independently on each grid (the RNG is sampled
with length N).  Those three runs are therefore distinct realizations from
t = 0, and their spread mixes the resolution effect with the realization
effect.

This script removes that ambiguity.  A single initial condition is generated
ONCE on the reference grid N_ref = 16384, stored, and band-limited Fourier
resampled to N = 8192 and N = 4096.  All three runs then start from the same
physical field, represented at three resolutions, so that whatever is left is
a property of the grid rather than of the realization.

Expected output (matches Table II):

    N = 4096   r_max = 90.1   kurtosis = 24.75
    N = 8192   r_max = 30.0   kurtosis =  2.77
    N = 16384  r_max = 27.5   kurtosis =  2.55

so that the N = 8192 spike of the uncontrolled comparison (r_max = 257.3)
does not survive the change of the initial field, whereas the N = 4096 grid
inflates the maximum by a factor 3.3.

Cost: about 20 minutes on four cores (N = 16384 dominates).

Usage
-----
    python 04_same_field_control.py --selftest   # verify the resampling only
    python 04_same_field_control.py              # run the three grids
    python 04_same_field_control.py --quick      # short T, for a smoke test
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fnls_core import (make_grid, semiclassical_ic, evolve, rmax, kurtosis,
                       fourier_resample)

# --- production parameters (match the manuscript) ---------------------------
L = 500.0
A0, ELL_A, ELL_S, EPS, NOISE_AMP = 2.0, 50.0, 50.0, 0.1, 1e-3
SEED = 300876
DT = 2e-4
T_TOTAL = 100.0
N_REF = 16384
GRIDS = [4096, 8192, 16384]

# published Table I values, for the comparison printed in the summary
TABLE1 = {4096: 90.4, 8192: 257.3, 16384: 27.5}
# published Table II values
TABLE2 = {4096: 90.1, 8192: 30.0, 16384: 27.5}


def reference_field():
    """The single physical initial condition, on the N_ref grid."""
    x, k, dx = make_grid(L, N_REF)
    psi = semiclassical_ic(x, A0=A0, ell_A=ELL_A, ell_S=ELL_S, eps=EPS,
                           noise_amp=NOISE_AMP, seed=SEED)
    return x, k, dx, psi


def selftest():
    """Check the two properties the control experiment relies on.

    Going down to a coarser grid necessarily discards whatever content lies
    above that grid's Nyquist wavenumber; for the published initial condition
    this is only the grid-level uniform phase of amplitude 1e-3, so the
    discarded amplitude is small.  What must hold exactly is (i) that the
    mapping is idempotent -- resampling the same field down twice gives the
    same result, so the three grids really do carry one field -- and (ii) that
    the mean intensity is preserved to machine accuracy, so the comparison of
    r_max (which is normalised by the mean intensity) is not biased.
    """
    x, k, dx, psi_ref = reference_field()
    print("selftest: N_ref = %d, max|psi| = %.6f, <|psi|^2> = %.6f"
          % (N_REF, np.abs(psi_ref).max(), np.mean(np.abs(psi_ref) ** 2)))
    ok = True
    for N in [4096, 8192]:
        psi_coarse = fourier_resample(psi_ref, N_REF, N)
        psi_back = fourier_resample(psi_coarse, N, N_REF)
        psi_coarse2 = fourier_resample(fourier_resample(psi_back, N_REF, N),
                                       N, N_REF)
        idem = np.abs(psi_coarse2 - psi_back).max() / np.abs(psi_back).max()
        trunc = np.abs(psi_back - psi_ref).max() / np.abs(psi_ref).max()
        mean_err = (abs(np.mean(np.abs(psi_coarse) ** 2)
                        - np.mean(np.abs(psi_ref) ** 2))
                    / np.mean(np.abs(psi_ref) ** 2))
        print("  N=%-6d idempotence err = %.3e   discarded amplitude = %.3e"
              "   mean-intensity rel.err = %.3e"
              % (N, idem, trunc, mean_err))
        ok = ok and (idem < 1e-12) and (mean_err < 1e-4)
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


def reference_field_coarse(N, psi_ref):
    x, k, dx = make_grid(L, N)
    psi = fourier_resample(psi_ref, N_REF, N)
    return x, k, dx, psi


def run_one(args):
    N, psi_ref, T = args
    x, k, dx, psi0 = reference_field_coarse(N, psi_ref)
    t0 = time.time()
    psi, t_axis, rmax_axis, _ = evolve(psi0, k, DT, T, dealias=False)
    return {
        "N": N,
        "dx": dx,
        "rmax": float(np.max(rmax_axis)),
        "rmax_final": float(rmax_axis[-1]),
        "kurtosis": float(kurtosis(psi)),
        "mass_drift": float(abs(np.mean(np.abs(psi) ** 2)
                                - np.mean(np.abs(psi0) ** 2))
                            / np.mean(np.abs(psi0) ** 2)),
        "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="verify the Fourier resampling and exit")
    ap.add_argument("--quick", action="store_true",
                    help="short integration time, for a smoke test")
    ap.add_argument("--T", type=float, default=None, help="override final time")
    ap.add_argument("--jobs", type=int, default=3, help="parallel workers")
    ap.add_argument("--out", default="same_field_control.json")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    T = a.T if a.T is not None else (2.0 if a.quick else T_TOTAL)
    print("same-field control experiment: N_ref = %d, T = %g, dt = %g"
          % (N_REF, T, DT))

    x, k, dx, psi_ref = reference_field()
    print("reference field ready: <|psi|^2> = %.6f"
          % np.mean(np.abs(psi_ref) ** 2))

    task_args = [(N, psi_ref, T) for N in GRIDS]
    recs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, ta): ta[0] for ta in task_args}
        for fut in as_completed(futs):
            r = fut.result()
            recs.append(r)
            print("[done] N=%-6d rmax=%.3f  kurt=%.3f  mdrift=%.1e  t=%.0fs"
                  % (r["N"], r["rmax"], r["kurtosis"], r["mass_drift"],
                     r["elapsed"]))
    recs.sort(key=lambda r: r["N"])

    json.dump(recs, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)

    if abs(T - T_TOTAL) < 1e-12:
        print("\n%-9s %-10s %-10s %-10s %-10s" %
              ("N", "r_max", "TableII", "kurtosis", "TableI r_max"))
        for r in recs:
            print("%-9d %-10.1f %-10.1f %-10.2f %-10.1f" %
                  (r["N"], r["rmax"], TABLE2[r["N"]], r["kurtosis"],
                   TABLE1[r["N"]]))
        print("\nRatio r_max / r_max(N=16384):")
        ref = [r for r in recs if r["N"] == 16384][0]["rmax"]
        for r in recs:
            print("  N=%-6d  %.2f" % (r["N"], r["rmax"] / ref))
    print("\ntotal wall time: %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
