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
"""Low-pass (delta) fingerprint test at a fixed time.

Reproduces the numbers of Table 2 and Fig. 2 of the manuscript.

The test is performed within a single resolution.  Two snapshots of the field at
t = 60, computed on different grids with dealiasing OFF, are each filtered to a
sequence of cutoffs k_c / k_Nyq, and the peak-intensity ratio r retained after
filtering is recorded.

  * A spurious, single-grid-cell peak behaves like a delta function: its
    spectral energy is spread over all resolved wavenumbers, so the retained
    peak decays linearly with the cutoff.
  * A resolved peak keeps its spectral content at low wavenumbers and is
    essentially unaffected until the filter is very sharp.

Usage
-----
    python 02_lowpass_fingerprint.py results/psi_N8192_t60.npy \
                                     results/psi_N16384_t60.npy

If no arguments are given, the two files above are read from ./results/.

Dependencies: numpy only.
"""

import os
import sys

import numpy as np

from fnls_core import make_grid, rmax, lowpass

L = 500.0
CUTOFFS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]
LABELS = ["1", "3/4", "1/2", "3/8", "1/4", "3/16", "1/8"]


def report(path):
    psi = np.load(path)
    N = psi.size
    _, k, _ = make_grid(L, N)
    base = rmax(psi)
    vals = [rmax(lowpass(psi, k, c)) for c in CUTOFFS]
    print("%-24s  (N = %d,  r_max before filtering = %.4f)"
          % (os.path.basename(path), N, base))
    print("    %-8s" % "k_c/kNyq", end="")
    for lab in LABELS:
        print(" %9s" % lab, end="")
    print()
    print("    %-8s" % "r_max", end="")
    for v in vals:
        print(" %9.1f" % v, end="")
    print()
    print("    %-8s" % "lin.base", end="")
    for c in CUTOFFS:
        print(" %9.1f" % (base * c), end="")
    print()
    return base, vals


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    args = sys.argv[1:]
    if len(args) != 2:
        args = [os.path.join(here, "results", "psi_N8192_t60.npy"),
                os.path.join(here, "results", "psi_N16384_t60.npy")]
        print("No file arguments given; using\n  %s\n  %s\n" % tuple(args))

    print("Low-pass (delta) fingerprint test at t = 60")
    print("-" * 72)
    for p in args:
        if not os.path.exists(p):
            print("MISSING:", p)
            continue
        report(p)
        print("-" * 72)

    print("Reading the table:")
    print("  * a spurious peak tracks the linear baseline (decays linearly);")
    print("  * a resolved peak stays flat until the filter becomes very sharp.")


if __name__ == "__main__":
    main()
