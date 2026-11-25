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
Bootstrap confidence intervals for the Table 4 statistics at n = 1000.

Complements 07_bootstrap_statistics.py, which studies how *few* draws suffice.
This script reports the percentile-bootstrap 95% confidence intervals of the two
summary statistics quoted in the manuscript, at the full sample size n = 1000
that Table 4 actually uses.

Reads
-----
  ensemble_results_1000.npz   (three grids, n = 1000 each, 3000 records)

Writes
------
  figures/bootstrap_ci_summary.txt

The numbers printed here are the ones quoted in the manuscript, on
N = 4096, 8192, 16384 respectively:

  bootstrap 95% CI for the median kappa : [23.03, 23.55] / [2.87, 2.94] / [2.63, 2.66]
  bootstrap 95% CI for P(kappa > 3.9)   : [0.990, 0.999] / [0.215, 0.268] / [0.006, 0.019]
"""
import os
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA_THRESH = 3.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='ensemble_results_1000.npz')
    ap.add_argument('--outdir', default='figures')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--boot', type=int, default=20000)
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, args.npz), allow_pickle=True)
    Ns, kf, rm = d['Ns'], d['kurt_final'], d['rmax_final']
    rng = np.random.default_rng(args.seed)

    lines = [f'percentile bootstrap, B = {args.boot}, seed = {args.seed}',
             f'{"grid":>8} {"n":>5}  {"median kappa [95% CI]":>28}   '
             f'{"P(kappa>3.9) [95% CI]":>28}   {"median rmax [95% CI]":>26}']
    for N in sorted(set(int(v) for v in Ns)):
        k, r = kf[Ns == N], rm[Ns == N]
        n = len(k)
        idx = rng.integers(0, n, size=(args.boot, n))
        bm = np.median(k[idx], axis=1)
        bt = (k[idx] > KAPPA_THRESH).mean(axis=1)
        br = np.median(r[idx], axis=1)
        lo_mk, hi_mk = np.percentile(bm, [2.5, 97.5])
        lo_bt, hi_bt = np.percentile(bt, [2.5, 97.5])
        lo_mr, hi_mr = np.percentile(br, [2.5, 97.5])
        lines.append(
            f'{N:>8} {n:>5}  {np.median(k):9.4f} [{lo_mk:.4f},{hi_mk:.4f}]   '
            f'{(k > KAPPA_THRESH).mean():11.4f} [{lo_bt:.4f},{hi_bt:.4f}]   '
            f'{np.median(r):11.3f} [{lo_mr:.3f},{hi_mr:.3f}]')
    out = '\n'.join(lines)
    print(out)
    os.makedirs(os.path.join(HERE, args.outdir), exist_ok=True)
    with open(os.path.join(HERE, args.outdir, 'bootstrap_ci_summary.txt'), 'w') as f:
        f.write(out + '\n')


if __name__ == '__main__':
    main()
