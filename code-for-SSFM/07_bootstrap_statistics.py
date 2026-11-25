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
Bootstrap analysis of sampling-variability for ensemble medians and tail
probabilities.

Motivation
----------
The ensemble of Table 4 uses n = 1000 independent realizations at each of
N = 4096, N = 8192 and N = 16384.  A natural question is: "how few
realizations are enough?"  This script answers it empirically by drawing
bootstrap sub-samples of size m from the thousand-realization population and
recording the sampling distribution of two statistics

    * the median              --  location statistic (robust, used in Table 4)
    * P(kappa > 3.9)         --  a tail probability (fragile at small n)

Produces
--------
  figures/bootstrap_median_kappa_N4096.pdf, _N8192.pdf : median 5-95% band vs m
  figures/bootstrap_tailfrac_N4096.pdf, _N8192.pdf     : tail-prob. 5-95% band vs m
  figures/small_sample_gaps.pdf        :  frequency of apparent "gaps" in
                                          histograms drawn from a
                                          single-component heavy-tailed
                                          distribution (demonstrates that
                                          apparent bimodality at m = 20 is a
                                          small-sample artifact)
  figures/bootstrap_summary.txt        :  numerical summary

The conclusion reported in the manuscript is: medians are reliably estimated at
m = 20 (standard error of the median ~ 14% for r_max and ~ 5% for kappa at
N = 8192), whereas tail probabilities have a five-to-ninety-five percentile
range of [0.10, 0.40] at m = 20 and require hundreds of realizations to
converge; the frequency of an apparent two-peak histogram falls from about 96%
at m = 20 to about 1% at m = 200.
"""
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA_THRESH = 3.9   # tail threshold used in the paper
N_BOOT = 2000        # bootstrap replicates per (m, grid)
# Per-call RNGs are seeded inside bootstrap_bands()/gap_frequency() so that
# results depend only on the sample size m, not on call order across grids.


def load_summary(path):
    """Return {N: dict(rmax=array, kappa=array)} from the merged thousand-run npz.

    Expects the array layout written by the ensemble assembly step:
    keys 'Ns', 'kurt_final' and 'rmax_final', one row per realization.
    """
    z = np.load(path, allow_pickle=True)
    Ns = np.asarray(z['Ns'])
    kf = np.asarray(z['kurt_final'], dtype=float)
    rm = np.asarray(z['rmax_final'], dtype=float)
    out = {}
    for N in sorted(set(int(v) for v in Ns)):
        m = Ns == N
        out[N] = {'rmax': rm[m], 'kappa': kf[m]}
    return out


def median_se(x):
    """Standard error of the median (Maritz-Jarrett, for continuous data)."""
    n = len(x)
    s = np.sort(x)
    if n % 2 == 1:
        m = (n - 1) // 2
        return 1.2533 * (s[m + 1] - s[m - 1]) / (2 * np.sqrt(n)) if n > 3 \
            else float('nan')
    else:
        m = n // 2 - 1
        return 1.2533 * (s[m + 2] - s[m]) / (2 * np.sqrt(n)) if n > 4 \
            else float('nan')


def bootstrap_bands(pop, ms, stat_fn, n_boot=N_BOOT, seed=20261102):
    """Return 5%, 50%, 95% percentiles of stat_fn(sample_m) for each m.

    A fresh RNG is created for each m so the reported bands depend only on the
    sample size, not on how many grids were processed before.
    """
    bands = np.zeros((len(ms), 3))
    for j, m in enumerate(ms):
        rng = np.random.default_rng(seed + j)
        vals = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(pop), size=m)
            vals[b] = stat_fn(pop[idx])
        bands[j] = np.percentile(vals, [5, 50, 95])
    return bands


def gap_frequency(pop, m, n_boot=N_BOOT, min_gap_width=2.0, bin_width=1.0,
                  seed=12345):
    """Fraction of m-sized draws whose histogram of kappa contains an empty bin
    of width at least min_gap_width (in units of bin_width) within [2, 10].

    This quantifies how often a small sample from a single-component heavy-tailed
    distribution appears to be "bimodal" to the eye.  The RNG seed depends on m
    so the answer is deterministic for a given sample size.
    """
    edges = np.arange(2.0, 12.0, bin_width)
    rng = np.random.default_rng(seed + int(m))
    cnt = 0
    for b in range(n_boot):
        x = pop[rng.integers(0, len(pop), size=m)]
        h, _ = np.histogram(x, bins=edges)
        # longest run of zero counts
        run = 0
        maxrun = 0
        for c in h:
            if c == 0:
                run += 1
                maxrun = max(maxrun, run)
            else:
                run = 0
        if maxrun >= min_gap_width:
            cnt += 1
    return cnt / n_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='ensemble_results_1000.npz',
                    help='merged thousand-run ensemble npz '
                         '(keys: Ns, rmax_final, kurt_final)')
    ap.add_argument('--outdir', default='figures')
    args = ap.parse_args()

    data = load_summary(os.path.join(HERE, args.npz))

    os.makedirs(os.path.join(HERE, args.outdir), exist_ok=True)
    ms = [5, 10, 20, 50, 100, 200, 500, 1000]

    lines = ['bootstrap sampling-variability for ensemble statistics',
             '=' * 60]
    for N, d in sorted(data.items()):
        kappa = d['kappa']
        rmax = d['rmax']
        print(f'[N={N}] n={len(kappa)}, kappa median={np.median(kappa):.3f}, '
              f'r_max median={np.median(rmax):.2f}')

        # Median bands
        bk = bootstrap_bands(kappa, ms, np.median)
        br = bootstrap_bands(rmax, ms, np.median)
        # Tail-fraction bands
        bt = bootstrap_bands(kappa, ms,
                              lambda x: float(np.mean(x > KAPPA_THRESH)))

        # Print summary at m=20
        j20 = ms.index(20)
        se_k = (bk[j20, 2] - bk[j20, 0]) / (2 * 1.645)   # ~ 1 sigma
        se_r = (br[j20, 2] - br[j20, 0]) / (2 * 1.645)
        lines.append(f'N={N}: true medians (n={len(kappa)}): '
                     f'kappa={np.median(kappa):.3f}, rmax={np.median(rmax):.2f}')
        lines.append(f'  m=20: kappa median 5-95% = [{bk[j20,0]:.3f}, '
                     f'{bk[j20,2]:.3f}]  (SE ~ {se_k:.3f})')
        lines.append(f'  m=20: r_max median 5-95% = [{br[j20,0]:.2f}, '
                     f'{br[j20,2]:.2f}]  (SE ~ {se_r:.2f})')
        lines.append(f'  m=20: P(kappa>{KAPPA_THRESH}) 5-95% = '
                     f'[{bt[j20,0]:.3f}, {bt[j20,2]:.3f}]')

        # Plot: median kappa vs m
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.fill_between(ms, bk[:, 0], bk[:, 2], color='C3', alpha=0.18,
                        label='5--95% (bootstrap)')
        ax.semilogx(ms, bk[:, 1], 'o-', color='C3', label='median of medians')
        ax.axhline(np.median(kappa), color='k', ls=':', lw=1,
                   label=f'true median = {np.median(kappa):.3f}')
        ax.set_xlabel('sample size $m$')
        ax.set_ylabel(f'median $\\kappa$ (N={N})')
        ax.set_title(f'Convergence of median $\\kappa$ at N={N}')
        ax.grid(alpha=0.28)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, args.outdir,
                                 f'bootstrap_median_kappa_N{N}.pdf'))
        plt.close(fig)

        # Plot: tail fraction vs m
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.fill_between(ms, bt[:, 0], bt[:, 2], color='C2', alpha=0.18,
                        label='5--95% (bootstrap)')
        ax.semilogx(ms, bt[:, 1], 's-', color='C2', label='median')
        ax.axhline(np.mean(kappa > KAPPA_THRESH), color='k', ls=':', lw=1,
                   label=f'true fraction = {np.mean(kappa>KAPPA_THRESH):.3f}')
        ax.set_xlabel('sample size $m$')
        ax.set_ylabel(f'$P(\\kappa>{KAPPA_THRESH})$ (N={N})')
        ax.set_title(f'Convergence of tail fraction at N={N}')
        ax.grid(alpha=0.28)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, args.outdir,
                                 f'bootstrap_tailfrac_N{N}.pdf'))
        plt.close(fig)

    # Gap-frequency demonstration (small-sample "bimodality")
    if 8192 in data:
        k8 = data[8192]['kappa']
        ms_gap = [20, 50, 100, 200, 500, 1000]
        freqs = []
        for m in ms_gap:
            freqs.append(gap_frequency(k8, m))
            lines.append(f'N=8192: apparent-gap fraction at m={m}: {freqs[-1]:.2%}')
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.semilogx(ms_gap, freqs, 'o-', color='#7a4d9e')
        ax.set_xlabel('sample size $m$')
        ax.set_ylabel('fraction of samples with an apparent gap')
        ax.set_title('Apparent bimodality as a small-sample artifact (N=8192)')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.28)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, args.outdir, 'small_sample_gaps.pdf'))
        plt.close(fig)

    with open(os.path.join(HERE, args.outdir, 'bootstrap_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
