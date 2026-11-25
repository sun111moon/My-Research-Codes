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
Analyze the n=1000 ensemble (N=4096/8192/16384) and produce Figure 2 of the
manuscript: two-panel log-scale scatter + median/IQR for r_max and kappa.

Reads:
  ensemble_results_1000.npz   (three grids, n=1000 each, 3000 records total)

Writes:
  figures/fig2_ensemble.pdf
  figures/fig2_ensemble.png
  figures/fig2_summary.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'figures')
DATA  = os.path.join(HERE, 'ensemble_results_1000.npz')
KAPPA_GAUSS = 2.0


def median_iqr(x):
    return float(np.median(x)), float(np.percentile(x,25)), float(np.percentile(x,75))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = np.load(DATA, allow_pickle=True)
    Ns = sorted(set(int(v) for v in d['Ns']))
    rmax = {N: d['rmax_final'][d['Ns']==N] for N in Ns}
    kurt = {N: d['kurt_final'][d['Ns']==N] for N in Ns}

    lines = ['N        n   r_max med  IQR              kappa med  IQR              P(k>3.9)  skew(k)']
    for N in Ns:
        rm,rq1,rq3 = median_iqr(rmax[N])
        km,kq1,kq3 = median_iqr(kurt[N])
        sk = float(((kurt[N]-kurt[N].mean())**3).mean() / kurt[N].std()**3)
        lines.append(f'{N:5d}  {len(rmax[N]):4d}  {rm:7.2f}   [{rq1:6.2f},{rq3:7.2f}]   '
                     f'{km:8.3f}   [{kq1:6.3f},{kq3:6.3f}]   {np.mean(kurt[N]>3.9)*100:5.1f}%   {sk:+.2f}')
    summary = '\n'.join(lines)
    print(summary)
    with open(os.path.join(OUTDIR, 'fig2_summary.txt'), 'w') as f:
        f.write(summary+'\n')

    # Reproducible jitter seed
    rng = np.random.RandomState(2024)
    colors = {4096: '#1f4e79', 8192: '#c76200', 16384: '#117a37'}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    for ax, data, ylabel, ref in [
            (axes[0], rmax, r'global maximum $r_{\max} = \max(|\psi|^2)/\langle|\psi|^2\rangle$', None),
            (axes[1], kurt, r'final kurtosis $\kappa = \langle I^2\rangle/\langle I\rangle^2$', KAPPA_GAUSS)]:
        # log y axis
        ax.set_yscale('log')
        meds, lo, hi = [], [], []
        xs = np.arange(len(Ns))
        for i, N in enumerate(Ns):
            v = data[N]
            m, q1, q3 = median_iqr(v)
            meds.append(m); lo.append(m/q1); hi.append(q3/m)  # asymmetric on log scale
            # jittered scatter
            j = 0.07*(rng.rand(len(v))-0.5)
            ax.scatter(np.full(len(v), i, dtype=float)+j, v,
                       color=colors[N], alpha=0.20, s=14, zorder=2,
                       edgecolors='none')
        ax.errorbar(xs, meds, yerr=[lo, hi], fmt='s-', color='black',
                    capsize=5, capthick=1.4, lw=1.6, ms=8,
                    label='median (IQR)', zorder=4)
        if ref is not None:
            ax.axhline(ref, color='0.35', ls='--', lw=1.3, zorder=1,
                       label=f'Gaussian reference $\\kappa={ref:.0f}$')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'$N={N}$' for N in Ns])
        ax.set_xlim(-0.45, len(Ns)-0.55)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.28, lw=0.6, which='both')
        ax.legend(loc='upper right', framealpha=0.92, fontsize=8.8)

    # --- mark the heavy upper tail of the N=8192 column in panel (a) ---
    if 8192 in Ns:
        i82 = Ns.index(8192)
        q3_82 = float(np.percentile(rmax[8192], 75))
        axes[0].hlines(q3_82, i82 - 0.30, i82 + 0.30, color=colors[8192],
                       ls='--', lw=1.6, zorder=5)
        axes[0].annotate(f'$N{{=}}8192$ heavy tail (IQR edge $={q3_82:.0f}$)',
                         xy=(i82, q3_82), xytext=(i82, q3_82 * 3.4),
                         fontsize=8.2, color=colors[8192], ha='center',
                         arrowprops=dict(arrowstyle='-', color=colors[8192], lw=0.9))
    axes[0].set_title('(a) Peak intensity ratio', fontsize=10.5)
    axes[1].set_title('(b) Final kurtosis', fontsize=10.5)
    fig.tight_layout()
    for ext in ['pdf', 'png']:
        out = os.path.join(OUTDIR, f'fig2_ensemble.{ext}')
        fig.savefig(out, dpi=200)
        print('[fig] saved', out)
    plt.close(fig)


if __name__ == '__main__':
    main()
