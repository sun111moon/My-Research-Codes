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
"""Draw Figure 1 and Figure 3 of the manuscript from the two t = 60 snapshots.

Reads (defaults; override on the command line):
    results/psi_N8192_t60.npy
    results/psi_N16384_t60.npy

Writes:
    figures/fig1_lowpass_test.pdf          figures/fig1_lowpass_test.png
    figures/fig3_resolution_snapshot.pdf   figures/fig3_resolution_snapshot.png

Figure 1 -- the low-pass ("delta") fingerprint.  For each grid the retained
peak ratio r is recorded as the spectral cutoff k_c/k_Nyq is lowered through
1, 3/4, 1/2, 3/8, 1/4, 3/16, 1/8.  A single-cell spike behaves like a delta
function and follows the linear baseline; a resolved peak stays flat until the
filter bites into its spectral core.  The convention block below is copied from
02_lowpass_fingerprint.py verbatim so the two cannot disagree about CUTOFFS.

Figure 3 -- the resolution snapshot.  The same ten-unit window of the local
density |psi(x)|^2 is shown for N = 8192 (spurious single-cell spike) and
N = 16384 (smooth in the same window).  The window is centred on the spike of
the coarse grid.

Every annotated number (r, the peak-to-background ratio, and the factor by
which the spike exceeds its immediate neighbours) is computed from the
snapshots -- nothing is hard-coded, so the figure cannot claim a value that the
data does not support.  In particular the dashed baseline is labelled with the
r_max that was actually measured at the first cutoff; an earlier draft of this
script printed the manuscript's value here instead, which made the legend
disagree with the plotted points.

REPRODUCIBILITY NOTE
--------------------
The snapshots are dumped by fnls_core.evolve at the first step with
t >= dump_at, and 01_run_convergence.py sets dump_at = T_DUMP = 60.0.  The
dump time is therefore independent of the integration end time T_FINAL, so a
truncated run (T_FINAL = 60.1) and a full run (T_FINAL = 100) produce the same
t = 60 snapshots.  If the r values printed here disagree with the numbers
quoted in the manuscript, the manuscript figure came from an earlier run or a
different seed, and the package cannot reproduce it as it stands -- the
manuscript numbers must then be regenerated from the reproducible run.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fnls_core import make_grid, rmax, lowpass

# --- conventions shared with 02_lowpass_fingerprint.py (kept identical) -----
L = 500.0
CUTOFFS = [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]
LABELS = ["1", "3/4", "1/2", "3/8", "1/4", "3/16", "1/8"]

# Values quoted in the manuscript, used only to print a reconciliation line so
# that any drift between the package and the paper is visible immediately.
MANUSCRIPT_R_N8192_AT_CUTOFF_1 = 233.2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "figures")

HALF_WINDOW = 5.0
COLORS = {8192: "#c76200", 16384: "#117a37"}


def spatial_axis(n):
    grid = make_grid(L, n)
    x = np.asarray(grid[0], dtype=float)
    k = np.asarray(grid[1], dtype=float)
    if x.shape != (n,):
        x = np.linspace(-L / 2.0, L / 2.0, n, endpoint=False)
    return x, k


def load(path):
    psi = np.asarray(np.load(path))
    x, k = spatial_axis(psi.size)
    return psi, x, k


def save(fig, stem):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTDIR, "%s.%s" % (stem, ext))
        fig.savefig(out, dpi=200)
        print("[fig] saved", out)
    plt.close(fig)


def figure_lowpass(snap):
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    base8 = None
    baseline = None

    for n in (8192, 16384):
        psi, _, k = snap[n]
        base = rmax(psi)
        vals = np.array([rmax(lowpass(psi, k, c)) for c in CUTOFFS])
        print("N = %-6d r_max before filtering = %.4f" % (n, base))
        for c, v, lab in zip(CUTOFFS, vals, LABELS):
            print("    k_c/k_Nyq = %-5s  r = %8.2f   lin.base = %8.2f"
                  % (lab, v, base * c))
        if n == 8192:
            base8 = base
            model = base * np.array(CUTOFFS)
            rel = np.max(np.abs(vals - model) / model)
            print("    deviation from linear baseline: max %.1f%%" % (100.0 * rel))
            print("    manuscript quotes r = %.1f at k_c/k_Nyq = 1 for N = 8192;"
                  % MANUSCRIPT_R_N8192_AT_CUTOFF_1)
            print("    recomputed from this snapshot:      r = %.1f" %
                  vals[0])
            baseline = model
        ax.plot(CUTOFFS, vals, marker="o" if n == 8192 else "s",
                ls="none", color=COLORS[n], ms=6.5, zorder=3,
                label=r"$N=%d$" % n)

    # The label must state the value the data produced, not a remembered one.
    ax.plot(CUTOFFS, baseline, ls="--", lw=1.4, color="0.35", zorder=2,
            label=r"linear baseline $%.1f\,k_c/k_{\mathrm{Nyq}}$" % base8)
    ax.set_xlim(1.06, 0.115)
    ax.set_xlabel(r"cutoff  $k_c/k_{\mathrm{Nyq}}$")
    ax.set_ylabel(r"retained peak  $r$")
    ax.set_xticks(CUTOFFS)
    ax.set_xticklabels(LABELS)
    ax.grid(alpha=0.28, lw=0.6)
    ax.legend(fontsize=9, framealpha=0.92)
    fig.tight_layout()
    save(fig, "fig1_lowpass_test")


def figure_resolution(snap):
    psi8, x8, _ = snap[8192]
    psi16, x16, _ = snap[16384]
    I8 = np.abs(psi8) ** 2
    I16 = np.abs(psi16) ** 2

    ipk = int(np.argmax(I8))
    xpk = float(x8[ipk])
    r8 = rmax(psi8)
    neighbours = I8[max(ipk - 1, 0):ipk + 2]
    nb = float(I8[ipk] / np.max(np.delete(neighbours, np.argmax(neighbours))))
    print("spike at x = %.3f:  r = %.1f,  peak/neighbour = %.1f" % (xpk, r8, nb))

    left, right = xpk - HALF_WINDOW, xpk + HALF_WINDOW
    w8 = (x8 >= left) & (x8 <= right)
    w16 = (x16 >= left) & (x16 <= right)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharex=True)
    axes[0].plot(x8[w8], I8[w8] / np.mean(I8), "-", color=COLORS[8192], lw=1.4)
    axes[0].plot([xpk], [I8[ipk] / np.mean(I8)], "*", color="black", ms=13, zorder=5)
    axes[0].annotate(r"$r=%.0f$,  $%.1f\times$ neighbours" % (r8, nb),
                     xy=(xpk, I8[ipk] / np.mean(I8)),
                     xytext=(min(xpk + 2.4, right - 1.0), I8[ipk] / np.mean(I8) * 0.78),
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.9))
    axes[0].set_title(r"(a) $N=8192$: spurious single-cell spike", fontsize=10.5)
    axes[0].set_ylabel(r"normalized density  $|\psi|^2 / \langle |\psi|^2 \rangle$")

    axes[1].plot(x16[w16], I16[w16] / np.mean(I16), "-", color=COLORS[16384], lw=1.4)
    axes[1].set_title(r"(b) $N=16384$: smooth in the same window", fontsize=10.5)

    for ax in axes:
        ax.grid(alpha=0.28, lw=0.6)
        ax.set_xlabel(r"$x$")
    axes[1].set_xlim(left, right)
    fig.tight_layout()
    save(fig, "fig3_resolution_snapshot")


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        args = [os.path.join(RESULTS, "psi_N8192_t60.npy"),
                os.path.join(RESULTS, "psi_N16384_t60.npy")]
        print("No file arguments given; using\n  %s\n  %s\n" % tuple(args))

    snap = {}
    for n, path in ((8192, args[0]), (16384, args[1])):
        if not os.path.exists(path):
            print("MISSING:", path, "— run 01_run_convergence.py first")
            return
        snap[n] = load(path)
    figure_lowpass(snap)
    figure_resolution(snap)


if __name__ == "__main__":
    main()
