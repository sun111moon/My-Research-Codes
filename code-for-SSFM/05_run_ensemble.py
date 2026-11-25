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
Ensemble driver for the fNLS resolution-trap study.

Runs multiple independent realizations (different seeds) at multiple
resolutions, and stores per-realization scalars (rmax_max, rmax_final,
kurtosis_final, mass_drift) in a single .npz file.

Supports
  * multiprocessing parallelism (one process per realization);
  * checkpointing: already-completed (seed, N, dt, dealias, T) tuples
    are skipped, so the script can be interrupted and restarted freely;
  * command-line control of all key parameters.

Usage examples
--------------
  # smoke test: 5 seeds x 3 grids, short T
  python 05_run_ensemble.py --n-seeds 5 --seed-start 100 --T 5

  # production: 20 seeds x 3 grids x 2 A0 values, T=100 (production)
  python 05_run_ensemble.py --n-seeds 20 --seed-start 100 --T 100 --A0-list 2.0 2.5

Output
------
  ensemble_results.npz with keys:
    seeds      (n_tasks,)
    Ns         (n_tasks,)
    A0s        (n_tasks,)
    dts        (n_tasks,)
    dealias    (n_tasks,) bool
    Ts         (n_tasks,)
    rmax_max   (n_tasks,)
    rmax_final (n_tasks,)
    kurt_final (n_tasks,)
    mass_drift (n_tasks,)
"""
import os, sys, time, argparse
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fnls_core import (make_grid, initial_cond_semiclassical,
                       evolve_track, _set_threads, _PYFFTW_OK)


# ---------------------------------------------------------------------------
# Physical parameters (fixed; match the published setup)
# ---------------------------------------------------------------------------
L = 500.0
ELL_A = 50.0
ELL_S = 50.0
EPS = 0.1
NOISE_AMP = 1e-3


def run_one(task):
    """Run one realization; return a dict of scalars.

    task is a tuple (seed, N, A0, dt, T, dealias).
    Each worker disables pyFFTW (fork-unsafe) and uses numpy FFT with a
    single thread; parallelism comes from processes, not threads.
    """
    # CRITICAL: pyfftw plans are not fork-safe; disable in child processes
    import fnls_core as _fc
    _fc._disable_pyfftw()

    seed, N, A0, dt, T, dealias = task

    x, k, dx = make_grid(L, N)
    psi0 = initial_cond_semiclassical(x, A0=A0, ell_A=ELL_A, ell_S=ELL_S,
                                       eps=EPS, noise_amp=NOISE_AMP, seed=seed)
    t0 = time.time()
    res = evolve_track(psi0, k, dt, T, dx=dx, dealias=dealias,
                       check_mass=True, mass_tol=1e-6)
    elapsed = time.time() - t0
    if res['mass_drift'] > 1e-6:
        print(f'[warn] seed={seed} N={N} mass_drift={res["mass_drift"]:.2e} > 1e-6', flush=True)
    print(f'[done] seed={seed:5d} N={N:5d} A0={A0} dt={dt:.0e} T={T:5.1f} '
          f'dealias={int(dealias)}: rmax_max={res["rmax_max"]:8.2f} '
          f'kurt={res["kurt_final"]:.3f} mdrift={res["mass_drift"]:.1e} '
          f't={elapsed/60:.1f}min', flush=True)
    return {**res, 'seed': seed, 'N': N, 'A0': A0, 'dt': dt,
            'T': T, 'dealias': bool(dealias), 'elapsed_s': elapsed}


def load_existing(path):
    """Load previously completed results; return set of done keys and data arrays."""
    done = set()
    data = {k: [] for k in ['seeds','Ns','A0s','dts','dealias','Ts',
                            'rmax_max','rmax_final','kurt_final','mass_drift']}
    if not os.path.exists(path):
        return done, data
    try:
        npz = np.load(path, allow_pickle=True)
        n = len(npz['seeds'])
        for i in range(n):
            key = (int(npz['seeds'][i]), int(npz['Ns'][i]),
                   float(npz['A0s'][i]), float(npz['dts'][i]),
                   float(npz['Ts'][i]), bool(npz['dealias'][i]))
            done.add(key)
        for k in data:
            data[k] = list(npz[k])
    except Exception as e:
        print(f'[load] could not read existing {path}: {e}', flush=True)
    return done, data


def save(path, data):
    np.savez(path, **{k: np.array(v) for k, v in data.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-start', type=int, default=100,
                    help='first seed index')
    ap.add_argument('--n-seeds', type=int, default=20,
                    help='number of independent realizations')
    ap.add_argument('--Ns', type=int, nargs='+', default=[4096, 8192, 16384],
                    help='grid sizes')
    ap.add_argument('--A0-list', type=float, nargs='+', default=[2.0],
                    help='background amplitudes')
    ap.add_argument('--dt', type=float, default=2e-4)
    ap.add_argument('--T', type=float, default=100.0)
    ap.add_argument('--dealias', type=int, nargs='+', default=[0],
                    help='0 = no dealiasing, 1 = 2/3 Orszag; default [0]')
    ap.add_argument('--n-proc', type=int, default=max(1, cpu_count()-1))
    ap.add_argument('--out', type=str, default='ensemble_results.npz')
    ap.add_argument('--start-method', type=str, default=None,
                    choices=['fork', 'spawn', 'forkserver'],
                    help='multiprocessing start method; default = platform '
                         'default (spawn on macOS, fork on Linux)')
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    dealias_flags = [bool(x) for x in args.dealias]

    # Build full task list
    tasks = []
    for s in seeds:
        for N in args.Ns:
            for A0 in args.A0_list:
                for da in dealias_flags:
                    tasks.append((s, N, A0, args.dt, args.T, da))

    # Load checkpoint
    done, data = load_existing(args.out)
    todo = [t for t in tasks if t not in done]
    print(f'[ensemble] total tasks: {len(tasks)}, already done: {len(done)}, '
          f'to run: {len(todo)}, n_proc={args.n_proc}, pyfftw={_PYFFTW_OK}', flush=True)
    print(f'[ensemble] grids={args.Ns}, A0={args.A0_list}, dt={args.dt}, '
          f'T={args.T}, dealias={dealias_flags}', flush=True)

    if not todo:
        print('[ensemble] nothing to do; all tasks completed.', flush=True)
        return

    # Run with multiprocessing; imap_unordered preserves checkpointing robustness.
    # Start method: explicit if requested, else platform default (macOS = spawn).
    if args.start_method:
        import multiprocessing as _mp
        ctx = _mp.get_context(args.start_method)
    else:
        import multiprocessing as _mp
        ctx = _mp.get_context()
    t_start = time.time()
    n_done = 0
    with ctx.Pool(processes=args.n_proc) as pool:
        for res in pool.imap_unordered(run_one, todo):
            # record
            data['seeds'].append(res['seed'])
            data['Ns'].append(res['N'])
            data['A0s'].append(res['A0'])
            data['dts'].append(res['dt'])
            data['dealias'].append(res['dealias'])
            data['Ts'].append(res['T'])
            data['rmax_max'].append(res['rmax_max'])
            data['rmax_final'].append(res['rmax_final'])
            data['kurt_final'].append(res['kurt_final'])
            data['mass_drift'].append(res['mass_drift'])
            n_done += 1
            # save incrementally
            save(args.out, data)
            if n_done % max(1, len(todo) // 20) == 0 or n_done == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(todo) - n_done) / 60
                print(f'[progress] {n_done}/{len(todo)} done, '
                      f'elapsed={elapsed/60:.1f}min, ETA={eta:.1f}min', flush=True)

    elapsed = time.time() - t_start
    print(f'[ensemble] finished {len(todo)} new tasks in {elapsed/60:.1f} min; '
          f'output -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
