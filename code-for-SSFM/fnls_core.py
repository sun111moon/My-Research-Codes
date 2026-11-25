"""Core utilities for the split-step Fourier study of fNLS integrable turbulence.

Companion code for

  H. Ma and A. Deng, "Spurious extreme events in under-resolved simulations of
  integrable turbulence: a single-grid delta fingerprint and a convergence
  protocol".

The model is the one-dimensional focusing nonlinear Schrodinger equation

    i psi_t + 0.5 psi_xx + |psi|^2 psi = 0,

integrated on a periodic domain with a symmetrized (Strang) split-step Fourier
method.  The 2/3 Orszag dealiasing rule is OFF by default and can be switched on
with dealias=True.

NUMERICAL CONVENTION -- must match the production runs quoted in the paper
-------------------------------------------------------------------------
The paper reports that the production runs used a *symmetrized* SSFM.  The
specific symmetrization behind the published numbers is

    exp(-0.5j k^2 dt/2)  exp(+1j |psi|^2 dt)  exp(-0.5j k^2 dt/2)      (L-N-L)

i.e. a half linear step, one FULL nonlinear step, then the second half linear
step, with r_max recorded at EVERY step.

An equally second-order, but *not numerically identical*, arrangement is

    exp(+0.5j |psi|^2 dt/2)  exp(-0.5j k^2 dt)  exp(+0.5j |psi|^2 dt/2)  (N-L-N)

which this module used in an earlier revision.  Both are correct Strang
splittings of the same equation -- a stationary-soliton study gives clean second
order (error ratio 4.0 per halving of dt) for each, with the same absolute error
to within a few percent -- so this is NOT a bug.  But in the chaotic, spiky
regime the paper studies, the two arrangements do not give the same
single-realization maximum.  Measured on the production seed at N = 8192,
T = 100, dt = 2e-4, no dealiasing:

    L-N-L, r_max every step : r_max ~ 250.8   (matches the quoted 257.3)
    N-L-N, r_max every step : r_max ~ 144.1
    N-L-N, r_max every 2500 steps : r_max ~ 138.4

so the arrangement accounts for the bulk of the gap and the coarse sampling for
about 4%.  The module below therefore uses the L-N-L arrangement of the
production code and records r_max at every step, so that it reproduces the
published table values instead of a different realization of the same chaos.

The Fourier transforms use pyFFTW when it is available -- the production runs
used pyFFTW, and the package is licensed GPL-3.0-or-later accordingly -- and
fall back to numpy.fft otherwise.  The residual difference between the two
libraries is at the round-off level and only matters through the same chaos
amplification.
"""

import numpy as np

try:                                     # production runs used pyFFTW
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fftw
    pyfftw.interfaces.cache.enable()
    pyfftw.config.NUM_THREADS = 1
    _FFT = _fftw.fft
    _IFFT = _fftw.ifft
except Exception:                        # numpy-only fallback
    _FFT = np.fft.fft
    _IFFT = np.fft.ifft

__all__ = [
    "make_grid",
    "semiclassical_ic",
    "rmax",
    "kurtosis",
    "lowpass",
    "evolve",
    "fourier_resample",
]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def make_grid(L, N):
    """Return the periodic grid x in [-L/2, L/2), the wavenumbers k and dx."""
    dx = L / N
    x = np.linspace(-L / 2.0, L / 2.0 - dx, N)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, k, dx


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
def semiclassical_ic(x, A0=2.0, ell_A=50.0, ell_S=50.0, eps=0.1,
                     noise_amp=1e-3, seed=300876):
    """Semiclassical, partially coherent initial state.

        psi(x, 0) = [A0 + a(x)] exp( i S(x)/eps + i phi(x) )

    a and S are independent real Gaussian random fields with Gaussian spectra
    exp(-(k ell)^2 / 2) and correlation lengths ell_A, ell_S, normalised so
    that std(a) = 0.3 A0 and std(S) = 1; phi is an independent uniform phase of
    amplitude noise_amp.  The defaults are the values used for the published
    data set.

    NOTE
    ----
    The random fields are drawn with length N, so a call with a different grid
    returns different fields: runs on different grids are distinct realizations
    from t = 0.  To put one physical field on several grids, generate it once on
    a reference grid and map it with fourier_resample.
    """
    rng = np.random.default_rng(seed)
    N = len(x)
    dx = x[1] - x[0]
    k_ = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    spec_A = np.exp(-(k_ * ell_A) ** 2 / 2)
    a_k = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(spec_A)
    a_x = np.real(np.fft.ifft(a_k))
    a_x = a_x / np.std(a_x) * 0.3 * A0

    spec_S = np.exp(-(k_ * ell_S) ** 2 / 2)
    S_k = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(spec_S)
    S_x = np.real(np.fft.ifft(S_k))
    S_x = S_x / np.std(S_x)

    phi = noise_amp * rng.uniform(-np.pi, np.pi, N)
    return (A0 + a_x) * np.exp(1j * (S_x / eps + phi))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def rmax(psi):
    """Peak-intensity ratio r_max = max|psi|^2 / <|psi|^2>."""
    I = np.abs(psi) ** 2
    return float(I.max() / I.mean())


def kurtosis(psi):
    """Kurtosis of the intensity, kappa = <I^2> / <I>^2  (equals 2 for
    Gaussian statistics)."""
    I = np.abs(psi) ** 2
    m = I.mean()
    return float(np.mean(I ** 2) / m ** 2)


def lowpass(psi, k, frac):
    """Sharp low-pass filter keeping |k| <= frac * k_Nyq.

    With frac = 1 nothing is removed; with frac = 2/3 this is the standard
    2/3 Orszag truncation.
    """
    kc = frac * np.max(np.abs(k))
    pk = np.fft.fft(psi)
    pk[np.abs(k) > kc] = 0.0
    return np.fft.ifft(pk)


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
def evolve(psi0, k, dt, T, dealias=False, n_save=200, dump_at=None,
           verbose=False):
    """Symmetrized split-step Fourier integration of the fNLS equation.

    The step is the L-N-L arrangement of the production code,

        psi <- exp(-0.5j k^2 dt/2) psi                 (half linear step)
        psi <- exp(+1j |psi|^2 dt)  psi                (full nonlinear step)
        psi <- exp(-0.5j k^2 dt/2) psi                 (half linear step)

    Parameters
    ----------
    psi0    : complex ndarray, initial field
    k       : wavenumber grid from make_grid
    dt      : time step
    T       : final time
    dealias : if True, apply the 2/3 Orszag truncation after each step
    n_save  : kept for backward compatibility; it no longer controls the
              sampling of r_max(t).  r_max is recorded at EVERY step, because
              the spurious spike of the paper is short-lived and a coarse
              sampling underestimates the maximum by ~4% at N = 8192.
    dump_at : optional time; if given, the full field at the first step with
              t >= dump_at is appended to the return value
    verbose : print progress every 10 percent

    Returns
    -------
    psi_final, t_axis, rmax_axis [, psi_dump]

    t_axis and rmax_axis have one entry per time step, so that
    ``max(rmax_axis)`` is the true maximum over the run.
    """
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    N = psi.size
    n_steps = int(round(T / dt))

    lin_half = np.exp(-0.5j * k ** 2 * dt / 2.0)

    if dealias:
        kmax = np.max(np.abs(k))
        keep = np.abs(k) <= (2.0 / 3.0) * kmax

    t_axis = np.empty(n_steps, dtype=float)
    rmax_axis = np.empty(n_steps, dtype=float)
    psi_dump = None
    dumped = dump_at is None

    for n in range(n_steps):
        # half linear step
        psi = _IFFT(lin_half * _FFT(psi))
        # full nonlinear step
        psi = psi * np.exp(1j * np.abs(psi) ** 2 * dt)
        # half linear step
        psi = _IFFT(lin_half * _FFT(psi))
        if dealias:
            pk = _FFT(psi)
            pk[~keep] = 0.0
            psi = _IFFT(pk)

        t = (n + 1) * dt
        if not dumped and t >= dump_at:
            psi_dump = psi.copy()
            dumped = True
        t_axis[n] = t
        rmax_axis[n] = rmax(psi)
        if verbose and n_steps >= 10 and n % (n_steps // 10) == 0:
            print("    step %d/%d (t=%.1f)" % (n, n_steps, t))

    if dump_at is not None and psi_dump is None:
        psi_dump = psi.copy()
    return psi, t_axis, rmax_axis, psi_dump


# ---------------------------------------------------------------------------
# Fourier resampling between grids (same physical initial condition)
# ---------------------------------------------------------------------------
def fourier_resample(psi_ref, N_ref, N_new):
    """Band-limited resampling of a field from an N_ref grid to an N_new grid.

    Zero-padding (N_new > N_ref) refines the grid; truncation
    (N_new < N_ref) coarsens it.  The physical field is preserved.
    """
    if N_ref == N_new:
        return psi_ref.copy()
    pk = np.fft.fft(psi_ref)
    half = min(N_ref, N_new) // 2
    pk_new = np.zeros(N_new, dtype=complex)
    pk_new[:half] = pk[:half]
    pk_new[-half:] = pk[-half:]
    return np.fft.ifft(pk_new) * (N_new / float(N_ref))
