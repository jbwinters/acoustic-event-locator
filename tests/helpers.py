"""Shared helpers for the test suite (imported as `helpers`; tests/ is on sys.path via conftest)."""
import numpy as np

import locate_event as le

C = 343.0
FS = 48000


def square_xyz(side=20.0):
    return np.array([[0, 0, 1.5], [side, 0, 2.0], [side, side, 1.0], [0, side, 3.0]], dtype=float)


def l_xyz():
    return np.array([[0, 0, 2], [0, 22, 2.2], [0, 44, 1.8], [16.6, 0, 3.0], [33.1, 0, 2.5], [49.7, 0, 2.8]], dtype=float)


def linear_xyz(n=5, spacing=20.0):
    return np.array([[0.0, spacing * i, 0.0] for i in range(n)], dtype=float)


def arrivals(src_xy, XYZ, c=C, t0=1.0, offsets=None, source_z=0.0, noise=0.0, rng=None):
    """True arrival times on each recording's clock (optionally noisy)."""
    d = le.distances_3d(np.asarray(src_xy, float), np.asarray(XYZ, float), source_z)
    t = t0 + d / c
    if offsets is not None:
        t = t + np.asarray(offsets, float)
    if noise > 0:
        rng = np.random.default_rng(0) if rng is None else rng
        t = t + noise * rng.standard_normal(len(t))
    return t


def pos_error(sol, true_xy):
    return float(np.linalg.norm(sol.s_xy - np.asarray(true_xy, float)))


def nearest_solution_error(sol, true_xy):
    """Error to the closest of the best solution and any reported alternative."""
    errs = [pos_error(sol, true_xy)] + [
        float(np.linalg.norm(np.array([a["x"], a["y"]]) - np.asarray(true_xy, float))) for a in sol.alternatives
    ]
    return min(errs)


def click_track(fs, duration_s, onset_s, amp=0.5, noise=0.0, tau_s=0.003, rng=None, noise_rng=None):
    """A decaying broadband click at a (fractional) onset time plus optional white noise.
    The click waveform is drawn from `rng`; noise from `noise_rng` (defaults to `rng`)."""
    rng = np.random.default_rng(0) if rng is None else rng
    noise_rng = rng if noise_rng is None else noise_rng
    n = int(duration_s * fs)
    m = int(0.05 * fs)
    ev = rng.standard_normal(m) * np.exp(-np.arange(m) / fs / tau_s)
    ev /= np.max(np.abs(ev))
    k = int(np.floor(onset_s * fs))
    x = np.zeros(n)
    x[k : k + m] += amp * ev[: max(0, min(m, n - k))]
    frac = onset_s * fs - k
    if frac > 0:
        f = np.fft.rfftfreq(n)
        x = np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * frac), n=n)
    return x + noise * noise_rng.standard_normal(n)
