#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locate a single impulsive acoustic event (gunshot, explosion, firework burst) from
audio recorded by several spatially separated devices, using time-difference-of-arrival
(TDOA) multilateration.

Inputs
  --videos_dir   directory with the recordings (any ffmpeg-readable video/audio, or .wav)
  --positions    JSON describing each recording's position and site constants:
      {
        "temperature_C": 20,                          // optional, sets the speed of sound
        "speed_of_sound": null,                       // optional, overrides temperature
        "reference": {"lat": 41.881, "lon": -87.629}, // optional local-frame origin
                                                      // ("reference_point" is also accepted)
        "mics": [
          {"file": "cam1.mp4", "lat": 41.88110, "lon": -87.62970, "height_m": 1.6},
          {"file": "cam2.mp4", "lat": 41.88125, "lon": -87.62920, "height_m": 1.5},
          ...
        ]
      }

Outputs (written to --out)
  results.json   location (local meters and WGS84), 95% ellipse, per-recording arrivals,
                 clock offsets, residuals, fit diagnostics, alternative solutions
  sync.csv       per-recording arrival time and the seek offset that aligns it on the event
  layout.png     geometry, estimate, 95% ellipse
  wav/           mono audio at --fs extracted from each recording

Model
  t_i = t0 + ||s - x_i|| / c + delta_i
    s        event position; x, y are solved, z is fixed at --source_height_m
    x_i      recording position (lat/lon projected to a local tangent plane, plus height)
    t0       emission time on the common clock
    delta_i  clock offset of recording i

Clock synchronization
  One event cannot determine per-device clock offsets: for every candidate position there
  is a set of offsets that fits the arrivals exactly. The position is therefore only as
  good as the clock synchronization of the recordings. Two modes are offered:
    --clock_sigma_ms 0   (default) devices are assumed synchronized; offsets fixed at 0
    --clock_sigma_ms S   offsets are estimated under a Gaussian prior N(0, S^2); the
                         result is a MAP estimate and the position uncertainty grows with S
  Recordings whose clocks differ by more than the array's maximum propagation delay cannot
  be localized from a single event. sync.csv still gives the offset that aligns each
  recording on the event.

Pipeline
  1. Load or extract audio, resample to --fs, causal bandpass (no pre-ringing).
  2. Per recording, detect onset candidates from a moving-RMS envelope against a robust
     noise floor; choose one candidate per recording so that the set is physically
     consistent with the geometry (event association).
  3. Fine first-arrival pick with an AIC change-point picker around each candidate.
  4. Pairwise band-limited generalized cross-correlation between recordings for
     sub-sample relative timing, fused by weighted least squares.
  5. Vectorized grid search for starting points, Levenberg-Marquardt refinement with
     Huber reweighting, multi-start so that ambiguous geometries are reported.
  6. Covariance from the full parameter Fisher matrix (position, t0, offsets), inflated by
     the reduced chi-square when the residuals exceed the assumed timing noise.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from numpy.fft import irfft, rfft
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt, windows

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VERBOSE = False
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


class LocatorError(Exception):
    """User-facing error: bad inputs or data that cannot be localized."""


# ------------------------------ Utilities ------------------------------


def log(msg: str, level: str = "INFO"):
    if level == "DEBUG" and not VERBOSE:
        return
    print(f"[{level}] {msg}", flush=True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ------------------------------ Geo conversion (local tangent plane) ------------------------------


def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """WGS-84 meters per degree of latitude and longitude at the given latitude."""
    phi = math.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = (
        111412.84 * math.cos(phi)
        - 93.5 * math.cos(3 * phi)
        + 0.118 * math.cos(5 * phi)
    )
    return m_lat, m_lon


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular projection about (lat0, lon0); x east, y north, meters.
    Accurate to ~1e-4 relative over a few kilometers."""
    m_lat, m_lon = meters_per_degree(lat0)
    return (lon - lon0) * m_lon, (lat - lat0) * m_lat


def local_xy_to_latlon(x, y, lat0, lon0):
    m_lat, m_lon = meters_per_degree(lat0)
    return y / m_lat + lat0, x / m_lon + lon0


def speed_of_sound_mps(temperature_C: float) -> float:
    """Speed of sound in dry air, m/s."""
    return 20.05 * math.sqrt(273.15 + temperature_C)


# ------------------------------ Positions ------------------------------


@dataclass
class Mic:
    file: str
    lat: float
    lon: float
    height_m: float = 0.0
    height_sigma_m: float = 0.0  # 0 = height known; > 0 = prior std, height is estimated


def parse_positions(J: dict) -> Tuple[List[Mic], Tuple[float, float], float]:
    """Parse a positions dict. Does not touch the filesystem (file names kept as given)."""
    if not isinstance(J.get("mics"), list) or len(J["mics"]) == 0:
        raise LocatorError("positions file must contain a non-empty 'mics' list")
    mics = []
    for m in J["mics"]:
        for key in ("file", "lat", "lon"):
            if key not in m:
                raise LocatorError(f"mic entry is missing '{key}': {m}")
        mics.append(
            Mic(
                file=str(m["file"]),
                lat=float(m["lat"]),
                lon=float(m["lon"]),
                height_m=float(m.get("height_m", 0.0)),
                height_sigma_m=float(m.get("height_sigma_m", 0.0)),
            )
        )
        if mics[-1].height_sigma_m < 0:
            raise LocatorError(f"height_sigma_m must be >= 0: {m}")
    ref = J.get("reference") or J.get("reference_point")
    if ref:
        lat0, lon0 = float(ref["lat"]), float(ref["lon"])
    else:
        lat0 = float(np.mean([m.lat for m in mics]))
        lon0 = float(np.mean([m.lon for m in mics]))
    c = J.get("speed_of_sound", None)
    if c is None:
        c = speed_of_sound_mps(float(J.get("temperature_C", 20.0)))
    c = float(c)
    if not (250.0 < c < 450.0):
        raise LocatorError(f"speed of sound {c} m/s is not plausible for air")
    return mics, (lat0, lon0), c


def resolve_media_path(videos_dir: str, name: str) -> str:
    p = os.path.join(videos_dir, name)
    if os.path.exists(p):
        return p
    stem = os.path.splitext(p)[0]
    for ext in (".wav", ".flac"):
        if os.path.exists(stem + ext):
            log(f"{name} not found; using {os.path.basename(stem + ext)}", "DEBUG")
            return stem + ext
    raise LocatorError(f"Recording listed in positions not found: {p}")


def load_positions(positions_json: str, videos_dir: str):
    """Returns (mics with resolved paths, (lat0, lon0), speed_of_sound, raw_json)."""
    J = read_json(positions_json)
    mics, origin, c = parse_positions(J)
    for m in mics:
        m.file = resolve_media_path(videos_dir, m.file)
    return mics, origin, c, J


def mic_local_xyz(mics: Sequence[Mic], lat0: float, lon0: float) -> np.ndarray:
    XYZ = np.zeros((len(mics), 3), dtype=float)
    for i, m in enumerate(mics):
        x, y = latlon_to_local_xy(m.lat, m.lon, lat0, lon0)
        XYZ[i] = (x, y, m.height_m)
    return XYZ


def mic_height_sigma(mics: Sequence[Mic]) -> np.ndarray:
    return np.array([m.height_sigma_m for m in mics], dtype=float)


# ------------------------------ Audio loading ------------------------------


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise LocatorError(
            "ffmpeg is required to read video files but was not found on PATH "
            "(WAV/FLAC inputs do not need it)."
        )


def extract_audio_ffmpeg(in_video: str, out_wav: str, target_sr: int = 48000):
    """Extract a mono WAV at target_sr from any ffmpeg-readable file."""
    cmd = [
        "ffmpeg", "-y", "-i", in_video, "-vn", "-ac", "1",
        "-ar", str(target_sr), "-f", "wav", out_wav,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        log(
            f"ffmpeg failed on {in_video}:\n{e.stderr.decode('utf-8', errors='ignore')}",
            "ERROR",
        )
        raise


def _to_mono_at_rate(x: np.ndarray, sr: int, fs: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    if sr != fs:
        g = math.gcd(int(fs), int(sr))
        x = resample_poly(x, fs // g, sr // g)
    return x - float(np.mean(x))


def load_audio(path: str, fs: int, wav_dir: str) -> np.ndarray:
    """Return a zero-mean mono float64 signal at rate fs; also writes wav_dir/<name>.wav."""
    ensure_dir(wav_dir)
    out_wav = os.path.join(wav_dir, os.path.basename(path) + ".wav")
    if path.lower().endswith(AUDIO_EXTS):
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        x = _to_mono_at_rate(x, sr, fs)
        sf.write(out_wav, x.astype(np.float32), fs)
    else:
        require_ffmpeg()
        extract_audio_ffmpeg(path, out_wav, target_sr=fs)
        x, sr = sf.read(out_wav, dtype="float64", always_2d=True)
        x = _to_mono_at_rate(x, sr, fs)
    if len(x) < fs // 10:
        raise LocatorError(f"{os.path.basename(path)}: audio shorter than 0.1 s")
    return x


# ------------------------------ Signal processing ------------------------------


def design_bandpass(low_hz: float, high_hz: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), 0.95 * nyq)
    if not low < high:
        raise LocatorError(f"invalid bandpass {low_hz}-{high_hz} Hz for fs={fs}")
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def apply_bandpass(x, fs, low_hz=200.0, high_hz=4000.0, order=4, zero_phase=False):
    """Bandpass filter. Causal by default so an onset never appears before it happened
    (zero-phase filtering pre-rings and biases first-arrival picks early)."""
    sos = design_bandpass(low_hz, high_hz, fs, order)
    return sosfiltfilt(sos, x) if zero_phase else sosfilt(sos, x)


def rms_envelope(x, fs, win_ms=20.0):
    """Trailing (causal) moving RMS over win_ms; same length as x."""
    n = max(1, int(round(fs * win_ms / 1000.0)))
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - n)
    return np.sqrt((c[idx] - c[lo]) / (idx - lo))


def noise_floor(env: np.ndarray, percentile: float = 20.0) -> float:
    """Robust background level of an envelope (low percentile over the whole track)."""
    return float(max(np.percentile(env, percentile), 1e-12))


def aic_picker(x: np.ndarray, min_len: int = 64, guard_frac: float = 0.02) -> Optional[int]:
    """Akaike Information Criterion change-point picker for the first arrival in a window.
    Returns the split index k such that x[:k] is 'noise' and x[k:] is 'signal', or None if
    the window is too short. The first/last guard_frac of the window is excluded because
    the segment variances are unreliable there."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < min_len:
        return None
    s1 = np.cumsum(x)
    s2 = np.cumsum(x * x)
    g = max(4, int(guard_frac * N))
    k = np.arange(g, N - g)
    n1 = k.astype(np.float64)
    n2 = (N - k).astype(np.float64)
    mu1 = s1[k - 1] / n1
    var1 = s2[k - 1] / n1 - mu1 * mu1
    mu2 = (s1[-1] - s1[k - 1]) / n2
    var2 = (s2[-1] - s2[k - 1]) / n2 - mu2 * mu2
    eps = 1e-30
    aic = n1 * np.log(np.maximum(var1, eps)) + n2 * np.log(np.maximum(var2, eps))
    return int(k[int(np.argmin(aic))])


def sta_lta_picker(x, fs, sta_ms=5.0, lta_ms=200.0, thr=3.5) -> Optional[int]:
    """Classic STA/LTA trigger; returns the first index whose ratio exceeds thr, else None."""
    x = np.asarray(x, dtype=np.float64)
    sta_n = max(1, int(fs * sta_ms / 1000.0))
    lta_n = max(sta_n + 1, int(fs * lta_ms / 1000.0))
    if len(x) <= lta_n + 1:
        return None
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    idx = np.arange(lta_n, len(x) + 1)
    sta = (c[idx] - c[idx - sta_n]) / sta_n
    lta = (c[idx] - c[idx - lta_n]) / lta_n
    ratio = sta / (lta + 1e-20)
    hits = np.flatnonzero(ratio > thr)
    if len(hits) == 0:
        return None
    return int(idx[hits[0]] - 1)


def find_onset_candidates(
    env: np.ndarray,
    fs: int,
    floor: float,
    min_ratio: float = 6.0,
    merge_gap_s: float = 0.5,
    max_candidates: int = 8,
) -> List[Tuple[int, float]]:
    """Onset candidates from an envelope: each is (index, strength) where index is the first
    sample above min_ratio * floor and strength is the peak ratio reached before the envelope
    stays below threshold for merge_gap_s. Sorted by strength, strongest first."""
    ratio = env / floor
    above = ratio > min_ratio
    if not above.any():
        return []
    d = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    gap = int(merge_gap_s * fs)
    cands: List[List[float]] = []  # [onset, strength, end]
    for s0, e0 in zip(starts, ends):
        peak = float(ratio[s0:e0].max())
        if cands and (s0 - cands[-1][2]) < gap:
            cands[-1][1] = max(cands[-1][1], peak)
            cands[-1][2] = e0
        else:
            cands.append([int(s0), peak, int(e0)])
    out = sorted(((int(c0), float(c1)) for c0, c1, _ in cands), key=lambda t: -t[1])
    return out[:max_candidates]


def fine_pick(xf: np.ndarray, fs: int, coarse_idx: int, pre_s=0.3, post_s=0.2) -> int:
    """Refine a coarse onset index to the first arrival with the AIC picker."""
    a = max(0, coarse_idx - int(pre_s * fs))
    b = min(len(xf), coarse_idx + int(post_s * fs))
    k = aic_picker(xf[a:b])
    if k is None:
        return int(coarse_idx)
    k_abs = a + k
    # The AIC pick should sit at or shortly before the envelope crossing; if it wandered
    # off (e.g. a second event inside the window) fall back to STA/LTA, then to the coarse pick.
    if not (coarse_idx - int(0.1 * fs) <= k_abs <= coarse_idx + int(0.02 * fs)):
        k2 = sta_lta_picker(xf[a:b], fs)
        if k2 is not None and coarse_idx - int(0.1 * fs) <= a + k2 <= coarse_idx + int(0.02 * fs):
            return int(a + k2)
        return int(coarse_idx)
    return int(k_abs)


# ------------------------------ Cross-correlation ------------------------------


def quadratic_subsample_peak(y, i):
    """Parabolic interpolation around the discrete peak y[i]. Returns (x_hat, y_hat)."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2 * y1 + y2  # negative at a maximum
    if abs(denom) < 1e-30:
        return float(i), y1
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return i + delta, y1 - 0.25 * (y0 - y2) * delta


def gcc(
    sig,
    ref,
    fs,
    max_tau: Optional[float] = None,
    interp: int = 8,
    weighting: str = "phat",
    band: Optional[Tuple[float, float]] = None,
    phat_eps: float = 1e-2,
):
    """Generalized cross-correlation of sig against ref.

    Returns (tau_s, quality, lags_s, cc). tau > 0 means sig is delayed relative to ref.
    weighting: 'cc' (plain), 'phat' (regularized phase transform), 'scot'.
    band: (low, high) Hz; bins outside are zeroed so empty spectrum regions do not add noise.
    quality: peak height over the largest secondary peak more than 1 ms away (>= 1)."""
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = 1 << int(np.ceil(np.log2(len(sig) + len(ref))))
    S = rfft(sig, n=n)
    R = rfft(ref, n=n)
    X = S * np.conj(R)
    mag = np.abs(X)
    if weighting == "phat":
        X = X / (mag + phat_eps * (mag.max() + 1e-300))
    elif weighting == "scot":
        X = X / (np.sqrt(np.abs(S) ** 2 * np.abs(R) ** 2) + 1e-12 * (mag.max() + 1e-300))
    elif weighting != "cc":
        raise ValueError(f"unknown gcc weighting {weighting!r}")
    if band is not None:
        f = np.fft.rfftfreq(n, 1.0 / fs)
        X[(f < band[0]) | (f > band[1])] = 0.0
    cc = irfft(X, n=interp * n)
    half = interp * n // 2
    cc = np.concatenate((cc[-half:], cc[:half]))
    lags = np.arange(-half, half)
    if max_tau is not None:
        lim = int(round(interp * fs * max_tau))
        sel = slice(max(0, half - lim), min(len(cc), half + lim + 1))
        cc = cc[sel]
        lags = lags[sel]
    i = int(np.argmax(cc))
    xi, yi = quadratic_subsample_peak(cc, i)
    tau = (lags[0] + xi) / float(interp * fs)
    excl = int(round(interp * fs * 0.001))
    mask = np.ones(len(cc), dtype=bool)
    mask[max(0, i - excl): i + excl + 1] = False
    second = float(cc[mask].max()) if mask.any() else 0.0
    quality = float(yi / second) if second > 0 else 1e6
    quality = float(np.clip(quality, 0.0, 1e6))
    return float(tau), quality, lags / float(interp * fs), cc


def _event_window(x: np.ndarray, k: int, pre: int, post: int) -> np.ndarray:
    """Slice x[k-pre:k+post], zero-padded so the pick is always at index pre."""
    w = np.zeros(pre + post, dtype=np.float64)
    a, b = k - pre, k + post
    sa, sb = max(0, a), min(len(x), b)
    if sb > sa:
        w[sa - a: sb - a] = x[sa:sb]
    return w


def refine_arrivals_pairwise(
    picks_idx: Sequence[int],
    signals: Sequence[np.ndarray],
    fs: int,
    pre_s: float = 0.03,
    post_s: float = 0.08,
    max_tau: float = 0.015,
    weighting: str = "phat",
    band: Optional[Tuple[float, float]] = None,
    min_quality: float = 1.5,
    ref_idx: Optional[int] = None,
) -> Dict[str, object]:
    """Sub-sample relative timing between recordings.

    Each pair (i, j) is cross-correlated around its picks, giving tau_ij = eps_j - eps_i where
    eps_i is the correction to pick i. Corrections are fused by weighted least squares with
    eps[ref_idx] = 0 (the reference pick is left where the AIC put it, which only shifts t0).
    Returns dict with arrival_s, eps_s, pair_tau_s, pair_quality, pairs, used_pairs."""
    M = len(signals)
    pre, post = int(pre_s * fs), int(post_s * fs)
    taper = windows.tukey(pre + post, alpha=0.25)
    segs = [_event_window(signals[i], int(picks_idx[i]), pre, post) * taper for i in range(M)]
    segs = [s - s.mean() for s in segs]
    if ref_idx is None:
        ref_idx = int(np.argmax([float(np.sum(s * s)) for s in segs]))
    pairs, taus, quals = [], [], []
    for i in range(M):
        for j in range(i + 1, M):
            tau, q, _, _ = gcc(segs[j], segs[i], fs, max_tau=max_tau, interp=8, weighting=weighting, band=band)
            pairs.append((i, j))
            taus.append(tau)
            quals.append(q)
    pairs_arr = np.array(pairs, dtype=int).reshape(-1, 2)
    taus_arr = np.array(taus, dtype=float)
    quals_arr = np.array(quals, dtype=float)
    used = quals_arr >= min_quality
    eps = np.zeros(M)
    if M >= 2 and used.sum() >= M - 1:
        A = np.zeros((int(used.sum()), M))
        rows = np.flatnonzero(used)
        for r, p in enumerate(rows):
            i, j = pairs_arr[p]
            A[r, j] = 1.0
            A[r, i] = -1.0
        wts = np.sqrt(np.log(quals_arr[rows]))  # heavier weight for sharper, unambiguous peaks
        Ar = np.delete(A, ref_idx, axis=1) * wts[:, None]
        br = taus_arr[rows] * wts
        sol, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        eps = np.insert(sol, ref_idx, 0.0)
        # A recording with no usable pair is left uncorrected.
        connected = np.zeros(M, dtype=bool)
        connected[ref_idx] = True
        for _ in range(M):
            for p in rows:
                i, j = pairs_arr[p]
                if connected[i] or connected[j]:
                    connected[i] = connected[j] = True
        eps[~connected] = 0.0
    arrival_s = np.asarray(picks_idx, dtype=float) / fs + eps
    return {
        "arrival_s": arrival_s,
        "eps_s": eps,
        "pairs": pairs_arr,
        "pair_tau_s": taus_arr,
        "pair_quality": quals_arr,
        "used_pairs": used,
        "ref_idx": int(ref_idx),
    }


# ------------------------------ Event association ------------------------------


def max_pairwise_lag(XYZ: np.ndarray, c: float, slack_s: float = 0.005, clock_sigma_s: float = 0.0, max_detour_s: float = 0.0):
    """Matrix of the largest physically possible |t_i - t_j| for each pair. max_detour_s widens
    the gate so that an arrival delayed by an obstacle (up to that extra path time) still
    associates with the event; the solver then decides whether it was occluded."""
    D = np.linalg.norm(XYZ[:, None, :] - XYZ[None, :, :], axis=-1)
    return D / c + slack_s + 3.0 * clock_sigma_s + max_detour_s


def associate_onsets(
    cands: Sequence[Sequence[Tuple[int, float]]],
    fs: int,
    max_lag: np.ndarray,
    anchor_depth: int = 3,
) -> Tuple[Dict[int, int], List[int]]:
    """Choose one onset candidate per recording so that all chosen arrivals are mutually
    consistent with the geometry. Tries the strongest candidates of each recording as the
    anchor and keeps the assignment covering the most recordings (ties: largest total
    strength). Returns (chosen {track: index}, tracks_without_consistent_onset)."""
    M = len(cands)
    best_key, best = None, {}
    for a in range(M):
        for ia, sa in list(cands[a])[:anchor_depth]:
            chosen = {a: (ia, sa)}
            for j in range(M):
                if j == a:
                    continue
                feas = [(i, s) for (i, s) in cands[j] if abs(i - ia) / fs <= max_lag[a, j]]
                if feas:
                    chosen[j] = max(feas, key=lambda t: t[1])
            # enforce pairwise consistency: drop the weaker member of any violating pair
            while True:
                bad = None
                keys = sorted(chosen)
                for u in range(len(keys)):
                    for v in range(u + 1, len(keys)):
                        i, j = keys[u], keys[v]
                        if abs(chosen[i][0] - chosen[j][0]) / fs > max_lag[i, j]:
                            weaker = i if chosen[i][1] < chosen[j][1] else j
                            if weaker == a:
                                weaker = j if weaker == i else i
                            bad = weaker
                            break
                    if bad is not None:
                        break
                if bad is None:
                    break
                del chosen[bad]
            key = (len(chosen), sum(s for _, s in chosen.values()))
            if best_key is None or key > best_key:
                best_key, best = key, chosen
    chosen_idx = {k: int(v[0]) for k, v in best.items()}
    missing = [i for i in range(M) if i not in chosen_idx]
    return chosen_idx, missing


# ------------------------------ TDOA solver ------------------------------


def distances_3d(s, XYZ, source_z=0.0):
    """Distances from a source given as (x, y) with height source_z, or as (x, y, z)."""
    s = np.asarray(s, dtype=float)
    z = s[2] if s.shape[0] > 2 else source_z
    dx = XYZ[:, 0] - s[0]
    dy = XYZ[:, 1] - s[1]
    dz = XYZ[:, 2] - z
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def predict_arrivals(s, XYZ, c, source_z=0.0, t0=0.0, delta=None):
    """t_i = t0 + ||s - x_i||/c + delta_i. Returns (pred, distances)."""
    d = distances_3d(s, XYZ, source_z)
    pred = t0 + d / c
    if delta is not None:
        pred = pred + delta
    return pred, d


def profile_t0(s, t, XYZ, c, w, source_z=0.0):
    """Weighted least-squares emission time for a fixed position (offsets = 0)."""
    _, d = predict_arrivals(s, XYZ, c, source_z)
    return float(np.sum(w * (t - d / c)) / np.sum(w))


def grid_search_init(
    t, XYZ, c, w, source_z, bounds, res, n_best=6, min_sep=None, max_points=300_000
):
    """Vectorized grid search of the synchronized-clock cost with t0 profiled out, at a fixed
    source height. Returns up to n_best well separated low-cost grid points as
    [((x, y), cost), ...]."""
    xmin, xmax, ymin, ymax = bounds
    area = max(xmax - xmin, res) * max(ymax - ymin, res)
    res_eff = max(res, math.sqrt(area / max_points))
    if res_eff > res:
        log(f"grid resolution coarsened from {res:.2f} to {res_eff:.2f} m to bound the search", "DEBUG")
    xs = np.arange(xmin, xmax + 0.5 * res_eff, res_eff)
    ys = np.arange(ymin, ymax + 0.5 * res_eff, res_eff)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    dx = GX[..., None] - XYZ[:, 0]
    dy = GY[..., None] - XYZ[:, 1]
    dz = source_z - XYZ[:, 2]
    D = np.sqrt(dx * dx + dy * dy + dz * dz) / c
    y = np.asarray(t, dtype=float) - D
    wn = np.asarray(w, dtype=float)
    wn = wn / np.sum(wn)
    t0 = np.sum(wn * y, axis=-1, keepdims=True)
    cost = np.sum(np.asarray(w, dtype=float) * (y - t0) ** 2, axis=-1)
    order = np.argsort(cost, axis=None)
    sep = min_sep if min_sep is not None else max(10 * res_eff, 0.05 * max(xmax - xmin, ymax - ymin))
    picks: List[Tuple[np.ndarray, float]] = []
    for f in order[:20000]:
        i, j = np.unravel_index(int(f), cost.shape)
        p = np.array([xs[i], ys[j]], dtype=float)
        if all(np.linalg.norm(p - q) > sep for q, _ in picks):
            picks.append((p, float(cost[i, j])))
            if len(picks) >= n_best:
                break
    return picks


class _Layout:
    """Index map of the parameter vector theta = [x, y, (z), t0, (delta_0..M-1), (z_i for
    recordings with uncertain height)]."""

    def __init__(self, M: int, solve_z: bool, estimate_offsets: bool, height_sigma: np.ndarray):
        self.M = M
        self.solve_z = bool(solve_z)
        self.estimate_offsets = bool(estimate_offsets)
        self.hidx = np.flatnonzero(np.asarray(height_sigma, dtype=float) > 0)
        n = 2
        self.i_z = n if self.solve_z else None
        n += 1 if self.solve_z else 0
        self.i_t0 = n
        n += 1
        self.i_delta = n if self.estimate_offsets else None
        n += M if self.estimate_offsets else 0
        self.i_h = n if len(self.hidx) else None
        n += len(self.hidx)
        self.n = n
        self.n_prior = (1 if self.solve_z else 0) + (M if self.estimate_offsets else 0) + len(self.hidx)
        self.n_pos = 3 if self.solve_z else 2

    def pack(self, xy, z, t0, delta=None, heights=None):
        th = np.zeros(self.n)
        th[0], th[1] = xy[0], xy[1]
        if self.solve_z:
            th[self.i_z] = z
        th[self.i_t0] = t0
        if self.estimate_offsets and delta is not None:
            th[self.i_delta : self.i_delta + self.M] = delta
        if self.i_h is not None and heights is not None:
            th[self.i_h : self.i_h + len(self.hidx)] = np.asarray(heights)[self.hidx]
        return th

    def unpack(self, th, XYZ, z_fixed):
        s = np.array([th[0], th[1], th[self.i_z] if self.solve_z else z_fixed])
        t0 = th[self.i_t0]
        delta = th[self.i_delta : self.i_delta + self.M] if self.estimate_offsets else np.zeros(self.M)
        XYZc = XYZ
        if self.i_h is not None:
            XYZc = XYZ.copy()
            XYZc[self.hidx, 2] = th[self.i_h : self.i_h + len(self.hidx)]
        return s, t0, delta, XYZc


@dataclass
class TDOASolution:
    s_xy: np.ndarray
    t0: float
    delta: np.ndarray
    cov: np.ndarray
    cov_xy: np.ndarray
    residuals_s: np.ndarray
    weights: np.ndarray
    sigma_t: np.ndarray
    chi2: float
    dof: int
    scale: float
    cost: float
    converged: bool
    iterations: int
    alternatives: List[dict]
    condition_number: float
    degenerate: bool
    estimate_offsets: bool
    ambiguous: bool
    rejected: List[int]
    at_boundary: bool = False
    s_xyz: Optional[np.ndarray] = None
    z_std: float = 0.0
    solve_z: bool = False
    z_at_bound: bool = False
    cov_pos: Optional[np.ndarray] = None
    mic_heights: Optional[np.ndarray] = None
    mic_height_std: Optional[np.ndarray] = None
    occluded: List[int] = field(default_factory=list)
    occlusion_prob: Optional[np.ndarray] = None
    detour_m: Optional[np.ndarray] = None
    loss: str = "huber"
    sigma_eff: Optional[np.ndarray] = None
    n_direct: int = 0

    @property
    def rmse_s(self) -> float:
        """RMS residual of the recordings that carry weight (direct-path arrivals)."""
        if not len(self.residuals_s):
            return float("nan")
        w = self.weights if self.weights is not None else np.ones(len(self.residuals_s))
        sel = w >= 0.5
        r = self.residuals_s[sel] if sel.any() else self.residuals_s
        return float(np.sqrt(np.mean(r**2)))


def _huber_irls_weights(raw: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(raw)
    w = np.ones_like(a)
    big = a > k
    w[big] = k / a[big]
    return w


class _HuberLoss:
    """Symmetric Huber loss on raw residuals with threshold k (seconds)."""

    name = "huber"

    def __init__(self, k: float):
        self.k = float(k)

    def weights(self, raw, sigma):
        return _huber_irls_weights(raw, self.k)

    def cost(self, raw, sigma):
        u = np.abs(raw) / sigma
        kn = self.k / sigma
        rho = np.where(u <= kn, 0.5 * u * u, kn * u - 0.5 * kn * kn)
        return float(np.sum(rho))

    def occlusion_prob(self, raw, sigma):
        return np.zeros_like(np.asarray(raw, dtype=float))


class _OcclusionLoss:
    """Negative log-likelihood of a three-component residual model:
      direct   (prob 1 - p - q): r ~ N(0, sigma^2)
      occluded (prob p):         r = extra path / c + noise, extra path ~ Exponential(mean tau * c),
                                 i.e. an exponentially modified Gaussian; only positive residuals
      blunder  (prob q):         r ~ Uniform(-U, U), a mis-pick of either sign
    Large positive residuals cost ~r/tau and pull almost nothing; residuals that are impossible
    for a direct or delayed arrival (large negative) become blunders with ~zero weight instead
    of dragging the solution. Weights are exact IRLS weights psi(r)/r of this loss."""

    name = "occlusion"

    def __init__(self, p: float, tau: float, q: float = 0.05, blunder_window: float = 0.5):
        self.p = float(p)
        self.q = float(q)
        self.tau = float(tau)
        self.U = float(blunder_window)
        if not (self.p > 0 and self.q >= 0 and self.p + self.q < 1):
            raise LocatorError("occlusion_prob and blunder_prob must be positive and sum to less than 1")

    def _log_components(self, r, sigma):
        from scipy.special import log_ndtr

        log_n = math.log(1 - self.p - self.q) - 0.5 * (r / sigma) ** 2 - np.log(sigma * math.sqrt(2 * math.pi))
        z = sigma / self.tau - r / sigma
        log_e = math.log(self.p) - math.log(self.tau) + 0.5 * (sigma / self.tau) ** 2 - r / self.tau + log_ndtr(-z)
        log_u = np.full_like(r, math.log(self.q / (2 * self.U)) if self.q > 0 else -np.inf)
        return log_n, log_e, log_u, z

    def responsibilities(self, raw, sigma):
        """(gamma_direct, gamma_occluded, gamma_blunder) per residual."""
        r = np.asarray(raw, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        ln, le_, lu, _ = self._log_components(r, sigma)
        m = np.maximum(np.maximum(ln, le_), lu)
        en, ee, eu = np.exp(ln - m), np.exp(le_ - m), np.exp(lu - m)
        s = en + ee + eu
        return en / s, ee / s, eu / s

    def occlusion_prob(self, raw, sigma):
        return self.responsibilities(raw, sigma)[1]

    def blunder_prob(self, raw, sigma):
        return self.responsibilities(raw, sigma)[2]

    def cost(self, raw, sigma):
        r = np.asarray(raw, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        ln, le_, lu, _ = self._log_components(r, sigma)
        m = np.maximum(np.maximum(ln, le_), lu)
        return float(-np.sum(m + np.log(np.exp(ln - m) + np.exp(le_ - m) + np.exp(lu - m))))

    def psi(self, raw, sigma):
        from scipy.special import log_ndtr

        r = np.asarray(raw, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        gn, ge, _ = self.responsibilities(r, sigma)
        z = sigma / self.tau - r / sigma
        log_phi = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
        mills = np.exp(log_phi - log_ndtr(-z))  # phi(-z) / Phi(-z)
        return gn * r / sigma**2 + ge * (1.0 / self.tau - mills / sigma)

    def weights(self, raw, sigma):
        """IRLS weights psi(r)/r relative to the Gaussian weight 1/sigma^2. Within half a sigma
        of zero the loss is quadratic to a very good approximation and psi/r is ill-conditioned
        (the mixture's minimum sits a few microseconds right of zero), so unit weight is used."""
        r = np.asarray(raw, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        psi = self.psi(r, sigma)
        small = np.abs(r) < 0.5 * sigma
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(small, 1.0, psi / np.where(small, 1.0, r) * sigma**2)
        return np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)


def _lm_refine(theta0, t, XYZ, c, sigma_t, layout, priors, loss, box=None, max_iter=200, trace=None):
    """Levenberg-Marquardt with IRLS weights from `loss` over the parameters in `layout`.

    priors: dict with z0, z_sigma, clock_sigma, height_mu (M,), height_sigma (M,).
    box: (xmin, xmax, ymin, ymax, zmin, zmax); steps leaving it are rejected.
    trace: optional list that receives theta after each accepted step."""
    M = len(t)
    z0, z_sigma = priors["z0"], priors["z_sigma"]
    clock_sigma = priors["clock_sigma"]
    height_mu, height_sigma = priors["height_mu"], priors["height_sigma"]
    nP = layout.n_prior

    def evaluate(th):
        s, t0, delta, XYZc = layout.unpack(th, XYZ, z0)
        pred, d = predict_arrivals(s, XYZc, c, s[2], t0, delta)
        raw = t - pred
        r = raw / sigma_t
        J = np.zeros((M, layout.n))
        dsafe = np.maximum(d, 1e-9)
        J[:, 0] = (s[0] - XYZc[:, 0]) / (c * dsafe)
        J[:, 1] = (s[1] - XYZc[:, 1]) / (c * dsafe)
        if layout.solve_z:
            J[:, layout.i_z] = (s[2] - XYZc[:, 2]) / (c * dsafe)
        J[:, layout.i_t0] = 1.0
        if layout.estimate_offsets:
            J[:, layout.i_delta : layout.i_delta + M] = np.eye(M)
        if layout.i_h is not None:
            for k, i in enumerate(layout.hidx):
                J[i, layout.i_h + k] = (XYZc[i, 2] - s[2]) / (c * dsafe[i])
        J = -J / sigma_t[:, None]
        if nP == 0:
            return raw, r, J
        rp = np.zeros(nP)
        Jp = np.zeros((nP, layout.n))
        row = 0
        if layout.solve_z:
            rp[row] = (z0 - s[2]) / z_sigma
            Jp[row, layout.i_z] = -1.0 / z_sigma
            row += 1
        if layout.estimate_offsets:
            rp[row : row + M] = -delta / clock_sigma
            Jp[row : row + M, layout.i_delta : layout.i_delta + M] = -np.eye(M) / clock_sigma
            row += M
        if layout.i_h is not None:
            for k, i in enumerate(layout.hidx):
                rp[row] = (height_mu[i] - XYZc[i, 2]) / height_sigma[i]
                Jp[row, layout.i_h + k] = -1.0 / height_sigma[i]
                row += 1
        return raw, np.concatenate([r, rp]), np.vstack([J, Jp])

    def total_cost(raw, r):
        return loss.cost(raw, sigma_t) + 0.5 * float(np.sum(r[M:] ** 2))

    def inside(th):
        if box is None:
            return True
        if not (box[0] <= th[0] <= box[1] and box[2] <= th[1] <= box[3]):
            return False
        return (not layout.solve_z) or (box[4] <= th[layout.i_z] <= box[5])

    theta = np.array(theta0, dtype=float)
    raw, r, J = evaluate(theta)
    cost = total_cost(raw, r)
    if trace is not None:
        trace.append(theta.copy())
    lam = 1e-3
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        hw = loss.weights(raw, sigma_t)
        sw = np.sqrt(np.concatenate([hw, np.ones(len(r) - M)]))
        Jw = J * sw[:, None]
        JTJ = Jw.T @ Jw
        g = Jw.T @ (r * sw)
        diag = np.maximum(np.diag(JTJ), 1e-18)
        accepted = False
        for _ in range(12):
            A = JTJ + lam * np.diag(diag)
            try:
                step = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A, g, rcond=None)[0]
            th_new = theta + step
            if not inside(th_new):
                lam *= 10.0
                if lam > 1e12:
                    break
                continue
            raw_new, r_new, J_new = evaluate(th_new)
            cost_new = total_cost(raw_new, r_new)
            if cost_new <= cost:
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not accepted:
            converged = True
            break
        rel = (cost - cost_new) / max(abs(cost), 1e-300)
        moved = float(np.linalg.norm(step[: layout.n_pos]))
        theta, raw, r, J, cost = th_new, raw_new, r_new, J_new, cost_new
        if trace is not None:
            trace.append(theta.copy())
        lam = max(lam / 10.0, 1e-15)
        if rel < 1e-12 or moved < 1e-7:
            converged = True
            break
    return theta, raw, r, J, cost, converged, it


def _fit_all_starts(t, XYZ, c, sigma_t, layout, priors, loss, search_radius, grid_res, init, n_starts, trace=None):
    M = len(t)
    w = 1.0 / sigma_t**2
    z0, z_sigma = priors["z0"], priors["z_sigma"]
    zmin, zmax = priors["z_bounds"]
    span = float(max(np.ptp(XYZ[:, 0]), np.ptp(XYZ[:, 1]), 1.0))
    R = float(search_radius) if search_radius is not None else max(200.0, 3.0 * span)
    bounds = (XYZ[:, 0].min() - R, XYZ[:, 0].max() + R, XYZ[:, 1].min() - R, XYZ[:, 1].max() + R)
    res = float(grid_res) if grid_res is not None else max(0.25, (2 * R + span) / 400.0)
    # height levels for the grid: the prior mean plus a coarse ladder over its +-3 sigma range
    if layout.solve_z:
        lo, hi = max(zmin, z0 - 3 * z_sigma), min(zmax, z0 + 3 * z_sigma)
        levels = np.unique(np.concatenate([[np.clip(z0, zmin, zmax)], np.linspace(lo, hi, 7)]))
    else:
        levels = np.array([z0])
    starts = []
    for zl in levels:
        for xy, cst in grid_search_init(t, XYZ, c, w, zl, bounds, res, n_best=n_starts, min_sep=3 * res):
            starts.append((xy, zl, cst))
    starts.sort(key=lambda q: q[2])
    starts = starts[: max(n_starts, 2 * n_starts if layout.solve_z else n_starts)]
    if init is not None:
        init = np.asarray(init, dtype=float)
        starts.insert(0, (init[:2], init[2] if len(init) > 2 else z0, float("inf")))
    # LM may leave the grid but not the search area (plus a margin); a fit that wants to run to
    # infinity (plane-wave degeneracy, inconsistent arrivals) ends up flagged at the boundary.
    box = (bounds[0] - R, bounds[1] + R, bounds[2] - R, bounds[3] + R, zmin, zmax)
    sols = []
    for k, (xy, zl, _) in enumerate(starts):
        theta0 = layout.pack(xy, zl, profile_t0(np.array([xy[0], xy[1], zl]), t, XYZ, c, w), None, priors["height_mu"])
        tr = trace if (trace is not None and k == 0) else None
        sols.append(_lm_refine(theta0, t, XYZ, c, sigma_t, layout, priors, loss, box=box, trace=tr))
    sols.sort(key=lambda z: z[4])
    return sols, box


def _subset_search(t, XYZ, c, sigma_t, z0, loss, huber, search_radius, m0=None, max_subsets=300, n_keep=6, seed=0):
    """Fit the core model (x, y, t0 at height z0) to every m0-recording subset and score each
    solution by the mixture likelihood of ALL residuals. Returns up to n_keep hypotheses
    [(theta_core [x, y, t0], cost), ...] sorted by cost and distinct in position, so that
    competing explanations (which recordings are direct) surface as alternatives."""
    from itertools import combinations

    M = len(t)
    if M < 4:
        return []
    sizes = [m0] if m0 is not None else ([4, 3] if M <= 8 else [4])  # 3-subsets: exact fits, catch a 3-recording direct set
    combos = [c_ for m in sizes if M > m for c_ in combinations(range(M), m)]
    if len(combos) > max_subsets:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_subsets, replace=False)]
    span = float(max(np.ptp(XYZ[:, 0]), np.ptp(XYZ[:, 1]), 1.0))
    R = float(search_radius) if search_radius is not None else max(200.0, 3.0 * span)
    coarse = max(0.5, (2 * R + span) / 120.0)
    hyps = []
    for sub in combos:
        sub = np.asarray(sub)
        m = len(sub)
        lay = _Layout(m, False, False, np.zeros(m))
        pri = {"z0": z0, "z_sigma": 1.0, "z_bounds": (z0 - 1.0, z0 + 1.0), "clock_sigma": 0.0,
               "height_mu": XYZ[sub, 2], "height_sigma": np.zeros(m)}
        sols, _ = _fit_all_starts(t[sub], XYZ[sub], c, sigma_t[sub], lay, pri, huber, R, coarse, None, 4)
        for th, *_ in sols[:2]:  # best two starts per subset (mirror geometries)
            pred, _ = predict_arrivals(np.array([th[0], th[1], z0]), XYZ, c, z0, th[2])
            hyps.append((th.copy(), loss.cost(t - pred, sigma_t)))
    hyps.sort(key=lambda h: h[1])
    keep = []
    for th, cst in hyps:
        if all(np.linalg.norm(th[:2] - k[0][:2]) > max(1.0, 0.02 * span) for k in keep):
            keep.append((th, cst))
        if len(keep) >= n_keep:
            break
    return keep


def _mahalanobis(p, q, C):
    d = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    try:
        return float(math.sqrt(max(d @ np.linalg.solve(C, d), 0.0)))
    except np.linalg.LinAlgError:
        return float(math.sqrt(max(d @ np.linalg.pinv(C) @ d, 0.0)))


def solve_tdoa(
    t: Sequence[float],
    XYZ: np.ndarray,
    c: float,
    sigma_t: Optional[Sequence[float]] = None,
    source_z: float = 0.0,
    source_z_sigma: float = 0.0,
    source_z_bounds: Tuple[float, float] = (0.0, 5000.0),
    height_sigma: Optional[Sequence[float]] = None,
    clock_sigma: float = 0.0,
    huber_k: float = 0.002,
    reject_k: Optional[float] = None,
    occlusion: bool = True,
    occlusion_prob: float = 0.2,
    occlusion_scale_m: float = 5.0,
    blunder_prob: float = 0.05,
    search_radius: Optional[float] = None,
    grid_res: Optional[float] = None,
    init: Optional[Sequence[float]] = None,
    n_starts: int = 8,
    trace: Optional[list] = None,
) -> TDOASolution:
    """Robust weighted TDOA multilateration.

    t               arrival times (s) on each recording's own clock
    XYZ             (M, 3) recording positions in meters (z = height, or its prior mean)
    sigma_t         per-recording timing standard deviation (s); default 0.5 ms
    source_z        event height (m): fixed when source_z_sigma == 0, else the prior mean
    source_z_sigma  prior std of the event height (m); > 0 solves z as a parameter
    source_z_bounds (min, max) height allowed when solving z (rules out the mirror image
                    below a horizontal camera plane)
    height_sigma    per-recording prior std of the height (m); 0 = known. Recordings with
                    height_sigma > 0 get their height estimated jointly with a prior at XYZ[i, 2]
    clock_sigma     prior std of per-recording clock offsets (s); 0 = synchronized (offsets fixed)
    huber_k         residual magnitude (s) beyond which observations are down-weighted (and,
                    with occlusion on, below which negative residuals stop being Gaussian)
    reject_k        residual magnitude (s) beyond which an observation is dropped and the fit
                    repeated, as long as at least 4 recordings remain (default 3 * huber_k;
                    0 disables). With occlusion on only early (negative) residuals are dropped,
                    late ones are explained as detours instead.
    occlusion       model late arrivals as detours: mixture loss in which an arrival is direct,
                    delayed by an exponential extra path of mean occlusion_scale_m (probability
                    occlusion_prob), or a mis-pick of either sign (probability blunder_prob);
                    plus a search over 4-recording subsets (3 with only 4 recordings) for the
                    set of direct-path recordings, whose runner-up explanations are reported as
                    alternatives. Off = symmetric Huber plus rejection and leave-one-out.
    search_radius   grid search extends this far beyond the array bounding box (default
                    max(200 m, 3 x array extent)); grid_res defaults to ~400 steps across it
    trace           optional list receiving the parameter vector after each accepted
                    Levenberg-Marquardt step of the best start (for visualization)
    Alternative minima outside the 95% region of the best solution are reported; the solution
    is flagged ambiguous when one of them fits within delta_cost <= 3.
    """
    t = np.asarray(t, dtype=float)
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise LocatorError("XYZ must be (M, 3)")
    M = len(t)
    if M != XYZ.shape[0]:
        raise LocatorError("number of arrival times must match number of positions")
    if M < 3:
        raise LocatorError(f"need at least 3 recordings with a usable arrival, got {M}")
    if not np.all(np.isfinite(t)):
        raise LocatorError("arrival times contain NaN/inf")
    sigma_t = np.full(M, 0.5e-3) if sigma_t is None else np.asarray(sigma_t, dtype=float)
    if sigma_t.shape != (M,) or np.any(sigma_t <= 0):
        raise LocatorError("sigma_t must be positive and have one entry per recording")
    height_sigma = np.zeros(M) if height_sigma is None else np.asarray(height_sigma, dtype=float)
    if height_sigma.shape != (M,) or np.any(height_sigma < 0):
        raise LocatorError("height_sigma must be non-negative and have one entry per recording")
    if source_z_sigma < 0:
        raise LocatorError("source_z_sigma must be >= 0")
    if not (0.0 < occlusion_prob < 1.0) or occlusion_scale_m <= 0 or not (0.0 <= blunder_prob < 1.0):
        raise LocatorError("occlusion_prob must be in (0, 1), occlusion_scale_m > 0, blunder_prob in [0, 1)")
    zmin, zmax = float(source_z_bounds[0]), float(source_z_bounds[1])
    if not zmin < zmax:
        raise LocatorError("source_z_bounds must be (min, max) with min < max")
    solve_z = source_z_sigma > 0
    if solve_z and not (zmin <= source_z <= zmax):
        raise LocatorError("source_z (prior mean) must lie within source_z_bounds")
    estimate_offsets = clock_sigma > 0
    reject_k = 3.0 * huber_k if reject_k is None else float(reject_k)
    huber = _HuberLoss(huber_k)
    loss = _OcclusionLoss(occlusion_prob, occlusion_scale_m / c, blunder_prob) if occlusion else huber
    priors_full = {
        "z0": float(source_z), "z_sigma": float(source_z_sigma) if solve_z else 1.0, "z_bounds": (zmin, zmax),
        "clock_sigma": float(clock_sigma), "height_mu": XYZ[:, 2].copy(), "height_sigma": height_sigma,
    }

    def fit(idx, tr=None):
        lay = _Layout(len(idx), solve_z, estimate_offsets, height_sigma[idx])
        pri = dict(priors_full, height_mu=XYZ[idx, 2], height_sigma=height_sigma[idx])
        sols, box = _fit_all_starts(t[idx], XYZ[idx], c, sigma_t[idx], lay, pri, huber,
                                    search_radius, grid_res, init, n_starts, trace=tr)
        if occlusion:
            sig = sigma_t[idx]
            # candidate starts: every Huber-stage minimum plus the subset-search hypotheses;
            # all are refined under the mixture loss so their costs are comparable
            cands = [s[0] for s in sols]
            for th_core, _ in _subset_search(t[idx], XYZ[idx], c, sig, priors_full["z0"], loss, huber, search_radius):
                cands.append(lay.pack(th_core[:2], priors_full["z0"], th_core[2], None, pri["height_mu"]))
            refined = [_lm_refine(th0, t[idx], XYZ[idx], c, sig, lay, pri, loss, box=box) for th0 in cands]
            refined.sort(key=lambda z: z[4])
            sols = refined
        return sols, box, lay, pri

    active = np.ones(M, dtype=bool)
    rejected: List[int] = []
    sigma_assumed = sigma_t.copy()
    polished = None
    while True:
        idx = np.flatnonzero(active)
        if trace is not None:
            trace.clear()
        sols, box, layout, priors = fit(idx, trace)
        theta, raw, r, J, cost, converged, iters = sols[0]
        if occlusion:
            # Final polish: the mixture decides which recordings are direct; the position itself is
            # then re-fitted on that set alone, so the exponential detour prior exerts no residual
            # pull on the estimate (the pull is ~1/tau per late recording and biases by centimetres).
            gd, go, gb = loss.responsibilities(raw, sigma_t[idx])
            direct = gd > 0.5
            if direct.sum() >= 3 and (~direct).any():
                sub = idx[direct]
                lay_d = _Layout(len(sub), solve_z, estimate_offsets, height_sigma[sub])
                pri_d = dict(priors_full, height_mu=XYZ[sub, 2], height_sigma=height_sigma[sub])
                # start from the mixture solution restricted to the direct set
                s0, t00, d0, XYZc0 = layout.unpack(theta, XYZ[idx], priors_full["z0"])
                th0 = lay_d.pack(s0[:2], s0[2], t00, d0[direct] if estimate_offsets else None, XYZc0[direct, 2])
                th_d, raw_d, r_d, J_d, cost_d, conv_d, it_d = _lm_refine(th0, t[sub], XYZ[sub], c, sigma_t[sub], lay_d, pri_d, huber, box=box)
                polished = (th_d, raw_d, r_d, J_d, cost_d, conv_d, it_d, sub, lay_d)
            else:
                polished = None
            break  # with the mixture loss, blunders get ~zero weight instead of being dropped
        if reject_k <= 0 or len(idx) - 1 < 4:
            break
        worst = int(np.argmax(np.abs(raw)))
        if abs(raw[worst]) > reject_k:
            active[idx[worst]] = False
            rejected.append(int(idx[worst]))
            continue
        # A single bad arrival can be absorbed by moving the source so that every residual stays
        # small (leverage). When the fit is clearly worse than the timing noise allows, try
        # leave-one-out: if dropping one recording removes most of the misfit and that recording
        # then disagrees with the rest by more than 1.5 * huber_k, drop it.
        dof_full = len(r) - len(theta)
        chi2_full = float(np.sum(huber.weights(raw, sigma_t[idx]) * r[: len(idx)] ** 2))
        if dof_full <= 0 or chi2_full / dof_full < 2.0:
            break
        best_i, best_cost, best_pred = None, cost, None
        for k, i in enumerate(idx):
            sub = np.delete(idx, k)
            s_sols, _, s_lay, _ = fit(sub)
            th_s, _, _, _, cost_s, _, _ = s_sols[0]
            if cost_s < best_cost:
                s_s, t0_s, d_s, XYZc_s = s_lay.unpack(th_s, XYZ[sub], priors_full["z0"])
                pred_i, _ = predict_arrivals(s_s, XYZ[i : i + 1], c, s_s[2], t0_s, None)
                best_i, best_cost, best_pred = i, cost_s, float(pred_i[0])
        if best_i is not None and best_cost < 0.25 * cost and abs(t[best_i] - best_pred) > 1.5 * huber_k:
            active[best_i] = False
            rejected.append(int(best_i))
            continue
        break
    classify_raw, classify_sigma = raw.copy(), sigma_t[idx].copy()
    if occlusion and polished is not None:
        theta, raw, r, J, cost, converged, iters, idx, layout = polished
        priors = dict(priors_full, height_mu=XYZ[idx, 2], height_sigma=height_sigma[idx])
    Ma = len(idx)

    hw = huber.weights(raw, sigma_t[idx]) if (occlusion and polished is not None) else loss.weights(raw, sigma_t[idx])
    wfull = np.concatenate([hw, np.ones(len(r) - Ma)])
    F = (J * wfull[:, None]).T @ J
    cond = float(np.linalg.cond(F)) if np.all(np.isfinite(F)) else float("inf")
    degenerate = not np.isfinite(cond) or cond > 1e12
    cov = np.linalg.pinv(F) if degenerate else np.linalg.inv(F)
    dof = int(len(r) - len(theta))
    dof_eff = float(np.sum(hw)) + (len(r) - Ma) - len(theta)  # occluded recordings carry ~no weight
    chi2 = float(np.sum(hw * r[:Ma] ** 2))
    inflation = max(1.0, chi2 / dof_eff) if dof_eff > 0.5 else 1.0
    cov = cov * inflation
    # total uncertainty scale relative to what the caller assumed (sigma widening x chi-square)
    scale = inflation * float(np.median(sigma_t / sigma_assumed)) ** 2
    n_pos = layout.n_pos
    cov_pos = cov[:n_pos, :n_pos].copy()
    cov_xy = cov[:2, :2].copy()

    s_xyz, t0, delta_sub, XYZc_sub = layout.unpack(theta, XYZ[idx], priors_full["z0"])
    s_xy = s_xyz[:2].copy()
    delta_full = np.zeros(M)
    if estimate_offsets:
        delta_full[idx] = delta_sub
    heights_full = XYZ[:, 2].copy()
    heights_full[idx] = XYZc_sub[:, 2]
    height_std_full = np.zeros(M)
    if layout.i_h is not None:
        height_std_full[idx[layout.hidx]] = np.sqrt(np.maximum(np.diag(cov)[layout.i_h : layout.i_h + len(layout.hidx)], 0.0))
    XYZ_full = XYZ.copy()
    XYZ_full[:, 2] = heights_full
    pred_full, _ = predict_arrivals(s_xyz, XYZ_full, c, s_xyz[2], t0, delta_full)
    raw_full = t - pred_full
    w_full = np.zeros(M)
    w_full[idx] = hw
    gamma_full = np.zeros(M)
    if occlusion:
        # classify every active recording at the final solution
        act = np.flatnonzero(active)
        _, go_all, gb_all = loss.responsibilities(raw_full[act], sigma_t[act])
        gamma_full[act] = go_all
        occluded = [int(i) for i in act[go_all > 0.5]]
        rejected = [int(i) for i in act[gb_all > 0.5]]  # mis-picks: ~zero weight either side
    else:
        occluded = []
    detour_full = np.where(gamma_full > 0.5, np.maximum(raw_full, 0.0) * c, 0.0)
    z_std = float(math.sqrt(max(cov[layout.i_z, layout.i_z], 0.0))) if solve_z else 0.0

    def cost_at(th):
        s_, t0_, d_, XYZc_ = layout.unpack(th, XYZ[idx], priors_full["z0"])
        pred_, _ = predict_arrivals(s_, XYZc_, c, s_[2], t0_, d_)
        return loss.cost(t[idx] - pred_, sigma_t[idx])

    alternatives = []
    thr = math.sqrt(7.815 if solve_z else 5.991)
    pos_of = lambda th: th[:n_pos]  # noqa: E731
    best_pos = pos_of(theta)
    for th, _, _, _, _, _, _ in sols[1:]:
        p = pos_of(th)
        if _mahalanobis(p, best_pos, cov_pos) <= thr:
            continue
        if any(_mahalanobis(p, [a["x"], a["y"]] + ([a["z"]] if solve_z else []), cov_pos) <= thr for a in alternatives):
            continue
        dc = float((cost_at(th) - cost) / scale)
        if dc > 50.0:
            continue
        alt = {"x": float(th[0]), "y": float(th[1]), "z": float(th[layout.i_z]) if solve_z else float(source_z),
               "cost": float(cost_at(th)), "delta_cost": dc, "distance_m": float(np.linalg.norm(p - best_pos))}
        alternatives.append(alt)
    ambiguous = any(a["delta_cost"] <= 3.0 for a in alternatives)
    margin = 0.02 * max(box[1] - box[0], box[3] - box[2])
    at_boundary = bool(
        s_xy[0] - box[0] < margin or box[1] - s_xy[0] < margin or s_xy[1] - box[2] < margin or box[3] - s_xy[1] < margin
    )
    z_at_bound = bool(solve_z and (abs(s_xyz[2] - zmin) < 1e-6 or abs(s_xyz[2] - zmax) < 1e-6) and abs(source_z - s_xyz[2]) > 1e-6)

    return TDOASolution(
        s_xy=s_xy, t0=float(t0), delta=delta_full, cov=cov, cov_xy=cov_xy, residuals_s=raw_full,
        weights=w_full, sigma_t=sigma_assumed, chi2=chi2, dof=dof, scale=float(scale), cost=float(cost),
        converged=bool(converged), iterations=int(iters), alternatives=alternatives,
        condition_number=cond, degenerate=bool(degenerate), estimate_offsets=bool(estimate_offsets),
        ambiguous=bool(ambiguous), rejected=rejected, at_boundary=at_boundary,
        s_xyz=s_xyz.copy(), z_std=z_std, solve_z=bool(solve_z), z_at_bound=z_at_bound, cov_pos=cov_pos,
        mic_heights=heights_full, mic_height_std=height_std_full,
        occluded=occluded, occlusion_prob=gamma_full, detour_m=detour_full, loss=loss.name,
        sigma_eff=sigma_t.copy(), n_direct=int(np.sum(w_full >= 0.5)),
    )


def ellipse_from_cov2(cov, conf=0.95):
    """2x2 covariance -> (semi-major m, semi-minor m, angle of major axis from +x, degrees)."""
    from scipy.stats import chi2 as _chi2

    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    scale = math.sqrt(_chi2.ppf(conf, df=2))
    a = scale * math.sqrt(max(vals[0], 0.0))
    b = scale * math.sqrt(max(vals[1], 0.0))
    return a, b, math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))


def mahalanobis_xy(sol: TDOASolution, true_xy) -> float:
    d = np.asarray(true_xy, dtype=float) - sol.s_xy
    return float(math.sqrt(d @ np.linalg.solve(sol.cov_xy, d)))


def mahalanobis_pos(sol: TDOASolution, true_pos) -> float:
    """Mahalanobis distance in the solved position space (2D, or 3D when z was solved)."""
    n = sol.cov_pos.shape[0]
    d = np.asarray(true_pos, dtype=float)[:n] - sol.s_xyz[:n]
    return float(math.sqrt(d @ np.linalg.solve(sol.cov_pos, d)))


# ------------------------------ Pipeline ------------------------------


@dataclass
class PipelineParams:
    band: Tuple[float, float] = (200.0, 4000.0)
    env_ms: float = 2.0
    min_snr: float = 4.0
    merge_gap_s: float = 0.5
    slack_s: float = 0.005
    clock_sigma_s: float = 0.0
    source_z: float = 0.0
    source_z_sigma: float = 0.0
    source_z_bounds: Tuple[float, float] = (0.0, 5000.0)
    huber_k_s: float = 0.002
    occlusion: bool = True
    occlusion_prob: float = 0.2
    occlusion_scale_m: float = 5.0
    blunder_prob: float = 0.05
    max_detour_m: float = 30.0
    timing_sigma_s: float = 0.5e-3
    gcc_weighting: str = "phat"
    refine: bool = True
    search_radius_m: Optional[float] = None
    grid_res_m: Optional[float] = None


@dataclass
class TrackResult:
    index: int
    used: bool
    coarse_idx: Optional[int] = None
    pick_idx: Optional[int] = None
    arrival_s: Optional[float] = None
    snr: float = 0.0
    sigma_t: Optional[float] = None
    residual_s: Optional[float] = None
    weight: Optional[float] = None
    clock_offset_s: Optional[float] = None
    height_m: Optional[float] = None
    height_std_m: Optional[float] = None
    occlusion_prob: Optional[float] = None
    detour_m: Optional[float] = None
    note: str = ""


def timing_sigma_from_snr(snr: Sequence[float], base_sigma_s: float) -> np.ndarray:
    """Timing noise grows for weaker onsets: sigma_i = base * clip(sqrt(snr_max / snr_i), 1, 10)."""
    snr = np.maximum(np.asarray(snr, dtype=float), 1e-6)
    return base_sigma_s * np.clip(np.sqrt(snr.max() / snr), 1.0, 10.0)


def locate_from_signals(
    signals: Sequence[np.ndarray], fs: int, XYZ: np.ndarray, c: float, p: PipelineParams,
    height_sigma: Optional[Sequence[float]] = None,
) -> dict:
    """Full pipeline on in-memory mono signals. height_sigma (M,) gives the prior std of each
    recording's height (0 = known). Returns dict with 'solution' (TDOASolution), 'tracks'
    (List[TrackResult]), 'refinement', 'warnings', 'used'."""
    M = len(signals)
    XYZ = np.asarray(XYZ, dtype=float)
    height_sigma = np.zeros(M) if height_sigma is None else np.asarray(height_sigma, dtype=float)
    warnings_out: List[str] = []
    tracks = [TrackResult(index=i, used=False) for i in range(M)]

    xf = [apply_bandpass(x, fs, p.band[0], p.band[1]) for x in signals]
    envs = [rms_envelope(x, fs, win_ms=p.env_ms) for x in xf]
    floors = [noise_floor(e) for e in envs]
    # Detect onsets; if fewer than 3 recordings trigger, relax the threshold stepwise (never
    # below 3x, above the ~2x that white noise alone reaches). Anything found this way must
    # also pass the fit-consistency gate below.
    min_ratio = p.min_snr
    while True:
        cands = [
            find_onset_candidates(envs[i], fs, floors[i], min_ratio=min_ratio, merge_gap_s=p.merge_gap_s)
            for i in range(M)
        ]
        n_hit = sum(1 for ci in cands if ci)
        if n_hit >= 3 or min_ratio <= 3.0:
            break
        min_ratio = max(3.0, min_ratio * 0.8)
    if min_ratio < p.min_snr:
        warnings_out.append(
            f"only {sum(1 for ci in cands if ci)} recording(s) reached {p.min_snr:g}x the noise floor; "
            f"detection threshold relaxed to {min_ratio:.1f}x"
        )
    for i in range(M):
        ci = cands[i]
        if not ci:
            tracks[i].note = f"no onset exceeds {min_ratio:g}x the noise floor"
        log(f"track {i}: {len(ci)} onset candidate(s)" + (f", strongest {ci[0][1]:.1f}x at {ci[0][0]/fs:.3f}s" if ci else ""), "DEBUG")

    max_lag = max_pairwise_lag(XYZ, c, p.slack_s, p.clock_sigma_s, (p.max_detour_m / c) if p.occlusion else 0.0)
    chosen, missing = associate_onsets(cands, fs, max_lag)
    for i in missing:
        if cands[i]:
            tracks[i].note = "no onset consistent with the other recordings"
            tracks[i].coarse_idx = cands[i][0][0]
            tracks[i].snr = cands[i][0][1]
            warnings_out.append(f"recording {i}: strongest onset at {cands[i][0][0]/fs:.3f}s is not consistent with the others; excluded")
    used = sorted(chosen)
    if len(used) < 3:
        raise LocatorError(
            f"only {len(used)} recording(s) have a mutually consistent onset; need at least 3. "
            "Check that the event is audible in each recording and that positions/clocks are right."
        )

    for i in used:
        tr = tracks[i]
        tr.used = True
        tr.coarse_idx = chosen[i]
        tr.snr = dict(cands[i])[chosen[i]]
        tr.pick_idx = fine_pick(xf[i], fs, chosen[i])

    picks = [tracks[i].pick_idx for i in used]
    refinement = None
    if p.refine and len(used) >= 2:
        refinement = refine_arrivals_pairwise(
            picks, [xf[i] for i in used], fs, weighting=p.gcc_weighting, band=p.band
        )
        arrivals = refinement["arrival_s"]
        if not refinement["used_pairs"].all():
            n_bad = int((~refinement["used_pairs"]).sum())
            n_pairs = len(refinement["used_pairs"])
            if n_bad * 2 >= n_pairs:
                warnings_out.append(
                    f"{n_bad} of {n_pairs} cross-correlation pairs had no clear peak: the picked onsets may not be "
                    "the same sound in every recording; treat the result with caution"
                )
            else:
                warnings_out.append(f"{n_bad} cross-correlation pair(s) had no clear peak and were ignored in refinement")
    else:
        arrivals = np.asarray(picks, dtype=float) / fs

    snrs = [tracks[i].snr for i in used]
    sigma_t = timing_sigma_from_snr(snrs, p.timing_sigma_s)
    sol = solve_tdoa(
        arrivals, XYZ[used], c, sigma_t=sigma_t, source_z=p.source_z, source_z_sigma=p.source_z_sigma,
        source_z_bounds=p.source_z_bounds, height_sigma=height_sigma[used], clock_sigma=p.clock_sigma_s,
        huber_k=p.huber_k_s, occlusion=p.occlusion, occlusion_prob=p.occlusion_prob,
        occlusion_scale_m=p.occlusion_scale_m, blunder_prob=p.blunder_prob,
        search_radius=p.search_radius_m, grid_res=p.grid_res_m,
    )
    relaxed = min_ratio < p.min_snr
    if relaxed and sol.dof > 0 and sol.chi2 / sol.dof > 25.0:
        raise LocatorError(
            f"onsets were only found at a relaxed detection threshold ({min_ratio:.1f}x) and they are "
            f"not mutually consistent (reduced chi2 {sol.chi2 / sol.dof:.0f}). The event is probably too "
            "faint in some recordings; try a narrower --bandpass, --min_snr 3, or drop weak recordings."
        )
    for k, i in enumerate(used):
        tr = tracks[i]
        tr.arrival_s = float(arrivals[k])
        tr.sigma_t = float(sigma_t[k])
        tr.residual_s = float(sol.residuals_s[k])
        tr.weight = float(sol.weights[k])
        tr.clock_offset_s = float(sol.delta[k])
        tr.height_m = float(sol.mic_heights[k])
        tr.height_std_m = float(sol.mic_height_std[k])
        tr.occlusion_prob = float(sol.occlusion_prob[k])
        tr.detour_m = float(sol.detour_m[k])
        if k in sol.occluded:
            tr.note = (f"arrived {sol.residuals_s[k]*1000:.1f} ms late: treated as occluded "
                       f"(about {sol.detour_m[k]:.1f} m of extra path)")
        elif k in sol.rejected:
            tr.note = f"treated as a mis-pick (residual {sol.residuals_s[k]*1000:.2f} ms, weight {sol.weights[k]:.2f})"
        elif sol.weights[k] < 0.5:
            tr.note = f"down-weighted outlier (residual {sol.residuals_s[k]*1000:.2f} ms)"
    for k in sol.rejected:
        warnings_out.append(f"recording {used[k]}: arrival treated as a mis-pick (residual {sol.residuals_s[k]*1000:.2f} ms)")
    if sol.occluded:
        det = ", ".join(f"{used[k]}: {sol.detour_m[k]:.1f} m" for k in sol.occluded)
        warnings_out.append(f"{len(sol.occluded)} recording(s) arrived late and were treated as occluded (detour {det}); "
                            "the position rests on the direct-path recordings")
    n_par = 3 + (1 if sol.solve_z else 0)
    if (sol.occluded or sol.rejected) and sol.n_direct - n_par < 2:
        warnings_out.append(
            f"only {sol.n_direct} recordings carry the solution after {len(sol.occluded)} occluded and "
            f"{len(sol.rejected)} mis-pick classification(s): little redundancy, so which recordings are direct may "
            "be ambiguous (check alternatives) and the ellipse assumes --timing_sigma_ms is realistic"
        )
    if sol.dof <= 0:
        warnings_out.append(
            "no redundancy (as many unknowns as observations): the fit cannot be cross-checked"
            + (f"; only {len(used)} of {M} recordings were usable" if M > len(used) else "")
        )
    if sol.degenerate:
        warnings_out.append("degenerate geometry: position is not determined in at least one direction")
    if sol.at_boundary:
        warnings_out.append("solution sits at the edge of the search area: the arrivals do not pin down a location "
                            "(increase --search_radius_m only if the event really was that far away)")
    if sol.solve_z and sol.z_at_bound:
        warnings_out.append(f"event height ran into its bound ({sol.s_xyz[2]:.1f} m); widen --source_height_bounds "
                            "if that height is physically possible")
    if sol.solve_z and sol.z_std > 10.0:
        warnings_out.append(f"event height is weakly determined (std {sol.z_std:.1f} m): the cameras have little "
                            "vertical aperture relative to this source")
    if sol.ambiguous:
        alt = min(sol.alternatives, key=lambda a: a["delta_cost"])
        warnings_out.append(f"ambiguous geometry: an alternative solution {alt['distance_m']:.1f} m away fits nearly as well")
    if sol.scale > 4.0:
        warnings_out.append(f"residuals are {math.sqrt(sol.scale):.1f}x larger than the assumed timing noise; uncertainty inflated accordingly")
    for w in warnings_out:
        log(w, "WARN")
    return {"solution": sol, "tracks": tracks, "refinement": refinement, "warnings": warnings_out, "used": used,
            "height_sigma": height_sigma}


# ------------------------------ Plot & I/O ------------------------------


def plot_layout(XY, s_xy, cov, out_png, labels=None, alternatives=None, unused=None, elevation=None):
    """Plan view of recordings, estimate and 95% ellipse. If `elevation` is given (dict with
    mic_z, mic_z_std, source_z, z_std, z_prior=(mean, sigma) or None) a side view is added."""
    a, b, ang_deg = ellipse_from_cov2(cov)
    XY = np.asarray(XY, dtype=float)
    unused = set(unused or [])
    used_mask = np.array([i not in unused for i in range(len(XY))])
    if elevation is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), gridspec_kw={"width_ratios": [1.15, 1]})
        ax = axes[0]
    ax.scatter(XY[used_mask, 0], XY[used_mask, 1], marker="^", s=60, label="Recordings")
    if (~used_mask).any():
        ax.scatter(XY[~used_mask, 0], XY[~used_mask, 1], marker="x", s=60, c="gray", label="Not used")
    if labels:
        for (x, y), lb in zip(XY, labels):
            ax.annotate(lb, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.scatter([s_xy[0]], [s_xy[1]], marker="*", s=140, label="Estimated event")
    theta = np.linspace(0, 2 * np.pi, 200)
    ca, sa = np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))
    R = np.array([[ca, -sa], [sa, ca]])
    ell = (R @ np.vstack([a * np.cos(theta), b * np.sin(theta)])).T
    ax.plot(s_xy[0] + ell[:, 0], s_xy[1] + ell[:, 1], label="95% ellipse")
    if alternatives:
        ax.scatter([q["x"] for q in alternatives], [q["y"] for q in alternatives], marker="o",
                   facecolors="none", edgecolors="tab:red", s=80, label="Alternative minima")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.set_title("Plan view")
    if elevation is not None:
        ax2 = axes[1]
        mic_z = np.asarray(elevation["mic_z"], dtype=float)
        mic_z_std = np.asarray(elevation.get("mic_z_std", np.zeros(len(mic_z))), dtype=float)
        ax2.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax2.errorbar(XY[used_mask, 0], mic_z[used_mask], yerr=2 * mic_z_std[used_mask], fmt="^", ms=7,
                     capsize=3, label="Recordings (height, 95%)")
        if (~used_mask).any():
            ax2.scatter(XY[~used_mask, 0], mic_z[~used_mask], marker="x", c="gray", s=60)
        zp = elevation.get("z_prior")
        if zp is not None and zp[1] > 0:
            ax2.axhspan(zp[0] - 2 * zp[1], zp[0] + 2 * zp[1], color="tab:orange", alpha=0.12, label="Height prior (95%)")
        ax2.errorbar([s_xy[0]], [elevation["source_z"]], yerr=[[2 * elevation["z_std"]], [2 * elevation["z_std"]]],
                     fmt="*", ms=14, capsize=4, color="tab:orange", label="Estimated event (height, 95%)")
        if alternatives:
            ax2.scatter([q["x"] for q in alternatives], [q.get("z", elevation["source_z"]) for q in alternatives],
                        marker="o", facecolors="none", edgecolors="tab:red", s=80)
        if labels:
            for (x, _), z, lb in zip(XY, mic_z, labels):
                ax2.annotate(lb, (x, z), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_xlabel("x east (m)")
        ax2.set_ylabel("height (m)")
        ax2.set_title("Elevation view (looking north)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_sync_csv(out_csv: str, files: Sequence[str], tracks: Sequence[TrackResult]):
    """align_to_event_offset_s = -arrival_time_s (seek so the event sits at t=0)."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "used", "arrival_time_s", "clock_offset_s", "align_to_event_offset_s", "residual_ms"])
        for fn, tr in zip(files, tracks):
            if tr.arrival_s is None:
                w.writerow([os.path.basename(fn), int(tr.used), "", "", "", ""])
            else:
                w.writerow([
                    os.path.basename(fn), int(tr.used), f"{tr.arrival_s:.6f}",
                    f"{(tr.clock_offset_s or 0.0):.6f}", f"{-tr.arrival_s:.6f}",
                    f"{(tr.residual_s or 0.0)*1000:.3f}",
                ])


def build_results(res: dict, files, XYZ, lat0, lon0, c, p: PipelineParams, fs: int) -> dict:
    sol: TDOASolution = res["solution"]
    a95, b95, ang = ellipse_from_cov2(sol.cov_xy)
    lat, lon = local_xy_to_latlon(sol.s_xy[0], sol.s_xy[1], lat0, lon0)
    alts = []
    for q in sol.alternatives:
        qlat, qlon = local_xy_to_latlon(q["x"], q["y"], lat0, lon0)
        alts.append({**q, "lat": qlat, "lon": qlon})
    per = []
    for fn, tr in zip(files, res["tracks"]):
        per.append({
            "file": os.path.basename(fn),
            "used": tr.used,
            "arrival_time_s": tr.arrival_s,
            "clock_offset_s": tr.clock_offset_s,
            "align_to_event_offset_s": (-tr.arrival_s if tr.arrival_s is not None else None),
            "onset_snr": tr.snr,
            "timing_sigma_ms": (tr.sigma_t * 1000 if tr.sigma_t is not None else None),
            "residual_ms": (tr.residual_s * 1000 if tr.residual_s is not None else None),
            "weight": tr.weight,
            "height_m": tr.height_m,
            "height_std_m": tr.height_std_m,
            "occlusion_probability": tr.occlusion_prob,
            "detour_m": tr.detour_m,
            "note": tr.note,
        })
    refinement = None
    if res["refinement"] is not None:
        rf = res["refinement"]
        used = res["used"]
        refinement = {
            "reference": os.path.basename(files[used[rf["ref_idx"]]]),
            "pairs": [
                {"i": os.path.basename(files[used[i]]), "j": os.path.basename(files[used[j]]),
                 "tau_ms": float(tau * 1000), "quality": float(q), "used": bool(u)}
                for (i, j), tau, q, u in zip(rf["pairs"], rf["pair_tau_s"], rf["pair_quality"], rf["used_pairs"])
            ],
        }
    return {
        "event_location_local_m": {"x": float(sol.s_xy[0]), "y": float(sol.s_xy[1]), "z": float(sol.s_xyz[2])},
        "event_location_wgs84": {"lat": float(lat), "lon": float(lon), "alt_m": float(sol.s_xyz[2])},
        "local_frame": {"origin_lat": lat0, "origin_lon": lon0, "x": "east (m)", "y": "north (m)"},
        "confidence_ellipse_95": {"semi_major_m": a95, "semi_minor_m": b95, "angle_deg": ang},
        "position_std_m": {"x": float(math.sqrt(max(sol.cov_xy[0, 0], 0))), "y": float(math.sqrt(max(sol.cov_xy[1, 1], 0))),
                           "z": float(sol.z_std)},
        "height_model": {
            "source": {"prior_mean_m": p.source_z, "prior_sigma_m": p.source_z_sigma, "solved": sol.solve_z,
                       "estimate_m": float(sol.s_xyz[2]), "std_m": float(sol.z_std), "bounds_m": list(p.source_z_bounds),
                       "at_bound": sol.z_at_bound},
            "recordings": [
                {"file": os.path.basename(fn), "prior_m": float(XYZ[i, 2]), "sigma_m": float(hs),
                 "estimate_m": tr.height_m, "std_m": tr.height_std_m}
                for i, (fn, tr, hs) in enumerate(zip(files, res["tracks"], res.get("height_sigma", np.zeros(len(files)))))
                if hs > 0
            ],
        },
        "speed_of_sound_mps": c,
        "emission_time_s": sol.t0,
        "clock_model": {"mode": "prior" if sol.estimate_offsets else "synchronized", "clock_sigma_ms": p.clock_sigma_s * 1000},
        "fit": {
            "recordings_used": len(res["used"]),
            "recordings_total": len(files),
            "chi2": sol.chi2,
            "dof": sol.dof,
            "reduced_chi2": (sol.chi2 / sol.dof if sol.dof > 0 else None),
            "uncertainty_scale": sol.scale,
            "rmse_ms": sol.rmse_s * 1000,
            "rejected": [os.path.basename(files[res["used"][k]]) for k in sol.rejected],
            "occluded": [os.path.basename(files[res["used"][k]]) for k in sol.occluded],
            "direct_path_recordings": sol.n_direct,
            "loss": sol.loss,
            "converged": sol.converged,
            "iterations": sol.iterations,
            "condition_number": sol.condition_number,
            "degenerate": sol.degenerate,
            "ambiguous": sol.ambiguous,
            "at_search_boundary": sol.at_boundary,
            "z_at_bound": sol.z_at_bound,
            "alternatives": alts,
        },
        "per_recording": per,
        "refinement": refinement,
        "warnings": res["warnings"],
        "parameters": {
            "fs": fs, "bandpass_hz": list(p.band), "env_ms": p.env_ms, "min_snr": p.min_snr, "merge_gap_s": p.merge_gap_s,
            "slack_ms": p.slack_s * 1000, "source_height_m": p.source_z, "source_height_sigma_m": p.source_z_sigma,
            "source_height_bounds_m": list(p.source_z_bounds), "huber_k_ms": p.huber_k_s * 1000,
            "occlusion": p.occlusion, "occlusion_prob": p.occlusion_prob, "occlusion_scale_m": p.occlusion_scale_m,
            "blunder_prob": p.blunder_prob, "max_detour_m": p.max_detour_m,
            "timing_sigma_ms": p.timing_sigma_s * 1000, "gcc_weighting": p.gcc_weighting, "refine": p.refine,
            "search_radius_m": p.search_radius_m, "grid_res_m": p.grid_res_m,
        },
    }


# ------------------------------ Main ------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Locate a single impulsive acoustic event from multiple recordings.")
    ap.add_argument("--videos_dir", required=True, help="Directory containing the recordings (video or WAV).")
    ap.add_argument("--positions", required=True, help="positions.json mapping file -> lat/lon.")
    ap.add_argument("--out", default="./out", help="Output directory.")
    ap.add_argument("--fs", type=int, default=48000, help="Working sample rate (Hz).")
    ap.add_argument("--bandpass", type=float, nargs=2, default=(200.0, 4000.0), metavar=("LOW", "HIGH"), help="Bandpass (Hz).")
    ap.add_argument("--env_ms", type=float, default=2.0, help="Moving-RMS window (ms) used to detect onsets.")
    ap.add_argument("--min_snr", type=float, default=4.0, help="Onset must exceed this multiple of the noise floor (moving RMS).")
    ap.add_argument("--merge_gap_s", type=float, default=0.5, help="Onsets closer than this to a previous burst are treated as its coda.")
    ap.add_argument("--slack_ms", type=float, default=5.0, help="Extra tolerance on the physical arrival-time gate.")
    ap.add_argument("--clock_sigma_ms", type=float, default=0.0, help="Prior std of per-recording clock offsets; 0 = synchronized clocks.")
    ap.add_argument("--source_height_m", type=float, default=0.0, help="Event height (m) in the same datum as height_m; fixed unless --source_height_sigma_m > 0, then the prior mean.")
    ap.add_argument("--source_height_sigma_m", type=float, default=0.0, help="Prior std of the event height (m); > 0 solves the height (3D).")
    ap.add_argument("--source_height_bounds", type=float, nargs=2, default=(0.0, 5000.0), metavar=("MIN", "MAX"), help="Allowed event height range when solving it.")
    ap.add_argument("--timing_sigma_ms", type=float, default=0.5, help="Assumed arrival-time noise for the strongest recording.")
    ap.add_argument("--huber_k_ms", type=float, default=2.0, help="Residuals beyond this are down-weighted (robustness).")
    ap.add_argument("--no_occlusion", action="store_true", help="Disable the occlusion model (symmetric robust loss instead).")
    ap.add_argument("--occlusion_prob", type=float, default=0.2, help="Prior probability that a recording's direct path is blocked.")
    ap.add_argument("--occlusion_scale_m", type=float, default=5.0, help="Typical extra path length (m) of an occluded arrival.")
    ap.add_argument("--blunder_prob", type=float, default=0.05, help="Prior probability that a pick is simply wrong (either sign).")
    ap.add_argument("--max_detour_m", type=float, default=30.0, help="Largest extra path an occluded arrival may have when associating onsets.")
    ap.add_argument("--search_radius_m", type=float, default=None, help="Search this far beyond the array (default max(200, 3x extent)).")
    ap.add_argument("--grid_res_m", type=float, default=None, help="Grid resolution for initialization (default auto).")
    ap.add_argument("--gcc_weight", choices=("phat", "cc", "scot"), default="phat", help="Cross-correlation weighting for refinement.")
    ap.add_argument("--no_refine", action="store_true", help="Skip cross-correlation refinement (AIC picks only).")
    ap.add_argument("--verbose", action="store_true", help="Debug logging.")
    return ap


def params_from_args(args) -> PipelineParams:
    return PipelineParams(
        band=(float(args.bandpass[0]), float(args.bandpass[1])),
        env_ms=args.env_ms,
        min_snr=args.min_snr,
        merge_gap_s=args.merge_gap_s,
        slack_s=args.slack_ms / 1000.0,
        clock_sigma_s=args.clock_sigma_ms / 1000.0,
        source_z=args.source_height_m,
        source_z_sigma=args.source_height_sigma_m,
        source_z_bounds=(float(args.source_height_bounds[0]), float(args.source_height_bounds[1])),
        huber_k_s=args.huber_k_ms / 1000.0,
        occlusion=not args.no_occlusion,
        occlusion_prob=args.occlusion_prob,
        occlusion_scale_m=args.occlusion_scale_m,
        blunder_prob=args.blunder_prob,
        max_detour_m=args.max_detour_m,
        timing_sigma_s=args.timing_sigma_ms / 1000.0,
        gcc_weighting=args.gcc_weight,
        refine=not args.no_refine,
        search_radius_m=args.search_radius_m,
        grid_res_m=args.grid_res_m,
    )


def run(args) -> dict:
    global VERBOSE
    VERBOSE = bool(args.verbose)
    p = params_from_args(args)
    ensure_dir(args.out)
    wav_dir = os.path.join(args.out, "wav")

    mics, (lat0, lon0), c, _ = load_positions(args.positions, args.videos_dir)
    if len(mics) < 3:
        raise LocatorError("need at least 3 recordings to localize in 2D (4 or more recommended)")
    XYZ = mic_local_xyz(mics, lat0, lon0)
    hsig = mic_height_sigma(mics)
    files = [m.file for m in mics]
    zmodel = (f"solved, prior {p.source_z:g} +- {p.source_z_sigma:g} m" if p.source_z_sigma > 0 else f"fixed at {p.source_z:g} m")
    log(f"{len(mics)} recordings, speed of sound {c:.2f} m/s, clock model "
        f"{'prior sigma=%.1f ms' % args.clock_sigma_ms if p.clock_sigma_s > 0 else 'synchronized'}, event height {zmodel}"
        + (f", {int((hsig > 0).sum())} recording height(s) uncertain" if (hsig > 0).any() else ""))

    signals = []
    for m in mics:
        log(f"Loading audio: {os.path.basename(m.file)}")
        signals.append(load_audio(m.file, args.fs, wav_dir))

    res = locate_from_signals(signals, args.fs, XYZ, c, p, height_sigma=hsig)
    sol: TDOASolution = res["solution"]
    results = build_results(res, files, XYZ, lat0, lon0, c, p, args.fs)
    write_json(os.path.join(args.out, "results.json"), results)
    write_sync_csv(os.path.join(args.out, "sync.csv"), files, res["tracks"])
    elevation = None
    if sol.solve_z or (hsig > 0).any():
        elevation = {"mic_z": sol.mic_heights, "mic_z_std": sol.mic_height_std, "source_z": float(sol.s_xyz[2]),
                     "z_std": sol.z_std, "z_prior": (p.source_z, p.source_z_sigma) if sol.solve_z else None}
    plot_layout(
        XYZ[:, :2], sol.s_xy, sol.cov_xy, os.path.join(args.out, "layout.png"),
        labels=[os.path.basename(f) for f in files], alternatives=sol.alternatives,
        unused=[i for i in range(len(files)) if i not in res["used"]], elevation=elevation,
    )
    for tr, fn in zip(res["tracks"], files):
        if tr.used:
            log(f"{os.path.basename(fn)}: arrival={tr.arrival_s:.6f}s snr={tr.snr:.1f} residual={tr.residual_s*1000:+.3f} ms weight={tr.weight:.2f}" + (f" ({tr.note})" if tr.note else ""))
        else:
            log(f"{os.path.basename(fn)}: not used ({tr.note})")
    e = results["confidence_ellipse_95"]
    log(f"Estimated location (local m): x={sol.s_xy[0]:.2f}, y={sol.s_xy[1]:.2f}"
        + (f", z={sol.s_xyz[2]:.2f} (std {sol.z_std:.2f})" if sol.solve_z else f", z fixed at {sol.s_xyz[2]:.2f}"))
    log(f"Estimated location (lat/lon): lat={results['event_location_wgs84']['lat']:.7f}, lon={results['event_location_wgs84']['lon']:.7f}")
    log(f"95% ellipse: a={e['semi_major_m']:.2f} m, b={e['semi_minor_m']:.2f} m, angle={e['angle_deg']:.1f} deg; "
        f"rmse={sol.rmse_s*1000:.3f} ms, reduced chi2={(sol.chi2/sol.dof if sol.dof>0 else float('nan')):.2f}")
    log(f"Results written to: {args.out}")
    return results


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run(args)
    except LocatorError as e:
        log(str(e), "ERROR")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
