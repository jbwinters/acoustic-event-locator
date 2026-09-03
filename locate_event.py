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
  results.json   location (local metres and WGS84), 95% ellipse, per-recording arrivals,
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

Clock synchronisation
  One event cannot determine per-device clock offsets: for every candidate position there
  is a set of offsets that fits the arrivals exactly. The position is therefore only as
  good as the clock synchronisation of the recordings. Two modes are offered:
    --clock_sigma_ms 0   (default) devices are assumed synchronised; offsets fixed at 0
    --clock_sigma_ms S   offsets are estimated under a Gaussian prior N(0, S^2); the
                         result is a MAP estimate and the position uncertainty grows with S
  Recordings whose clocks differ by more than the array's maximum propagation delay cannot
  be localised from a single event. sync.csv still gives the offset that aligns each
  recording on the event.

Pipeline
  1. Load or extract audio, resample to --fs, causal bandpass (no pre-ringing).
  2. Per recording, detect onset candidates from a moving-RMS envelope against a robust
     noise floor; choose one candidate per recording so that the set is physically
     consistent with the geometry (event association).
  3. Fine first-arrival pick with an AIC change-point picker around each candidate.
  4. Pairwise band-limited generalised cross-correlation between recordings for
     sub-sample relative timing, fused by weighted least squares.
  5. Vectorised grid search for starting points, Levenberg-Marquardt refinement with
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
from dataclasses import dataclass
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
    """User-facing error: bad inputs or data that cannot be localised."""


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
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ------------------------------ Geo conversion (local tangent plane) ------------------------------


def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """WGS-84 metres per degree of latitude and longitude at the given latitude."""
    phi = math.radians(lat_deg)
    m_lat = 111132.954 - 559.822 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = (
        111412.84 * math.cos(phi)
        - 93.5 * math.cos(3 * phi)
        + 0.118 * math.cos(5 * phi)
    )
    return m_lat, m_lon


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Equirectangular projection about (lat0, lon0); x east, y north, metres.
    Accurate to ~1e-4 relative over a few kilometres."""
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
            )
        )
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
    """Generalised cross-correlation of sig against ref.

    Returns (tau_s, quality, lags_s, cc). tau > 0 means sig is delayed relative to ref.
    weighting: 'cc' (plain), 'phat' (regularised phase transform), 'scot'.
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


def max_pairwise_lag(XYZ: np.ndarray, c: float, slack_s: float = 0.005, clock_sigma_s: float = 0.0):
    """Matrix of the largest physically possible |t_i - t_j| for each pair."""
    D = np.linalg.norm(XYZ[:, None, :] - XYZ[None, :, :], axis=-1)
    return D / c + slack_s + 3.0 * clock_sigma_s


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


def distances_3d(s_xy, XYZ, source_z=0.0):
    dx = XYZ[:, 0] - s_xy[0]
    dy = XYZ[:, 1] - s_xy[1]
    dz = XYZ[:, 2] - source_z
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def predict_arrivals(s_xy, XYZ, c, source_z=0.0, t0=0.0, delta=None):
    """t_i = t0 + ||s - x_i||/c + delta_i. Returns (pred, distances)."""
    d = distances_3d(s_xy, XYZ, source_z)
    pred = t0 + d / c
    if delta is not None:
        pred = pred + delta
    return pred, d


def profile_t0(s_xy, t, XYZ, c, w, source_z=0.0):
    """Weighted least-squares emission time for a fixed position (offsets = 0)."""
    _, d = predict_arrivals(s_xy, XYZ, c, source_z)
    return float(np.sum(w * (t - d / c)) / np.sum(w))


def grid_search_init(
    t, XYZ, c, w, source_z, bounds, res, n_best=6, min_sep=None, max_points=300_000
):
    """Vectorised grid search of the synchronised-clock cost with t0 profiled out.
    Returns up to n_best well separated low-cost grid points as [((x, y), cost), ...]."""
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

    @property
    def rmse_s(self) -> float:
        return float(np.sqrt(np.mean(self.residuals_s**2))) if len(self.residuals_s) else float("nan")


def _huber_irls_weights(raw: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(raw)
    w = np.ones_like(a)
    big = a > k
    w[big] = k / a[big]
    return w


def _huber_cost(r_data: np.ndarray, k_norm: np.ndarray, r_prior: np.ndarray) -> float:
    u = np.abs(r_data)
    rho = np.where(u <= k_norm, 0.5 * u * u, k_norm * u - 0.5 * k_norm * k_norm)
    return float(np.sum(rho) + 0.5 * np.sum(r_prior * r_prior))


def _lm_refine(theta0, t, XYZ, c, sigma_t, source_z, clock_sigma, huber_k, estimate_offsets, max_iter=200, box=None):
    """Levenberg-Marquardt with Huber IRLS weights on theta = [x, y, t0, (delta_0..delta_{M-1})]."""
    M = len(t)
    k_norm = huber_k / sigma_t

    def unpack(th):
        return th[:2], th[2], (th[3:] if estimate_offsets else np.zeros(M))

    def evaluate(th):
        s, t0, delta = unpack(th)
        pred, d = predict_arrivals(s, XYZ, c, source_z, t0, delta)
        raw = t - pred
        r = raw / sigma_t
        n_par = len(th)
        J = np.zeros((M, n_par))
        dsafe = np.maximum(d, 1e-9)
        J[:, 0] = (s[0] - XYZ[:, 0]) / (c * dsafe)
        J[:, 1] = (s[1] - XYZ[:, 1]) / (c * dsafe)
        J[:, 2] = 1.0
        if estimate_offsets:
            J[:, 3:] = np.eye(M)
        J = -J / sigma_t[:, None]
        if estimate_offsets:
            rp = -delta / clock_sigma
            Jp = np.zeros((M, n_par))
            Jp[:, 3:] = -np.eye(M) / clock_sigma
            return raw, np.concatenate([r, rp]), np.vstack([J, Jp])
        return raw, r, J

    theta = np.array(theta0, dtype=float)
    raw, r, J = evaluate(theta)
    cost = _huber_cost(r[:M], k_norm, r[M:])
    lam = 1e-3
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        hw = _huber_irls_weights(raw, huber_k)
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
            if box is not None and not (box[0] <= th_new[0] <= box[1] and box[2] <= th_new[1] <= box[3]):
                lam *= 10.0  # step leaves the search area: damp and retry
                if lam > 1e12:
                    break
                continue
            raw_new, r_new, J_new = evaluate(th_new)
            cost_new = _huber_cost(r_new[:M], k_norm, r_new[M:])
            if cost_new <= cost:
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not accepted:
            converged = True
            break
        rel = (cost - cost_new) / max(cost, 1e-300)
        moved = float(np.linalg.norm(step[:2]))
        theta, raw, r, J, cost = th_new, raw_new, r_new, J_new, cost_new
        lam = max(lam / 10.0, 1e-15)
        if rel < 1e-12 or moved < 1e-7:
            converged = True
            break
    return theta, raw, r, J, cost, converged, it


def _fit_all_starts(t, XYZ, c, sigma_t, source_z, clock_sigma, huber_k, estimate_offsets,
                    search_radius, grid_res, init, n_starts):
    M = len(t)
    w = 1.0 / sigma_t**2
    span = float(max(np.ptp(XYZ[:, 0]), np.ptp(XYZ[:, 1]), 1.0))
    R = float(search_radius) if search_radius is not None else max(200.0, 3.0 * span)
    bounds = (XYZ[:, 0].min() - R, XYZ[:, 0].max() + R, XYZ[:, 1].min() - R, XYZ[:, 1].max() + R)
    res = float(grid_res) if grid_res is not None else max(0.25, (2 * R + span) / 400.0)
    starts = grid_search_init(t, XYZ, c, w, source_z, bounds, res, n_best=n_starts, min_sep=3 * res)
    if init is not None:
        starts.insert(0, (np.asarray(init, dtype=float)[:2], float("inf")))
    # LM may leave the grid but not the search area (plus a margin); a fit that wants to run to
    # infinity (plane-wave degeneracy, inconsistent arrivals) ends up flagged at the boundary.
    box = (bounds[0] - R, bounds[1] + R, bounds[2] - R, bounds[3] + R)
    sols = []
    for s0, _ in starts:
        theta0 = np.concatenate(
            [s0, [profile_t0(s0, t, XYZ, c, w, source_z)], np.zeros(M if estimate_offsets else 0)]
        )
        sols.append(_lm_refine(theta0, t, XYZ, c, sigma_t, source_z, clock_sigma, huber_k, estimate_offsets, box=box))
    sols.sort(key=lambda z: z[4])
    return sols, box


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
    clock_sigma: float = 0.0,
    huber_k: float = 0.002,
    reject_k: Optional[float] = None,
    search_radius: Optional[float] = None,
    grid_res: Optional[float] = None,
    init: Optional[Sequence[float]] = None,
    n_starts: int = 8,
) -> TDOASolution:
    """Robust weighted TDOA multilateration.

    t             arrival times (s) on each recording's own clock
    XYZ           (M, 3) recording positions in metres
    sigma_t       per-recording timing standard deviation (s); default 0.5 ms
    clock_sigma   prior std of per-recording clock offsets (s); 0 = synchronised (offsets fixed)
    huber_k       residual magnitude (s) beyond which observations are down-weighted
    reject_k      residual magnitude (s) beyond which an observation is dropped and the fit
                  repeated, as long as at least 4 recordings remain (default 3 * huber_k;
                  0 disables). A leave-one-out check additionally catches a single bad
                  arrival that the fit would otherwise absorb by shifting the source.
    search_radius grid search extends this far beyond the array bounding box (default
                  max(200 m, 3 x array extent)); grid_res defaults to ~400 steps across it
    Alternative minima outside the 95% ellipse of the best solution are reported; the
    solution is flagged ambiguous when one of them fits within delta_cost <= 3 (~95%).
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
    estimate_offsets = clock_sigma > 0
    reject_k = 3.0 * huber_k if reject_k is None else float(reject_k)

    def fit(idx):
        return _fit_all_starts(
            t[idx], XYZ[idx], c, sigma_t[idx], source_z, clock_sigma, huber_k, estimate_offsets,
            search_radius, grid_res, init, n_starts,
        )

    active = np.ones(M, dtype=bool)
    rejected: List[int] = []
    while True:
        idx = np.flatnonzero(active)
        sols, box = fit(idx)
        theta, raw, r, J, cost, converged, iters = sols[0]
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
        chi2_full = float(np.sum(_huber_irls_weights(raw, huber_k) * r[:len(idx)] ** 2))
        if dof_full <= 0 or chi2_full / dof_full < 2.0:
            break
        best_i, best_cost, best_pred = None, cost, None
        for k, i in enumerate(idx):
            sub = np.delete(idx, k)
            s_sols, _ = fit(sub)
            th_s, _, _, _, cost_s, _, _ = s_sols[0]
            if cost_s < best_cost:
                d_s = np.zeros(M)
                if estimate_offsets:
                    d_s[sub] = th_s[3:]
                pred_i, _ = predict_arrivals(th_s[:2], XYZ[i:i + 1], c, source_z, th_s[2], d_s[i:i + 1])
                best_i, best_cost, best_pred = i, cost_s, float(pred_i[0])
        if best_i is not None and best_cost < 0.25 * cost and abs(t[best_i] - best_pred) > 1.5 * huber_k:
            active[best_i] = False
            rejected.append(int(best_i))
            continue
        break
    Ma = len(idx)

    hw = _huber_irls_weights(raw, huber_k)
    wfull = np.concatenate([hw, np.ones(len(r) - Ma)])
    F = (J * wfull[:, None]).T @ J
    cond = float(np.linalg.cond(F)) if np.all(np.isfinite(F)) else float("inf")
    degenerate = not np.isfinite(cond) or cond > 1e12
    cov = np.linalg.pinv(F) if degenerate else np.linalg.inv(F)
    dof = int(len(r) - len(theta))
    chi2 = float(np.sum(hw * r[:Ma] ** 2))
    scale = max(1.0, chi2 / dof) if dof > 0 else 1.0
    cov = cov * scale
    cov_xy = cov[:2, :2].copy()

    s_xy = theta[:2].copy()
    t0 = float(theta[2])
    delta_full = np.zeros(M)
    if estimate_offsets:
        delta_full[idx] = theta[3:]
    pred_full, _ = predict_arrivals(s_xy, XYZ, c, source_z, t0, delta_full)
    raw_full = t - pred_full
    w_full = np.zeros(M)
    w_full[idx] = hw

    alternatives = []
    thr = math.sqrt(5.991)
    for th, _, _, _, cst, _, _ in sols[1:]:
        p = th[:2]
        if _mahalanobis(p, s_xy, cov_xy) <= thr:
            continue
        if any(_mahalanobis(p, (a["x"], a["y"]), cov_xy) <= thr for a in alternatives):
            continue
        dc = float((cst - cost) / scale)
        if dc > 50.0:
            continue
        alternatives.append(
            {"x": float(th[0]), "y": float(th[1]), "cost": float(cst), "delta_cost": dc,
             "distance_m": float(np.linalg.norm(p - s_xy))}
        )
    ambiguous = any(a["delta_cost"] <= 3.0 for a in alternatives)
    margin = 0.02 * max(box[1] - box[0], box[3] - box[2])
    at_boundary = bool(
        s_xy[0] - box[0] < margin or box[1] - s_xy[0] < margin or s_xy[1] - box[2] < margin or box[3] - s_xy[1] < margin
    )

    return TDOASolution(
        s_xy=s_xy, t0=t0, delta=delta_full, cov=cov, cov_xy=cov_xy, residuals_s=raw_full,
        weights=w_full, sigma_t=sigma_t, chi2=chi2, dof=dof, scale=float(scale), cost=float(cost),
        converged=bool(converged), iterations=int(iters), alternatives=alternatives,
        condition_number=cond, degenerate=bool(degenerate), estimate_offsets=bool(estimate_offsets),
        ambiguous=bool(ambiguous), rejected=rejected, at_boundary=at_boundary,
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
    huber_k_s: float = 0.002
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
    note: str = ""


def timing_sigma_from_snr(snr: Sequence[float], base_sigma_s: float) -> np.ndarray:
    """Timing noise grows for weaker onsets: sigma_i = base * clip(sqrt(snr_max / snr_i), 1, 10)."""
    snr = np.maximum(np.asarray(snr, dtype=float), 1e-6)
    return base_sigma_s * np.clip(np.sqrt(snr.max() / snr), 1.0, 10.0)


def locate_from_signals(signals: Sequence[np.ndarray], fs: int, XYZ: np.ndarray, c: float, p: PipelineParams) -> dict:
    """Full pipeline on in-memory mono signals. Returns dict with 'solution' (TDOASolution),
    'tracks' (List[TrackResult]), 'refinement', 'warnings', 'used'."""
    M = len(signals)
    XYZ = np.asarray(XYZ, dtype=float)
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

    max_lag = max_pairwise_lag(XYZ, c, p.slack_s, p.clock_sigma_s)
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
        arrivals, XYZ[used], c, sigma_t=sigma_t, source_z=p.source_z, clock_sigma=p.clock_sigma_s,
        huber_k=p.huber_k_s, search_radius=p.search_radius_m, grid_res=p.grid_res_m,
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
        if k in sol.rejected:
            tr.note = f"rejected as outlier (residual {sol.residuals_s[k]*1000:.2f} ms)"
        elif sol.weights[k] < 0.5:
            tr.note = f"down-weighted outlier (residual {sol.residuals_s[k]*1000:.2f} ms)"
    for k in sol.rejected:
        warnings_out.append(f"recording {used[k]}: arrival rejected as an outlier (residual {sol.residuals_s[k]*1000:.2f} ms)")
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
    if sol.ambiguous:
        alt = min(sol.alternatives, key=lambda a: a["delta_cost"])
        warnings_out.append(f"ambiguous geometry: an alternative solution {alt['distance_m']:.1f} m away fits nearly as well")
    if sol.scale > 4.0:
        warnings_out.append(f"residuals are {math.sqrt(sol.scale):.1f}x larger than the assumed timing noise; uncertainty inflated accordingly")
    for w in warnings_out:
        log(w, "WARN")
    return {"solution": sol, "tracks": tracks, "refinement": refinement, "warnings": warnings_out, "used": used}


# ------------------------------ Plot & I/O ------------------------------


def plot_layout(XY, s_xy, cov, out_png, labels=None, alternatives=None, unused=None):
    a, b, ang_deg = ellipse_from_cov2(cov)
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = plt.gca()
    XY = np.asarray(XY, dtype=float)
    unused = set(unused or [])
    used_mask = np.array([i not in unused for i in range(len(XY))])
    ax.scatter(XY[used_mask, 0], XY[used_mask, 1], marker="^", s=60, label="Recordings")
    if (~used_mask).any():
        ax.scatter(XY[~used_mask, 0], XY[~used_mask, 1], marker="x", s=60, c="grey", label="Not used")
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
        "event_location_local_m": {"x": float(sol.s_xy[0]), "y": float(sol.s_xy[1]), "z": float(p.source_z)},
        "event_location_wgs84": {"lat": float(lat), "lon": float(lon), "alt_m": float(p.source_z)},
        "local_frame": {"origin_lat": lat0, "origin_lon": lon0, "x": "east (m)", "y": "north (m)"},
        "confidence_ellipse_95": {"semi_major_m": a95, "semi_minor_m": b95, "angle_deg": ang},
        "position_std_m": {"x": float(math.sqrt(max(sol.cov_xy[0, 0], 0))), "y": float(math.sqrt(max(sol.cov_xy[1, 1], 0)))},
        "speed_of_sound_mps": c,
        "emission_time_s": sol.t0,
        "clock_model": {"mode": "prior" if sol.estimate_offsets else "synchronised", "clock_sigma_ms": p.clock_sigma_s * 1000},
        "fit": {
            "recordings_used": len(res["used"]),
            "recordings_total": len(files),
            "chi2": sol.chi2,
            "dof": sol.dof,
            "reduced_chi2": (sol.chi2 / sol.dof if sol.dof > 0 else None),
            "uncertainty_scale": sol.scale,
            "rmse_ms": sol.rmse_s * 1000,
            "rejected": [os.path.basename(files[res["used"][k]]) for k in sol.rejected],
            "converged": sol.converged,
            "iterations": sol.iterations,
            "condition_number": sol.condition_number,
            "degenerate": sol.degenerate,
            "ambiguous": sol.ambiguous,
            "at_search_boundary": sol.at_boundary,
            "alternatives": alts,
        },
        "per_recording": per,
        "refinement": refinement,
        "warnings": res["warnings"],
        "parameters": {
            "fs": fs, "bandpass_hz": list(p.band), "env_ms": p.env_ms, "min_snr": p.min_snr, "merge_gap_s": p.merge_gap_s,
            "slack_ms": p.slack_s * 1000, "source_height_m": p.source_z, "huber_k_ms": p.huber_k_s * 1000,
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
    ap.add_argument("--clock_sigma_ms", type=float, default=0.0, help="Prior std of per-recording clock offsets; 0 = synchronised clocks.")
    ap.add_argument("--source_height_m", type=float, default=0.0, help="Assumed event height (m) in the same datum as height_m.")
    ap.add_argument("--timing_sigma_ms", type=float, default=0.5, help="Assumed arrival-time noise for the strongest recording.")
    ap.add_argument("--huber_k_ms", type=float, default=2.0, help="Residuals beyond this are down-weighted (robustness).")
    ap.add_argument("--search_radius_m", type=float, default=None, help="Search this far beyond the array (default max(200, 3x extent)).")
    ap.add_argument("--grid_res_m", type=float, default=None, help="Grid resolution for initialisation (default auto).")
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
        huber_k_s=args.huber_k_ms / 1000.0,
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
        raise LocatorError("need at least 3 recordings to localise in 2D (4 or more recommended)")
    XYZ = mic_local_xyz(mics, lat0, lon0)
    files = [m.file for m in mics]
    log(f"{len(mics)} recordings, speed of sound {c:.2f} m/s, clock model "
        f"{'prior sigma=%.1f ms' % args.clock_sigma_ms if p.clock_sigma_s > 0 else 'synchronised'}")

    signals = []
    for m in mics:
        log(f"Loading audio: {os.path.basename(m.file)}")
        signals.append(load_audio(m.file, args.fs, wav_dir))

    res = locate_from_signals(signals, args.fs, XYZ, c, p)
    sol: TDOASolution = res["solution"]
    results = build_results(res, files, XYZ, lat0, lon0, c, p, args.fs)
    write_json(os.path.join(args.out, "results.json"), results)
    write_sync_csv(os.path.join(args.out, "sync.csv"), files, res["tracks"])
    plot_layout(
        XYZ[:, :2], sol.s_xy, sol.cov_xy, os.path.join(args.out, "layout.png"),
        labels=[os.path.basename(f) for f in files], alternatives=sol.alternatives,
        unused=[i for i in range(len(files)) if i not in res["used"]],
    )
    for tr, fn in zip(res["tracks"], files):
        if tr.used:
            log(f"{os.path.basename(fn)}: arrival={tr.arrival_s:.6f}s snr={tr.snr:.1f} residual={tr.residual_s*1000:+.3f} ms" + (f" ({tr.note})" if tr.note else ""))
        else:
            log(f"{os.path.basename(fn)}: not used ({tr.note})")
    e = results["confidence_ellipse_95"]
    log(f"Estimated location (local m): x={sol.s_xy[0]:.2f}, y={sol.s_xy[1]:.2f}")
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
