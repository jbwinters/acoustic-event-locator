#!/usr/bin/env python3
"""
Synthesize test recordings for the acoustic event locator.

For each scenario directory under test_data/ (a positions.json with an "event" block) this
writes one audio track per recording plus metadata.json holding the ground truth in the same
local frame that locate_event.py uses (origin = "reference"/"reference_point", else the
centroid of the recordings).

Physics included
  * exact (fractional-sample) propagation delays from the true source to each recording
  * 1/r amplitude spreading, so the signal-to-noise ratio falls with distance
  * independent background noise per recording
  * per-recording clock offsets (each recording runs on its own clock)
  * optional early reflections (echoes) with random delays and gains per recording
Not modeled: wind, temperature gradients, microphone directivity, clipping/AGC.

Output format: WAV by default (no external tools needed). --format mp4 wraps each track in an
MP4 with a black video track and needs ffmpeg. locate_event.py reads either.

The synthesis functions (event_waveform, render_track, synthesize_scenario) are also used by
the unit tests.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import locate_event as le  # noqa: E402

EVENT_KINDS = ("gunshot", "explosion", "fireworks")
DEFAULT_CLOCK_OFFSETS_MS = [0.0] * 8  # synchronized by default; see --clock_offsets_ms / --random_clock_ms
SCENARIOS = ("scenario1_gunshot", "scenario2_explosion", "scenario3_fireworks", "scenario4_window_shot")


# ------------------------------ Synthesis ------------------------------


def event_waveform(kind: str, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Event pressure waveform with its onset at index 0, normalized to peak 1."""
    if kind == "gunshot":
        n = int(0.06 * fs)
        t = np.arange(n) / fs
        burst = rng.standard_normal(n) * np.exp(-t / 0.003)  # broadband muzzle blast
        blast = np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.008)  # low-frequency push
        w = burst + 0.8 * blast
    elif kind == "explosion":
        n = int(1.0 * fs)
        t = np.arange(n) / fs
        sos = butter(2, [40 / (fs / 2), 1500 / (fs / 2)], btype="band", output="sos")
        rumble = sosfilt(sos, rng.standard_normal(n)) * np.exp(-t / 0.15)
        front = rng.standard_normal(n) * np.exp(-t / 0.01)  # shock front
        w = front + 3.0 * rumble
    elif kind == "fireworks":
        n = int(1.3 * fs)
        w = np.zeros(n)
        nc = int(0.015 * fs)
        w[:nc] = rng.standard_normal(nc) * np.exp(-np.arange(nc) / fs / 0.003)  # burst report
        m = int(0.003 * fs)
        for _ in range(40):  # crackling stars
            k = int(rng.uniform(0.05, 1.1) * fs)
            w[k : k + m] += rng.uniform(0.05, 0.35) * rng.standard_normal(m) * np.exp(-np.arange(m) / fs / 0.001)
    else:
        raise ValueError(f"unknown event kind {kind!r}; choose from {EVENT_KINDS}")
    return w / np.max(np.abs(w))


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Band-limited (sinc) delay of x by a possibly fractional number of samples."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * delay_samples), n=n)


def render_track(event, fs, duration_s, arrival_s, gain, noise_rms, rng, reflections=()):
    """Place `event` so its onset is at arrival_s (fractional sample accurate), scaled by gain,
    add reflections [(delay_s, relative_gain), ...] and white noise. Returns (track, clean)."""
    n = int(round(duration_s * fs))
    k = int(np.floor(arrival_s * fs))
    frac = arrival_s * fs - k
    clean = np.zeros(n)

    def place(k0, g):
        a, b = max(0, k0), min(n, k0 + len(event))
        if b > a:
            clean[a:b] += g * event[a - k0 : b - k0]

    place(k, gain)
    for d_s, g_r in reflections:
        place(k + int(round(d_s * fs)), gain * g_r)
    if frac > 0:
        clean = fractional_delay(clean, frac)
    return clean + noise_rms * rng.standard_normal(n), clean


def synthesize_scenario(
    XYZ,
    source_xyz,
    c,
    kind="gunshot",
    fs=48000,
    duration_s=10.0,
    emission_s=5.0,
    clock_offsets_s=None,
    noise_rms=0.003,
    level_at_10m=0.5,
    reflections=True,
    rng=None,
):
    """Synthesize one track per recording. Returns (tracks, truth dict)."""
    rng = np.random.default_rng(0) if rng is None else rng
    XYZ = np.asarray(XYZ, dtype=float)
    source_xyz = np.asarray(source_xyz, dtype=float)
    M = len(XYZ)
    clock = np.zeros(M) if clock_offsets_s is None else np.asarray(clock_offsets_s, dtype=float)[:M]
    if len(clock) < M:
        clock = np.concatenate([clock, np.zeros(M - len(clock))])
    d = np.linalg.norm(XYZ - source_xyz, axis=1)
    t_true = emission_s + d / c + clock
    event = event_waveform(kind, fs, rng)
    tracks, snr_db, refl_all = [], [], []
    for i in range(M):
        gain = level_at_10m * 10.0 / max(d[i], 1.0)
        refl = []
        if reflections:
            for _ in range(int(rng.integers(1, 4))):
                refl.append((float(rng.uniform(0.008, 0.06)), float(rng.uniform(0.15, 0.5))))
        x, clean = render_track(event, fs, duration_s, t_true[i], gain, noise_rms, rng, refl)
        tracks.append(x)
        snr_db.append(float(20 * np.log10(np.max(np.abs(clean)) / noise_rms)))
        refl_all.append(refl)
    truth = {
        "event_type": kind,
        "source_position_m": [float(source_xyz[0]), float(source_xyz[1])],
        "source_height_m": float(source_xyz[2]),
        "microphone_positions_m": XYZ.tolist(),
        "clock_offsets_s": clock.tolist(),
        "arrival_times_s": t_true.tolist(),
        "distances_m": d.tolist(),
        "emission_time_s": float(emission_s),
        "speed_of_sound_ms": float(c),
        "sample_rate_hz": int(fs),
        "duration_s": float(duration_s),
        "noise_rms": float(noise_rms),
        "snr_db": snr_db,
        "reflections": refl_all,
    }
    return tracks, truth


# ------------------------------ Scenario files ------------------------------


def load_scenario(scenario_dir: str):
    """Returns (J, mics, (lat0, lon0), c, XYZ_true, source_xyz, kind). XYZ_true uses each entry's
    `true_height_m` when present (the locator only sees `height_m`, the prior mean)."""
    J = le.read_json(os.path.join(scenario_dir, "positions.json"))
    mics, (lat0, lon0), c = le.parse_positions(J)
    XYZ = le.mic_local_xyz(mics, lat0, lon0)
    for i, m in enumerate(J["mics"]):
        if "true_height_m" in m:
            XYZ[i, 2] = float(m["true_height_m"])
    ev = J.get("event", {})
    loc = ev.get("true_location") or ev.get("estimated_location")
    if loc is None:
        raise ValueError(f"{scenario_dir}: positions.json needs event.true_location")
    sx, sy = le.latlon_to_local_xy(float(loc["lat"]), float(loc["lon"]), lat0, lon0)
    source = np.array([sx, sy, float(loc.get("height_m", 0.0))])
    kind = ev.get("type", "gunshot")
    return J, mics, (lat0, lon0), c, XYZ, source, kind


def write_wav(path: str, x: np.ndarray, fs: int):
    peak = float(np.max(np.abs(x)))
    if peak > 0.999:
        print(f"    note: {os.path.basename(path)} peaks at {peak:.2f}, clipping to +-1")
        x = np.clip(x, -0.999, 0.999)
    sf.write(path, x.astype(np.float32), fs, subtype="PCM_16")


def wrap_mp4(wav_path: str, mp4_path: str, duration_s: float):
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=black:size=320x240:duration={duration_s}:rate=10",
        "-i", wav_path, "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-shortest", mp4_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def generate_scenario(scenario_dir, fmt="wav", seed=0, noise_rms=0.003, clock_offsets_ms=None,
                      random_clock_ms=None, reflections=True, duration_s=10.0, emission_s=5.0, level=0.5):
    J, mics, (lat0, lon0), c, XYZ, source, kind = load_scenario(scenario_dir)
    rng = np.random.default_rng(seed)
    M = len(mics)
    if random_clock_ms is not None:
        clock = rng.normal(0.0, random_clock_ms / 1000.0, M)
    else:
        ms = DEFAULT_CLOCK_OFFSETS_MS if clock_offsets_ms is None else clock_offsets_ms
        clock = np.array(ms, dtype=float)[:M] / 1000.0
    tracks, truth = synthesize_scenario(
        XYZ, source, c, kind, fs=48000, duration_s=duration_s, emission_s=emission_s,
        clock_offsets_s=clock, noise_rms=noise_rms, level_at_10m=level, reflections=reflections, rng=rng,
    )
    print(f"{os.path.basename(scenario_dir)}: {kind}, {M} recordings, c={c:.1f} m/s, source=({source[0]:.1f}, {source[1]:.1f}, {source[2]:.1f}) m")
    files = []
    for i, m in enumerate(mics):
        stem = os.path.splitext(m.file)[0]
        wav_path = os.path.join(scenario_dir, stem + ".wav")
        write_wav(wav_path, tracks[i], 48000)
        out_name = stem + ".wav"
        if fmt == "mp4":
            mp4_path = os.path.join(scenario_dir, stem + ".mp4")
            wrap_mp4(wav_path, mp4_path, duration_s)
            os.remove(wav_path)
            out_name = stem + ".mp4"
        files.append(out_name)
        hnote = f"  height {XYZ[i, 2]:.1f} m" + (f" (prior {m.height_m:.1f} +- {m.height_sigma_m:.1f})" if m.height_sigma_m > 0 else "")
        print(f"  {out_name:12s} d={truth['distances_m'][i]:6.1f} m  snr={truth['snr_db'][i]:5.1f} dB  clock={clock[i]*1000:+.1f} ms  arrival={truth['arrival_times_s'][i]:.6f} s{hnote}")
    lat_s, lon_s = le.local_xy_to_latlon(source[0], source[1], lat0, lon0)
    hp = J.get("event", {}).get("height_prior")
    truth.update({
        "files": files,
        "format": fmt,
        "seed": seed,
        "source_latlon": {"lat": lat_s, "lon": lon_s},
        "local_frame": {"origin_lat": lat0, "origin_lon": lon0},
        "microphone_height_prior_m": [m.height_m for m in mics],
        "microphone_height_sigma_m": [m.height_sigma_m for m in mics],
        "height_prior_m": ({"mean": float(hp["mean_m"]), "sigma": float(hp["sigma_m"])} if hp else None),
    })
    with open(os.path.join(scenario_dir, "metadata.json"), "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic test recordings.")
    ap.add_argument("--scenarios", nargs="+", default=["all"], help="Scenario directory names under test_data/, or 'all'.")
    ap.add_argument("--format", choices=("auto", "wav", "mp4"), default="auto", help="auto = mp4 if ffmpeg is available, else wav.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise_rms", type=float, default=0.003, help="Background noise RMS (full scale = 1; 0.003 is about -50 dBFS).")
    ap.add_argument("--clock_offsets_ms", type=float, nargs="+", default=None, help="Per-recording clock offsets (ms).")
    ap.add_argument("--random_clock_ms", type=float, default=None, help="Draw clock offsets from N(0, sigma) instead.")
    ap.add_argument("--no_reflections", action="store_true")
    ap.add_argument("--duration_s", type=float, default=10.0)
    ap.add_argument("--emission_s", type=float, default=5.0)
    ap.add_argument("--level", type=float, default=0.5, help="Peak amplitude at 10 m (full scale = 1).")
    args = ap.parse_args(argv)

    fmt = args.format
    if fmt == "auto":
        fmt = "mp4" if shutil.which("ffmpeg") else "wav"
        if fmt == "wav":
            print("ffmpeg not found: writing WAV tracks (locate_event.py reads them directly)")
    elif fmt == "mp4" and not shutil.which("ffmpeg"):
        print("Error: --format mp4 needs ffmpeg on PATH")
        return 1
    names = list(SCENARIOS) if "all" in args.scenarios else args.scenarios
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    for name in names:
        sdir = os.path.join(base, name)
        if not os.path.isdir(sdir):
            print(f"Warning: {sdir} does not exist, skipping")
            continue
        generate_scenario(
            sdir, fmt=fmt, seed=args.seed, noise_rms=args.noise_rms, clock_offsets_ms=args.clock_offsets_ms,
            random_clock_ms=args.random_clock_ms, reflections=not args.no_reflections,
            duration_s=args.duration_s, emission_s=args.emission_s, level=args.level,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
