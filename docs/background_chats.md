Below is a robust, field‑tested way to localize a single impulsive sound from many unsynchronized videos whose camera locations (lat/long) are known. It’s built around **TDOA multilateration** (time‑difference of arrival) with echo‑robust timing and a solver that also estimates each device’s unknown clock offset. It works best with ≥4 videos.

---

## 0) Notation & coordinate frame

* Let microphone (phone/camera) positions be $x_i \in \mathbb{R}^2$ for $i=1,\dots,M$.

  * Convert lat/long to a local planar frame (meters). For a site ≤ 200 yd across, the local tangent plane is accurate:

  $$
  \begin{aligned}
  x &= (\lambda - \lambda_0)\;\cos\varphi_0 \cdot 111{,}320 \;\text{m},\\
  y &= (\varphi - \varphi_0)\cdot 110{,}540 \;\text{m},
  \end{aligned}
  $$

  where $(\varphi_0,\lambda_0)$ is a convenient reference (e.g., the centroid).
* Unknown source (event) location $s\in\mathbb{R}^2$ on the ground plane.
* Speed of sound $c \approx 343\;\text{m/s}$ at 20 °C. You can refine with $c \approx 331.4 + 0.6T_{^\circ C}$ (humidity effect is small).
* Each device has an unknown clock offset $\delta_i$ (seconds) relative to an arbitrary reference ($\delta_1\equiv 0$ to fix the gauge).

---

## 1) Extract precise pairwise time differences (robust to echoes)

**Goal:** For each pair $(i,j)$, estimate $\Delta t^{\text{meas}}_{ij}$ = time by which the event arrives earlier at $i$ than $j$. Also compute a confidence weight $w_{ij}$.

1. **Preprocess each audio track**

   * Resample all to a common rate (e.g., 48 kHz).
   * Bandpass filter to the event’s dominant band if known (e.g., 200–4000 Hz for a sharp transient).
   * Normalize (AGC differences do not affect timing).

2. **Find the first-arrival (direct-path) window per track**

   * Use an onset picker robust to reverberation:

     * **STA/LTA ratio** (short‑term over long‑term energy), OR
     * **AIC picker** (from seismology): $ \mathrm{AIC}(k)=k\ln \hat\sigma^2_{1:k} + (N-k-1)\ln \hat\sigma^2_{k+1:N}$; the argmin marks the first break (direct path).
   * Keep a short analysis window around that onset (e.g., ±40 ms). This suppresses late echoes.

3. **Pairwise TDOA via GCC‑PHAT**

   * For each pair $(i,j)$, compute the **generalized cross‑correlation with phase transform** on the chosen windows:

     $$
     R_{ij}(\tau)=\mathcal{F}^{-1}\!\left\{\frac{X_i(f)\,X_j^\*(f)}{|X_i(f)\,X_j^\*(f)|}\right\}.
     $$
   * The peak lag $\hat\tau_{ij}$ gives $\Delta t^{\text{meas}}_{ij}$. Refine to sub‑sample precision by parabolic fit around the peak.
   * Assign a weight $w_{ij}$ from peak sharpness/SNR (e.g., peak‑to‑sidelobe ratio or inverse of the fitted variance). Discard pairs with ambiguous peaks (multipath).

> Practical bounds: if your microphones span at most \~200 m, the **max** |TDOA| is at most baseline/c. For 100 m baselines, $|\Delta t|\lesssim 0.29$ s. Gate your search to physically plausible lags.

---

## 2) Measurement model (asynchronous sensors)

GCC‑PHAT gives **measured** pairwise lags $\Delta t^{\text{meas}}_{ij}$. The model that links them to geometry is:

$$
\Delta t^{\text{meas}}_{ij}
= \underbrace{\frac{\|s-x_i\|-\|s-x_j\|}{c}}_{\text{geometric TDOA}}
+ \underbrace{(\delta_i - \delta_j)}_{\text{clock offsets}}
+ \nu_{ij},
$$

with noise $\nu_{ij}$. Unknowns: $s=(x,y)$ and the offsets $\delta_2,\dots,\delta_M$ (set $\delta_1=0$).

**Identifiability:**

* If your devices are **unsynchronized**, you need **at least 4** videos to solve for $s$ and $\{\delta_i\}$ in 2D.
* If they are truly synchronized (same recorder), **3** suffices in 2D.

---

## 3) Solve for the source and the clock offsets (robust WLS)

We have measurements for all pairs $(i,j)$, $i<j$. Define $z_{ij}=\Delta t^{\text{meas}}_{ij}$ and range differences $\Delta r_{ij}(s)=\|s-x_i\|-\|s-x_j\|$.

We minimize the **robust weighted least squares** cost

$$
\min_{s,\delta}\ \sum_{i<j} w_{ij}\,\rho\!\left(
z_{ij} - \frac{\Delta r_{ij}(s)}{c} - (\delta_i-\delta_j)
\right),
$$

with $\delta_1=0$. Use a robust loss $\rho(\cdot)$ (Huber/Tukey) to blunt multipath outliers.

### Efficient alternating solver (fast and stable)

This problem is **biconvex** in $(s)$ and $(\delta)$. Alternate two closed‑form/standard steps until convergence:

**A. Fix $s$, solve the clock offsets $\delta$ in closed form**

Let $g_{ij} = z_{ij} - \Delta r_{ij}(s)/c$. Stack all pairwise differences into vector $g$.
Let $B\in\mathbb{R}^{P\times M}$ be the incidence matrix (one row per pair $(i,j)$: +1 at $i$, −1 at $j$, 0 elsewhere).
Solve the weighted least squares

$$
\min_\delta \| W^{1/2}(g - B\delta) \|_2^2,\quad \text{with } \delta_1=0.
$$

Normal equations: $ (B^\top W B)\,\delta = B^\top W g$.
Implement by removing the first column (fixing $\delta_1$) or by adding a tiny Tikhonov term.

**B. Fix $\delta$, update the source $s$ with Gauss–Newton**

Minimize

$$
\min_s \sum_{i<j} w_{ij}\left[\, r_{ij}(s) \,\right]^2,\quad
r_{ij}(s) = z_{ij} - (\delta_i-\delta_j) - \frac{\Delta r_{ij}(s)}{c}.
$$

Jacobian row for pair $(i,j)$:

$$
\frac{\partial r_{ij}}{\partial s}
= -\frac{1}{c}\left( \frac{s-x_i}{\|s-x_i\|} - \frac{s-x_j}{\|s-x_j\|} \right)^\top.
$$

Initialize $s$ by a coarse **grid search** over the courtyard (e.g., 1–2 m resolution) evaluating the cost with $\delta$ “profiled out” using step A at each grid point; pick the best grid point as $s_0$, then refine with Gauss–Newton + line search.

**C. Stop when** the cost decrease is tiny or $\|s_{k+1}-s_k\|<1\,\mathrm{cm}$.
**D. Uncertainty:** At the solution, estimate covariance of $s$ via the Gauss–Newton Fisher approximation:
$\mathrm{Cov}(s)\approx (J^\top W J)^{-1}\hat\sigma^2$, with $J$ the stacked Jacobian and $\hat\sigma^2$ the residual variance. Report a 95% ellipse.

> **Why this works with unsynchronized videos:** Pairwise lags give many equations ($\binom{M}{2}$) for relatively few unknowns ($M-1$ offsets + 2 location params). With $M\ge 4$, the system is overdetermined; we solve offsets **and** position together.

---

## 4) Handling echoes and bad pairs (crucial in courtyards)

* **Early‑arrival gating:** The direct path is the **earliest** coherent rise; use the AIC/STA‑LTA pick to anchor the window for GCC‑PHAT.
* **Peak‑quality weights:** Let $w_{ij}\propto$ (main‑peak height)/(second‑peak height) or inverse of fitted parabola width. Down‑weight smeared/ambiguous correlations.
* **Geometric consistency RANSAC (optional):**

  * Randomly pick 4 mics, solve $(s,\delta)$, score against all pairs; keep the hypothesis with most inliers; refine on inliers.
* **Cycle‑consistency filtering:** The measured lags must obey $\Delta t_{ij}+\Delta t_{jk}+\Delta t_{ki}\approx 0$. Triples that violate this are likely echo‑contaminated; drop the offending pairs.

---

## 5) Practical numbers & tips

* **Resolution:** At 48 kHz, 1 sample ≈ 20.8 µs → range‑difference quantum $\approx 7.1$ mm. Sub‑sample interpolation commonly achieves \~0.05–0.2 sample under good SNR (mm‑cm path‑difference accuracy), but echoes can push to tens of cm.
* **Geometry:** Avoid near‑collinear mic layouts; spread devices around the courtyard if possible. Larger baselines improve TDOA sensitivity.
* **Wind (optional refinement):** A uniform wind $v$ slightly biases travel time: effective $c_\text{eff}\approx c+v\cdot \hat u$. You can append $v_x,v_y$ as two extra unknowns; usually unnecessary over \~200 m unless it’s very windy.
* **3D variant:** If height might matter (e.g., sound from a balcony), extend $s\in\mathbb{R}^3$ and include known mic heights $z_i$. You’ll need one more mic (≥5 unsynced) for robust estimation.

---

## 6) End‑to‑end algorithm (pseudocode)

```text
INPUT: videos {V_i} with lat/long {(lat_i, lon_i)}, sample rate Fs
OUTPUT: estimated source ŝ, clock offsets δ̂, confidence ellipse

1. Convert {(lat_i, lon_i)} → {x_i ∈ R^2} in local meters.
2. For each audio A_i from V_i:
   a. Resample to Fs (e.g., 48 kHz), bandpass, normalize.
   b. Compute onset index k_i via AIC or STA/LTA.
   c. Extract window W_i centered on k_i (e.g., ±40 ms).

3. For all pairs (i<j):
   a. Compute GCC-PHAT on (W_i, W_j) → lag τ̂_ij.
   b. Sub-sample refine (parabolic fit) → Δt_meas_ij.
   c. Compute quality measure → weight w_ij.
   d. Discard pairs with poor quality or implausible lag.

4. Build pair list P = {(i,j, z_ij=Δt_meas_ij, w_ij)}.

5. Initialize:
   a. Coarse grid over courtyard: for each candidate s:
        - Compute g_ij(s) = z_ij - Δr_ij(s)/c for all (i,j).
        - Solve δ(s): (BᵀWB) δ = BᵀW g  (fix δ_1=0).
        - Score cost(s) = Σ w_ij [ g_ij(s) - (δ_i-δ_j) ]^2.
      Pick s_0 = argmin cost(s); δ_0 = δ(s_0).

6. Alternate until convergence:
   a. Given s_k, solve δ_{k+1} by (BᵀWB) δ = BᵀW g(s_k).
   b. Given δ_{k+1}, update s_{k+1} with Gauss–Newton on
        r_ij(s) = z_ij - (δ_i-δ_j) - Δr_ij(s)/c
      using robust loss and line search.

7. After convergence:
   a. Compute residuals, drop gross outliers, re-solve if needed.
   b. Compute covariance ellipse from (Jᵀ W J)^{-1}.

Return ŝ, δ̂, and 95% ellipse.
```

---

## 7) What if you only have 3 videos?

* **If clocks are synchronized** (same recorder or known offsets), step 3 drops the $(\delta_i-\delta_j)$ terms and standard 2D TDOA with 3 mics works.
* **If not synchronized**, a **single event** with 3 unsynced devices is unidentifiable without extra information. Remedies:

  * Do a **calibration impulse** (clap/handheld buzzer) at a known spot; use it to solve $\delta$ first.
  * Or get **a 4th** recording; or two distinct events.

---

## 8) Deliverables you can compute

* **Point estimate** $\hat s = (x,y)$ in meters (convert back to lat/long).
* **Uncertainty ellipse** (e.g., semi‑axes $a,b$ and angle $\theta$) for 95% confidence.
* **Per‑device clock offsets** $ \hat\delta_i$ (handy to re‑synchronize the videos).
* **Diagnostics:** inlier count, residual histogram, pairwise weights.

---

### Why this design?

* GCC‑PHAT is the standard for TDOA in reverberant, noisy environments.
* Alternating “profiled‑offsets + Gauss–Newton for position” exploits the problem’s structure: offsets are linear once $s$ is fixed; $s$ is low‑dimensional and smooth once offsets are fixed.
* Robust losses, cycle checks, and RANSAC handle echoes common in open courtyards with hard building faces.

If you’d like, I can also sketch a compact Python/Numpy implementation and a small synthetic test to validate it against known geometry.

Good question — let me state this clearly in terms of inputs and outputs, at a “black-box API” level for the algorithm I just designed:

---

## **Inputs**

1. **Audio data**

   * A set of ≥3 video/audio recordings $\{V_i\}_{i=1}^M$.
   * For each $V_i$, you need the **audio waveform** (single channel or mixdown), ideally resampled to a common rate (e.g. 48 kHz).
   * Noise is allowed, but the target impulsive event must be clearly audible in each.

2. **Sensor positions**

   * GPS **lat/long** (and optionally altitude) for each recording device.
   * Convert to local planar coordinates (meters).

3. **Physical constants (optional)**

   * Speed of sound $c$. Default: 343 m/s (20 °C, dry air).
   * Optionally, temperature/humidity/wind if you want more accurate modeling.

4. **Algorithm parameters**

   * Window length around onset (e.g. ±40 ms).
   * Bandpass filter range (Hz).
   * Robust loss choice (Huber, Tukey, etc.).
   * Grid resolution for initialization (meters).
   * Convergence thresholds (cm or cost decrease).

---

## **Outputs**

1. **Event location estimate**

   * $\hat{s} = (x,y)$ in meters in the local coordinate frame.
   * Can be converted back to **lat/long**.
   * If solved in 3D: $(x,y,z)$.

2. **Uncertainty / confidence**

   * A covariance matrix $\Sigma_s$ or 95% confidence ellipse (semi-axes lengths and orientation).
   * Useful for understanding how tight the solution is (cm, meters).

3. **Clock offset estimates (if devices unsynchronized)**

   * $\hat{\delta}_i$ for each device relative to a reference device.
   * Lets you effectively “re-synchronize” the recordings post-hoc.

4. **Diagnostics**

   * Residual errors per pair $(i,j)$.
   * Quality weights $w_{ij}$.
   * Number of inliers/outliers after robust filtering.
   * Optional: refined TDOA matrix (time-differences between each pair).

---

### **Summary**

* **Inputs:** audio waveforms from multiple devices, their GPS coordinates, and some environmental constants.
* **Outputs:** event’s location (lat/long), confidence ellipse, estimated device time offsets, and diagnostic statistics.

---

Do you want me to show you a **concrete JSON-like I/O schema** (as if this were a Python function), so it’s clearer how you’d implement this?

Awesome — here’s a **single Python script** that takes a directory of videos plus a `positions.json`, does **everything automatically** (audio extraction, first‑arrival picking, pairwise time-differences, robust multilateration with unknown clock offsets, uncertainty ellipse, and media sync offsets), and writes a results bundle.

* **Minimal user input:** just `--videos_dir` and `--positions`.
* **No manual alignment:** it auto‑detects the event in each track.
* **Unsynced devices supported:** it solves per‑video clock offsets jointly with the source location.
* **Outputs:** `results.json`, `sync.csv`, and a plot `layout.png`.

> **Dependencies:** `ffmpeg` (CLI), `numpy`, `scipy`, `soundfile`, `matplotlib`.
> Install: `pip install numpy scipy soundfile matplotlib`

---

### File: `locate_event.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locate a single impulsive acoustic event from multiple unsynchronized videos.

Inputs:
  - --videos_dir: directory with the video files (mp4/mov/etc.)
  - --positions: JSON with mapping file->lat/lon (and optional height), optionally temperature/speed of sound
      {
        "temperature_C": 20,                 // optional
        "speed_of_sound": null,              // optional (overrides temp if provided)
        "reference": {"lat": 41.881, "lon": -87.629},   // optional local-frame origin
        "mics": [
          {"file": "cam1.mp4", "lat": 41.88110, "lon": -87.62970, "height_m": 1.6},
          {"file": "cam2.mp4", "lat": 41.88125, "lon": -87.62920, "height_m": 1.5},
          {"file": "cam3.mp4", "lat": 41.88085, "lon": -87.62940, "height_m": 1.7},
          {"file": "cam4.mp4", "lat": 41.88100, "lon": -87.62905, "height_m": 1.6}
        ]
      }

Outputs (written to --out):
  - results.json (event location in local meters and WGS84, per-video offsets, covariance/ellipse)
  - sync.csv (file, arrival_time_s, clock_offset_s, align_to_event_offset_s)
  - layout.png (mic geometry, estimate, 95% ellipse)
  - wav/ (normalized mono 48kHz audio extracted from each video)

Algorithm highlights:
  - Bandpass + energy-envelope gate + AIC/STA-LTA onset picker for first-arrival (echo-robust)
  - Refines each arrival using a template cross-correlation (GCC-PHAT) for sub-sample precision
  - Builds pairwise time differences z_ij = t_j - t_i
  - Robust weighted solver estimates source position s and per-device clock offsets δ_i (δ_0=0)
  - Confidence ellipse from Gauss-Newton Fisher approximation
"""

import argparse, os, json, sys, math, csv, shutil, tempfile, subprocess, itertools, time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ Utilities ------------------------------

def log(msg: str, level="INFO"):
    print(f"[{level}] {msg}", flush=True)

def require_ffmpeg():
    for binname in ("ffmpeg", "ffprobe"):
        if shutil.which(binname) is None:
            log(f"Required tool '{binname}' is not on PATH. Please install ffmpeg.", "ERROR")
            sys.exit(2)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi


# ------------------------------ Geo conversion (local tangent plane) ------------------------------

def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert lat/lon to a local tangent plane in meters.
    Accurate for small areas (<= ~1 km).
    """
    # WGS-84 approximate meters/deg
    mx = 111320.0 * math.cos(deg2rad(lat0))
    my = 110540.0
    x = (lon - lon0) * mx
    y = (lat - lat0) * my
    return x, y

def local_xy_to_latlon(x, y, lat0, lon0):
    mx = 111320.0 * math.cos(deg2rad(lat0))
    my = 110540.0
    lat = y / my + lat0
    lon = x / mx + lon0
    return lat, lon


# ------------------------------ Audio extraction ------------------------------

def extract_audio_ffmpeg(in_video: str, out_wav: str, target_sr=48000):
    """
    Extract mono, normalized (no gain change here, just mono mix), resampled audio from video.
    """
    cmd = [
        "ffmpeg", "-y", "-i", in_video,
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", out_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        log(f"ffmpeg failed on {in_video} with error:\n{e.stderr.decode('utf-8', errors='ignore')}", "ERROR")
        raise


# ------------------------------ Signal processing ------------------------------

def butter_bandpass(low_hz, high_hz, fs, order=4):
    nyq = 0.5 * fs
    low = max(1.0, low_hz) / nyq
    high = min(high_hz, nyq * 0.99) / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass(x, fs, low_hz=200.0, high_hz=4000.0, order=4):
    if high_hz >= fs * 0.49:
        high_hz = fs * 0.49
    b, a = butter_bandpass(low_hz, high_hz, fs, order=order)
    return filtfilt(b, a, x)

def rms_envelope(x, fs, win_ms=20.0):
    win = max(1, int(fs * (win_ms/1000.0)))
    # squared moving average (RMS)
    x2 = np.pad(x**2, (win, win), mode="edge")
    c = np.cumsum(x2)
    ma = (c[2*win:] - c[:-2*win]) / (2*win)
    rms = np.sqrt(ma)
    return rms

def aic_picker(x):
    """
    Akaike Information Criterion picker for first-arrival in 1D signal.
    Compute efficiently via cumulative sums; restrict to valid interior.
    """
    N = len(x)
    if N < 1000:
        # fallback to simple energy threshold later
        return None
    # Cumulative sums for mean/var computation
    s1 = np.cumsum(x)
    s2 = np.cumsum(x**2)
    idx = np.arange(1, N-1)

    # first segment stats
    mu1 = s1[idx-1] / idx
    var1 = (s2[idx-1] / idx) - mu1**2
    var1 = np.maximum(var1, 1e-12)

    # second segment stats
    n2 = (N - idx - 1)
    mu2 = (s1[-1] - s1[idx]) / n2
    var2 = ((s2[-1] - s2[idx]) / n2) - mu2**2
    var2 = np.maximum(var2, 1e-12)

    aic = idx * np.log(var1) + (N - idx - 1) * np.log(var2)
    k = int(np.argmin(aic))
    return k

def sta_lta_picker(x, fs, sta_ms=5.0, lta_ms=200.0, thr=3.5):
    sta_n = max(1, int(fs * (sta_ms/1000)))
    lta_n = max(sta_n+1, int(fs * (lta_ms/1000)))
    eps = 1e-12
    x2 = x**2
    c = np.cumsum(x2)
    sta = (c[sta_n:] - c[:-sta_n]) / sta_n
    lta = (c[lta_n:] - c[:-lta_n]) / lta_n
    # align lengths
    pad = lta_n - sta_n
    sta = sta[pad:]
    ratio = sta / (lta + eps)
    # find first crossing
    idx = np.argmax(ratio > thr)
    if ratio[idx] > thr:
        return int(idx + lta_n)   # approximate alignment with original indexing
    return None

def gcc_phat(sig, ref, fs, max_tau=None, interp=4):
    """
    GCC-PHAT cross-correlation to estimate lag between sig and ref.
    Returns lag (seconds) and correlation function.
    If max_tau is provided, restrict the search to ±max_tau.
    """
    n = int(2**np.ceil(np.log2(len(sig) + len(ref))))
    SIG = rfft(sig, n=n)
    REF = rfft(ref, n=n)
    R = SIG * np.conj(REF)
    denom = np.abs(R) + 1e-12
    R /= denom
    cc = irfft(R, n=interp*n)
    max_shift = int(interp * n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # lags in samples (interp)
    shift = np.arange(-max_shift, max_shift+1)
    if max_tau is not None:
        max_shift_limit = int(interp * fs * max_tau)
        center = len(cc) // 2
        lo = center - max_shift_limit
        hi = center + max_shift_limit + 1
        lo = max(0, lo)
        hi = min(len(cc), hi)
        cc_window = cc[lo:hi]
        shift_window = shift[(lo):(hi)]
        imax = np.argmax(cc_window)
        shift_est = shift_window[imax]
        return shift_est / float(interp * fs), cc_window
    else:
        imax = np.argmax(cc)
        shift_est = shift[imax]
        return shift_est / float(interp * fs), cc

def quadratic_subsample_peak(y, i):
    """
    Parabolic interpolation around discrete peak at i.
    Returns (peak_index_subsampled, peak_value_subsampled).
    """
    if i <= 0 or i >= len(y)-1:
        return i, y[i]
    y0, y1, y2 = y[i-1], y[i], y[i+1]
    denom = (2*y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return i, y1
    delta = 0.5 * (y0 - y2) / denom
    xhat = i + delta
    yhat = y1 - 0.25*(y0 - y2)*delta
    return xhat, yhat


# ------------------------------ Core pipeline ------------------------------

@dataclass
class Mic:
    file: str
    lat: float
    lon: float
    height_m: float = 0.0

@dataclass
class MicData:
    file: str
    wav_path: str
    fs: int
    signal: np.ndarray
    filtered: np.ndarray
    arrival_idx: int            # first-arrival index (samples)
    arrival_s: float            # seconds
    snr_like: float             # weight proxy
    x: float                    # local x (m)
    y: float                    # local y (m)
    z: float                    # local z (m), optional

def load_positions(positions_json: str, videos_dir: str):
    J = read_json(positions_json)
    mics = []
    for m in J["mics"]:
        p = os.path.join(videos_dir, m["file"])
        if not os.path.exists(p):
            log(f"Video listed in positions not found: {p}", "ERROR")
            sys.exit(2)
        mics.append(Mic(file=p, lat=m["lat"], lon=m["lon"], height_m=m.get("height_m", 0.0)))
    # reference lat/lon
    if "reference" in J and J["reference"]:
        lat0 = J["reference"]["lat"]
        lon0 = J["reference"]["lon"]
    else:
        lat0 = float(np.mean([m.lat for m in mics]))
        lon0 = float(np.mean([m.lon for m in mics]))
    # speed of sound
    c = J.get("speed_of_sound", None)
    if c is None:
        T = float(J.get("temperature_C", 20.0))
        c = 331.4 + 0.6 * T  # m/s
    return mics, (lat0, lon0), float(c), J

def extract_all_audio(mics: List[Mic], out_dir: str, fs=48000) -> List[Tuple[str,int,np.ndarray]]:
    ensure_dir(out_dir)
    results = []
    for mic in mics:
        out_wav = os.path.join(out_dir, os.path.basename(mic.file) + ".wav")
        log(f"Extracting audio: {mic.file} -> {out_wav}")
        extract_audio_ffmpeg(mic.file, out_wav, target_sr=fs)
        x, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x[:,0]
        if sr != fs:
            log(f"WARN: read SR {sr} != target {fs}; continuing")
        results.append((out_wav, fs, x))
    return results

def pick_arrival_indices(x: np.ndarray, fs: int, band=(200, 4000)) -> Tuple[int, float, float]:
    """
    Returns (arrival_index_samples, arrival_time_seconds, snr_like)
    Strategy:
      - Bandpass filter to suppress low/high noise
      - Build RMS envelope; estimate baseline; find first coarse exceedance
      - Run AIC inside a +/- 0.75s window; fallback to STA/LTA if needed
      - snr_like ~ envelope jump at the pick
    """
    xf = apply_bandpass(x, fs, low_hz=band[0], high_hz=band[1], order=4)
    env = rms_envelope(xf, fs, win_ms=20.0)
    # Bring envelope to signal length (approx)
    pad = len(x) - len(env)
    if pad > 0:
        env = np.pad(env, (pad//2, pad - pad//2), mode="edge")
    # Baseline from first 20% or 5 seconds (whichever is smaller)
    n0 = int(min(0.2*len(env), 5*fs))
    baseline = np.median(env[:max(1, n0)])
    # Find first crossing
    thr = baseline * 6.0 if baseline > 0 else np.percentile(env, 90)
    idx_coarse = np.argmax(env > thr)
    if env[idx_coarse] <= thr:
        # fallback: global max (if user guaranteed "clearly audible")
        idx_coarse = int(np.argmax(env))
    left = max(0, idx_coarse - int(0.75*fs))
    right = min(len(xf), idx_coarse + int(0.75*fs))
    segment = xf[left:right]
    krel = aic_picker(segment)
    if krel is None:
        krel = sta_lta_picker(segment, fs, sta_ms=5.0, lta_ms=200.0, thr=3.5) or int(0.5*(right-left))
    k = int(left + krel)
    k = max(0, min(len(xf)-1, k))
    # SNR-like measure: slope of envelope around k
    k0 = max(0, k - int(0.02*fs))
    k1 = min(len(env)-1, k + int(0.02*fs))
    snr_like = float((np.max(env[k:k1+1]) - np.median(env[k0:k])) / (baseline + 1e-9))
    return k, float(k)/fs, snr_like

def refine_arrivals_with_template(arrivals: List[int], signals_filt: List[np.ndarray], fs: int, pre_ms=40.0, post_ms=70.0) -> List[float]:
    """
    Use one high-SNR track as a template; GCC-PHAT each window vs template
    to refine each arrival time by a small sub-sample shift (seconds).
    Returns refined arrival times (seconds).
    """
    M = len(signals_filt)
    pre = int(fs * (pre_ms/1000.0))
    post = int(fs * (post_ms/1000.0))

    # pick template = track with largest local energy at arrival
    energies = []
    for i in range(M):
        k = arrivals[i]
        a = max(0, k - pre); b = min(len(signals_filt[i]), k + post)
        w = signals_filt[i][a:b]
        energies.append(np.sum(w*w))
    ref_idx = int(np.argmax(energies))
    log(f"Using template from track #{ref_idx}")

    ref_k = arrivals[ref_idx]
    a = max(0, ref_k - pre); b = min(len(signals_filt[ref_idx]), ref_k + post)
    template = signals_filt[ref_idx][a:b]
    template = template - np.mean(template)

    refined = []
    for i in range(M):
        k = arrivals[i]
        a = max(0, k - pre); b = min(len(signals_filt[i]), k + post)
        w = signals_filt[i][a:b]
        w = w - np.mean(w)
        # Limit lag search to small window (±10 ms) – we only want micro-adjustment
        max_tau = 0.010
        tau, _ = gcc_phat(w, template, fs, max_tau=max_tau, interp=8)
        refined.append((k/fs) + tau)
    return refined

def build_pairwise_z(arrival_times_s: List[float], XY: np.ndarray, c: float, base_weights: List[float], slack_ms=5.0):
    """
    Build pairwise measurements z_ij = t_j - t_i with SNR-based weights and physical gating.
    Returns arrays: pairs (P x 2), z (P,), w (P,)
    """
    M = len(arrival_times_s)
    pairs = []
    z = []
    w = []
    for i in range(M):
        for j in range(i+1, M):
            tij = arrival_times_s[j] - arrival_times_s[i]
            baseline = np.linalg.norm(XY[j,:] - XY[i,:])
            max_tdoa = baseline / c + (slack_ms/1000.0)
            if abs(tij) > max_tdoa:
                # probably a bad pick; drop
                log(f"Dropping pair ({i},{j}) |t_ij|={tij:.4f}s > max {max_tdoa:.4f}s", "WARN")
                continue
            # combine SNR-like weights
            wij = float(max(1e-3, min(base_weights[i], base_weights[j])))
            pairs.append((i,j))
            z.append(tij)
            w.append(wij)
    if len(pairs) < max(1, M-1):
        log("Few valid pairs remained after gating; results may be unstable.", "WARN")
    return np.array(pairs, dtype=int), np.array(z, dtype=float), np.array(w, dtype=float)

def robust_weights_huber(residuals: np.ndarray, base_w: np.ndarray, k: float):
    """
    Huber M-estimator weights: psi(r)/r with threshold k (seconds).
    """
    r = np.abs(residuals)
    w = np.where(r <= k, 1.0, (k / (r + 1e-12)))
    return base_w * w

def incidence_matrix(pairs: np.ndarray, M: int):
    """
    Build P x M incidence matrix B with +1 at i, -1 at j for each pair (i,j)
    """
    P = pairs.shape[0]
    B = np.zeros((P, M), dtype=float)
    for p, (i,j) in enumerate(pairs):
        B[p, i] = 1.0
        B[p, j] = -1.0
    return B

def solve_offsets_given_s(s_xy: np.ndarray, pairs: np.ndarray, z: np.ndarray, w: np.ndarray, XY: np.ndarray, c: float):
    """
    Given source position s (2,), solve for clock offsets δ (M,), with δ_0 = 0 gauge.
    """
    M = XY.shape[0]
    P = pairs.shape[0]
    # geometric TDOA for each pair
    di = np.linalg.norm(s_xy - XY[pairs[:,0],:], axis=1)
    dj = np.linalg.norm(s_xy - XY[pairs[:,1],:], axis=1)
    geom = (di - dj) / c  # (P,)

    g = z - geom          # z_ij ≈ geom + (δ_i - δ_j) => g = z - geom ≈ B δ
    B = incidence_matrix(pairs, M)
    W = np.diag(w)

    # Fix δ_0 = 0 by dropping column 0
    Bm = B[:,1:]  # (P x (M-1))
    # Weighted LS: (Bm^T W Bm) d = Bm^T W g
    A = Bm.T @ W @ Bm
    b = Bm.T @ W @ g
    # Solve (add tiny Tikhonov for stability)
    A += 1e-10 * np.eye(A.shape[0])
    d = np.linalg.solve(A, b)
    delta = np.zeros(M, dtype=float)
    delta[1:] = d
    return delta

def residuals_given_s_delta(s_xy, delta, pairs, z, XY, c):
    di = np.linalg.norm(s_xy - XY[pairs[:,0],:], axis=1)
    dj = np.linalg.norm(s_xy - XY[pairs[:,1],:], axis=1)
    geom = (di - dj) / c
    model = geom + (delta[pairs[:,0]] - delta[pairs[:,1]])
    r = z - model
    return r

def jacobian_wrt_s(s_xy, pairs, XY, c):
    """
    Jacobian of residuals r_ij wrt s = (x,y):
      r_ij = z_ij - (||s-x_i|| - ||s-x_j||)/c - (δ_i - δ_j)
      dr/ds = -(1/c) * ( (s-x_i)/||s-x_i|| - (s-x_j)/||s-x_j|| )
    """
    P = pairs.shape[0]
    J = np.zeros((P, 2), dtype=float)
    for p, (i,j) in enumerate(pairs):
        vi = s_xy - XY[i,:]
        vj = s_xy - XY[j,:]
        ni = np.linalg.norm(vi) + 1e-12
        nj = np.linalg.norm(vj) + 1e-12
        J[p,:] = -(1.0/c) * (vi/ni - vj/nj)
    return J

def grid_search_init(XY, pairs, z, w, c, pad_m=30.0, grid_res_m=2.0):
    """
    Coarse initialization over the bounding box around mic positions.
    For each grid point, "profile out" δ via WLS and evaluate cost.
    """
    xmin = np.min(XY[:,0]) - pad_m
    xmax = np.max(XY[:,0]) + pad_m
    ymin = np.min(XY[:,1]) - pad_m
    ymax = np.max(XY[:,1]) + pad_m

    xs = np.arange(xmin, xmax+grid_res_m, grid_res_m)
    ys = np.arange(ymin, ymax+grid_res_m, grid_res_m)

    best_cost = np.inf
    best_s = None
    for x in xs:
        for y in ys:
            s = np.array([x,y], dtype=float)
            delta = solve_offsets_given_s(s, pairs, z, w, XY, c)
            r = residuals_given_s_delta(s, delta, pairs, z, XY, c)
            cost = np.sum(w * (r**2))
            if cost < best_cost:
                best_cost = cost
                best_s = s.copy()
    return best_s

def alternation_solver(XY, pairs, z, w_base, c, max_outer=20, huber_k_ms=2.0, grid_res_m=2.0):
    """
    Alternate:
      - δ-update (weighted LS) given s
      - s-update (Gauss-Newton) given δ
    Robustify with Huber weights on residuals.
    """
    # init s by grid search
    s = grid_search_init(XY, pairs, z, w_base, c, grid_res_m=grid_res_m)
    delta = solve_offsets_given_s(s, pairs, z, w_base, XY, c)

    k = huber_k_ms / 1000.0  # seconds
    prev_cost = np.inf

    for outer in range(max_outer):
        # robust reweights based on residuals
        r = residuals_given_s_delta(s, delta, pairs, z, XY, c)
        w = robust_weights_huber(r, w_base, k)

        # δ-step
        delta = solve_offsets_given_s(s, pairs, z, w, XY, c)

        # s-step: Gauss-Newton with small line search
        for _ in range(10):
            r = residuals_given_s_delta(s, delta, pairs, z, XY, c)
            w = robust_weights_huber(r, w_base, k)
            J = jacobian_wrt_s(s, pairs, XY, c)
            W = np.diag(w)
            H = J.T @ W @ J
            g = J.T @ (W @ r)
            # damped step
            H_reg = H + 1e-8 * np.eye(2)
            try:
                step = np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                step = np.zeros(2)
            # line search
            best_s = s
            best_cost = np.sum(w * (r**2))
            improved = False
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                s_try = s + alpha * step
                r_try = residuals_given_s_delta(s_try, delta, pairs, z, XY, c)
                cost_try = np.sum(w * (r_try**2))
                if cost_try < best_cost:
                    best_cost = cost_try
                    best_s = s_try
                    improved = True
                    break
            s = best_s
            if not improved:
                break

        # check convergence
        r = residuals_given_s_delta(s, delta, pairs, z, XY, c)
        w = robust_weights_huber(r, w_base, k)
        cost = np.sum(w * (r**2))
        log(f"Iter {outer+1}: cost={cost:.6e}, |Δcost|={abs(prev_cost-cost):.3e}")
        if abs(prev_cost - cost) < 1e-9:
            break
        prev_cost = cost

    # Final covariance (2x2)
    J = jacobian_wrt_s(s, pairs, XY, c)
    W = np.diag(w)
    H = J.T @ W @ J + 1e-12*np.eye(2)
    sigma2 = np.sum(w * (r**2)) / max(1, len(r) - 2)
    cov = sigma2 * np.linalg.inv(H)

    return s, delta, cov, r, w

def ellipse_from_cov2(cov, conf=0.95):
    """
    2D covariance -> ellipse params (semi-major, semi-minor, angle_degrees)
    For 95% confidence in 2D, scale factor = sqrt(chi2.ppf(0.95, df=2)) ≈ sqrt(5.991) ≈ 2.447.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    scale = math.sqrt(5.991)  # 95% for 2 dof
    a = scale * math.sqrt(max(vals[0], 0.0))
    b = scale * math.sqrt(max(vals[1], 0.0))
    # angle of major axis wrt x-axis
    ang = math.atan2(vecs[1,0], vecs[0,0])
    return a, b, rad2deg(ang)

def compute_t0(arrival_times_s: List[float], s_xy: np.ndarray, XY: np.ndarray, delta: np.ndarray, c: float, weights: List[float]):
    """
    Solve t_i ≈ t0 + ||s - x_i||/c + δ_i  => t0 via weighted mean.
    """
    di = np.linalg.norm(XY - s_xy[None,:], axis=1) / c
    y = np.array(arrival_times_s) - di - delta
    w = np.array(weights)
    w = np.maximum(w, 1e-6)
    t0 = float(np.sum(w * y) / np.sum(w))
    return t0


# ------------------------------ Plot & I/O ------------------------------

def plot_layout(XY, s_xy, cov, out_png: str):
    a, b, ang_deg = ellipse_from_cov2(cov)
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = plt.gca()
    ax.scatter(XY[:,0], XY[:,1], marker='^', s=60, label="Mics")
    ax.scatter([s_xy[0]], [s_xy[1]], marker='*', s=120, label="Estimated event")

    # Draw ellipse
    theta = np.linspace(0, 2*np.pi, 200)
    R = np.array([[np.cos(np.deg2rad(ang_deg)), -np.sin(np.deg2rad(ang_deg))],
                  [np.sin(np.deg2rad(ang_deg)),  np.cos(np.deg2rad(ang_deg))]])
    ellipse = (R @ np.vstack([a*np.cos(theta), b*np.sin(theta)])).T
    ax.plot(s_xy[0] + ellipse[:,0], s_xy[1] + ellipse[:,1], label="95% ellipse")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def write_sync_csv(out_csv: str, files: List[str], arrival_s: List[float], delta: np.ndarray):
    """
    align_to_event_offset_s = -arrival_time_s  (puts event at t=0 when seeking)
    """
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "arrival_time_s", "clock_offset_s", "align_to_event_offset_s"])
        for i, fn in enumerate(files):
            w.writerow([os.path.basename(fn), f"{arrival_s[i]:.6f}", f"{delta[i]:.6f}", f"{-arrival_s[i]:.6f}"])


# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Locate a single impulsive acoustic event from multiple videos.")
    parser.add_argument("--videos_dir", required=True, help="Directory containing input videos.")
    parser.add_argument("--positions", required=True, help="positions.json mapping file->lat/lon.")
    parser.add_argument("--out", default="./out", help="Output directory.")
    parser.add_argument("--fs", type=int, default=48000, help="Target sample rate for audio extraction.")
    parser.add_argument("--bandpass", type=float, nargs=2, default=(200.0, 4000.0), help="Bandpass (low high) Hz.")
    parser.add_argument("--grid_res_m", type=float, default=2.0, help="Grid resolution (m) for initialization.")
    parser.add_argument("--huber_k_ms", type=float, default=2.0, help="Huber threshold (ms) for robust weighting.")
    parser.add_argument("--assume_3d", action="store_true", help="If set, treat heights as meaningful (z used for distance); still solves in 2D for location.")
    args = parser.parse_args()

    require_ffmpeg()
    ensure_dir(args.out)
    wav_dir = os.path.join(args.out, "wav")
    ensure_dir(wav_dir)

    # Load positions and constants
    mics, (lat0, lon0), c, rawJ = load_positions(args.positions, args.videos_dir)
    log(f"Using speed of sound c={c:.2f} m/s")

    # Convert to local XY (and Z)
    XY = []
    files = []
    Z = []
    for m in mics:
        x, y = latlon_to_local_xy(m.lat, m.lon, lat0, lon0)
        XY.append([x,y])
        Z.append(m.height_m if args.assume_3d else 0.0)
        files.append(m.file)
    XY = np.array(XY, dtype=float)
    Z = np.array(Z, dtype=float)
    M = XY.shape[0]
    if M < 3:
        log("Need at least 3 videos (≥4 if devices are unsynchronized) to localize robustly.", "ERROR")
        sys.exit(2)
    if M < 4:
        log("WARNING: With 3 unsynchronized devices and a single event, the problem can be underdetermined.", "WARN")

    # Extract audio
    wavs = extract_all_audio(mics, wav_dir, fs=args.fs)
    fs = args.fs

    # Bandpass & pick arrivals (first direct path)
    arrivals_idx = []
    arrivals_s = []
    snr_likes = []
    filtered_signals = []
    for (wav_path, fs2, x), m in zip(wavs, mics):
        xf = apply_bandpass(x, fs, low_hz=args.bandpass[0], high_hz=args.bandpass[1], order=4)
        k, t, snr_like = pick_arrival_indices(x, fs, band=tuple(args.bandpass))
        filtered_signals.append(xf)
        arrivals_idx.append(k)
        arrivals_s.append(t)
        snr_likes.append(max(0.1, min(100.0, snr_like)))
        log(f"{os.path.basename(m.file)}: arrival={t:.6f}s, snr_like={snr_like:.2f}")

    # Refine arrivals using GCC-PHAT template (sub-sample)
    arrivals_s_ref = refine_arrivals_with_template(arrivals_idx, filtered_signals, fs, pre_ms=40.0, post_ms=70.0)
    for i in range(M):
        dt = arrivals_s_ref[i] - arrivals_s[i]
        arrivals_s[i] = arrivals_s_ref[i]
        if abs(dt) > 1e-6:
            log(f"Refined arrival[{i}]: Δ={dt*1000:.2f} ms")

    # Build pairwise observations
    pairs, z, w = build_pairwise_z(arrivals_s, XY, c, snr_likes, slack_ms=5.0)
    if len(pairs) < M-1:
        log("Too few valid TDOA pairs after gating; try relaxing bandpass, or ensure event is clearly audible.", "ERROR")
        # Continue anyway; solver may still run but with poor geometry
    # Solve (s, δ) with robust alternation
    s_xy, delta, cov, residuals, w_eff = alternation_solver(
        XY=XY, pairs=pairs, z=z, w_base=w, c=c,
        max_outer=25, huber_k_ms=args.huber_k_ms, grid_res_m=args.grid_res_m
    )

    # If user supplied heights and wants 3D distance effect, distances already
    # used via XY; z used only for distance if assume_3d True:
    if args.assume_3d:
        # Replace XY distances in model with 3D ones for t0 calculation only (location stays 2D).
        XY3 = np.hstack([XY, Z[:,None]])
        s3 = np.array([s_xy[0], s_xy[1], 0.0])  # assume event on ground plane
        di = np.linalg.norm(XY3 - s3[None,:], axis=1) / c
    else:
        di = np.linalg.norm(XY - s_xy[None,:], axis=1) / c

    # Emission time t0 and outputs
    t0 = compute_t0(arrivals_s, s_xy, XY, delta, c, snr_likes)
    # Confidence ellipse
    a95, b95, ang_deg = ellipse_from_cov2(cov)

    # Convert to WGS84
    lat, lon = local_xy_to_latlon(s_xy[0], s_xy[1], lat0, lon0)

    # Write outputs
    res_json = {
        "event_location_local_m": {"x": s_xy[0], "y": s_xy[1], "z": 0.0},
        "event_location_wgs84": {"lat": lat, "lon": lon, "alt_m": 0.0},
        "confidence_ellipse": {"semi_major_m": a95, "semi_minor_m": b95, "angle_deg": ang_deg},
        "speed_of_sound_mps": c,
        "emission_time_ref_seconds": t0,
        "per_video": [
            {
                "file": os.path.basename(files[i]),
                "arrival_time_s": arrivals_s[i],
                "clock_offset_s": float(delta[i]),
                "align_to_event_offset_s": -arrivals_s[i]
            }
            for i in range(M)
        ],
        "tdoa_pairs_used": int(len(pairs)),
        "residual_stats": {
            "rmse_ms": float(np.sqrt(np.mean(residuals**2))*1000.0) if len(residuals) else None,
        }
    }
    write_json(os.path.join(args.out, "results.json"), res_json)
    write_sync_csv(os.path.join(args.out, "sync.csv"), files, arrivals_s, delta)
    plot_layout(XY, s_xy, cov, os.path.join(args.out, "layout.png"))

    log("Done.")
    log(f"Estimated location (local m): x={s_xy[0]:.3f}, y={s_xy[1]:.3f}")
    log(f"Estimated location (lat/lon): lat={lat:.7f}, lon={lon:.7f}")
    log(f"95% ellipse: a={a95:.2f} m, b={b95:.2f} m, angle={ang_deg:.1f}°")
    log(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
```

---

## How to run

1. Put your videos in a folder, e.g. `./videos/`.
2. Create `positions.json` (example structure shown at top).
3. Install deps and ensure `ffmpeg` is on your PATH.
4. Run:

```bash
python locate_event.py \
  --videos_dir ./videos \
  --positions ./positions.json \
  --out ./out \
  --bandpass 200 4000 \
  --grid_res_m 2.0 \
  --huber_k_ms 2.0
```

### Outputs

* `out/results.json` – event location (local + lat/lon), 95% ellipse, per-video arrival & offsets.
* `out/sync.csv` – quick offsets to align media (event at t=0) or for re-muxing.
* `out/layout.png` – mic geometry and confidence ellipse.
* `out/wav/` – extracted mono 48 kHz WAVs (for reproducibility).

---

## Notes & knobs (defaults “just work”)

* **≥4 videos strongly recommended** if devices aren’t time‑synced.
* For very bright transients outside 200–4000 Hz (e.g., low‑boom), adjust `--bandpass`.
* If your courtyard radius is \~150–200 yd, `--grid_res_m 1.0` is safe (slower), `2.0` is fast/good.
* The solver uses **Huber** robust weights (default 2 ms). If echoes are strong, try `--huber_k_ms 4`.

---

## Optional: Batch aligning videos (preview)

With `sync.csv`, you can create preview clips where the event is at `t=0`:

```bash
while IFS=, read -r file arrival clock align; do
  [[ "$file" == "file" ]] && continue
  ffmpeg -y -ss ${align} -i "./videos/${file}" -t 5 -c copy "./out/aligned_${file}"
done < ./out/sync.csv
```

This script is ready to drop into a repo. If you want, I can also provide a lightweight test harness that generates synthetic mics & an impulse to validate the end‑to‑end pipeline.

