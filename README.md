# Acoustic Event Locator

Locate a single impulsive sound (gunshot, explosion, firework burst) from audio recorded by
several spatially separated devices, using time-difference-of-arrival (TDOA) multilateration.
Input is a directory of recordings (video files via ffmpeg, or WAV/FLAC directly) and a JSON file
with each device's GPS position. Output is the event position in local meters and WGS84, a 95%
confidence ellipse, per-recording arrival times, and the offsets that align every recording on
the event.

![Demo](docs/demo_diagram.svg)

The diagram is the output of an actual run: `python docs/generate_demo_diagram.py` synthesizes
the fireworks scenario with a fixed seed, runs the locator on it, and draws what came back (the
estimate with its 95% ellipse to scale and zoomed, the solved height against the truth, and
every recording's arrival, onset SNR and residual).

### How it works, animated

![How TDOA localization works](docs/anim/how_it_works_2d.gif)

Four acts on a 20 m square of four microphones: the wavefront reaches each microphone at a
different time (right: the band-passed recordings with the refined arrival picks); each pair's
time difference confines the source to a hyperbola; the misfit surface is searched on a grid
and refined by Levenberg-Marquardt; the result is reported with its 95% ellipse. Every frame
is computed by the same generator and locator code the tests run
(`python docs/make_animations.py`). Per-scenario animations are under
[Scenario walkthroughs](#scenario-walkthroughs).

## What it does, and what it cannot do

- With **time-synchronized recordings** and a non-collinear array the estimate is accurate to
  centimeters on synthetic data (see [Accuracy](#accuracy)) and the reported ellipse is calibrated.
- With recordings whose clocks disagree by a known amount (say a few milliseconds of NTP jitter)
  you can supply that as a prior (`--clock_sigma_ms`); the position error and the ellipse grow
  accordingly.
- **One event cannot synchronize unsynchronized recordings.** For any candidate position there
  is a set of per-device clock offsets that fits the arrivals exactly, so the offsets and the
  position cannot both be determined from a single event. Recordings started by hand on
  different phones (offsets of seconds) cannot be localized with this tool; it says so instead
  of guessing. `sync.csv` still gives the seek offset that aligns each recording on the event.
  Joint estimation from several events is the way to lift this and is listed under
  [Limitations](#limitations-and-roadmap).
- Cameras in a straight line produce a mirror-image ambiguity. Both solutions are reported and
  the result is flagged `ambiguous`.
- **Height is optional and honest.** The event height can be fixed, given a prior, or solved
  freely (`--source_height_sigma_m`), and any recording's height can be marked uncertain
  (`height_sigma_m`). Whether the height is actually observable depends on the vertical
  aperture of the cameras, and the reported standard deviation says so. See
  [Heights and 3D](#heights-and-3d).

## Quick start

```bash
git clone <this repository> && cd acoustic-event-locator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt        # numpy, scipy, soundfile, matplotlib, pytest

python generate_test_data.py                # synthetic scenarios (WAV; no ffmpeg needed)
python locate_event.py --videos_dir test_data/scenario1_gunshot \
                       --positions test_data/scenario1_gunshot/positions.json --out out/gunshot
python run_test_scenarios.py                # score all scenarios against their ground truth
python -m pytest                            # 157 tests, ~45 s
```

ffmpeg is only needed to read video files. If it is missing, generate the test data as WAV
(the default when ffmpeg is absent) and point `--videos_dir` at WAV recordings.

## Inputs

**Recordings**: anything ffmpeg can decode (MP4, MOV, MKV, ...) or WAV/FLAC/OGG. Any sample rate
and channel count; audio is mixed to mono and resampled to `--fs` (default 48 kHz). If a file
listed in `positions.json` is missing but a `.wav` with the same stem exists, the WAV is used.

**positions.json**

```json
{
  "temperature_C": 20.0,
  "speed_of_sound": null,
  "reference": {"lat": 41.881832, "lon": -87.623177},
  "mics": [
    {"file": "cam1.mp4", "lat": 41.881832, "lon": -87.623177, "height_m": 3.0},
    {"file": "cam2.mp4", "lat": 41.881832, "lon": -87.622977, "height_m": 3.5},
    {"file": "cam3.mp4", "lat": 41.881632, "lon": -87.622977, "height_m": 2.8},
    {"file": "cam4.mp4", "lat": 41.881632, "lon": -87.623177, "height_m": 4.2}
  ]
}
```

- `reference` (or `reference_point`) is the origin of the local x-east/y-north frame; default is
  the centroid of the recordings. Results are reported in this frame and in WGS84.
- `height_m` is used in the 3D distance to the event. Add `height_sigma_m` when a height is only
  roughly known (a hand-held phone: `1.5` with `height_sigma_m: 0.5`); that height is then
  estimated jointly under a prior instead of being trusted. Ignoring a 6 m camera height on a
  20 m array moves the estimate by ~0.6 m.
- The event height is `--source_height_m` (default 0, fixed). With `--source_height_sigma_m S`
  it becomes the mean of a prior with std S and the height is solved within
  `--source_height_bounds` (default 0 to 5000 m, which excludes the mirror image below a
  horizontal camera plane).
- The speed of sound comes from `speed_of_sound` if given, else from `temperature_C`.
- At least 3 recordings are needed; 4 or more give the redundancy required to check the fit
  and reject a bad arrival.

## Outputs (`--out`, default `./out`)

- `results.json`: `event_location_local_m`, `event_location_wgs84`, `confidence_ellipse_95`,
  `position_std_m`, `emission_time_s`, `clock_model`, `fit` (chi-square, degrees of freedom,
  uncertainty scale, convergence, ambiguity flag, alternative solutions, rejected recordings),
  `per_recording` (arrival time, clock offset, alignment offset, onset SNR, residual, weight,
  notes), `refinement` (pairwise cross-correlation lags and quality), `warnings`, `parameters`.
- `sync.csv`: one row per recording with `arrival_time_s`, `clock_offset_s`,
  `align_to_event_offset_s` (seek offset that puts the event at t = 0) and the residual.
- `layout.png`: recordings, estimate, 95% ellipse, alternative minima, unused recordings; when
  any height is solved a second panel shows the elevation view with the height prior, the
  estimate and per-recording height error bars.
- `results.json` also carries `height_model`: the source height prior, estimate and std, and the
  same for every recording whose height was uncertain.
- `wav/`: the mono audio actually analyzed.

Read the `warnings` list. The locator prefers a clear failure or a flagged result to a silent
wrong answer; the flags that matter are `ambiguous`, `at_search_boundary`, `degenerate`, a
relaxed detection threshold, and an uncertainty scale well above 1.

## How it works

1. **Audio**: load or extract, mix to mono, resample, causal Butterworth bandpass
   (`--bandpass`, default 200 to 4000 Hz). A causal filter is used because zero-phase filtering
   pre-rings and biases first-arrival picks early.
2. **Onset detection**: a short trailing RMS envelope (`--env_ms`, 2 ms) against a robust noise
   floor (20th percentile of the envelope). Every burst exceeding `--min_snr` times the floor
   becomes a candidate; a burst's coda (crackles, echoes within `--merge_gap_s`) is merged into
   it. White noise alone peaks at about 2x the floor, so the default 4x has a 2x margin.
3. **Association**: one candidate per recording is chosen so that all chosen arrivals are
   mutually consistent with the geometry (|t_i - t_j| <= d_ij / c + slack). A louder unrelated
   sound in one recording is rejected in favor of the consistent onset; a recording with no
   consistent onset is excluded and reported.
4. **Fine pick**: an AIC change-point picker in a window around each candidate.
5. **Refinement**: pairwise band-limited, regularized GCC-PHAT between recordings with parabolic
   sub-sample interpolation, fused by weighted least squares into consistent arrival times.
   Pairs without a clear correlation peak are ignored and reported.
6. **Solve**: `t_i = t0 + |s - x_i| / c + delta_i`. Vectorized grid search over a wide area,
   multi-start Levenberg-Marquardt with Huber reweighting, bounded to the search area. A bad
   arrival is rejected when at least 4 recordings remain, either from its residual or from a
   leave-one-out check that catches an arrival the fit would otherwise absorb by moving the
   source. Distinct minima outside the 95% ellipse of the best solution are reported as
   alternatives.
7. **Uncertainty**: covariance from the full Fisher matrix (position, emission time, offsets)
   with per-recording timing sigmas (`--timing_sigma_ms`, scaled up for weaker onsets), inflated
   by the reduced chi-square when the residuals exceed the assumed timing noise.

## Heights and 3D

The model is fully 3D; what changes between cases is how many height parameters are free and
how well the geometry constrains them. All four scenarios below use the same solver.

| Scenario | Cameras | Event height, prior | Recovered height | x, y error |
|---|---|---|---|---|
| Window shot (scenario 4) | 4 phones at 1.5 ± 0.5 m, doorbell 1.4 m, mast 6 m, rooftop 22 m | 9 m, prior 5 ± 10 m | -0.02 ± 0.36 m | 0.05 m |
| Fireworks (scenario 3) | 6 cameras at 1.8 to 3 m, burst 25 m up | 25 m, prior 20 ± 30 m | -0.04 ± 1.97 m | 0.00 m |
| Gunshot (scenario 1) | 4 cameras at 2.8 to 4.2 m, shot at 1.2 m | free | -0.01 ± 3.1 m, mirror above the camera plane flagged | 0.00 m |

What to expect:
- A source well above or below the cameras is 3D-locatable from a flat array once timing is
  sub-millisecond; one camera with real height (a rooftop) shrinks the height uncertainty by
  a factor of three to five.
- A source at the same height as the cameras is not: every arrival responds to height the
  same way, which is absorbed by the emission time. The std reported for z is large and the
  x, y solution is unaffected.
- Cameras all in one horizontal plane cannot tell a source h meters above the plane from one h
  meters below it. The default height bounds remove the underground image; if both are
  physically possible the second appears under `alternatives`.
- An uncertain camera height only affects that camera's arrival, by
  `|z_event - z_camera| / (c d)` seconds per meter. For a ground-level shot and phones at
  chest height that is about 0.15 ms per meter, so ±0.5 m is harmless; for a 25 m aerial burst
  it is 1.5 to 1.9 ms per meter, and a camera whose height is unknown to a building storey
  contributes almost nothing. Giving the prior lets the solver weight it correctly instead of
  being biased by it.

Reproduce: `python run_test_scenarios.py --z prior` (or `--z free`, `--z truth`).

## Clock synchronization

| Situation | Setting | What you get |
|---|---|---|
| Devices share a clock (common recorder, GPS/PTP-disciplined, or aligned beforehand) | default (`--clock_sigma_ms 0`) | Full accuracy; residuals validate the fit |
| Clocks agree to within a known jitter S ms | `--clock_sigma_ms S` | MAP estimate; ellipse widens with S. On a 20 m array, S = 2 ms costs about 1 m |
| Clocks unknown by more than the array's propagation time | no setting helps | Clear error; use `sync.csv` to align recordings on the event |

If you assume synchronized clocks but they are not, the misfit shows up as a large reduced
chi-square and an inflated ellipse that still covers the truth in the tested cases, plus a
warning.

## Accuracy

All figures below are from the included synthetic generator (exact fractional-sample delays,
1/r spreading, independent noise, random early reflections). Synthetic waveforms are identical
at every recording apart from noise and echoes, which makes cross-correlation more precise than
it will be with real microphones in reverberant spaces; for real data set `--timing_sigma_ms`
to what your picks actually achieve (0.5 to 2 ms is typical) so the ellipse stays honest.

**Included scenarios** (`python run_test_scenarios.py`, synchronized clocks):

| Scenario | Array | Recordings | Error | 95% ellipse |
|---|---|---|---|---|
| Gunshot, urban intersection | 17 x 22 m square | 4 | 0.00 m | 0.43 x 0.32 m |
| Explosion, factory fence | 88 m straight line | 5 | 0.00 m to the nearest of two mirror solutions, flagged ambiguous | 1.5 x 0.3 m |
| Fireworks, 25 m aerial burst | L-shape, 50 x 44 m | 6 | 0.00 m | 0.73 x 0.33 m |
| Window shot, 9 m up, mixed cameras | 60 x 55 m, heights 1.2 to 22 m | 7 | 0.05 m | 0.35 x 0.29 m |

With clock offsets drawn from N(0, 2 ms) and `--clock_sigma_ms 2`: gunshot 1.40 m, fireworks
0.21 m, both inside their ellipses.

### Scenario walkthroughs

Each animation runs the real pipeline on that scenario's synthetic recordings: the wavefront
from the true source, the picks appearing in each recording as it is reached, then the estimate
with its 95% ellipse, any alternative solution, and (3D scenarios) the elevation view with the
height prior and the solved height.

| Gunshot, four cameras on an intersection | Explosion, five cameras in a straight line |
|---|---|
| ![scenario 1](docs/anim/scenario1_gunshot.gif) | ![scenario 2](docs/anim/scenario2_explosion.gif) |
| **Fireworks 25 m up, height solved from a 20 ± 30 m prior** | **Shot from a 9 m window, phones with 1.5 ± 0.5 m heights plus surveyed cameras** |
| ![scenario 3](docs/anim/scenario3_fireworks.gif) | ![scenario 4](docs/anim/scenario4_window_shot.gif) |

**Signal-to-noise** (gunshot inside a 20 m square, 4 recordings, 12 trials per row):

| Peak SNR in band | Median error | 90th pct | Solved |
|---|---|---|---|
| 35 to 48 dB | 0.000 m | 0.000 m | 12/12 |
| 24 to 37 dB | 0.001 m | 0.001 m | 12/12 |
| 18 to 31 dB | 0.001 m | 0.003 m | 12/12 |
| 15 to 28 dB | 0.002 m | 0.004 m | 11/12 |
| 10 to 23 dB | 0.004 m | 0.006 m | 4/12 (detection floor) |

**Distance** (same array, source moved outward, 6 trials per row):

| Distance from array | Median error | Ellipse semi-major | Solved |
|---|---|---|---|
| 10 m | 0.00 m | 0.7 m | 6/6 |
| 30 m | 0.00 m | 5.8 m | 6/6 |
| 60 m | 0.02 m | 17 m | 6/6 |
| 100 m | 0.11 m | 45 m | 6/6 |
| 200 m | 0.88 m (one 185 m miss) | 170 m | 5/6 |
| 400 m | not detected | | 0/6 |

Far outside the array the bearing is well determined and the range is not; the ellipse says so.
A wider array, not a better algorithm, is what fixes that.

Runtime is about 0.1 s for six 10 s recordings after audio loading.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--fs` | 48000 | Working sample rate |
| `--bandpass LOW HIGH` | 200 4000 | Analysis band (Hz) |
| `--env_ms` | 2.0 | Onset-detection RMS window |
| `--min_snr` | 4.0 | Onset must exceed this multiple of the noise floor; relaxed stepwise to 3.0 if fewer than 3 recordings trigger, with a consistency gate |
| `--merge_gap_s` | 0.5 | Bursts closer than this to a previous burst are treated as its coda |
| `--slack_ms` | 5 | Extra tolerance on the physical arrival gate |
| `--clock_sigma_ms` | 0 | Prior std of clock offsets; 0 = synchronized |
| `--source_height_m` | 0 | Event height in the same datum as `height_m`; fixed unless a sigma is given |
| `--source_height_sigma_m` | 0 | Prior std of the event height; > 0 solves the height |
| `--source_height_bounds MIN MAX` | 0 5000 | Allowed height range when solving it |
| `--timing_sigma_ms` | 0.5 | Assumed timing noise of the strongest recording |
| `--huber_k_ms` | 2 | Residuals beyond this are down-weighted; 3x this rejects |
| `--search_radius_m` | max(200, 3x array extent) | Search area beyond the array |
| `--grid_res_m` | auto | Initial grid resolution |
| `--gcc_weight` | phat | `phat`, `cc` or `scot` weighting for refinement |
| `--no_refine` | | Skip cross-correlation refinement (AIC picks only) |
| `--verbose` | | Debug logging |

## Troubleshooting

- **"only N recording(s) have a mutually consistent onset"**: the event is inaudible in some
  recordings, the positions are wrong, or the clocks differ by more than the propagation time.
  Check `--verbose` output for the candidates found per recording.
- **"onsets were only found at a relaxed detection threshold and they are not mutually
  consistent"**: the event is too faint; try a narrower `--bandpass` around its energy or drop
  the weak recordings.
- **"ambiguous geometry"**: cameras are (nearly) collinear; both solutions are in `results.json`.
  Add a recording off the line.
- **"solution sits at the edge of the search area"**: the arrivals do not pin down a location.
  Only raise `--search_radius_m` if the event really was that far away.
- **"residuals are Nx larger than the assumed timing noise"**: picks are noisier than
  `--timing_sigma_ms`, the clocks are not synchronized, or `--source_height_m` is wrong. The
  ellipse has been inflated to match.
- **ffmpeg errors**: install ffmpeg or convert the recordings to WAV.

## Testing

```bash
python -m pytest                    # full suite
python -m pytest tests/test_solver.py -q
python -m pytest --cov=locate_event --cov-report=term-missing
```

The suite asserts against ground truth with centimeter and sub-0.1 ms tolerances: geometry,
filters and pickers, cross-correlation, association, the solver (perfect data, Monte Carlo
coverage of the 95% ellipse, outliers, clock prior, mirror ambiguity, heights, validation),
event-height and recording-height priors (recovery, 3D coverage, the below-plane mirror,
vertical aperture), the in-memory pipeline, and the command line end to end on WAV input,
including the four checked-in scenarios generated on the fly. See [tests/README.md](tests/README.md).

## Limitations and roadmap

- Single event only. Estimating clock offsets jointly from several events (fireworks shows,
  multiple shots) would make truly unsynchronized recordings usable; the solver's parameter
  layout allows it but the detector and association would need to handle multiple events.
- Height observability depends on vertical aperture; with cameras all near one height the
  solved height of a ground-level event is uninformative (the std says so).
- No wind or temperature-gradient model; the speed of sound is a single number.
- Not validated on real recordings. Clipping, automatic gain control, microphone directivity
  and reverberation will degrade timing precision below the synthetic figures above.

## Acknowledgments

GCC-PHAT after Knapp and Carter (1976); Huber M-estimation (1981); AIC onset picking after
Maeda (1985).

## License

This project is licensed under the MIT License.
