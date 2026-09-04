# Scenario results

Produced by `python generate_test_data.py` followed by `python run_test_scenarios.py`
(seed 0, noise RMS 0.003, reflections on). Errors are 2D position errors against the ground
truth in `metadata.json`; "nearest" is the closest of the best and any alternative solution.

## Synchronized clocks (default)

| Scenario | Recordings used | Error | Nearest | 95% ellipse | Truth inside | Timing RMSE |
|---|---|---|---|---|---|---|
| scenario1_gunshot | 4/4 | 0.00 m | 0.00 m | 0.43 x 0.32 m | yes | 0.000 ms |
| scenario2_explosion | 5/5 | 16.60 m | 0.00 m | 1.48 x 0.29 m | no (mirror) | 0.000 ms |
| scenario3_fireworks | 6/6 | 0.00 m | 0.00 m | 0.73 x 0.33 m | yes | 0.001 ms |
| scenario4_window_shot | 7/7 | 0.05 m | 0.05 m | 0.35 x 0.29 m | yes | 0.057 ms |
| scenario5_urban_canyon | 7/7 (3 flagged occluded) | 0.00 m | 0.00 m | 1.00 x 0.32 m | yes | 0.000 ms |

The five explosion cameras lie on one straight line, so a source at x = +8.3 m and its mirror
image at x = -8.3 m produce identical arrival times. The locator reports both and flags the
result `ambiguous`; the one it lists first is a coin toss. Camera heights do not break this
symmetry because all cameras sit in the same vertical plane.

## Occlusion (scenario 5)

Cameras 3, 4 and 5 sit behind buildings with detours of 7, 4 and 12 m. The locator flags
exactly those three as occluded with implied detours of 6.9, 3.9 and 12.0 m, places the shot
0.00 m from the truth on the four direct-path cameras, and notes that with four direct
recordings an alternative explanation (one more camera occluded by a short detour) is within
the 95% region; it lists it. Variants on the same geometry (6 seeds each): two mild 1.5 to 2 m
detours are identified every time at 0.00 m; four occluded of seven leaves three direct
recordings and a genuinely ambiguous, flagged answer (0.21 m to the nearest reported solution).
`--no_occlusion` also reaches 0.00 m here by rejecting the three late arrivals as outliers, but
says nothing about why they were late.

## Event height solved (`--z prior`, and `--z free`)

| Scenario | Prior | Height error ± std | x, y error | Note |
|---|---|---|---|---|
| scenario1_gunshot | 0 ± 50 m (none given) | -0.01 ± 3.13 m | 0.00 m | cameras nearly coplanar at 2.8 to 4.2 m: the image of the 1.2 m shot above the plane (~5.6 m) is reported as an ambiguous alternative |
| scenario2_explosion | 0 ± 50 m | +0.00 ± 6.12 m | mirror | height weakly observable for a ground-level source |
| scenario3_fireworks | 20 ± 30 m | -0.04 ± 1.97 m | 0.00 m | 25 m burst seen from cameras at 2 to 3 m |
| scenario4_window_shot | 5 ± 10 m | -0.02 ± 0.36 m | 0.05 m | rooftop camera at 22 m gives vertical aperture; four phone heights carried as 1.5 ± 0.5 m priors |
| scenario5_urban_canyon | 1.5 ± 1 m | +0.19 ± 0.96 m | 0.01 m | with three of seven cameras down-weighted as occluded |

`--z free` (prior std 1000 m) gives the same numbers to two decimals: the priors above are not
doing the work, the geometry is.

## Clock offsets drawn from N(0, 2 ms)

Generated with `--random_clock_ms 2 --seed 3`, solved with `--clock_sigma_ms 2`:

| Scenario | Error | 95% ellipse | Truth inside | Max offset error |
|---|---|---|---|---|
| scenario1_gunshot | 1.40 m | 1.60 x 1.12 m | yes | 3.5 ms |
| scenario2_explosion | 16.43 m (nearest 0.6 m) | 4.86 x 0.92 m | mirror | 1.2 ms |
| scenario3_fireworks | 0.21 m | 2.76 x 1.29 m | yes | 0.5 ms |

The same data solved with the default synchronized model: gunshot 1.94 m (ellipse 2.6 x 1.8 m,
truth inside), fireworks 0.58 m (ellipse 3.9 x 1.6 m, truth inside), each with a warning that
the residuals are several times the assumed timing noise. One event cannot determine the
offsets themselves; the prior only limits the damage.

## Reading the numbers

- Millimeter errors on synthetic data reflect identical waveforms at every recording. Real
  microphones, reverberation and clipping will limit timing precision to roughly 0.5 to 2 ms,
  i.e. 0.2 to 0.7 m of range difference; set `--timing_sigma_ms` accordingly.
- Accuracy is governed by geometry: inside or near a non-collinear array errors stay small; far
  outside, range becomes poorly determined and the ellipse stretches along the line of sight.
