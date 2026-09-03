# Scenario results

Produced by `python generate_test_data.py` followed by `python run_test_scenarios.py`
(seed 0, noise RMS 0.003, reflections on). Errors are 2D position errors against the ground
truth in `metadata.json`; "nearest" is the closest of the best and any alternative solution.

## Synchronised clocks (default)

| Scenario | Recordings used | Error | Nearest | 95% ellipse | Truth inside | Timing RMSE |
|---|---|---|---|---|---|---|
| scenario1_gunshot | 4/4 | 0.00 m | 0.00 m | 0.43 x 0.32 m | yes | 0.000 ms |
| scenario2_explosion | 5/5 | 16.60 m | 0.00 m | 1.48 x 0.29 m | no (mirror) | 0.000 ms |
| scenario3_fireworks | 6/6 | 0.00 m | 0.00 m | 0.73 x 0.33 m | yes | 0.001 ms |

The five explosion cameras lie on one straight line, so a source at x = +8.3 m and its mirror
image at x = -8.3 m produce identical arrival times. The locator reports both and flags the
result `ambiguous`; the one it lists first is a coin toss. Camera heights do not break this
symmetry because all cameras sit in the same vertical plane.

## Clock offsets drawn from N(0, 2 ms)

Generated with `--random_clock_ms 2 --seed 3`, solved with `--clock_sigma_ms 2`:

| Scenario | Error | 95% ellipse | Truth inside | Max offset error |
|---|---|---|---|---|
| scenario1_gunshot | 1.40 m | 1.60 x 1.12 m | yes | 3.5 ms |
| scenario2_explosion | 16.43 m (nearest 0.6 m) | 4.86 x 0.92 m | mirror | 1.2 ms |
| scenario3_fireworks | 0.21 m | 2.76 x 1.29 m | yes | 0.5 ms |

The same data solved with the default synchronised model: gunshot 1.94 m (ellipse 2.6 x 1.8 m,
truth inside), fireworks 0.58 m (ellipse 3.9 x 1.6 m, truth inside), each with a warning that
the residuals are several times the assumed timing noise. One event cannot determine the
offsets themselves; the prior only limits the damage.

## Reading the numbers

- Millimetre errors on synthetic data reflect identical waveforms at every recording. Real
  microphones, reverberation and clipping will limit timing precision to roughly 0.5 to 2 ms,
  i.e. 0.2 to 0.7 m of range difference; set `--timing_sigma_ms` accordingly.
- Accuracy is governed by geometry: inside or near a non-collinear array errors stay small; far
  outside, range becomes poorly determined and the ellipse stretches along the line of sight.
