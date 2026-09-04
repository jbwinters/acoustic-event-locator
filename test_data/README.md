# Synthetic test scenarios

Each scenario directory holds a `positions.json` (recording positions plus an `event` block
with the true location) and, after generation, one audio track per recording and a
`metadata.json` with the ground truth. Tracks are not checked in; generate them with

```bash
python generate_test_data.py                         # all scenarios, WAV (MP4 if ffmpeg is on PATH)
python generate_test_data.py --scenarios scenario1_gunshot --format wav --seed 7
python generate_test_data.py --random_clock_ms 2     # unsynchronized clocks, N(0, 2 ms)
python generate_test_data.py --noise_rms 0.01        # 10 dB more background noise
```

and score the locator against them with

```bash
python run_test_scenarios.py                         # synchronized clocks
python run_test_scenarios.py --clock_sigma_ms 2      # after generating with --random_clock_ms 2
python run_test_scenarios.py --z prior               # solve the event height with each scenario's prior
```

In `positions.json`, `height_m` is what the locator sees (the prior mean when `height_sigma_m`
is set); `true_height_m` is used only by the generator. `event.height_prior` is the prior the
scorer passes with `--z prior`.

| Scenario | Event | Array | Notes |
|---|---|---|---|
| `scenario1_gunshot` | gunshot, 1.2 m high | 4 cameras on the corners of a 17 x 22 m intersection | good geometry, source off-center |
| `scenario2_explosion` | explosion at ground level | 5 cameras in an 88 m straight line | collinear: the mirror solution is reported as an alternative |
| `scenario3_fireworks` | aerial burst 25 m up | 6 cameras in an L (50 x 44 m) | event height solved from a 20 ± 30 m prior |
| `scenario5_urban_canyon` | gunshot at street level | 7 cameras around a block, 3 of them behind buildings (`occlusion_detour_m` 4, 7 and 12 m) | occlusion: late, low-passed arrivals must be identified, not averaged in |
| `scenario4_window_shot` | gunshot from a window 9 m up | 4 phones (height 1.5 ± 0.5 m, true 1.2 to 1.8 m), doorbell 1.4 m, mast 6 m, rooftop 22 m | uncertain recording heights (`height_sigma_m`, `true_height_m`) plus a 5 ± 10 m event-height prior |

## What the generator models

- exact fractional-sample propagation delays from the true source to each recording
- 1/r amplitude spreading, so signal-to-noise ratio falls with distance
- independent white background noise per recording (`--noise_rms`, default 0.003 ≈ -50 dBFS)
- per-recording clock offsets (`--clock_offsets_ms` or `--random_clock_ms`)
- one to three early reflections per recording with random delay (8 to 60 ms) and gain
- occlusion per recording (`occlusion_detour_m`): arrival delayed by the detour, waveform
  low-passed at 1.2 kHz and attenuated 6 dB
- event waveforms: gunshot (broadband decaying burst plus low-frequency push), explosion
  (shock front plus 40 to 1500 Hz rumble), fireworks (report followed by a second of crackles)

Not modeled: wind, temperature gradients, microphone directivity, clipping and automatic gain
control. Waveforms are identical at every recording apart from noise and echoes.

## metadata.json

`source_position_m` (x east, y north in the same local frame the locator uses, i.e. relative to
`reference`/`reference_point`, else the centroid), `source_height_m`, `source_latlon`,
`microphone_positions_m` (true x, y, z), `microphone_height_prior_m`, `microphone_height_sigma_m`,
`height_prior_m`, `occlusion_detour_m`, `clock_offsets_s`, `arrival_times_s` (true arrival of the
event in each track, on that recording's own clock), `emission_time_s`, `distances_m`,
`snr_db` (peak signal over noise RMS), `reflections`, `speed_of_sound_ms`, `sample_rate_hz`,
`duration_s`, `files`, `format`, `seed`.

Current results are in [RESULTS.md](RESULTS.md).
