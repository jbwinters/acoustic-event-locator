# Test suite

```bash
python -m pytest              # all tests (~15 s)
python -m pytest -k solver    # one area
python -m pytest --cov=locate_event --cov-report=term-missing
```

Every test compares against a known ground truth with tolerances that would catch a real
regression (centimetres for positions, tens of microseconds for relative timing, correct
coverage for the 95% ellipse). Randomness is seeded.

| File | Covers |
|---|---|
| `test_geo_positions.py` | Local projection round trip and scale, speed of sound, positions.json parsing (`reference`/`reference_point`, temperature vs `speed_of_sound`, validation), file resolution with WAV fallback |
| `test_dsp.py` | Bandpass design and causality, moving RMS envelope, noise floor, AIC and STA/LTA pickers, onset candidates (ranking, coda merging, no false alarms on noise), fine pick consistency |
| `test_gcc_refine.py` | GCC weightings on fractional delays with and without noise, sign convention, bounded search, peak quality, parabolic interpolation, pairwise refinement recovering relative timing to 30 us and isolating an unrelated recording |
| `test_association.py` | Physical gate matrix, choosing consistent onsets, rejecting louder decoys, dropping inconsistent recordings, pairwise consistency |
| `test_solver.py` | Exact recovery on perfect data, 3 recordings, far sources, heights, determinism; Monte Carlo accuracy and ellipse coverage; outlier rejection including leverage cases and no false rejections; clock prior; mirror ambiguity of a linear array; validation errors; ellipse geometry |
| `test_pipeline.py` | In-memory pipeline on synthetic gunshot/explosion/fireworks (all recordings used, < 5 cm, relative timing < 0.1 ms), decoy sound, silent recording, clock prior, seconds-scale offsets fail clearly, faint events never give a silent wrong answer; command line end to end on WAV (results.json, sync.csv, layout.png, WGS84 round trip), prior mode, video path through a mocked extractor, missing ffmpeg, resampling and stereo input; the three checked-in scenarios generated on the fly |
| `test_io.py` | JSON serialisation of numpy types, sync.csv, results document, plotting, the synthetic generator (waveforms, fractional delay, track rendering, truth, files) |

`helpers.py` holds shared geometries and a click synthesiser; `conftest.py` puts the repo root
on `sys.path` and silences the locator's logging.
