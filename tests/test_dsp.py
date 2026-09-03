import numpy as np
import pytest

import locate_event as le
from helpers import FS, click_track


class TestFilters:
    def test_invalid_band_raises(self, fs):
        with pytest.raises(le.LocatorError):
            le.design_bandpass(5000.0, 4000.0, fs)
        with pytest.raises(le.LocatorError):
            le.design_bandpass(3900.0, 30000.0, 8000)  # high clamps to 0.95*nyquist, below low

    def test_causal_filter_has_no_pre_ringing(self, fs):
        x = np.zeros(fs)
        x[fs // 2] = 1.0
        y = le.apply_bandpass(x, fs)
        assert np.all(y[: fs // 2] == 0.0)
        assert np.max(np.abs(y[fs // 2 :])) > 0.01

    def test_zero_phase_filter_pre_rings(self, fs):
        # documents why the pipeline uses the causal filter for onset picking
        x = np.zeros(fs)
        x[fs // 2] = 1.0
        y = le.apply_bandpass(x, fs, zero_phase=True)
        assert np.max(np.abs(y[: fs // 2])) > 1e-3

    def test_passband_gain_and_stopband_rejection(self, fs):
        t = np.arange(fs) / fs
        for f0, keep in [(1000.0, True), (20.0, False), (12000.0, False)]:
            y = le.apply_bandpass(np.sin(2 * np.pi * f0 * t), fs, 200.0, 4000.0)
            rms = np.sqrt(np.mean(y[fs // 2 :] ** 2))
            assert (rms > 0.6) if keep else (rms < 0.05)


class TestEnvelope:
    def test_length_and_constant_signal(self, fs):
        x = np.full(1000, 0.3)
        env = le.rms_envelope(x, fs, win_ms=1.0)
        assert env.shape == x.shape
        assert np.allclose(env, 0.3)

    def test_causal_rise_at_onset(self, fs):
        x = np.zeros(2000)
        x[1000:] = 1.0
        env = le.rms_envelope(x, fs, win_ms=1.0)  # 48-sample window
        assert np.all(env[:1000] == 0.0)
        assert env[1000] > 0 and abs(env[1047] - 1.0) < 1e-9

    def test_noise_floor_tracks_sigma(self, fs, rng):
        x = 0.02 * rng.standard_normal(5 * fs)
        env = le.rms_envelope(x, fs, win_ms=2.0)
        fl = le.noise_floor(env)
        assert 0.8 * 0.02 < fl < 1.0 * 0.02

    def test_noise_floor_ignores_short_event(self, fs, rng):
        x = 0.01 * rng.standard_normal(5 * fs)
        x[2 * fs : 2 * fs + fs // 2] += 0.5  # half a second of loud signal
        env = le.rms_envelope(x, fs, win_ms=2.0)
        assert le.noise_floor(env) < 0.012


class TestPickers:
    @pytest.mark.parametrize("seed", range(5))
    def test_aic_finds_variance_change(self, seed):
        rng = np.random.default_rng(seed)
        x = np.concatenate([0.01 * rng.standard_normal(4000), 0.1 * rng.standard_normal(4000)])
        k = le.aic_picker(x)
        assert abs(k - 4000) <= 5

    def test_aic_short_input(self):
        assert le.aic_picker(np.zeros(10)) is None

    def test_sta_lta_trigger(self, fs, rng):
        x = 0.01 * rng.standard_normal(fs)
        x[fs // 2 :] += 0.3 * rng.standard_normal(fs - fs // 2)
        k = le.sta_lta_picker(x, fs)
        assert k is not None and 0 <= k - fs // 2 <= int(0.001 * fs)
        assert le.sta_lta_picker(0.01 * rng.standard_normal(fs), fs) is None
        assert le.sta_lta_picker(np.zeros(100), fs) is None


class TestOnsetCandidates:
    def test_two_events_ranked_and_timed(self, fs, rng):
        x = 0.003 * rng.standard_normal(6 * fs)
        x += click_track(fs, 6.0, 1.5, amp=0.2, rng=rng)
        x += click_track(fs, 6.0, 4.0, amp=0.5, rng=rng)
        xf = le.apply_bandpass(x, fs)
        env = le.rms_envelope(xf, fs, 2.0)
        cands = le.find_onset_candidates(env, fs, le.noise_floor(env), min_ratio=4.0, merge_gap_s=0.5)
        assert len(cands) == 2
        assert cands[0][1] > cands[1][1]
        assert abs(cands[0][0] / fs - 4.0) < 0.002 and abs(cands[1][0] / fs - 1.5) < 0.002

    def test_coda_merged_into_one_candidate(self, fs, rng):
        x = 0.003 * rng.standard_normal(4 * fs)
        for dt in (0.0, 0.1, 0.25, 0.4):  # burst followed by crackles inside the merge gap
            x += click_track(fs, 4.0, 2.0 + dt, amp=0.4 if dt == 0 else 0.2, rng=rng)
        xf = le.apply_bandpass(x, fs)
        env = le.rms_envelope(xf, fs, 2.0)
        cands = le.find_onset_candidates(env, fs, le.noise_floor(env), min_ratio=4.0, merge_gap_s=0.5)
        assert len(cands) == 1
        assert abs(cands[0][0] / fs - 2.0) < 0.002

    @pytest.mark.parametrize("seed", range(4))
    def test_no_false_alarms_on_noise(self, fs, seed):
        rng = np.random.default_rng(seed)
        x = 0.003 * rng.standard_normal(10 * fs)
        xf = le.apply_bandpass(x, fs)
        env = le.rms_envelope(xf, fs, 2.0)
        fl = le.noise_floor(env)
        assert env.max() / fl < 3.0
        assert le.find_onset_candidates(env, fs, fl, min_ratio=4.0) == []

    def test_empty_when_silent(self, fs):
        env = np.zeros(1000)
        assert le.find_onset_candidates(env, fs, 1e-12, min_ratio=4.0) == []


class TestFinePick:
    def test_pick_is_at_or_just_after_true_onset(self, fs):
        # causal filtering can only delay an onset; the AIC pick must land within ~1 ms after it
        errs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            onset = 2.0 + rng.uniform(0, 0.01)
            x = click_track(fs, 4.0, onset, amp=0.4, noise=0.003, rng=rng)
            xf = le.apply_bandpass(x, fs)
            env = le.rms_envelope(xf, fs, 2.0)
            coarse = le.find_onset_candidates(env, fs, le.noise_floor(env), 4.0)[0][0]
            k = le.fine_pick(xf, fs, coarse)
            errs.append(k / fs - onset)
        errs = np.array(errs) * 1000
        assert np.all(errs > -0.3) and np.all(errs < 1.5)
        assert np.ptp(errs) < 0.5  # consistent across recordings, which is what TDOA needs

    def test_fallback_to_coarse_when_window_empty(self, fs):
        xf = np.zeros(fs)
        assert le.fine_pick(xf, fs, 1000) in range(0, 1001 + int(0.02 * fs))
