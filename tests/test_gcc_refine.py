import numpy as np
import pytest

import locate_event as le
from helpers import click_track


def _shift(x, d):
    f = np.fft.rfftfreq(len(x))
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d), n=len(x))


@pytest.fixture
def ref_window(fs, rng):
    x = click_track(fs, 0.2, 0.05, amp=1.0, rng=rng)
    return le.apply_bandpass(x, fs)[int(0.02 * fs) : int(0.15 * fs)]


class TestGCC:
    @pytest.mark.parametrize("weighting", ["cc", "phat", "scot"])
    @pytest.mark.parametrize("delay", [0.0, 0.37, -2.5, 7.8])
    def test_fractional_delay_noise_free(self, fs, ref_window, weighting, delay):
        sig = _shift(ref_window, delay)
        tau, quality, lags, cc = le.gcc(sig, ref_window, fs, max_tau=0.01, weighting=weighting, band=(200, 4000))
        assert abs(tau * fs - delay) < (0.05 if weighting != "cc" else 0.15)
        # whitened correlations have a sharp, unambiguous peak; plain CC keeps the click's own
        # (broad, oscillating) autocorrelation so its peak ratio is inherently lower
        assert quality > (3.0 if weighting != "cc" else 1.5)

    @pytest.mark.parametrize("weighting", ["cc", "phat"])
    def test_with_noise(self, fs, ref_window, rng, weighting):
        errs = []
        for delay in (0.2, 1.6, 4.4):
            sig = _shift(ref_window, delay) + 0.02 * rng.standard_normal(len(ref_window))
            tau, *_ = le.gcc(sig, ref_window, fs, max_tau=0.01, weighting=weighting, band=(200, 4000))
            errs.append(abs(tau * fs - delay))
        assert max(errs) < 0.3

    def test_sign_convention(self, fs, ref_window):
        tau, *_ = le.gcc(_shift(ref_window, 12.0), ref_window, fs, max_tau=0.01)
        assert tau > 0  # sig lags ref -> positive tau

    def test_max_tau_bounds_search(self, fs, ref_window):
        # a much stronger copy far away must not be picked when the search window excludes it
        sig = _shift(ref_window, 3.0) + 5.0 * _shift(ref_window, 500.0)
        tau_free, *_ = le.gcc(sig, ref_window, fs, max_tau=None, weighting="phat", band=(200, 4000))
        tau, *_ = le.gcc(sig, ref_window, fs, max_tau=0.002, weighting="phat", band=(200, 4000))
        assert abs(tau_free * fs - 500.0) < 0.5  # unconstrained search finds the strong copy
        assert abs(tau * fs - 3.0) < 0.1  # bounded search finds the near one

    def test_quality_low_for_unrelated_noise(self, fs, rng):
        a = rng.standard_normal(4000)
        b = rng.standard_normal(4000)
        _, quality, _, _ = le.gcc(a, b, fs, max_tau=0.01, weighting="phat", band=(200, 4000))
        assert quality < 1.6

    def test_unknown_weighting(self, fs, ref_window):
        with pytest.raises(ValueError):
            le.gcc(ref_window, ref_window, fs, weighting="bogus")

    def test_quadratic_subsample_peak(self):
        y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        x, v = le.quadratic_subsample_peak(y, 2)
        assert abs(x - 2.0) < 1e-12 and abs(v - 1.0) < 1e-12
        y = -((np.arange(5) - 2.3) ** 2)
        x, _ = le.quadratic_subsample_peak(y, 2)
        assert abs(x - 2.3) < 1e-9
        assert le.quadratic_subsample_peak(y, 0) == (0.0, y[0])


class TestPairwiseRefinement:
    def _tracks(self, fs, rng, true_onsets, noise=0.003, amp=0.5):
        # same click waveform in every recording (rng seed 0), independent noise per recording
        return [le.apply_bandpass(click_track(fs, 3.0, t, amp=amp, noise=noise, rng=np.random.default_rng(0),
                                              noise_rng=np.random.default_rng(100 + i)), fs)
                for i, t in enumerate(true_onsets)]

    def test_recovers_relative_timing_from_perturbed_picks(self, fs, rng):
        true = np.array([1.0, 1.0123, 1.0051, 1.0207])
        tracks = self._tracks(fs, rng, true)
        picks = (true * fs).astype(int) + np.array([3, -4, 6, -2])  # deliberately wrong by a few samples
        out = le.refine_arrivals_pairwise(picks, tracks, fs, weighting="phat", band=(200, 4000))
        rel_est = out["arrival_s"] - out["arrival_s"][out["ref_idx"]]
        rel_true = true - true[out["ref_idx"]]
        assert np.max(np.abs(rel_est - rel_true)) * 1000 < 0.03  # 30 microseconds
        assert out["used_pairs"].all()
        assert out["eps_s"][out["ref_idx"]] == 0.0

    def test_unrelated_track_is_left_uncorrected(self, fs, rng):
        true = np.array([1.0, 1.0123, 1.0051, 1.0207])
        tracks = self._tracks(fs, rng, true)
        tracks[3] = 0.003 * rng.standard_normal(len(tracks[3]))  # no event at all
        picks = (true * fs).astype(int)
        out = le.refine_arrivals_pairwise(picks, tracks, fs, weighting="phat", band=(200, 4000))
        bad = [k for k, (i, j) in enumerate(out["pairs"]) if 3 in (i, j)]
        assert not out["used_pairs"][bad].any()
        assert out["eps_s"][3] == 0.0
        good = [k for k in range(len(out["pairs"])) if k not in bad]
        assert out["used_pairs"][good].all()

    def test_window_padding_at_edges(self, fs):
        x = np.arange(100, dtype=float)
        w = le._event_window(x, 5, pre=10, post=10)
        assert len(w) == 20 and np.all(w[:5] == 0) and w[5] == 0.0 and w[6] == 1.0
        w = le._event_window(x, 95, pre=10, post=10)
        assert len(w) == 20 and np.all(w[15:] == 0)
