"""Occlusion (late arrivals) and mis-picks: one-sided mixture loss and subset search."""
import json
import math

import numpy as np
import pytest
import soundfile as sf

import generate_test_data as gen
import locate_event as le
from helpers import C, arrivals, l_xyz, nearest_solution_error, pos_error, square_xyz

FS = 48000
SRC = np.array([30.0, 20.0])


def _late(XYZ, src, delays_ms, seed=0, noise=0.2e-3):
    t = arrivals(src, XYZ, c=C, noise=noise, rng=np.random.default_rng(seed))
    for i, d in delays_ms.items():
        t[i] += d / 1000.0
    return t


class TestMixtureLoss:
    def setup_method(self):
        self.loss = le._OcclusionLoss(p=0.2, tau=5.0 / C, q=0.05)
        self.sig = np.full(1, 0.3e-3)

    def test_weights_shape(self):
        r = np.array([0.0, 0.3e-3, -0.3e-3, 3e-3, 30e-3, -3e-3, -30e-3])
        w = self.loss.weights(r, np.full(len(r), 0.3e-3))
        assert abs(w[0] - 1.0) < 1e-6
        assert w[1] > 0.9 and w[2] > 0.9
        assert w[3] < 0.05 and w[4] < 0.005  # late arrivals pull almost nothing
        assert w[5] < 0.2 and w[6] < 0.01  # impossible early arrivals become blunders

    def test_responsibilities(self):
        for r in (0.0, 1.5e-3, 20e-3, -5e-3):
            g = self.loss.responsibilities(np.array([r]), self.sig)
            assert abs(sum(x[0] for x in g) - 1.0) < 1e-9
        assert self.loss.occlusion_prob(np.array([1.5e-3]), self.sig)[0] > 0.9
        assert self.loss.occlusion_prob(np.array([0.0]), self.sig)[0] < 0.05
        assert self.loss.blunder_prob(np.array([-5e-3]), self.sig)[0] > 0.9
        assert self.loss.blunder_prob(np.array([200e-3]), self.sig)[0] > 0.5  # a 70 m "detour" is not a detour

    def test_cost_shape(self):
        r = np.array([-2e-3, -1e-3, 0.0, 1e-3, 2e-3, 10e-3, 40e-3]) 
        c = np.array([self.loss.cost(np.array([x]), self.sig) for x in r])
        assert np.argmin(c) in (2, 3)  # minimum at (about) zero
        slope = (c[6] - c[5]) / (40e-3 - 10e-3)  # far on the late side the loss is linear at 1/tau
        assert abs(slope - C / 5.0) / (C / 5.0) < 0.2
        assert c[5] - c[4] < 1.0  # going from 2 ms to 10 ms late costs less than one nat
        assert c[0] > c[4]  # early is worse than equally late

    def test_bad_parameters(self):
        with pytest.raises(le.LocatorError):
            le._OcclusionLoss(p=0.6, tau=0.01, q=0.5)


class TestSolverOcclusion:
    def test_clean_data_unchanged(self):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert pos_error(sol, SRC) < 0.1 and sol.occluded == [] and sol.rejected == [] and sol.loss == "occlusion"

    @pytest.mark.parametrize("delay_ms", [3.0, 9.0, 30.0])
    def test_one_late_recording(self, delay_ms):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: delay_ms}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert sol.occluded == [1] and sol.rejected == []
        assert pos_error(sol, SRC) < 0.1
        assert abs(sol.detour_m[1] - delay_ms / 1000 * C) < 0.3
        assert sol.weights[1] < 0.1 and le.mahalanobis_xy(sol, SRC) < 2.5

    def test_two_late_recordings(self):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: 30.0, 4: 15.0}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert sol.occluded == [1, 4] and pos_error(sol, SRC) < 0.1

    def test_three_late_of_seven(self):
        XYZ = np.vstack([l_xyz(), [[25.0, 25.0, 8.0]]])
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: 30.0, 4: 15.0, 5: 20.0}), XYZ, C, sigma_t=np.full(7, 0.3e-3))
        assert sol.occluded == [1, 4, 5] and pos_error(sol, SRC) < 0.3

    def test_three_late_of_six_is_ambiguous_but_truth_is_reported(self):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: 30.0, 4: 15.0, 5: 20.0}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert nearest_solution_error(sol, SRC) < 0.3
        assert sol.ambiguous or pos_error(sol, SRC) < 0.3

    def test_early_mispick_is_a_blunder_not_an_occlusion(self):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: -9.0}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert sol.rejected == [1] and sol.occluded == []
        assert pos_error(sol, SRC) < 0.1 and sol.weights[1] < 0.05

    def test_early_plus_late(self):
        XYZ = l_xyz()
        sol = le.solve_tdoa(_late(XYZ, SRC, {1: -9.0, 4: 20.0}), XYZ, C, sigma_t=np.full(6, 0.3e-3))
        assert nearest_solution_error(sol, SRC) < 0.3
        assert sol.ambiguous or pos_error(sol, SRC) < 0.3

    @pytest.mark.parametrize("delay_ms", [9.0, 30.0])
    def test_four_recordings_one_late(self, delay_ms):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(_late(XYZ, src, {2: delay_ms}), XYZ, C, sigma_t=np.full(4, 0.3e-3))
        assert nearest_solution_error(sol, src) < 0.1
        assert 2 in sol.occluded or sol.ambiguous

    def test_four_recordings_one_early(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        sol = le.solve_tdoa(_late(XYZ, src, {2: -9.0}), XYZ, C, sigma_t=np.full(4, 0.3e-3))
        assert nearest_solution_error(sol, src) < 0.1 and sol.ambiguous

    def test_no_false_flags_on_clean_noise(self):
        XYZ = l_xyz()
        n_occ = n_rej = 0
        errs = []
        for seed in range(30):
            t = arrivals(SRC, XYZ, c=C, noise=0.3e-3, rng=np.random.default_rng(seed))
            sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.3e-3))
            n_occ += len(sol.occluded)
            n_rej += len(sol.rejected)
            errs.append(pos_error(sol, SRC))
        assert n_occ <= 4 and n_rej == 0  # about 1 recording in 50 may be mildly flagged
        assert np.mean(errs) < 0.2

    def test_random_occlusions_coverage(self):
        XYZ = np.vstack([l_xyz(), [[25.0, 25.0, 8.0]]])
        rng = np.random.default_rng(7)
        inside, errs = [], []
        for _ in range(40):
            det = np.where(rng.random(7) < 0.2, rng.exponential(5.0, 7), 0.0)
            t = arrivals(SRC, XYZ, c=C, noise=0.3e-3, rng=rng) + det / C
            sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(7, 0.3e-3))
            errs.append(nearest_solution_error(sol, SRC))
            inside.append(le.mahalanobis_xy(sol, SRC) <= math.sqrt(5.991) or sol.ambiguous)
        assert np.median(errs) < 0.2 and np.mean(inside) >= 0.85

    def test_with_height_and_camera_priors(self):
        XYZ = l_xyz()
        aerial = np.array([24.9, 33.2])
        t = arrivals(aerial, XYZ, c=345.0, source_z=25.0)
        t[3] += 0.012
        hs = np.zeros(6)
        hs[[0, 2]] = 0.5
        sol = le.solve_tdoa(t, XYZ, 345.0, sigma_t=np.full(6, 0.2e-3), source_z=10.0, source_z_sigma=50.0, height_sigma=hs)
        assert sol.occluded == [3] and pos_error(sol, aerial) < 0.3 and abs(sol.s_xyz[2] - 25.0) < 2.0

    def test_legacy_mode(self):
        XYZ = l_xyz()
        t = _late(XYZ, SRC, {1: 9.0})
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.3e-3), occlusion=False)
        assert sol.loss == "huber" and sol.occluded == [] and sol.rejected == [1] and pos_error(sol, SRC) < 0.1

    def test_validation(self):
        XYZ = l_xyz()
        t = _late(XYZ, SRC, {})
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, occlusion_prob=0.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, occlusion_scale_m=-1.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, occlusion_prob=0.7, blunder_prob=0.5)


class TestPipelineOcclusion:
    XYZ = np.array([[0, 0, 3], [35, 5, 4], [20, 40, 3], [-15, 30, 2.5], [50, 35, 6], [10, -25, 3], [40, -20, 8]], float)
    SRC3 = np.array([18.0, 10.0, 1.3])
    DET = [0, 0, 7.0, 4.0, 12.0, 0, 0]

    def test_generator_occlusion_model(self):
        tracks, truth = gen.synthesize_scenario(self.XYZ, self.SRC3, C, "gunshot", noise_rms=0.003,
                                                detours_m=self.DET, rng=np.random.default_rng(0))
        d = np.linalg.norm(self.XYZ - self.SRC3, axis=1)
        assert np.allclose(truth["arrival_times_s"], 5.0 + (d + np.array(self.DET)) / C)
        assert truth["occlusion_detour_m"] == self.DET
        # occluded tracks are low-passed: less energy above 3 kHz relative to below
        def hf_ratio(x):
            xf_hi = le.apply_bandpass(x, FS, 3000, 8000)
            xf_lo = le.apply_bandpass(x, FS, 200, 1000)
            k = int(truth["arrival_times_s"][0] * FS)
            return np.sum(xf_hi[k:k + FS // 5] ** 2) / np.sum(xf_lo[k:k + FS // 5] ** 2)
        assert hf_ratio(tracks[2]) < 0.6 * hf_ratio(tracks[0])  # the rest is the noise floor

    def test_end_to_end_flags_occluded_recordings(self):
        tracks, truth = gen.synthesize_scenario(self.XYZ, self.SRC3, C, "gunshot", noise_rms=0.003,
                                                detours_m=self.DET, rng=np.random.default_rng(1))
        res = le.locate_from_signals(tracks, FS, self.XYZ, C, le.PipelineParams(source_z=1.3))
        sol = res["solution"]
        assert sorted(sol.occluded) == [2, 3, 4]
        assert pos_error(sol, self.SRC3[:2]) < 0.15
        assert all(res["tracks"][i].occlusion_prob > 0.5 and "occluded" in res["tracks"][i].note for i in (2, 3, 4))
        assert all(res["tracks"][i].occlusion_prob < 0.5 for i in (0, 1, 5, 6))
        assert any("occluded" in w for w in res["warnings"])
        for i, dm in enumerate(self.DET):
            if dm > 0:
                assert abs(res["tracks"][i].detour_m - dm) < 1.0

    def test_cli_outputs(self, tmp_path):
        lat0, lon0 = 41.879, -87.63
        mics = []
        for i, (x, y, z) in enumerate(self.XYZ):
            lat, lon = le.local_xy_to_latlon(x, y, lat0, lon0)
            mics.append({"file": f"cam{i+1}.wav", "lat": lat, "lon": lon, "height_m": float(z)})
        d = tmp_path / "s"
        d.mkdir()
        (d / "positions.json").write_text(json.dumps({"speed_of_sound": C, "reference": {"lat": lat0, "lon": lon0}, "mics": mics}))
        tracks, truth = gen.synthesize_scenario(self.XYZ, self.SRC3, C, "gunshot", noise_rms=0.003,
                                                detours_m=self.DET, rng=np.random.default_rng(2))
        for i, x in enumerate(tracks):
            sf.write(str(d / f"cam{i+1}.wav"), x.astype(np.float32), FS, subtype="PCM_16")
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out), "--source_height_m", "1.3"])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        assert sorted(r["fit"]["occluded"]) == ["cam3.wav", "cam4.wav", "cam5.wav"]
        assert r["fit"]["loss"] == "occlusion"
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        assert np.linalg.norm(est - self.SRC3[:2]) < 0.15
        for p in r["per_recording"]:
            assert "occlusion_probability" in p and "detour_m" in p
        # legacy mode still runs
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o2"),
                      "--source_height_m", "1.3", "--no_occlusion"])
        assert rc == 0
        assert json.loads((tmp_path / "o2" / "results.json").read_text())["fit"]["loss"] == "huber"
