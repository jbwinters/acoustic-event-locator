"""Event height (3D) and uncertain recording heights."""
import json
import math
import os

import numpy as np
import pytest
import soundfile as sf

import generate_test_data as gen
import locate_event as le
from helpers import C, arrivals, l_xyz, pos_error, square_xyz

FS = 48000
AERIAL = np.array([24.9, 33.2, 25.0])


class TestSourceHeight:
    def test_fixed_height_matches_2d(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        sol = le.solve_tdoa(t, XYZ, C, source_z=25.0)
        assert not sol.solve_z and sol.cov_pos.shape == (2, 2)
        assert sol.s_xyz[2] == 25.0 and sol.z_std == 0.0
        assert pos_error(sol, AERIAL[:2]) < 1e-6

    def test_recovers_height_from_perfect_data(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.2e-3), source_z=10.0, source_z_sigma=50.0)
        assert sol.solve_z and sol.cov_pos.shape == (3, 3)
        assert abs(sol.s_xyz[2] - 25.0) < 0.05
        assert pos_error(sol, AERIAL[:2]) < 1e-3
        assert sol.dof == 3  # the height parameter is paid for by its prior row
        assert 0 < sol.z_std < 2.0
        assert not sol.z_at_bound

    def test_ground_source_height_is_weakly_determined_and_flagged_by_std(self):
        XYZ = l_xyz()  # cameras at 1.8-3 m, source on the ground: little vertical aperture
        src = np.array([10.0, 12.0])
        t = arrivals(src, XYZ, source_z=0.0, noise=1e-3, rng=np.random.default_rng(0))
        sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 1e-3), source_z=0.0, source_z_sigma=100.0)
        assert sol.z_std > 2.0 and sol.z_std > 4 * math.sqrt(max(sol.cov_xy[0, 0], sol.cov_xy[1, 1]))
        assert pos_error(sol, src) < 1.5
        assert le.mahalanobis_pos(sol, [src[0], src[1], 0.0]) < math.sqrt(7.815) * 1.5

    def test_prior_and_data_blend(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        tight = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 1e-3), source_z=20.0, source_z_sigma=0.5)
        loose = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 1e-3), source_z=20.0, source_z_sigma=50.0)
        assert 20.0 < tight.s_xyz[2] < loose.s_xyz[2] <= 25.0 + 1e-6
        assert tight.z_std < loose.z_std

    def test_height_coverage_monte_carlo(self):
        XYZ = l_xyz()
        inside, zerr = [], []
        for seed in range(60):
            t = arrivals(AERIAL[:2], XYZ, source_z=25.0, noise=0.5e-3, rng=np.random.default_rng(seed))
            sol = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.5e-3), source_z=10.0, source_z_sigma=50.0)
            inside.append(le.mahalanobis_pos(sol, AERIAL) <= math.sqrt(7.815))
            zerr.append(sol.s_xyz[2] - 25.0)
        assert np.mean(inside) >= 0.88
        assert np.sqrt(np.mean(np.square(zerr))) < 4.0

    def test_mirror_below_camera_plane(self):
        XYZ = l_xyz()
        XYZ[:, 2] = 2.5  # exactly coplanar cameras
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        kw = dict(sigma_t=np.full(6, 0.2e-3), source_z=5.0, source_z_sigma=100.0)
        above = le.solve_tdoa(t, XYZ, C, **kw)
        assert abs(above.s_xyz[2] - 25.0) < 0.05 and not above.ambiguous
        both = le.solve_tdoa(t, XYZ, C, source_z_bounds=(-200.0, 5000.0), **kw)
        assert both.ambiguous
        assert any(abs(a["z"] - (2 * 2.5 - 25.0)) < 0.5 for a in both.alternatives)

    def test_vertical_aperture_improves_height(self):
        flat = l_xyz()
        roof = flat.copy()
        roof[0, 2], roof[3, 2] = 20.0, 12.0
        kw = dict(sigma_t=np.full(6, 1e-3), source_z=10.0, source_z_sigma=50.0)
        s_flat = le.solve_tdoa(arrivals(AERIAL[:2], flat, source_z=25.0), flat, C, **kw)
        s_roof = le.solve_tdoa(arrivals(AERIAL[:2], roof, source_z=25.0), roof, C, **kw)
        assert s_roof.z_std < 0.5 * s_flat.z_std

    def test_height_bound_is_flagged(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        sol = le.solve_tdoa(t, XYZ, C, source_z=5.0, source_z_sigma=50.0, source_z_bounds=(0.0, 10.0))
        assert sol.z_at_bound and abs(sol.s_xyz[2] - 10.0) < 1e-6

    def test_validation(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, source_z_sigma=-1.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, source_z_sigma=5.0, source_z_bounds=(10.0, 10.0))
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, source_z=50.0, source_z_sigma=5.0, source_z_bounds=(0.0, 10.0))

    def test_trace_records_convergence(self):
        XYZ = square_xyz(20.0)
        tr = []
        sol = le.solve_tdoa(arrivals([4.0, 7.0], XYZ), XYZ, C, trace=tr)
        assert len(tr) >= 2 and np.allclose(tr[-1][:2], sol.s_xy)


def _true_and_prior_heights():
    XYZ_true = l_xyz()
    XYZ_true[[0, 2], 2] = [1.9, 1.2]
    XYZ_prior = l_xyz()
    XYZ_prior[[0, 2], 2] = 1.5
    hs = np.zeros(6)
    hs[[0, 2]] = 0.5
    return XYZ_true, XYZ_prior, hs


class TestRecordingHeights:
    def test_ground_source_is_insensitive_to_camera_heights(self):
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        src = np.array([10.0, 12.0])
        t = arrivals(src, XYZ_true, source_z=0.0)
        with_prior = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.3e-3), height_sigma=hs)
        assumed = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.3e-3))
        assert pos_error(with_prior, src) < 0.05 and pos_error(assumed, src) < 0.05
        assert with_prior.dof == 3  # each height parameter comes with a prior row

    def test_aerial_source_needs_height_priors(self):
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        t = arrivals(AERIAL[:2], XYZ_true, source_z=25.0)
        with_prior = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.3e-3), height_sigma=hs, source_z=25.0)
        assumed = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.3e-3), source_z=25.0)
        assert pos_error(with_prior, AERIAL[:2]) < pos_error(assumed, AERIAL[:2])
        assert le.mahalanobis_xy(with_prior, AERIAL[:2]) < 2.0
        assert le.mahalanobis_xy(assumed, AERIAL[:2]) > le.mahalanobis_xy(with_prior, AERIAL[:2])
        a_p, _, _ = le.ellipse_from_cov2(with_prior.cov_xy)
        a_f, _, _ = le.ellipse_from_cov2(assumed.cov_xy)
        assert a_p > a_f  # the uncertainty honestly grows

    def test_estimated_heights_carry_prior_uncertainty(self):
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        t = arrivals(AERIAL[:2], XYZ_true, source_z=25.0)
        sol = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.3e-3), height_sigma=hs, source_z=25.0)
        assert sol.mic_heights.shape == (6,) and sol.mic_height_std.shape == (6,)
        assert np.all(sol.mic_height_std[[1, 3, 4, 5]] == 0.0)
        assert np.all(sol.mic_height_std[[0, 2]] <= 0.5 + 1e-9)
        assert np.all(sol.mic_heights[[1, 3, 4, 5]] == XYZ_prior[[1, 3, 4, 5], 2])

    def test_huge_sigma_equals_dropping_the_recording(self):
        XYZ = l_xyz()
        src = np.array([10.0, 12.0])
        t = arrivals(src, XYZ, source_z=0.0)
        hs = np.zeros(6)
        hs[4] = 100.0
        full = le.solve_tdoa(t, XYZ, C, sigma_t=np.full(6, 0.3e-3), height_sigma=hs)
        dropped = le.solve_tdoa(np.delete(t, 4), np.delete(XYZ, 4, axis=0), C, sigma_t=np.full(5, 0.3e-3))
        assert np.allclose(np.sqrt(np.diag(full.cov_xy)), np.sqrt(np.diag(dropped.cov_xy)), rtol=1e-3)

    def test_outlier_rejection_still_works(self):
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        t = arrivals(AERIAL[:2], XYZ_true, source_z=25.0)
        t[3] += 0.008
        sol = le.solve_tdoa(t, XYZ_prior, C, sigma_t=np.full(6, 0.2e-3), height_sigma=hs,
                            source_z=10.0, source_z_sigma=50.0)
        assert sol.occluded == [3] and sol.weights[3] < 0.05  # +8 ms is a detour under the occlusion model
        assert pos_error(sol, AERIAL[:2]) < 0.5 and abs(sol.s_xyz[2] - 25.0) < 1.5
        assert le.mahalanobis_pos(sol, AERIAL) < 3.0

    def test_validation(self):
        XYZ = l_xyz()
        t = arrivals(AERIAL[:2], XYZ, source_z=25.0)
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, height_sigma=[0.5, 0.5])
        with pytest.raises(le.LocatorError):
            le.solve_tdoa(t, XYZ, C, height_sigma=[-1, 0, 0, 0, 0, 0])


class TestPipeline3D:
    def test_pipeline_solves_event_height(self):
        XYZ = l_xyz()
        tracks, truth = gen.synthesize_scenario(XYZ, AERIAL, 345.0, "fireworks", noise_rms=0.003, rng=np.random.default_rng(2))
        p = le.PipelineParams(source_z=10.0, source_z_sigma=30.0)
        res = le.locate_from_signals(tracks, FS, XYZ, 345.0, p)
        sol = res["solution"]
        assert len(res["used"]) == 6
        assert abs(sol.s_xyz[2] - 25.0) < 1.0
        assert pos_error(sol, AERIAL[:2]) < 0.1
        assert le.mahalanobis_pos(sol, AERIAL) < math.sqrt(7.815) * 1.5

    def test_pipeline_with_uncertain_camera_heights(self):
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        tracks, truth = gen.synthesize_scenario(XYZ_true, AERIAL, 345.0, "gunshot", noise_rms=0.003, rng=np.random.default_rng(4))
        p = le.PipelineParams(source_z=25.0)
        res = le.locate_from_signals(tracks, FS, XYZ_prior, 345.0, p, height_sigma=hs)
        sol = res["solution"]
        assert pos_error(sol, AERIAL[:2]) < 0.3
        assert le.mahalanobis_xy(sol, AERIAL[:2]) < 3.0
        assert res["tracks"][0].height_std_m > 0 and res["tracks"][1].height_std_m == 0

    def test_cli_3d_outputs(self, tmp_path):
        lat0, lon0 = 41.8925, -87.6123
        XYZ_true, XYZ_prior, hs = _true_and_prior_heights()
        mics = []
        for i, (x, y, z) in enumerate(XYZ_prior):
            lat, lon = le.local_xy_to_latlon(x, y, lat0, lon0)
            m = {"file": f"cam{i+1}.wav", "lat": lat, "lon": lon, "height_m": float(z)}
            if hs[i] > 0:
                m["height_sigma_m"] = float(hs[i])
            mics.append(m)
        J = {"speed_of_sound": 345.0, "reference": {"lat": lat0, "lon": lon0}, "mics": mics}
        d = tmp_path / "s"
        d.mkdir()
        (d / "positions.json").write_text(json.dumps(J))
        tracks, truth = gen.synthesize_scenario(XYZ_true, AERIAL, 345.0, "fireworks", noise_rms=0.003, rng=np.random.default_rng(6))
        for i, x in enumerate(tracks):
            sf.write(str(d / f"cam{i+1}.wav"), x.astype(np.float32), FS, subtype="PCM_16")
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out),
                      "--source_height_m", "10", "--source_height_sigma_m", "30"])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        assert r["height_model"]["source"]["solved"] is True
        assert abs(r["event_location_local_m"]["z"] - 25.0) < 1.5
        assert abs(r["event_location_wgs84"]["alt_m"] - r["event_location_local_m"]["z"]) < 1e-9
        assert r["position_std_m"]["z"] > 0
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        assert np.linalg.norm(est - AERIAL[:2]) < 0.3
        recs = r["height_model"]["recordings"]
        assert [q["file"] for q in recs] == ["cam1.wav", "cam3.wav"]
        assert all(0 < q["std_m"] <= 0.5 for q in recs)
        assert r["per_recording"][0]["height_std_m"] > 0 and r["per_recording"][1]["height_std_m"] == 0
        assert (out / "layout.png").stat().st_size > 10000  # two-panel figure

    def test_cli_height_bound_warning(self, tmp_path):
        XYZ = l_xyz()
        lat0, lon0 = 41.8925, -87.6123
        mics = []
        for i, (x, y, z) in enumerate(XYZ):
            lat, lon = le.local_xy_to_latlon(x, y, lat0, lon0)
            mics.append({"file": f"cam{i+1}.wav", "lat": lat, "lon": lon, "height_m": float(z)})
        d = tmp_path / "s"
        d.mkdir()
        (d / "positions.json").write_text(json.dumps({"speed_of_sound": 345.0, "reference": {"lat": lat0, "lon": lon0}, "mics": mics}))
        tracks, _ = gen.synthesize_scenario(XYZ, AERIAL, 345.0, "fireworks", noise_rms=0.003, rng=np.random.default_rng(6))
        for i, x in enumerate(tracks):
            sf.write(str(d / f"cam{i+1}.wav"), x.astype(np.float32), FS, subtype="PCM_16")
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out),
                      "--source_height_m", "5", "--source_height_sigma_m", "50", "--source_height_bounds", "0", "12"])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        assert r["fit"]["z_at_bound"] is True and any("bound" in w for w in r["warnings"])
