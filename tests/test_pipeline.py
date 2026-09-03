import json
import math
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

import generate_test_data as gen
import locate_event as le
from helpers import C, l_xyz, nearest_solution_error, pos_error, square_xyz

FS = 48000


def _run(tracks, XYZ, c, src, **params):
    p = le.PipelineParams(source_z=float(src[2]) if len(src) > 2 else 0.0, **params)
    return le.locate_from_signals(tracks, FS, XYZ, c, p)


class TestInMemoryScenarios:
    @pytest.mark.parametrize("kind", gen.EVENT_KINDS)
    @pytest.mark.parametrize("seed", [0, 1])
    def test_square_array_centimetre_accuracy(self, kind, seed):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, truth = gen.synthesize_scenario(XYZ, src, C, kind, noise_rms=0.003, rng=np.random.default_rng(seed))
        res = _run(tracks, XYZ, C, src)
        sol = res["solution"]
        assert res["used"] == [0, 1, 2, 3]
        assert pos_error(sol, src[:2]) < 0.05
        assert le.mahalanobis_xy(sol, src[:2]) < math.sqrt(5.991)
        arr = np.array([tr.arrival_s for tr in res["tracks"]])
        rel_err = ((arr - arr[0]) - (np.array(truth["arrival_times_s"]) - truth["arrival_times_s"][0])) * 1000
        assert np.max(np.abs(rel_err[1:])) < 0.1  # relative timing within 0.1 ms

    def test_l_array_elevated_source(self):
        XYZ, src = l_xyz(), np.array([24.9, 33.2, 25.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, 345.0, "fireworks", noise_rms=0.003, rng=np.random.default_rng(3))
        res = _run(tracks, XYZ, 345.0, src)
        assert len(res["used"]) == 6 and pos_error(res["solution"], src[:2]) < 0.05

    def test_wrong_source_height_is_visible_in_residuals(self):
        XYZ, src = l_xyz(), np.array([24.9, 33.2, 25.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, 345.0, "fireworks", noise_rms=0.003, rng=np.random.default_rng(3))
        res = le.locate_from_signals(tracks, FS, XYZ, 345.0, le.PipelineParams(source_z=0.0))
        assert res["solution"].scale > 2.0  # residuals exceed the assumed timing noise

    def test_decoy_sound_in_one_recording(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", noise_rms=0.003, rng=np.random.default_rng(5))
        t = np.arange(int(0.3 * FS)) / FS
        tracks[2][int(7.5 * FS): int(7.5 * FS) + len(t)] += 0.6 * np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))
        res = _run(tracks, XYZ, C, src)
        assert res["used"] == [0, 1, 2, 3] and pos_error(res["solution"], src[:2]) < 0.05

    def test_silent_recording_is_excluded(self):
        XYZ, src = l_xyz(), np.array([10.0, 12.0, 0.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", noise_rms=0.003, rng=np.random.default_rng(7))
        tracks[4] = 0.003 * np.random.default_rng(8).standard_normal(len(tracks[4]))
        res = _run(tracks, XYZ, C, src)
        assert 4 not in res["used"] and len(res["used"]) == 5
        assert res["tracks"][4].used is False and "noise floor" in res["tracks"][4].note
        assert pos_error(res["solution"], src[:2]) < 0.05

    def test_clock_offsets_with_prior(self):
        XYZ, src = l_xyz(), np.array([10.0, 12.0, 0.0])
        offs = np.array([0.0, 0.002, -0.001, 0.0015, -0.0005, 0.001])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", clock_offsets_s=offs, noise_rms=0.003, rng=np.random.default_rng(9))
        res = _run(tracks, XYZ, C, src, clock_sigma_s=0.002)
        sol = res["solution"]
        assert pos_error(sol, src[:2]) < 0.6
        assert le.mahalanobis_xy(sol, src[:2]) < 3.0
        assert sol.estimate_offsets and np.corrcoef(sol.delta, offs)[0, 1] > 0.8

    def test_unsynchronised_by_seconds_fails_clearly(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", clock_offsets_s=[0.0, 1.5, -0.8, 3.0], noise_rms=0.003)
        with pytest.raises(le.LocatorError, match="consistent"):
            _run(tracks, XYZ, C, src)

    def test_too_few_recordings_with_event(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", noise_rms=0.003)
        rng = np.random.default_rng(1)
        tracks[2] = 0.003 * rng.standard_normal(len(tracks[2]))
        tracks[3] = 0.003 * rng.standard_normal(len(tracks[3]))
        with pytest.raises(le.LocatorError):
            _run(tracks, XYZ, C, src)

    @pytest.mark.parametrize("seed", range(3))
    def test_faint_event_never_gives_a_silent_wrong_answer(self, seed):
        XYZ, src = l_xyz(), np.array([24.9, 33.2, 25.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, 345.0, "fireworks", noise_rms=0.015, rng=np.random.default_rng(seed))
        try:
            res = _run(tracks, XYZ, 345.0, src)
        except le.LocatorError:
            return
        sol = res["solution"]
        flagged = sol.at_boundary or sol.ambiguous or any(
            any(key in w for key in ("relaxed", "larger than", "redundancy", "cross-correlation", "caution"))
            for w in res["warnings"]
        )
        assert le.mahalanobis_xy(sol, src[:2]) < 3.5 or flagged

    def test_no_refine_still_accurate(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, _ = gen.synthesize_scenario(XYZ, src, C, "gunshot", noise_rms=0.003)
        res = _run(tracks, XYZ, C, src, refine=False)
        assert res["refinement"] is None and pos_error(res["solution"], src[:2]) < 0.3

    def test_timing_sigma_from_snr(self):
        s = le.timing_sigma_from_snr([100.0, 25.0, 1.0, 0.01], 0.5e-3)
        assert np.allclose(s, [0.5e-3, 1.0e-3, 5.0e-3, 5.0e-3])


def _write_scenario(tmp_path, XYZ_latlon, src_latlon, kind="gunshot", seed=0, offsets=None, files=None, reference=True):
    """Write positions.json + WAV tracks for a synthetic scenario; returns (dir, truth, mics_xyz, src_xyz)."""
    lat0, lon0 = 41.8818, -87.6232
    files = files or [f"cam{i+1}.wav" for i in range(len(XYZ_latlon))]
    J = {"temperature_C": 20.0, "mics": [
        {"file": f, "lat": lat, "lon": lon, "height_m": h} for f, (lat, lon, h) in zip(files, XYZ_latlon)]}
    if reference:
        J["reference_point"] = {"lat": lat0, "lon": lon0}
    mics, (la0, lo0), c = le.parse_positions(J)
    XYZ = le.mic_local_xyz(mics, la0, lo0)
    sx, sy = le.latlon_to_local_xy(src_latlon[0], src_latlon[1], la0, lo0)
    src = np.array([sx, sy, src_latlon[2]])
    tracks, truth = gen.synthesize_scenario(XYZ, src, c, kind, clock_offsets_s=offsets, noise_rms=0.003, rng=np.random.default_rng(seed))
    d = tmp_path / "scn"
    d.mkdir()
    (d / "positions.json").write_text(json.dumps(J))
    for f, x in zip(files, tracks):
        sf.write(str(d / f), x.astype(np.float32), FS, subtype="PCM_16")
    return d, truth, XYZ, src, c


MICS_LL = [(41.8818, -87.6232, 1.5), (41.8818, -87.6229, 2.0), (41.8816, -87.6229, 1.0), (41.8816, -87.6232, 3.0)]
SRC_LL = (41.88172, -87.62305, 0.0)


class TestCommandLine:
    def test_end_to_end_wav(self, tmp_path):
        d, truth, XYZ, src, c = _write_scenario(tmp_path, MICS_LL, SRC_LL)
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out)])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        assert np.linalg.norm(est - src[:2]) < 0.05
        lat, lon = r["event_location_wgs84"]["lat"], r["event_location_wgs84"]["lon"]
        assert abs(lat - SRC_LL[0]) < 1e-6 and abs(lon - SRC_LL[1]) < 1e-6
        assert r["fit"]["recordings_used"] == 4 and r["clock_model"]["mode"] == "synchronised"
        assert r["confidence_ellipse_95"]["semi_major_m"] < 1.0
        for p, t_true in zip(r["per_recording"], truth["arrival_times_s"]):
            assert p["used"] and abs(p["arrival_time_s"] - t_true) < 0.002
            assert abs(p["align_to_event_offset_s"] + p["arrival_time_s"]) < 1e-12
        assert (out / "sync.csv").exists() and (out / "layout.png").stat().st_size > 1000
        assert len((out / "sync.csv").read_text().strip().splitlines()) == 5
        assert sorted(os.listdir(out / "wav")) == sorted(f + ".wav" for f in truth["files"]) if "files" in truth else True

    def test_end_to_end_with_prior(self, tmp_path):
        offs = [0.0, 0.002, -0.001, 0.0015]
        d, truth, XYZ, src, c = _write_scenario(tmp_path, MICS_LL, SRC_LL, offsets=offs, seed=2)
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out), "--clock_sigma_ms", "2", "--verbose"])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        assert np.linalg.norm(est - src[:2]) < 0.6
        assert r["clock_model"]["mode"] == "prior" and r["fit"]["dof"] == 1
        assert all(p["clock_offset_s"] is not None for p in r["per_recording"])

    def test_video_listed_but_wav_present(self, tmp_path):
        files = ["cam1.mp4", "cam2.mp4", "cam3.mp4", "cam4.mp4"]
        d, truth, XYZ, src, c = _write_scenario(tmp_path, MICS_LL, SRC_LL, files=[f.replace(".mp4", ".wav") for f in files])
        J = json.loads((d / "positions.json").read_text())
        for m, f in zip(J["mics"], files):
            m["file"] = f
        (d / "positions.json").write_text(json.dumps(J))
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 0

    def test_video_path_uses_ffmpeg_extractor(self, tmp_path, monkeypatch):
        d, truth, XYZ, src, c = _write_scenario(tmp_path, MICS_LL, SRC_LL)
        for i in range(4):
            shutil.move(str(d / f"cam{i+1}.wav"), str(d / f"src{i+1}.wav"))
            (d / f"cam{i+1}.mp4").write_bytes(b"not really a video")
        J = json.loads((d / "positions.json").read_text())
        for m in J["mics"]:
            m["file"] = m["file"].replace(".wav", ".mp4")
        (d / "positions.json").write_text(json.dumps(J))
        calls = []

        def fake_extract(in_video, out_wav, target_sr=48000):
            calls.append(in_video)
            shutil.copy(str(d / os.path.basename(in_video).replace("cam", "src").replace(".mp4", ".wav")), out_wav)

        monkeypatch.setattr(le, "require_ffmpeg", lambda: None)
        monkeypatch.setattr(le, "extract_audio_ffmpeg", fake_extract)
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 0 and len(calls) == 4

    def test_missing_ffmpeg_is_a_clear_error(self, tmp_path, monkeypatch):
        d, *_ = _write_scenario(tmp_path, MICS_LL, SRC_LL)
        (d / "cam1.mp4").write_bytes(b"x")
        J = json.loads((d / "positions.json").read_text())
        J["mics"][0]["file"] = "cam1.mp4"
        (d / "cam1.wav").unlink()
        (d / "positions.json").write_text(json.dumps(J))
        monkeypatch.setattr(shutil, "which", lambda name: None)
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 2

    def test_missing_recording_returns_2(self, tmp_path):
        d, *_ = _write_scenario(tmp_path, MICS_LL, SRC_LL)
        (d / "cam3.wav").unlink()
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 2

    def test_too_few_recordings_returns_2(self, tmp_path):
        d, *_ = _write_scenario(tmp_path, MICS_LL[:2], SRC_LL)
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 2

    def test_resampling_and_stereo_input(self, tmp_path):
        d, truth, XYZ, src, c = _write_scenario(tmp_path, MICS_LL, SRC_LL)
        x, sr = sf.read(str(d / "cam2.wav"))
        from scipy.signal import resample_poly
        y = resample_poly(x, 44100, 48000)
        sf.write(str(d / "cam2.wav"), np.stack([y, 0.5 * y], axis=1), 44100)
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(tmp_path / "o")])
        assert rc == 0
        r = json.loads((tmp_path / "o" / "results.json").read_text())
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        assert np.linalg.norm(est - src[:2]) < 0.1


class TestGeneratorScenarios:
    """The checked-in scenario definitions, generated on the fly, must localise accurately."""

    @pytest.mark.parametrize("name", gen.SCENARIOS)
    def test_scenario(self, name, tmp_path):
        src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data", name)
        d = tmp_path / name
        d.mkdir()
        shutil.copy(os.path.join(src_dir, "positions.json"), d / "positions.json")
        truth = gen.generate_scenario(str(d), fmt="wav", seed=1)
        out = tmp_path / "out"
        rc = le.main(["--videos_dir", str(d), "--positions", str(d / "positions.json"), "--out", str(out),
                      "--source_height_m", str(truth["source_height_m"])])
        assert rc == 0
        r = json.loads((out / "results.json").read_text())
        est = np.array([r["event_location_local_m"]["x"], r["event_location_local_m"]["y"]])
        tru = np.array(truth["source_position_m"])
        cands = [est] + [np.array([a["x"], a["y"]]) for a in r["fit"]["alternatives"]]
        err = min(np.linalg.norm(cnd - tru) for cnd in cands)
        # recordings whose height is only known to +-0.5 m cost a few centimetres
        assert err < (0.15 if any(s > 0 for s in truth["microphone_height_sigma_m"]) else 0.05)
        assert r["fit"]["recordings_used"] == len(truth["files"])
        if "linear" in name or name.startswith("scenario2"):
            assert r["fit"]["ambiguous"]  # collinear cameras: mirror solution reported
        m = json.loads((d / "metadata.json").read_text())
        assert m["source_position_m"] == truth["source_position_m"]
        assert abs(m["source_latlon"]["lat"] - json.loads((d / "positions.json").read_text())["event"]["true_location"]["lat"]) < 1e-9
