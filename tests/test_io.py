import csv
import json
import os

import numpy as np
import pytest

import generate_test_data as gen
import locate_event as le
from helpers import C, arrivals, square_xyz


class TestJsonAndCsv:
    def test_json_default_handles_numpy(self, tmp_path):
        p = tmp_path / "x.json"
        le.write_json(str(p), {"a": np.float64(1.5), "b": np.int64(2), "c": np.array([1, 2]), "d": np.bool_(True)})
        assert json.loads(p.read_text()) == {"a": 1.5, "b": 2, "c": [1, 2], "d": True}
        with pytest.raises(TypeError):
            le.write_json(str(p), {"x": object()})

    def test_sync_csv(self, tmp_path):
        tracks = [le.TrackResult(index=0, used=True, arrival_s=5.25, clock_offset_s=0.001, residual_s=0.0002),
                  le.TrackResult(index=1, used=False, note="silent")]
        p = tmp_path / "sync.csv"
        le.write_sync_csv(str(p), ["/a/cam1.wav", "/a/cam2.wav"], tracks)
        rows = list(csv.DictReader(open(p)))
        assert rows[0]["file"] == "cam1.wav" and rows[0]["used"] == "1"
        assert float(rows[0]["align_to_event_offset_s"]) == -5.25 and float(rows[0]["clock_offset_s"]) == 0.001
        assert rows[1]["used"] == "0" and rows[1]["arrival_time_s"] == ""

    def test_build_results_is_serialisable_and_complete(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0])
        t = arrivals(src, XYZ)
        sol = le.solve_tdoa(t, XYZ, C)
        tracks = [le.TrackResult(index=i, used=True, arrival_s=float(t[i]), snr=10.0, sigma_t=5e-4,
                                 residual_s=float(sol.residuals_s[i]), weight=1.0, clock_offset_s=0.0) for i in range(4)]
        res = {"solution": sol, "tracks": tracks, "refinement": None, "warnings": [], "used": [0, 1, 2, 3]}
        p = le.PipelineParams()
        out = le.build_results(res, ["c1.wav", "c2.wav", "c3.wav", "c4.wav"], XYZ, 41.88, -87.62, C, p, 48000)
        json.dumps(out, default=le._json_default)
        for key in ("event_location_local_m", "event_location_wgs84", "confidence_ellipse_95", "fit", "per_recording",
                    "clock_model", "parameters", "warnings", "local_frame", "position_std_m"):
            assert key in out
        assert abs(out["event_location_local_m"]["x"] - 4.0) < 1e-6
        assert out["fit"]["recordings_used"] == 4 and out["fit"]["rejected"] == []
        lat, lon = out["event_location_wgs84"]["lat"], out["event_location_wgs84"]["lon"]
        x, y = le.latlon_to_local_xy(lat, lon, 41.88, -87.62)
        assert abs(x - 4.0) < 1e-6 and abs(y - 7.0) < 1e-6


class TestPlot:
    def test_plot_layout_with_alternatives_and_unused(self, tmp_path):
        XY = square_xyz(20.0)[:, :2]
        out = tmp_path / "layout.png"
        le.plot_layout(XY, np.array([4.0, 7.0]), np.diag([0.04, 0.09]), str(out), labels=list("abcd"),
                       alternatives=[{"x": -4.0, "y": 7.0}], unused=[3])
        assert out.exists() and out.stat().st_size > 5000


class TestGenerator:
    @pytest.mark.parametrize("kind", gen.EVENT_KINDS)
    def test_event_waveform(self, kind):
        w = gen.event_waveform(kind, 48000, np.random.default_rng(0))
        assert abs(np.max(np.abs(w)) - 1.0) < 1e-12
        assert len(w) > 1000 and np.max(np.abs(w[: 48 * 5])) > 0.05  # energy right at the onset

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            gen.event_waveform("thunder", 48000, np.random.default_rng(0))

    def test_fractional_delay_shifts_centroid(self):
        n = 4096
        t = np.arange(n)
        x = np.exp(-0.5 * ((t - 1000) / 20.0) ** 2)
        y = gen.fractional_delay(x, 3.7)
        cx = np.sum(t * x) / np.sum(x)
        cy = np.sum(t * y) / np.sum(y)
        assert abs((cy - cx) - 3.7) < 1e-6

    def test_render_track_onset_and_noise(self):
        fs = 48000
        ev = gen.event_waveform("gunshot", fs, np.random.default_rng(0))
        x, clean = gen.render_track(ev, fs, 2.0, 1.23456, gain=0.4, noise_rms=0.01, rng=np.random.default_rng(1))
        k = int(1.23456 * fs)
        peak = np.max(np.abs(clean))
        # band-limited fractional delay pre-rings slightly (as a linear-phase ADC filter would)
        assert np.max(np.abs(clean[: k - 50])) < 1e-2 * peak
        assert np.max(np.abs(clean[: k - 5])) < 3e-2 * peak
        assert np.sum(clean[:k] ** 2) < 1e-3 * np.sum(clean**2)
        assert np.max(np.abs(clean[k : k + 200])) > 0.1
        assert abs(np.std(x[: fs // 2]) - 0.01) < 0.001

    def test_synthesize_scenario_truth(self):
        XYZ, src = square_xyz(20.0), np.array([4.0, 7.0, 0.0])
        tracks, truth = gen.synthesize_scenario(XYZ, src, C, "gunshot", clock_offsets_s=[0, 0.001], rng=np.random.default_rng(0))
        assert len(tracks) == 4 and all(len(x) == 480000 for x in tracks)
        d = np.linalg.norm(XYZ - src, axis=1)
        assert np.allclose(truth["arrival_times_s"], 5.0 + d / C + [0, 0.001, 0, 0])
        assert truth["snr_db"][np.argmin(d)] >= max(truth["snr_db"]) - 1e-9  # closest mic is loudest

    def test_generate_scenario_writes_files(self, tmp_path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import shutil
        d = tmp_path / "s"
        d.mkdir()
        shutil.copy(os.path.join(root, "test_data", "scenario1_gunshot", "positions.json"), d / "positions.json")
        truth = gen.generate_scenario(str(d), fmt="wav", seed=0, duration_s=6.0, emission_s=3.0)
        assert sorted(os.listdir(d)) == ["cam1.wav", "cam2.wav", "cam3.wav", "cam4.wav", "metadata.json", "positions.json"]
        assert truth["files"] == ["cam1.wav", "cam2.wav", "cam3.wav", "cam4.wav"]
        assert truth["emission_time_s"] == 3.0 and truth["duration_s"] == 6.0

    def test_cli_without_ffmpeg_falls_back_to_wav(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(gen.shutil, "which", lambda name: None)
        rc = gen.main(["--format", "mp4"])
        assert rc == 1
