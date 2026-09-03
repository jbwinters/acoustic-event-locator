import json
import math
import os

import numpy as np
import pytest

import locate_event as le


class TestGeo:
    def test_round_trip(self):
        lat0, lon0 = 41.8818, -87.6232
        for lat, lon in [(41.8825, -87.6221), (41.8790, -87.6260), (41.8818, -87.6232)]:
            x, y = le.latlon_to_local_xy(lat, lon, lat0, lon0)
            lat2, lon2 = le.local_xy_to_latlon(x, y, lat0, lon0)
            assert abs(lat2 - lat) < 1e-10 and abs(lon2 - lon) < 1e-10

    def test_scale_against_great_circle(self):
        # 0.001 deg steps at 41.88 N compared with a spherical great-circle distance (R = 6371 km)
        lat0, lon0 = 41.88, -87.62
        R = 6371008.8
        x, y = le.latlon_to_local_xy(lat0 + 0.001, lon0, lat0, lon0)
        assert abs(x) < 1e-9 and abs(y - R * math.radians(0.001)) / y < 0.005
        x, y = le.latlon_to_local_xy(lat0, lon0 + 0.001, lat0, lon0)
        assert abs(y) < 1e-9 and abs(x - R * math.cos(math.radians(lat0)) * math.radians(0.001)) / x < 0.005

    def test_axes_orientation(self):
        x, y = le.latlon_to_local_xy(10.001, 20.0, 10.0, 20.0)
        assert y > 0 and abs(x) < 1e-9  # north is +y
        x, y = le.latlon_to_local_xy(10.0, 20.001, 10.0, 20.0)
        assert x > 0 and abs(y) < 1e-9  # east is +x

    def test_meters_per_degree_reference_values(self):
        m_lat, m_lon = le.meters_per_degree(0.0)
        assert abs(m_lat - 110574) < 5 and abs(m_lon - 111320) < 5
        m_lat, m_lon = le.meters_per_degree(45.0)
        assert abs(m_lat - 111132) < 5 and abs(m_lon - 78847) < 5

    def test_speed_of_sound(self):
        assert abs(le.speed_of_sound_mps(20.0) - 343.2) < 0.5
        assert abs(le.speed_of_sound_mps(0.0) - 331.3) < 0.5
        assert le.speed_of_sound_mps(30.0) > le.speed_of_sound_mps(10.0)


def _positions(**over):
    J = {
        "temperature_C": 20.0,
        "mics": [
            {"file": "cam1.wav", "lat": 41.8811, "lon": -87.6297, "height_m": 1.6},
            {"file": "cam2.wav", "lat": 41.88125, "lon": -87.6292},
            {"file": "cam3.wav", "lat": 41.88085, "lon": -87.6294, "height_m": 1.7},
        ],
    }
    J.update(over)
    return J


class TestParsePositions:
    def test_centroid_origin_and_temperature(self):
        mics, (lat0, lon0), c = le.parse_positions(_positions())
        assert len(mics) == 3
        assert abs(lat0 - np.mean([41.8811, 41.88125, 41.88085])) < 1e-12
        assert abs(c - le.speed_of_sound_mps(20.0)) < 1e-9
        assert mics[1].height_m == 0.0 and mics[2].height_m == 1.7

    def test_reference_keys(self):
        _, (lat0, lon0), _ = le.parse_positions(_positions(reference={"lat": 1.0, "lon": 2.0}))
        assert (lat0, lon0) == (1.0, 2.0)
        _, (lat0, lon0), _ = le.parse_positions(_positions(reference_point={"lat": 3.0, "lon": 4.0}))
        assert (lat0, lon0) == (3.0, 4.0)

    def test_speed_override(self):
        _, _, c = le.parse_positions(_positions(speed_of_sound=340.0, temperature_C=35.0))
        assert c == 340.0

    @pytest.mark.parametrize("bad", [{"mics": []}, {"mics": "x"}, {}])
    def test_missing_mics(self, bad):
        with pytest.raises(le.LocatorError):
            le.parse_positions(bad)

    def test_missing_keys(self):
        J = _positions()
        del J["mics"][0]["lat"]
        with pytest.raises(le.LocatorError, match="lat"):
            le.parse_positions(J)

    def test_implausible_speed(self):
        with pytest.raises(le.LocatorError):
            le.parse_positions(_positions(speed_of_sound=1000.0))


class TestLoadPositions:
    def test_resolves_files_and_wav_fallback(self, tmp_path):
        J = _positions()
        J["mics"][0]["file"] = "cam1.mp4"  # listed as video, only wav present
        for name in ("cam1.wav", "cam2.wav", "cam3.wav"):
            (tmp_path / name).write_bytes(b"")
        pj = tmp_path / "positions.json"
        pj.write_text(json.dumps(J))
        mics, origin, c, raw = le.load_positions(str(pj), str(tmp_path))
        assert os.path.basename(mics[0].file) == "cam1.wav"
        assert all(os.path.exists(m.file) for m in mics)
        assert raw["temperature_C"] == 20.0

    def test_missing_recording(self, tmp_path):
        pj = tmp_path / "positions.json"
        pj.write_text(json.dumps(_positions()))
        with pytest.raises(le.LocatorError, match="not found"):
            le.load_positions(str(pj), str(tmp_path))

    def test_mic_local_xyz(self):
        mics, (lat0, lon0), _ = le.parse_positions(_positions(reference={"lat": 41.8811, "lon": -87.6297}))
        XYZ = le.mic_local_xyz(mics, lat0, lon0)
        assert XYZ.shape == (3, 3)
        assert np.allclose(XYZ[0], [0.0, 0.0, 1.6])
        assert XYZ[1, 1] > 0 and XYZ[1, 0] > 0  # cam2 is north-east of cam1
