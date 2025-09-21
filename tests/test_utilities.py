#!/usr/bin/env python3
"""
Tests for utility functions in locate_event.py
"""

import pytest
import numpy as np
import math
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_event import (
    deg2rad, rad2deg, latlon_to_local_xy, local_xy_to_latlon,
    read_json, write_json, ensure_dir, log, require_ffmpeg,
    load_positions, Mic
)


class TestAngularConversions:
    """Test angular conversion utilities."""
    
    def test_deg2rad(self):
        """Test degree to radian conversion."""
        assert deg2rad(0) == 0
        assert abs(deg2rad(90) - math.pi/2) < 1e-10
        assert abs(deg2rad(180) - math.pi) < 1e-10
        assert abs(deg2rad(360) - 2*math.pi) < 1e-10
        assert abs(deg2rad(-90) - (-math.pi/2)) < 1e-10
    
    def test_rad2deg(self):
        """Test radian to degree conversion."""
        assert rad2deg(0) == 0
        assert abs(rad2deg(math.pi/2) - 90) < 1e-10
        assert abs(rad2deg(math.pi) - 180) < 1e-10
        assert abs(rad2deg(2*math.pi) - 360) < 1e-10
        assert abs(rad2deg(-math.pi/2) - (-90)) < 1e-10
    
    def test_round_trip_conversion(self):
        """Test that deg->rad->deg conversion is identity."""
        test_angles = [0, 30, 45, 90, 135, 180, 270, 360, -45, -180]
        for angle in test_angles:
            recovered = rad2deg(deg2rad(angle))
            assert abs(recovered - angle) < 1e-10


class TestGeoConversion:
    """Test geographic coordinate conversion functions."""
    
    def test_latlon_to_local_xy_origin(self):
        """Test conversion at reference origin."""
        lat0, lon0 = 41.881, -87.629
        x, y = latlon_to_local_xy(lat0, lon0, lat0, lon0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
    
    def test_latlon_to_local_xy_known_values(self):
        """Test conversion with known approximate values."""
        # Chicago area coordinates
        lat0, lon0 = 41.881, -87.629
        lat1, lon1 = 41.882, -87.628
        
        x, y = latlon_to_local_xy(lat1, lon1, lat0, lon0)
        
        # Should be positive (north and east of reference)
        assert x > 0
        assert y > 0
        
        # Rough magnitude checks (should be ~100m scale)
        assert 50 < x < 150  # longitude difference
        assert 50 < y < 150  # latitude difference
    
    def test_local_xy_to_latlon_origin(self):
        """Test inverse conversion at origin."""
        lat0, lon0 = 41.881, -87.629
        lat, lon = local_xy_to_latlon(0, 0, lat0, lon0)
        assert abs(lat - lat0) < 1e-10
        assert abs(lon - lon0) < 1e-10
    
    def test_round_trip_geo_conversion(self):
        """Test that lat/lon -> xy -> lat/lon is identity."""
        lat0, lon0 = 41.881, -87.629
        test_coords = [
            (41.882, -87.628),
            (41.880, -87.630),
            (41.885, -87.625),
            (41.875, -87.635)
        ]
        
        for lat, lon in test_coords:
            x, y = latlon_to_local_xy(lat, lon, lat0, lon0)
            lat_recovered, lon_recovered = local_xy_to_latlon(x, y, lat0, lon0)
            
            assert abs(lat_recovered - lat) < 1e-10
            assert abs(lon_recovered - lon) < 1e-10
    
    def test_consistent_scale_factors(self):
        """Test that scale factors are reasonable for Chicago area."""
        lat0 = 41.881
        mx = 111320.0 * math.cos(deg2rad(lat0))
        my = 110540.0
        
        # Scale factors should be reasonable for this latitude
        assert 70000 < mx < 90000  # meters per degree longitude
        assert 110000 < my < 111000  # meters per degree latitude


class TestFileOperations:
    """Test file I/O utilities."""
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test that ensure_dir creates a directory."""
        test_dir = os.path.join(temp_dir, "test_subdir")
        assert not os.path.exists(test_dir)
        
        ensure_dir(test_dir)
        assert os.path.isdir(test_dir)
    
    def test_ensure_dir_existing_directory(self, temp_dir):
        """Test that ensure_dir works with existing directory."""
        # Should not raise an error
        ensure_dir(temp_dir)
        assert os.path.isdir(temp_dir)
    
    def test_read_write_json_round_trip(self, temp_dir):
        """Test JSON read/write round trip."""
        test_data = {
            "string": "hello",
            "number": 42,
            "float": 3.14159,
            "array": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        json_path = os.path.join(temp_dir, "test.json")
        write_json(json_path, test_data)
        
        assert os.path.exists(json_path)
        
        recovered_data = read_json(json_path)
        assert recovered_data == test_data
    
    def test_read_json_file_not_found(self, temp_dir):
        """Test read_json with non-existent file."""
        non_existent = os.path.join(temp_dir, "does_not_exist.json")
        with pytest.raises(FileNotFoundError):
            read_json(non_existent)


class TestFFmpegRequirement:
    """Test ffmpeg availability checking."""
    
    def test_require_ffmpeg_available(self, mock_ffmpeg_available):
        """Test require_ffmpeg when tools are available."""
        # Should not raise any exception
        require_ffmpeg()
    
    def test_require_ffmpeg_missing(self, monkeypatch):
        """Test require_ffmpeg when tools are missing."""
        def mock_which(name):
            return None
        
        import shutil
        monkeypatch.setattr(shutil, 'which', mock_which)
        
        with pytest.raises(SystemExit) as exc_info:
            require_ffmpeg()
        
        assert exc_info.value.code == 2


class TestLogging:
    """Test logging utility."""
    
    def test_log_output(self, capsys):
        """Test log function output format."""
        log("Test message")
        captured = capsys.readouterr()
        assert "[INFO] Test message" in captured.out
    
    def test_log_with_level(self, capsys):
        """Test log function with custom level."""
        log("Warning message", level="WARN")
        captured = capsys.readouterr()
        assert "[WARN] Warning message" in captured.out
    
    def test_log_with_error_level(self, capsys):
        """Test log function with error level."""
        log("Error message", level="ERROR")
        captured = capsys.readouterr()
        assert "[ERROR] Error message" in captured.out


class TestLoadPositions:
    """Test position loading functionality."""
    
    def test_load_positions_basic(self, sample_positions_json, temp_dir):
        """Test basic position loading."""
        # Create dummy video files
        for i in range(1, 5):
            video_path = os.path.join(temp_dir, f"cam{i}.mp4")
            with open(video_path, 'w') as f:
                f.write("dummy video file")
        
        mics, (lat0, lon0), c, raw_json = load_positions(sample_positions_json, temp_dir)
        
        assert len(mics) == 4
        assert isinstance(mics[0], Mic)
        assert abs(lat0 - 41.881) < 1e-6
        assert abs(lon0 - (-87.629)) < 1e-6
        assert abs(c - 343.4) < 0.1  # 331.4 + 0.6 * 20
        assert "mics" in raw_json
    
    def test_load_positions_custom_speed_of_sound(self, temp_dir):
        """Test position loading with custom speed of sound."""
        positions = {
            "speed_of_sound": 350.0,
            "mics": [
                {"file": "cam1.mp4", "lat": 41.881, "lon": -87.629, "height_m": 1.5}
            ]
        }
        
        json_path = os.path.join(temp_dir, "positions.json")
        with open(json_path, 'w') as f:
            json.dump(positions, f)
        
        # Create dummy video file
        video_path = os.path.join(temp_dir, "cam1.mp4")
        with open(video_path, 'w') as f:
            f.write("dummy")
        
        _, _, c, _ = load_positions(json_path, temp_dir)
        assert abs(c - 350.0) < 1e-6
    
    def test_load_positions_temperature_based_speed(self, temp_dir):
        """Test speed of sound calculation from temperature."""
        positions = {
            "temperature_C": 25.0,
            "mics": [
                {"file": "cam1.mp4", "lat": 41.881, "lon": -87.629}
            ]
        }
        
        json_path = os.path.join(temp_dir, "positions.json")
        with open(json_path, 'w') as f:
            json.dump(positions, f)
        
        # Create dummy video file
        video_path = os.path.join(temp_dir, "cam1.mp4")
        with open(video_path, 'w') as f:
            f.write("dummy")
        
        _, _, c, _ = load_positions(json_path, temp_dir)
        expected_c = 331.4 + 0.6 * 25.0
        assert abs(c - expected_c) < 1e-6
    
    def test_load_positions_missing_video_file(self, sample_positions_json, temp_dir):
        """Test error when video file is missing."""
        # Don't create the video files
        with pytest.raises(SystemExit) as exc_info:
            load_positions(sample_positions_json, temp_dir)
        
        assert exc_info.value.code == 2
    
    def test_load_positions_auto_reference(self, temp_dir):
        """Test automatic reference point calculation."""
        positions = {
            "mics": [
                {"file": "cam1.mp4", "lat": 41.880, "lon": -87.630},
                {"file": "cam2.mp4", "lat": 41.882, "lon": -87.628}
            ]
        }
        
        json_path = os.path.join(temp_dir, "positions.json")
        with open(json_path, 'w') as f:
            json.dump(positions, f)
        
        # Create dummy video files
        for i in range(1, 3):
            video_path = os.path.join(temp_dir, f"cam{i}.mp4")
            with open(video_path, 'w') as f:
                f.write("dummy")
        
        _, (lat0, lon0), _, _ = load_positions(json_path, temp_dir)
        
        # Should be the mean of the mic positions
        expected_lat0 = (41.880 + 41.882) / 2
        expected_lon0 = (-87.630 + -87.628) / 2
        
        assert abs(lat0 - expected_lat0) < 1e-10
        assert abs(lon0 - expected_lon0) < 1e-10
    
    def test_load_positions_height_handling(self, temp_dir):
        """Test height parameter handling."""
        positions = {
            "mics": [
                {"file": "cam1.mp4", "lat": 41.881, "lon": -87.629, "height_m": 2.5},
                {"file": "cam2.mp4", "lat": 41.882, "lon": -87.628}  # No height
            ]
        }
        
        json_path = os.path.join(temp_dir, "positions.json")
        with open(json_path, 'w') as f:
            json.dump(positions, f)
        
        # Create dummy video files
        for i in range(1, 3):
            video_path = os.path.join(temp_dir, f"cam{i}.mp4")
            with open(video_path, 'w') as f:
                f.write("dummy")
        
        mics, _, _, _ = load_positions(json_path, temp_dir)
        
        assert abs(mics[0].height_m - 2.5) < 1e-6
        assert abs(mics[1].height_m - 0.0) < 1e-6  # Default value