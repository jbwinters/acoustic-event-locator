#!/usr/bin/env python3
"""
Tests for I/O and plotting functions in locate_event.py
"""

import pytest
import numpy as np
import os
import csv
import json
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_event import (
    plot_layout, write_sync_csv, write_json, read_json,
    ellipse_from_cov2, extract_audio_ffmpeg
)


class TestPlotLayout:
    """Test layout plotting function."""
    
    def test_plot_layout_basic(self, temp_dir):
        """Test basic layout plotting."""
        # Sample data
        XY = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        s_xy = np.array([5, 5])
        cov = np.array([[1.0, 0.2], [0.2, 0.8]])
        
        out_png = os.path.join(temp_dir, "test_layout.png")
        
        # Should not raise any exceptions
        plot_layout(XY, s_xy, cov, out_png)
        
        # File should be created
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 0  # Non-empty file
    
    def test_plot_layout_edge_cases(self, temp_dir):
        """Test plotting with edge case inputs."""
        # Single microphone
        XY_single = np.array([[0, 0]])
        s_xy = np.array([1, 1])
        cov_small = np.array([[0.01, 0], [0, 0.01]])  # Very small uncertainty
        
        out_png = os.path.join(temp_dir, "test_single_mic.png")
        plot_layout(XY_single, s_xy, cov_small, out_png)
        assert os.path.exists(out_png)
        
        # Large uncertainty
        cov_large = np.array([[100.0, 10.0], [10.0, 50.0]])
        out_png2 = os.path.join(temp_dir, "test_large_uncertainty.png")
        plot_layout(XY_single, s_xy, cov_large, out_png2)
        assert os.path.exists(out_png2)
    
    def test_plot_layout_rotated_ellipse(self, temp_dir):
        """Test plotting with rotated confidence ellipse."""
        XY = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
        s_xy = np.array([2.5, 2.5])
        
        # Create rotated covariance matrix
        theta = np.pi / 4  # 45 degrees
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        D = np.array([[4.0, 0], [0, 1.0]])  # Elongated ellipse
        cov = R @ D @ R.T
        
        out_png = os.path.join(temp_dir, "test_rotated_ellipse.png")
        plot_layout(XY, s_xy, cov, out_png)
        assert os.path.exists(out_png)
    
    def test_plot_layout_aspect_ratio(self, temp_dir):
        """Test that plot maintains equal aspect ratio."""
        # Rectangular array to test aspect ratio
        XY = np.array([[0, 0], [100, 0], [100, 10], [0, 10]])
        s_xy = np.array([50, 5])
        cov = np.eye(2)
        
        out_png = os.path.join(temp_dir, "test_aspect.png")
        
        # Mock matplotlib to check if set_aspect was called correctly
        with patch('matplotlib.pyplot.gca') as mock_gca:
            mock_ax = MagicMock()
            mock_gca.return_value = mock_ax
            
            plot_layout(XY, s_xy, cov, out_png)
            
            # Check that set_aspect was called with "equal"
            mock_ax.set_aspect.assert_called_with("equal", adjustable="box")
    
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    def test_plot_layout_cleanup(self, mock_figure, mock_close, temp_dir):
        """Test that matplotlib resources are properly cleaned up."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        XY = np.array([[0, 0], [1, 1]])
        s_xy = np.array([0.5, 0.5])
        cov = np.eye(2) * 0.1
        
        out_png = os.path.join(temp_dir, "test_cleanup.png")
        plot_layout(XY, s_xy, cov, out_png)
        
        # Check that figure was closed
        mock_close.assert_called_with(mock_fig)


class TestWriteSyncCSV:
    """Test CSV writing function."""
    
    def test_write_sync_csv_basic(self, temp_dir):
        """Test basic CSV writing."""
        files = ['cam1.mp4', 'cam2.mp4', 'cam3.mp4']
        arrival_s = [1.234, 1.456, 1.789]
        delta = np.array([0.0, 0.001, -0.002])
        
        csv_path = os.path.join(temp_dir, "test_sync.csv")
        write_sync_csv(csv_path, files, arrival_s, delta)
        
        assert os.path.exists(csv_path)
        
        # Read and verify content
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        
        # Check header
        expected_headers = ['file', 'arrival_time_s', 'clock_offset_s', 'align_to_event_offset_s']
        assert list(rows[0].keys()) == expected_headers
        
        # Check data values
        for i, row in enumerate(rows):
            assert row['file'] == files[i]
            assert abs(float(row['arrival_time_s']) - arrival_s[i]) < 1e-6
            assert abs(float(row['clock_offset_s']) - delta[i]) < 1e-6
            assert abs(float(row['align_to_event_offset_s']) - (-arrival_s[i])) < 1e-6
    
    def test_write_sync_csv_precision(self, temp_dir):
        """Test CSV writing with high precision values."""
        files = ['test.mp4']
        arrival_s = [1.123456789]
        delta = np.array([0.000123456])
        
        csv_path = os.path.join(temp_dir, "test_precision.csv")
        write_sync_csv(csv_path, files, arrival_s, delta)
        
        # Read back and check precision
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        
        # Should preserve 6 decimal places
        assert '1.123457' in row['arrival_time_s']
        assert '0.000123' in row['clock_offset_s']
    
    def test_write_sync_csv_filename_extraction(self, temp_dir):
        """Test that only basenames are written to CSV."""
        files = ['/path/to/cam1.mp4', '/another/path/cam2.mov']
        arrival_s = [1.0, 2.0]
        delta = np.array([0.0, 0.001])
        
        csv_path = os.path.join(temp_dir, "test_basename.csv")
        write_sync_csv(csv_path, files, arrival_s, delta)
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert rows[0]['file'] == 'cam1.mp4'
        assert rows[1]['file'] == 'cam2.mov'


class TestJSONIO:
    """Test JSON input/output functions."""
    
    def test_json_roundtrip_basic(self, temp_dir):
        """Test basic JSON read/write roundtrip."""
        test_data = {
            "string": "test",
            "number": 42,
            "float": 3.14159,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        json_path = os.path.join(temp_dir, "test.json")
        
        # Write and read back
        write_json(json_path, test_data)
        recovered_data = read_json(json_path)
        
        assert recovered_data == test_data
    
    def test_json_numpy_serialization(self, temp_dir):
        """Test JSON writing with numpy arrays (should convert to lists)."""
        test_data = {
            "array": [1, 2, 3],  # Regular list
            "float_val": 3.14159,
            "int_val": 42
        }
        
        json_path = os.path.join(temp_dir, "test_numpy.json")
        write_json(json_path, test_data)
        
        # Should be readable as valid JSON
        recovered_data = read_json(json_path)
        assert recovered_data == test_data
    
    def test_json_pretty_formatting(self, temp_dir):
        """Test that JSON is formatted with proper indentation."""
        test_data = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        
        json_path = os.path.join(temp_dir, "test_format.json")
        write_json(json_path, test_data)
        
        # Read raw content and check formatting
        with open(json_path, 'r') as f:
            content = f.read()
        
        # Should have indentation (spaces)
        assert '  ' in content  # At least 2-space indentation
        assert content.count('\n') > 3  # Multiple lines due to formatting
    
    def test_read_json_file_not_found(self, temp_dir):
        """Test reading non-existent JSON file."""
        non_existent = os.path.join(temp_dir, "does_not_exist.json")
        
        with pytest.raises(FileNotFoundError):
            read_json(non_existent)
    
    def test_read_json_invalid_format(self, temp_dir):
        """Test reading invalid JSON file."""
        invalid_json_path = os.path.join(temp_dir, "invalid.json")
        
        # Write invalid JSON
        with open(invalid_json_path, 'w') as f:
            f.write("{ invalid json content }")
        
        with pytest.raises(json.JSONDecodeError):
            read_json(invalid_json_path)


class TestFFmpegInterface:
    """Test FFmpeg interface functions."""
    
    @patch('locate_event.subprocess.run')
    def test_extract_audio_ffmpeg_success(self, mock_run, temp_dir):
        """Test successful audio extraction."""
        mock_run.return_value = MagicMock(returncode=0)
        
        in_video = os.path.join(temp_dir, "input.mp4")
        out_wav = os.path.join(temp_dir, "output.wav")
        
        # Create dummy input file
        with open(in_video, 'w') as f:
            f.write("dummy video")
        
        # Should not raise exception
        extract_audio_ffmpeg(in_video, out_wav, target_sr=48000)
        
        # Check that subprocess was called with correct arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # First positional argument (command list)
        
        assert call_args[0] == 'ffmpeg'
        assert '-y' in call_args  # Overwrite output
        assert '-i' in call_args
        assert in_video in call_args
        assert '-ac' in call_args
        assert '1' in call_args  # Mono
        assert '-ar' in call_args
        assert '48000' in call_args
        assert out_wav in call_args
    
    @patch('locate_event.subprocess.run')
    def test_extract_audio_ffmpeg_failure(self, mock_run, temp_dir):
        """Test FFmpeg failure handling."""
        # Mock ffmpeg failure
        error_output = b"ffmpeg: error: Input file not found"
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ['ffmpeg'], stderr=error_output
        )
        
        in_video = os.path.join(temp_dir, "nonexistent.mp4")
        out_wav = os.path.join(temp_dir, "output.wav")
        
        with pytest.raises(subprocess.CalledProcessError):
            extract_audio_ffmpeg(in_video, out_wav)
    
    @patch('locate_event.subprocess.run')
    def test_extract_audio_ffmpeg_parameters(self, mock_run, temp_dir):
        """Test FFmpeg with different parameters."""
        mock_run.return_value = MagicMock(returncode=0)
        
        in_video = os.path.join(temp_dir, "input.mp4")
        out_wav = os.path.join(temp_dir, "output.wav")
        
        # Test with different sample rate
        extract_audio_ffmpeg(in_video, out_wav, target_sr=44100)
        
        call_args = mock_run.call_args[0][0]
        assert '44100' in call_args
        
        # Test default sample rate
        extract_audio_ffmpeg(in_video, out_wav)
        call_args = mock_run.call_args[0][0]
        assert '48000' in call_args  # Default


class TestEllipseFromCovariance:
    """Test confidence ellipse computation (already tested in localization, but verify plotting integration)."""
    
    def test_ellipse_from_cov2_output_format(self):
        """Test that ellipse function returns expected format for plotting."""
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        
        a, b, angle = ellipse_from_cov2(cov)
        
        # Should return three scalars
        assert isinstance(a, (int, float))
        assert isinstance(b, (int, float))
        assert isinstance(angle, (int, float))
        
        # Semi-major axis should be larger
        assert a >= b
        
        # All should be positive
        assert a > 0
        assert b > 0
        
        # Angle should be in reasonable range
        assert -180 <= angle <= 180
    
    def test_ellipse_consistent_with_eigenvalues(self):
        """Test that ellipse parameters are consistent with covariance eigenvalues."""
        # Diagonal covariance (no rotation)
        cov = np.array([[4.0, 0.0], [0.0, 1.0]])
        
        a, b, angle = ellipse_from_cov2(cov)
        
        # For diagonal matrix, should get clean results
        scale = np.sqrt(5.991)  # 95% confidence scale factor
        expected_a = scale * 2.0  # sqrt(4.0)
        expected_b = scale * 1.0  # sqrt(1.0)
        
        assert abs(a - expected_a) < 0.01
        assert abs(b - expected_b) < 0.01
        
        # Angle should be aligned with axes (0 or 90 degrees)
        assert abs(angle % 90) < 5.0


class TestIntegratedIOWorkflow:
    """Test complete I/O workflow as used in main pipeline."""
    
    def test_complete_output_generation(self, temp_dir, synthetic_mic_data):
        """Test generation of all output files."""
        data = synthetic_mic_data
        
        # Simulate solver results
        s_xy = data['source_pos']
        delta = data['clock_offsets']
        cov = np.array([[0.1, 0.02], [0.02, 0.08]])  # Example covariance
        
        # File paths
        files = [f"cam{i+1}.mp4" for i in range(len(data['mic_positions']))]
        arrival_s = data['observed_times'].tolist()
        
        # Reference coordinates
        lat0, lon0 = 41.881, -87.629
        
        # Convert to lat/lon
        from locate_event import local_xy_to_latlon
        lat, lon = local_xy_to_latlon(s_xy[0], s_xy[1], lat0, lon0)
        
        # Generate all outputs
        results_json = {
            "event_location_local_m": {"x": float(s_xy[0]), "y": float(s_xy[1]), "z": 0.0},
            "event_location_wgs84": {"lat": lat, "lon": lon, "alt_m": 0.0},
            "confidence_ellipse": {"semi_major_m": 1.0, "semi_minor_m": 0.8, "angle_deg": 45.0},
            "speed_of_sound_mps": data['speed_of_sound'],
            "per_video": [
                {
                    "file": files[i],
                    "arrival_time_s": arrival_s[i],
                    "clock_offset_s": float(delta[i]),
                    "align_to_event_offset_s": -arrival_s[i]
                }
                for i in range(len(files))
            ]
        }
        
        # Write outputs
        results_path = os.path.join(temp_dir, "results.json")
        sync_path = os.path.join(temp_dir, "sync.csv")
        plot_path = os.path.join(temp_dir, "layout.png")
        
        write_json(results_path, results_json)
        write_sync_csv(sync_path, files, arrival_s, delta)
        plot_layout(data['mic_positions'], s_xy, cov, plot_path)
        
        # Verify all files exist and are valid
        assert os.path.exists(results_path)
        assert os.path.exists(sync_path)
        assert os.path.exists(plot_path)
        
        # Verify content validity
        from tests.test_helpers import ValidationHelpers
        validator = ValidationHelpers()
        assert validator.validate_results_json(results_path)
        
        # Verify CSV content
        with open(sync_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == len(files)
        
        # Verify plot file is not empty
        assert os.path.getsize(plot_path) > 1000  # Should be at least 1KB for a real PNG
    
    def test_output_file_permissions(self, temp_dir):
        """Test that output files have correct permissions."""
        # Create test outputs
        test_json = os.path.join(temp_dir, "test.json")
        test_csv = os.path.join(temp_dir, "test.csv")
        test_png = os.path.join(temp_dir, "test.png")
        
        write_json(test_json, {"test": "data"})
        write_sync_csv(test_csv, ["file1.mp4"], [1.0], np.array([0.0]))
        
        # Create dummy plot
        XY = np.array([[0, 0], [1, 1]])
        s_xy = np.array([0.5, 0.5])
        cov = np.eye(2) * 0.1
        plot_layout(XY, s_xy, cov, test_png)
        
        # Check files are readable
        assert os.access(test_json, os.R_OK)
        assert os.access(test_csv, os.R_OK)
        assert os.access(test_png, os.R_OK)
        
        # Check files are not empty
        assert os.path.getsize(test_json) > 0
        assert os.path.getsize(test_csv) > 0
        assert os.path.getsize(test_png) > 0