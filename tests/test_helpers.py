#!/usr/bin/env python3
"""
Test helper functions and synthetic data generators for event location detector tests.
"""

import pytest
import numpy as np
import os
import tempfile
import json
import soundfile as sf
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_event import Mic, latlon_to_local_xy


class SyntheticDataGenerator:
    """Generate synthetic test data for various scenarios."""
    
    def __init__(self, fs: int = 48000, c: float = 343.0):
        self.fs = fs
        self.c = c
        np.random.seed(42)  # For reproducible tests
    
    def generate_impulse_signal(self, duration: float, impulse_time: float, 
                               noise_level: float = 0.01, impulse_amplitude: float = 0.5) -> np.ndarray:
        """Generate audio signal with impulse at specified time."""
        n_samples = int(duration * self.fs)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Background noise
        signal = noise_level * np.random.randn(n_samples)
        
        # Add impulse
        impulse_idx = int(impulse_time * self.fs)
        if 0 <= impulse_idx < n_samples - 200:
            # Exponentially decaying impulse
            decay_length = 200
            decay = np.exp(-np.linspace(0, 8, decay_length))
            signal[impulse_idx:impulse_idx + decay_length] += impulse_amplitude * decay
        
        return signal
    
    def generate_multipath_signal(self, duration: float, direct_time: float,
                                 reflection_delays: List[float], reflection_amplitudes: List[float],
                                 noise_level: float = 0.01) -> np.ndarray:
        """Generate signal with direct path and reflections."""
        signal = self.generate_impulse_signal(duration, direct_time, noise_level, 0.5)
        
        # Add reflections
        for delay, amplitude in zip(reflection_delays, reflection_amplitudes):
            reflection_time = direct_time + delay
            reflection_signal = self.generate_impulse_signal(duration, reflection_time, 0.0, amplitude)
            signal += reflection_signal
        
        return signal
    
    def create_scenario(self, scenario_type: str) -> Dict:
        """Create predefined test scenarios."""
        scenarios = {
            'square_array': {
                'mic_positions': np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),
                'source_pos': np.array([10, 15]),
                'clock_offsets': np.array([0.0, 0.001, -0.002, 0.0015])
            },
            'linear_array': {
                'mic_positions': np.array([[0, 0], [10, 0], [20, 0], [30, 0]]),
                'source_pos': np.array([15, 10]),
                'clock_offsets': np.array([0.0, 0.0005, -0.001, 0.002])
            },
            'l_shaped_array': {
                'mic_positions': np.array([[0, 0], [10, 0], [20, 0], [0, 10]]),
                'source_pos': np.array([5, 5]),
                'clock_offsets': np.array([0.0, 0.0008, -0.0015, 0.001])
            },
            'close_mics': {
                'mic_positions': np.array([[0, 0], [2, 0], [0, 2], [2, 2]]),
                'source_pos': np.array([1, 1]),
                'clock_offsets': np.array([0.0, 0.0002, -0.0003, 0.0001])
            }
        }
        
        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_type}")
        
        scenario = scenarios[scenario_type].copy()
        
        # Calculate derived quantities
        distances = np.linalg.norm(scenario['mic_positions'] - scenario['source_pos'][None, :], axis=1)
        scenario['arrival_times'] = distances / self.c
        scenario['observed_times'] = scenario['arrival_times'] + scenario['clock_offsets']
        scenario['speed_of_sound'] = self.c
        
        return scenario


class TestDataFactory:
    """Factory for creating test data files and configurations."""
    
    @staticmethod
    def create_positions_json(temp_dir: str, scenario: Dict, reference_latlon: Optional[Tuple[float, float]] = None) -> str:
        """Create positions.json file from scenario."""
        if reference_latlon is None:
            reference_latlon = (41.881, -87.629)  # Chicago
        
        lat0, lon0 = reference_latlon
        
        # Convert local positions to lat/lon
        mics_data = []
        for i, (x, y) in enumerate(scenario['mic_positions']):
            # Simple inverse conversion (approximate)
            mx = 111320.0 * np.cos(np.radians(lat0))
            my = 110540.0
            lat = y / my + lat0
            lon = x / mx + lon0
            
            mics_data.append({
                "file": f"cam{i+1}.mp4",
                "lat": lat,
                "lon": lon,
                "height_m": 1.5
            })
        
        positions = {
            "temperature_C": 20,
            "speed_of_sound": scenario['speed_of_sound'],
            "reference": {"lat": lat0, "lon": lon0},
            "mics": mics_data
        }
        
        json_path = os.path.join(temp_dir, "positions.json")
        with open(json_path, 'w') as f:
            json.dump(positions, f, indent=2)
        
        return json_path
    
    @staticmethod
    def create_audio_files(temp_dir: str, scenario: Dict, duration: float = 3.0,
                          noise_level: float = 0.01, fs: int = 48000) -> List[str]:
        """Create audio files from scenario."""
        generator = SyntheticDataGenerator(fs=fs, c=scenario['speed_of_sound'])
        audio_files = []
        
        for i, obs_time in enumerate(scenario['observed_times']):
            signal = generator.generate_impulse_signal(duration, obs_time, noise_level)
            
            filename = f"cam{i+1}.mp4.wav"
            filepath = os.path.join(temp_dir, filename)
            sf.write(filepath, signal, fs)
            audio_files.append(filepath)
        
        return audio_files
    
    @staticmethod
    def create_video_files(temp_dir: str, num_videos: int) -> List[str]:
        """Create dummy video files for testing."""
        video_files = []
        for i in range(num_videos):
            filename = f"cam{i+1}.mp4"
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"dummy video file {i+1}")
            video_files.append(filepath)
        return video_files


class ValidationHelpers:
    """Helper functions for validating test results."""
    
    @staticmethod
    def validate_position_estimate(estimated_pos: np.ndarray, true_pos: np.ndarray, 
                                 tolerance_m: float = 0.1) -> bool:
        """Validate position estimate within tolerance."""
        error = np.linalg.norm(estimated_pos - true_pos)
        return error <= tolerance_m
    
    @staticmethod
    def validate_clock_offsets(estimated_offsets: np.ndarray, true_offsets: np.ndarray,
                             tolerance_s: float = 1e-4) -> bool:
        """Validate clock offsets within tolerance (accounting for gauge freedom)."""
        # Normalize both to have first element = 0 (gauge condition)
        est_norm = estimated_offsets - estimated_offsets[0]
        true_norm = true_offsets - true_offsets[0]
        
        max_error = np.max(np.abs(est_norm - true_norm))
        return max_error <= tolerance_s
    
    @staticmethod
    def validate_covariance_matrix(cov: np.ndarray) -> bool:
        """Validate that covariance matrix is positive definite."""
        if cov.shape != (2, 2):
            return False
        
        eigenvals = np.linalg.eigvals(cov)
        return np.all(eigenvals > 0) and np.all(np.isfinite(eigenvals))
    
    @staticmethod
    def validate_results_json(results_path: str) -> bool:
        """Validate structure of results.json file."""
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            required_keys = [
                'event_location_local_m', 'event_location_wgs84',
                'confidence_ellipse', 'per_video', 'speed_of_sound_mps'
            ]
            
            if not all(key in results for key in required_keys):
                return False
            
            # Validate location format
            local_loc = results['event_location_local_m']
            if not all(key in local_loc for key in ['x', 'y', 'z']):
                return False
            
            wgs84_loc = results['event_location_wgs84']
            if not all(key in wgs84_loc for key in ['lat', 'lon', 'alt_m']):
                return False
            
            # Validate ellipse format
            ellipse = results['confidence_ellipse']
            if not all(key in ellipse for key in ['semi_major_m', 'semi_minor_m', 'angle_deg']):
                return False
            
            # Validate per-video data
            per_video = results['per_video']
            if not isinstance(per_video, list) or len(per_video) == 0:
                return False
            
            for video_data in per_video:
                required_video_keys = ['file', 'arrival_time_s', 'clock_offset_s', 'align_to_event_offset_s']
                if not all(key in video_data for key in required_video_keys):
                    return False
            
            return True
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False


class NoiseModels:
    """Different noise models for testing robustness."""
    
    @staticmethod
    def add_timing_noise(arrival_times: np.ndarray, noise_std_ms: float = 1.0) -> np.ndarray:
        """Add Gaussian noise to timing measurements."""
        noise_std_s = noise_std_ms / 1000.0
        noise = np.random.normal(0, noise_std_s, len(arrival_times))
        return arrival_times + noise
    
    @staticmethod
    def add_outlier_times(arrival_times: np.ndarray, outlier_fraction: float = 0.1,
                         outlier_magnitude_ms: float = 50.0) -> np.ndarray:
        """Add outlier measurements to timing."""
        outlier_times = arrival_times.copy()
        n_outliers = int(outlier_fraction * len(arrival_times))
        
        if n_outliers > 0:
            outlier_indices = np.random.choice(len(arrival_times), n_outliers, replace=False)
            outlier_magnitude_s = outlier_magnitude_ms / 1000.0
            outliers = np.random.normal(0, outlier_magnitude_s, n_outliers)
            outlier_times[outlier_indices] += outliers
        
        return outlier_times
    
    @staticmethod
    def add_systematic_bias(arrival_times: np.ndarray, bias_pattern: str = 'linear') -> np.ndarray:
        """Add systematic bias to timing measurements."""
        biased_times = arrival_times.copy()
        n = len(arrival_times)
        
        if bias_pattern == 'linear':
            # Linear drift across microphones
            bias = np.linspace(-0.001, 0.001, n)  # ±1ms
        elif bias_pattern == 'quadratic':
            # Quadratic pattern
            x = np.linspace(-1, 1, n)
            bias = 0.001 * x**2  # Up to 1ms
        elif bias_pattern == 'random':
            # Random systematic offsets
            bias = 0.002 * np.random.randn(n)  # ±2ms std
        else:
            bias = np.zeros(n)
        
        return biased_times + bias


# Test fixtures using the helper classes
@pytest.fixture
def synthetic_data_generator():
    """Provide synthetic data generator."""
    return SyntheticDataGenerator()

@pytest.fixture
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory()

@pytest.fixture
def validation_helpers():
    """Provide validation helpers."""
    return ValidationHelpers()

@pytest.fixture
def noise_models():
    """Provide noise models."""
    return NoiseModels()

@pytest.fixture(params=['square_array', 'linear_array', 'l_shaped_array'])
def scenario_data(request, synthetic_data_generator):
    """Parametrized fixture providing different array geometries."""
    return synthetic_data_generator.create_scenario(request.param)

@pytest.fixture
def complete_test_setup(temp_dir, scenario_data, test_data_factory):
    """Complete test setup with all necessary files."""
    # Create positions.json
    positions_json = test_data_factory.create_positions_json(temp_dir, scenario_data)
    
    # Create video files
    num_mics = len(scenario_data['mic_positions'])
    video_files = test_data_factory.create_video_files(temp_dir, num_mics)
    
    # Create audio files
    audio_files = test_data_factory.create_audio_files(temp_dir, scenario_data)
    
    return {
        'positions_json': positions_json,
        'video_files': video_files,
        'audio_files': audio_files,
        'scenario': scenario_data,
        'temp_dir': temp_dir
    }


class TestHelperFunctions:
    """Test the helper functions themselves."""
    
    def test_synthetic_data_generator_impulse(self, synthetic_data_generator):
        """Test impulse signal generation."""
        duration = 2.0
        impulse_time = 1.0
        signal = synthetic_data_generator.generate_impulse_signal(duration, impulse_time)
        
        assert len(signal) == int(duration * synthetic_data_generator.fs)
        
        # Find peak location
        peak_idx = np.argmax(np.abs(signal))
        peak_time = peak_idx / synthetic_data_generator.fs
        
        # Should be near expected impulse time
        assert abs(peak_time - impulse_time) < 0.1  # Within 100ms
    
    def test_synthetic_data_generator_scenarios(self, synthetic_data_generator):
        """Test scenario generation."""
        scenario = synthetic_data_generator.create_scenario('square_array')
        
        assert 'mic_positions' in scenario
        assert 'source_pos' in scenario
        assert 'clock_offsets' in scenario
        assert 'arrival_times' in scenario
        assert 'observed_times' in scenario
        
        # Verify derived quantities
        expected_distances = np.linalg.norm(
            scenario['mic_positions'] - scenario['source_pos'][None, :], axis=1
        )
        expected_arrivals = expected_distances / scenario['speed_of_sound']
        
        np.testing.assert_array_almost_equal(scenario['arrival_times'], expected_arrivals)
        
        expected_observed = expected_arrivals + scenario['clock_offsets']
        np.testing.assert_array_almost_equal(scenario['observed_times'], expected_observed)
    
    def test_test_data_factory_positions_json(self, temp_dir, scenario_data, test_data_factory):
        """Test positions.json creation."""
        json_path = test_data_factory.create_positions_json(temp_dir, scenario_data)
        
        assert os.path.exists(json_path)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert 'mics' in data
        assert len(data['mics']) == len(scenario_data['mic_positions'])
        
        for mic_data in data['mics']:
            assert 'file' in mic_data
            assert 'lat' in mic_data
            assert 'lon' in mic_data
            assert 'height_m' in mic_data
    
    def test_test_data_factory_audio_files(self, temp_dir, scenario_data, test_data_factory):
        """Test audio file creation."""
        audio_files = test_data_factory.create_audio_files(temp_dir, scenario_data)
        
        assert len(audio_files) == len(scenario_data['observed_times'])
        
        for filepath in audio_files:
            assert os.path.exists(filepath)
            
            # Verify audio can be read
            signal, fs = sf.read(filepath)
            assert len(signal) > 0
            assert fs == 48000
    
    def test_validation_helpers_position(self, validation_helpers):
        """Test position validation."""
        true_pos = np.array([5.0, 10.0])
        
        # Good estimate
        good_est = np.array([5.05, 10.02])
        assert validation_helpers.validate_position_estimate(good_est, true_pos, tolerance_m=0.1)
        
        # Bad estimate
        bad_est = np.array([6.0, 11.0])
        assert not validation_helpers.validate_position_estimate(bad_est, true_pos, tolerance_m=0.1)
    
    def test_validation_helpers_clock_offsets(self, validation_helpers):
        """Test clock offset validation."""
        true_offsets = np.array([0.0, 0.001, -0.002, 0.0015])
        
        # Perfect estimate (accounting for gauge)
        perfect_est = true_offsets.copy()
        assert validation_helpers.validate_clock_offsets(perfect_est, true_offsets)
        
        # Good estimate with gauge shift
        gauge_shifted = true_offsets + 0.0005  # Add constant to all
        assert validation_helpers.validate_clock_offsets(gauge_shifted, true_offsets)
        
        # Bad estimate
        bad_est = np.array([0.0, 0.005, -0.008, 0.001])
        assert not validation_helpers.validate_clock_offsets(bad_est, true_offsets, tolerance_s=1e-4)
    
    def test_noise_models(self, noise_models):
        """Test noise model functions."""
        clean_times = np.array([1.0, 1.1, 1.05, 1.15])
        
        # Timing noise
        noisy_times = noise_models.add_timing_noise(clean_times, noise_std_ms=1.0)
        assert len(noisy_times) == len(clean_times)
        assert not np.array_equal(noisy_times, clean_times)  # Should be different
        
        # Outliers
        outlier_times = noise_models.add_outlier_times(clean_times, outlier_fraction=0.5)
        assert len(outlier_times) == len(clean_times)
        
        # Systematic bias
        biased_times = noise_models.add_systematic_bias(clean_times, bias_pattern='linear')
        assert len(biased_times) == len(clean_times)