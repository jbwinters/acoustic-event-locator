#!/usr/bin/env python3
"""
Pytest configuration and fixtures for event location detector tests.
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
import json
from dataclasses import dataclass
from typing import List, Tuple
import soundfile as sf

# Import the main module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locate_event import Mic, MicData


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_mic_positions():
    """Sample microphone positions for testing."""
    return [
        Mic(file="cam1.mp4", lat=41.88110, lon=-87.62970, height_m=1.6),
        Mic(file="cam2.mp4", lat=41.88125, lon=-87.62920, height_m=1.5),
        Mic(file="cam3.mp4", lat=41.88085, lon=-87.62940, height_m=1.7),
        Mic(file="cam4.mp4", lat=41.88100, lon=-87.62905, height_m=1.6)
    ]


@pytest.fixture
def sample_positions_json(temp_dir):
    """Create a sample positions.json file."""
    positions = {
        "temperature_C": 20,
        "speed_of_sound": None,
        "reference": {"lat": 41.881, "lon": -87.629},
        "mics": [
            {"file": "cam1.mp4", "lat": 41.88110, "lon": -87.62970, "height_m": 1.6},
            {"file": "cam2.mp4", "lat": 41.88125, "lon": -87.62920, "height_m": 1.5},
            {"file": "cam3.mp4", "lat": 41.88085, "lon": -87.62940, "height_m": 1.7},
            {"file": "cam4.mp4", "lat": 41.88100, "lon": -87.62905, "height_m": 1.6}
        ]
    }
    
    json_path = os.path.join(temp_dir, "positions.json")
    with open(json_path, 'w') as f:
        json.dump(positions, f, indent=2)
    
    return json_path


@pytest.fixture
def sample_audio_signal():
    """Generate a sample audio signal with an impulse."""
    fs = 48000
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Background noise
    noise = 0.01 * np.random.randn(len(t))
    
    # Impulse at t=2.0 seconds
    impulse_idx = int(2.0 * fs)
    impulse = np.zeros_like(t)
    impulse[impulse_idx:impulse_idx+100] = 0.5 * np.exp(-np.linspace(0, 5, 100))
    
    signal = noise + impulse
    return signal, fs, impulse_idx


@pytest.fixture
def synthetic_mic_data():
    """Create synthetic microphone data for testing."""
    # Source position (meters in local coordinates) - positioned to create unique distances
    source_pos = np.array([3.0, 4.0])
    
    # Microphone positions (meters in local coordinates) - asymmetric layout
    mic_positions = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [8.0, 8.0],
        [0.0, 6.0]
    ])
    
    # Speed of sound
    c = 343.0  # m/s
    
    # Calculate true arrival times
    distances = np.linalg.norm(mic_positions - source_pos[None, :], axis=1)
    arrival_times = distances / c
    
    # Add some clock offsets (first mic is reference with offset 0)
    clock_offsets = np.array([0.0, 0.001, -0.002, 0.0015])
    observed_times = arrival_times + clock_offsets
    
    return {
        'source_pos': source_pos,
        'mic_positions': mic_positions,
        'arrival_times': arrival_times,
        'clock_offsets': clock_offsets,
        'observed_times': observed_times,
        'speed_of_sound': c
    }


@pytest.fixture
def synthetic_audio_files(temp_dir, synthetic_mic_data):
    """Create synthetic audio files with realistic timing."""
    fs = 48000
    duration = 3.0
    files = []
    
    for i, obs_time in enumerate(synthetic_mic_data['observed_times']):
        # Create signal with impulse at the observed time
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        noise = 0.01 * np.random.randn(len(t))
        
        # Place impulse at observed time
        impulse_idx = int(obs_time * fs)
        if 0 <= impulse_idx < len(t) - 200:
            impulse = np.zeros_like(t)
            decay = np.exp(-np.linspace(0, 8, 200))
            impulse[impulse_idx:impulse_idx+200] = 0.3 * decay
            signal = noise + impulse
        else:
            signal = noise
        
        # Save as WAV file
        filename = f"mic_{i}.wav"
        filepath = os.path.join(temp_dir, filename)
        sf.write(filepath, signal, fs)
        files.append(filepath)
    
    return files


@pytest.fixture
def mock_ffmpeg_available(monkeypatch):
    """Mock ffmpeg availability."""
    def mock_which(name):
        if name in ('ffmpeg', 'ffprobe'):
            return '/usr/bin/' + name
        return None
    
    import shutil
    monkeypatch.setattr(shutil, 'which', mock_which)


# Tolerance settings for numerical comparisons
@pytest.fixture
def numerical_tolerance():
    """Standard numerical tolerances for tests."""
    return {
        'rtol': 1e-5,
        'atol': 1e-8,
        'position_tol_m': 0.1,  # 10 cm position tolerance
        'time_tol_s': 1e-4,     # 0.1 ms time tolerance
        'angle_tol_deg': 1.0    # 1 degree angle tolerance
    }