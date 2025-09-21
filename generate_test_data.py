#!/usr/bin/env python3
"""
Generate realistic synthetic audio data for testing the event location detector.
Creates MP4 video files with audio tracks containing acoustic events at known times.
"""

import numpy as np
import soundfile as sf
import subprocess
import json
import os
import argparse
from pathlib import Path

# Import functions from the main script
import sys
sys.path.append('.')
from locate_event import latlon_to_local_xy, load_positions


def generate_impulse_signal(fs=48000, duration=10.0, impulse_time=5.0, noise_level=0.01):
    """Generate a signal with an impulse at a specified time."""
    n_samples = int(duration * fs)
    signal = noise_level * np.random.randn(n_samples)
    
    # Add impulse
    impulse_idx = int(impulse_time * fs)
    if 0 <= impulse_idx < n_samples:
        # Create a sharp impulse with some ringing
        impulse_duration = int(0.01 * fs)  # 10ms
        t_impulse = np.arange(impulse_duration) / fs
        impulse = np.exp(-t_impulse * 100) * np.sin(2 * np.pi * 200 * t_impulse)
        
        end_idx = min(impulse_idx + impulse_duration, n_samples)
        signal[impulse_idx:end_idx] += 0.5 * impulse[:end_idx-impulse_idx]
    
    return signal


def generate_explosion_signal(fs=48000, duration=10.0, event_time=5.0, noise_level=0.01):
    """Generate a signal with an explosion-like event."""
    n_samples = int(duration * fs)
    signal = noise_level * np.random.randn(n_samples)
    
    # Add explosion - longer duration with multiple frequency components
    event_idx = int(event_time * fs)
    if 0 <= event_idx < n_samples:
        explosion_duration = int(0.5 * fs)  # 500ms
        t_exp = np.arange(explosion_duration) / fs
        
        # Multiple frequency components with decay
        explosion = (
            0.8 * np.exp(-t_exp * 5) * np.sin(2 * np.pi * 80 * t_exp) +
            0.6 * np.exp(-t_exp * 10) * np.sin(2 * np.pi * 150 * t_exp) +
            0.4 * np.exp(-t_exp * 20) * np.sin(2 * np.pi * 300 * t_exp)
        )
        
        end_idx = min(event_idx + explosion_duration, n_samples)
        signal[event_idx:end_idx] += explosion[:end_idx-event_idx]
    
    return signal


def generate_firework_signal(fs=48000, duration=10.0, event_time=5.0, noise_level=0.01):
    """Generate a signal with a firework burst."""
    n_samples = int(duration * fs)
    signal = noise_level * np.random.randn(n_samples)
    
    # Add firework burst - sharp crack followed by crackles
    event_idx = int(event_time * fs)
    if 0 <= event_idx < n_samples:
        # Initial crack
        crack_duration = int(0.02 * fs)  # 20ms
        t_crack = np.arange(crack_duration) / fs
        crack = 0.7 * np.exp(-t_crack * 200) * np.sin(2 * np.pi * 500 * t_crack)
        
        # Add the crack
        end_idx = min(event_idx + crack_duration, n_samples)
        signal[event_idx:end_idx] += crack[:end_idx-event_idx]
        
        # Add some crackling afterwards
        crackle_start = event_idx + crack_duration
        crackle_duration = int(1.0 * fs)  # 1 second of crackling
        if crackle_start < n_samples:
            crackle_end = min(crackle_start + crackle_duration, n_samples)
            n_crackles = crackle_end - crackle_start
            crackles = 0.2 * np.random.exponential(0.1, n_crackles) * np.random.randn(n_crackles)
            signal[crackle_start:crackle_end] += crackles
    
    return signal


def calculate_arrival_times(source_pos, mic_positions, c=343.0):
    """Calculate arrival times for each microphone."""
    distances = np.linalg.norm(mic_positions - source_pos[np.newaxis, :], axis=1)
    travel_times = distances / c
    return travel_times


def generate_scenario_audio(scenario_dir, event_type='gunshot', source_pos=None, clock_offsets=None):
    """Generate audio files for a specific scenario."""
    positions_file = os.path.join(scenario_dir, 'positions.json')
    
    # Read positions JSON directly (can't use load_positions since videos don't exist yet)
    with open(positions_file, 'r') as f:
        raw_json = json.load(f)
    
    # Extract reference point
    if 'reference_point' in raw_json:
        lat0 = raw_json['reference_point']['lat']
        lon0 = raw_json['reference_point']['lon']
    else:
        # Calculate mean of mic positions
        lats = [mic['lat'] for mic in raw_json['mics']]
        lons = [mic['lon'] for mic in raw_json['mics']]
        lat0 = sum(lats) / len(lats)
        lon0 = sum(lons) / len(lons)
    
    # Get speed of sound
    if 'speed_of_sound' in raw_json:
        c = raw_json['speed_of_sound']
    elif 'temperature_C' in raw_json:
        c = 331.4 + 0.6 * raw_json['temperature_C']
    else:
        c = 343.0  # Default at 20°C
    
    # Convert mic positions to local coordinates
    mic_positions = []
    mic_filenames = []
    for mic_data in raw_json['mics']:
        x, y = latlon_to_local_xy(mic_data['lat'], mic_data['lon'], lat0, lon0)
        mic_positions.append([x, y])
        mic_filenames.append(mic_data['file'])
    
    mic_positions = np.array(mic_positions)
    
    # Use provided source position or estimate from JSON
    if source_pos is None:
        if 'event' in raw_json and 'estimated_location' in raw_json['event']:
            event_lat = raw_json['event']['estimated_location']['lat']
            event_lon = raw_json['event']['estimated_location']['lon']
            source_x, source_y = latlon_to_local_xy(event_lat, event_lon, lat0, lon0)
            source_pos = np.array([source_x, source_y])
        else:
            # Default to center of mic array
            source_pos = np.mean(mic_positions, axis=0)
    
    # Generate clock offsets if not provided
    if clock_offsets is None:
        clock_offsets = np.array([0.0, 0.002, -0.001, 0.0015, -0.0005, 0.001])[:len(mic_filenames)]
    
    # Calculate true arrival times
    arrival_times = calculate_arrival_times(source_pos, mic_positions, c)
    
    # Audio parameters
    fs = 48000
    duration = 10.0
    event_time = 5.0  # Event occurs at 5 seconds
    
    print(f"Generating audio for {len(mic_filenames)} microphones in {scenario_dir}")
    print(f"Source position: ({source_pos[0]:.1f}, {source_pos[1]:.1f}) m")
    print(f"Speed of sound: {c:.1f} m/s")
    
    # Generate signal for each microphone
    for i, filename in enumerate(mic_filenames):
        print(f"  Generating {filename}...")
        
        # Calculate when the event arrives at this microphone
        mic_event_time = event_time + arrival_times[i] + clock_offsets[i]
        
        # Generate appropriate signal type
        if event_type == 'gunshot':
            signal = generate_impulse_signal(fs, duration, mic_event_time)
        elif event_type == 'explosion':
            signal = generate_explosion_signal(fs, duration, mic_event_time)
        elif event_type == 'fireworks':
            signal = generate_firework_signal(fs, duration, mic_event_time)
        else:
            signal = generate_impulse_signal(fs, duration, mic_event_time)
        
        # Add distance-based attenuation
        distance = np.linalg.norm(mic_positions[i] - source_pos)
        attenuation = 1.0 / (1.0 + distance / 100.0)  # Simple distance attenuation
        signal *= attenuation
        
        # Save as temporary WAV file
        wav_path = os.path.join(scenario_dir, f'temp_{filename}.wav')
        sf.write(wav_path, signal, fs)
        
        # Convert to MP4 using FFmpeg
        mp4_path = os.path.join(scenario_dir, filename)
        cmd = [
            'ffmpeg', '-y',  # Overwrite existing files
            '-f', 'lavfi', '-i', f'color=black:size=640x480:duration={duration}:rate=30',  # Black video
            '-i', wav_path,  # Audio input
            '-c:v', 'libx264', '-c:a', 'aac',  # Codecs
            '-shortest',  # End when shortest stream ends
            mp4_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"    Created {mp4_path}")
            
            # Clean up temporary WAV file
            os.remove(wav_path)
            
        except subprocess.CalledProcessError as e:
            print(f"    Error creating {mp4_path}: {e}")
            print(f"    FFmpeg output: {e.stderr.decode()}")
    
    # Save metadata about the generated scenario
    metadata = {
        'source_position_m': source_pos.tolist(),
        'microphone_positions_m': mic_positions.tolist(),
        'clock_offsets_s': clock_offsets[:len(mic_filenames)].tolist(),
        'arrival_times_s': arrival_times.tolist(),
        'event_time_s': event_time,
        'speed_of_sound_ms': c,
        'sample_rate_hz': fs,
        'duration_s': duration,
        'event_type': event_type
    }
    
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved to {metadata_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Generate test data for event location detector')
    parser.add_argument('--scenarios', nargs='+', 
                       choices=['scenario1_gunshot', 'scenario2_explosion', 'scenario3_fireworks', 'all'],
                       default=['all'],
                       help='Which scenarios to generate')
    
    args = parser.parse_args()
    
    # Ensure we have ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg is required but not found. Please install FFmpeg.")
        return 1
    
    scenarios = args.scenarios
    if 'all' in scenarios:
        scenarios = ['scenario1_gunshot', 'scenario2_explosion', 'scenario3_fireworks']
    
    base_dir = 'test_data'
    
    for scenario in scenarios:
        scenario_dir = os.path.join(base_dir, scenario)
        if not os.path.exists(scenario_dir):
            print(f"Warning: {scenario_dir} does not exist, skipping...")
            continue
        
        # Determine event type from scenario name
        if 'gunshot' in scenario:
            event_type = 'gunshot'
        elif 'explosion' in scenario:
            event_type = 'explosion'
        elif 'fireworks' in scenario:
            event_type = 'fireworks'
        else:
            event_type = 'gunshot'  # default
        
        try:
            generate_scenario_audio(scenario_dir, event_type)
        except Exception as e:
            print(f"Error generating {scenario}: {e}")
    
    print("Test data generation complete!")
    return 0


if __name__ == '__main__':
    exit(main())