#!/usr/bin/env python3
"""
Run the event location detector on all test scenarios and compare results with ground truth.
"""

import subprocess
import json
import os
import sys
import numpy as np
from pathlib import Path

def load_metadata(scenario_dir):
    """Load ground truth metadata for a scenario."""
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def run_scenario(scenario_dir, verbose=True):
    """Run the location detector on a scenario and return results."""
    positions_file = os.path.join(scenario_dir, 'positions.json')
    
    if not os.path.exists(positions_file):
        print(f"Error: {positions_file} not found")
        return None
    
    if verbose:
        print(f"\n=== Running {os.path.basename(scenario_dir)} ===")
        print(f"Config: {positions_file}")
    
    # Run the location detector
    try:
        result = subprocess.run([
            sys.executable, 'locate_event.py', 
            '--videos_dir', scenario_dir,
            '--positions', positions_file
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Error running detector: {result.stderr}")
            return None
            
        if verbose:
            print("Detector output:")
            print(result.stdout)
            
        return result.stdout
        
    except subprocess.TimeoutExpired:
        print("Error: Detector timed out after 2 minutes")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_results(output_text):
    """Parse the detector output to extract results."""
    results = {}
    
    lines = output_text.split('\n')
    for line in lines:
        if 'Estimated location (local m):' in line:
            # Extract position coordinates: "x=1.712, y=0.946"
            try:
                parts = line.split('x=')[1].split(',')
                x = float(parts[0].strip())
                y = float(parts[1].split('y=')[1].strip())
                results['position'] = [x, y]
            except:
                pass
        elif line.strip().startswith('[INFO] Mic') and 'clock offset' in line:
            # Extract clock offsets from lines like "[INFO] Mic #1 clock offset: -0.002s"
            if 'clock_offsets' not in results:
                results['clock_offsets'] = []
            try:
                offset = float(line.split(':')[-1].strip().rstrip('s'))
                results['clock_offsets'].append(offset)
            except:
                pass
    
    return results

def compare_results(estimated, ground_truth, scenario_name):
    """Compare estimated results with ground truth."""
    print(f"\n--- {scenario_name} Validation ---")
    
    # Position comparison
    if 'position' in estimated:
        est_pos = np.array(estimated['position'])
        true_pos = np.array(ground_truth['source_position_m'])
        
        position_error = np.linalg.norm(est_pos - true_pos)
        print(f"Position Error: {position_error:.1f} m")
        print(f"  Estimated: ({est_pos[0]:.1f}, {est_pos[1]:.1f}) m")
        print(f"  True:      ({true_pos[0]:.1f}, {true_pos[1]:.1f}) m")
        
        # Assess accuracy
        if position_error < 5.0:
            print("  ✓ Excellent accuracy (< 5m)")
        elif position_error < 15.0:
            print("  ✓ Good accuracy (< 15m)")
        elif position_error < 50.0:
            print("  ⚠ Moderate accuracy (< 50m)")
        else:
            print("  ✗ Poor accuracy (> 50m)")
    else:
        print("  ✗ No position estimate found")
    
    # Clock offset comparison  
    if 'clock_offsets' in estimated and len(estimated['clock_offsets']) > 0:
        est_offsets = np.array(estimated['clock_offsets'])
        true_offsets = np.array(ground_truth['clock_offsets_s'])
        
        if len(est_offsets) == len(true_offsets):
            # Compare relative offsets (subtract first mic as reference)
            est_rel = est_offsets - est_offsets[0]
            true_rel = true_offsets - true_offsets[0]
            
            offset_errors = np.abs(est_rel - true_rel)
            max_offset_error = np.max(offset_errors)
            mean_offset_error = np.mean(offset_errors)
            
            print(f"Clock Offset Errors: max={max_offset_error*1000:.1f}ms, mean={mean_offset_error*1000:.1f}ms")
            
            if max_offset_error < 0.005:  # 5ms
                print("  ✓ Good clock synchronization (< 5ms)")
            elif max_offset_error < 0.010:  # 10ms
                print("  ✓ Acceptable synchronization (< 10ms)")
            else:
                print("  ⚠ Poor synchronization (> 10ms)")
        else:
            print(f"  ⚠ Clock offset count mismatch: {len(est_offsets)} vs {len(true_offsets)}")
    else:
        print("  ⚠ No clock offset estimates found")

def main():
    """Run all test scenarios and validate results."""
    test_data_dir = 'test_data'
    
    if not os.path.exists(test_data_dir):
        print("Error: test_data directory not found. Run generate_test_data.py first.")
        return 1
    
    scenarios = [
        'scenario1_gunshot',
        'scenario2_explosion', 
        'scenario3_fireworks'
    ]
    
    print("Event Location Detector - Test Scenario Runner")
    print("=" * 50)
    
    results_summary = []
    
    for scenario in scenarios:
        scenario_dir = os.path.join(test_data_dir, scenario)
        
        if not os.path.exists(scenario_dir):
            print(f"Warning: {scenario_dir} not found, skipping...")
            continue
        
        # Load ground truth
        try:
            metadata = load_metadata(scenario_dir)
        except Exception as e:
            print(f"Error loading metadata for {scenario}: {e}")
            continue
        
        # Run detector
        output = run_scenario(scenario_dir)
        if output is None:
            print(f"Failed to run {scenario}")
            continue
        
        # Parse and validate results
        estimated = parse_results(output)
        compare_results(estimated, metadata, scenario)
        
        # Store for summary
        results_summary.append({
            'scenario': scenario,
            'estimated': estimated,
            'ground_truth': metadata
        })
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for result in results_summary:
        scenario = result['scenario']
        est = result['estimated']
        truth = result['ground_truth']
        
        if 'position' in est:
            est_pos = np.array(est['position'])
            true_pos = np.array(truth['source_position_m'])
            error = np.linalg.norm(est_pos - true_pos)
            print(f"{scenario}: {error:.1f}m position error")
        else:
            print(f"{scenario}: FAILED")
    
    print("\nTest scenarios complete!")
    return 0

if __name__ == '__main__':
    exit(main())