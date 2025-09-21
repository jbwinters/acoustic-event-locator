#!/usr/bin/env python3
"""
Test runner script for event location detector.
"""

import sys
import subprocess
import os

def run_tests():
    """Run the complete test suite."""
    
    print("Event Location Detector - Test Suite")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest not found. Install with:")
        print("pip install -r requirements-test.txt")
        sys.exit(1)
    
    # Check if main dependencies are available
    try:
        import numpy
        import scipy
        import matplotlib
        import soundfile
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install -r requirements-test.txt")
        sys.exit(1)
    
    # Run tests with different configurations
    test_configs = [
        {
            'name': 'Unit Tests',
            'args': ['-m', 'pytest', 'tests/', '-v', '--tb=short']
        },
        {
            'name': 'Integration Tests',
            'args': ['-m', 'pytest', 'tests/test_integration.py', '-v']
        },
        {
            'name': 'Coverage Report',
            'args': ['-m', 'pytest', 'tests/', '--cov=locate_event', '--cov-report=term-missing']
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Running {config['name']} ---")
        try:
            result = subprocess.run([sys.executable] + config['args'], 
                                  capture_output=False, 
                                  check=False)
            if result.returncode != 0:
                print(f"WARNING: {config['name']} had some failures")
        except Exception as e:
            print(f"ERROR running {config['name']}: {e}")
    
    print("\n" + "=" * 50)
    print("Test suite completed!")


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    cmd = [sys.executable, '-m', 'pytest', test_path, '-v']
    subprocess.run(cmd)


def run_with_coverage():
    """Run tests with detailed coverage analysis."""
    cmd = [
        sys.executable, '-m', 'pytest', 
        'tests/', 
        '--cov=locate_event', 
        '--cov-report=html',
        '--cov-report=term-missing',
        '--cov-branch'
    ]
    subprocess.run(cmd)
    print("\nCoverage report generated in htmlcov/index.html")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--coverage":
            run_with_coverage()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py --coverage   # Run with coverage")
            print("  python run_tests.py test_file.py # Run specific test")
        else:
            run_specific_test(sys.argv[1])
    else:
        run_tests()