# Test Data for Event Location Detector

This directory contains synthetic test datasets for evaluating the acoustic event location detector system.

## Overview

The test data includes three realistic scenarios with different microphone array geometries and acoustic event types:

1. **scenario1_gunshot** - Urban gunshot detection with 4 cameras in square formation
2. **scenario2_explosion** - Industrial explosion with 5 cameras in linear array
3. **scenario3_fireworks** - Fireworks display with 6 cameras in L-shaped array

## File Structure

Each scenario directory contains:
- `positions.json` - Microphone positions and configuration
- `cam*.mp4` - Synthetic video files with audio tracks
- `metadata.json` - Ground truth data for validation

## Generating Test Data

To regenerate the test data:

```bash
# Generate all scenarios
python generate_test_data.py

# Generate specific scenarios
python generate_test_data.py --scenarios scenario1_gunshot scenario2_explosion
```

## Running Tests

To test the event location detector on these datasets:

```bash
# Test scenario 1 (gunshot)
python locate_event.py test_data/scenario1_gunshot/positions.json

# Test scenario 2 (explosion) 
python locate_event.py test_data/scenario2_explosion/positions.json

# Test scenario 3 (fireworks)
python locate_event.py test_data/scenario3_fireworks/positions.json
```

## Scenario Details

### Scenario 1: Urban Gunshot
- **Location**: Downtown Chicago intersection
- **Event**: Single gunshot at street level
- **Microphones**: 4 cameras at building corners (square ~200m array)
- **Environment**: Urban with moderate ambient noise
- **Expected accuracy**: ~5-10m due to good geometry

### Scenario 2: Industrial Explosion
- **Location**: Factory district 
- **Event**: Industrial explosion from storage building
- **Microphones**: 5 cameras along perimeter fence (linear ~600m array)
- **Environment**: Industrial with machinery noise
- **Expected accuracy**: ~10-20m due to linear geometry

### Scenario 3: Fireworks Display
- **Location**: Millennium Park during show
- **Event**: Aerial firework burst with crackling
- **Microphones**: 6 cameras in L-shaped array around stage/audience
- **Environment**: Outdoor with crowd noise
- **Expected accuracy**: ~5-15m due to good L-shaped geometry

## Validation Data

Each scenario includes ground truth data in `metadata.json`:
- True source position in local coordinates
- Microphone positions 
- Clock offsets between devices
- Theoretical arrival times
- Signal parameters

## Synthetic Signal Characteristics

The generated audio includes:
- **Gunshot**: Sharp impulse with brief ringing (~10ms duration)
- **Explosion**: Multi-frequency burst with decay (~500ms duration)  
- **Fireworks**: Initial crack followed by crackling (~1s total duration)
- **Background**: Gaussian white noise at realistic levels
- **Physics**: Distance-based attenuation and propagation delays
- **Clock drift**: Realistic timing offsets between unsynchronized devices

## Usage Tips

1. **Validate installation**: Run the scripts on test data before real deployment
2. **Parameter tuning**: Use metadata.json to verify algorithm accuracy
3. **Geometry effects**: Compare results across different array configurations
4. **Noise robustness**: Test with varying background noise levels
5. **Clock synchronization**: Observe impact of timing offsets on accuracy

## Expected Results

With default parameters, the location detector should achieve:
- Position accuracy within 5-20m depending on geometry
- Clock offset estimation within 1-5ms 
- Successful detection and localization for all scenarios
- Confidence ellipses reflecting geometry-dependent uncertainty

Results will vary based on:
- Microphone array geometry (square > L-shaped > linear)
- Event signal strength and characteristics
- Background noise levels
- Clock synchronization quality