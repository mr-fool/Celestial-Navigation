# Celestial Navigation for Military Vehicles

## Project Overview

This project simulates and analyzes the performance of celestial navigation systems for military ground vehicles in GPS-denied environments. The research focuses on developing and testing progressive star identification algorithms suitable for vehicle-mounted camera systems operating under various military operational scenarios.

## Research Context

Modern military operations face increasing vulnerabilities from GPS jamming and spoofing. This research addresses the critical need for alternative Position, Navigation, and Timing (PNT) systems by investigating celestial navigation techniques adapted for military truck platforms. The simulation models real-world environmental challenges including urban canyons, forest obstructions, dust storms, and vehicle motion.

## Key Features

- **Monte Carlo Simulation**: 1000 trials per operational scenario (6000 total trials)
- **Progressive Algorithm Strategy**: Three-tiered approach (Liebe → Voting → Pyramid)
- **Military Operational Scenarios**: Six realistic environmental conditions
- **Performance Metrics**: Success rates, computation times, algorithm effectiveness
- **Visual Analysis**: Four comprehensive figures for research publication

## File Structure
```
celestial_navigation/
├── celestial_nav_simulation.py # Main simulation engine
├── visualize.py # Figure generation for paper
├── figures/ # Generated visualization outputs
├── results/ # Monte Carlo simulation reports
└── README.md
```

## Technical Implementation

### Core Algorithms

1. **Liebe's Triangle Algorithm** - Fastest method using minimal star patterns
2. **Geometric Voting** - Robust approach with pattern consistency verification  
3. **Pyramid Algorithm** - Most reliable lost-in-space solution with redundancy

### Operational Scenarios

- Optimal WideField (Baseline performance)
- Clear Rural Base (Standard conditions)
- Urban Canyon Restricted (Building obstructions)
- Forest Canopy Obscured (Limited sky visibility)
- Dust Storm HighNoise (Atmospheric degradation)
- Vehicle Motion Extreme (Vibration and motion blur)

### Simulation Parameters

- **Star Catalog**: Tycho-2 catalog
- **Match Tolerance**: 15 arcseconds
- **Field of View**: 8-30 degrees (scenario-dependent)
- **Noise Levels**: 0.8x to 3.5x baseline measurement error

## Requirements

- Python 3.7+
- matplotlib
- numpy

Install dependencies:
```bash
pip install matplotlib numpy
```
## Usage

### Running the Simulation

Execute the main simulation to generate Monte Carlo results:
```bash
python celestial_nav_simulation.py
```
This will:

- Run 1000 trials for each of 6 operational scenarios

- Generate comprehensive performance statistics

- Save detailed results to timestamped files in `results/` directory

## Output Files

### Simulation Results
- `results/monte_carlo_results_[trials]_[timestamp].txt`: Detailed statistical analysis including success rates, algorithm performance, and environmental impact assessment

## Visualization Outputs
- `fig1_success_rates.png`: Bar chart of success rates across scenarios
- `fig2_algorithm_performance.png`: Stacked bar chart showing progressive strategy effectiveness
- `fig3_environmental_analysis.png`: Multi-panel analysis of noise, star availability, and computation time impacts
- `fig4_sensor_layout.png`: Technical diagram of vehicle-mounted sensor configuration

## Key Findings
The simulation demonstrates:
- **Progressive Strategy Effectiveness**: Fallback algorithms provide significant additional coverage beyond primary methods
- **Environmental Resilience**: System maintains functionality across diverse military operational environments
- **Computational Efficiency**: All algorithms complete within milliseconds, suitable for real-time vehicle applications
- **Operational Viability**: Overall system success rates support practical military implementation

## Military Applications
- GPS-denied navigation for convoy operations
- Backup positioning in electronic warfare environments
- Autonomous vehicle navigation in contested areas
- Special operations in denied territories

## Research Significance

This work contributes to:

- Development of robust PNT alternatives for military vehicles
- Understanding of celestial navigation limitations in ground applications
- Algorithm optimization for real-time embedded systems
- Environmental impact assessment for military operations

## Future Work

- Integration with inertial navigation systems
- Machine learning enhancements for star identification
- Real-world hardware implementation and testing
- Extended scenarios including daytime and adverse weather operations
