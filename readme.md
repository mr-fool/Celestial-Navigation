# Celestial Navigation for Ground Vehicles

## Project Overview

This project simulates and analyzes the performance of celestial navigation systems for ground vehicles operating in GPS-denied environments. The research develops and tests a progressive star identification algorithm strategy suitable for vehicle-mounted camera systems under challenging conditions, including sensor noise, limited sky visibility, and constrained computational resources.

## Research Context

Ground vehicle navigation is increasingly vulnerable to GPS outages, jamming, and spoofing. This research addresses the need for robust alternative Position, Navigation, and Timing (PNT) inputs by investigating celestial navigation techniques adapted for land-based platforms. Unlike prior work focused on maritime and aviation applications, this simulation targets the under-studied problem of land-based star identification under realistic sensor and sky-visibility constraints. This code evaluates star identification and camera-orientation recovery only; it does not implement a geographic position or timing fix. See the accompanying paper's Limitations section for the full scope discussion.

## Key Features

- **Monte Carlo Simulation**: 1,000 trials per scenario (6,000 total trials)
- **Progressive Algorithm Strategy**: Three-tiered approach (Liebe → Voting → Pyramid)
- **Challenging Environment Scenarios**: Six scenarios covering a range of angular noise, sky obstruction, and visibility conditions
- **Illustrative GPS Degradation Model**: Positional error growth curves comparing Dead Reckoning, Celestial Nav, and INS+Celestial hybrid over 60 minutes post-GPS denial. This model is exploratory only; its position-error outputs are parametric assumptions, not derived from the attitude errors measured in the Monte Carlo simulation, and are not treated as validated results.
- **Time-to-First-Identification (TTFI) Simulation**: Measures latency to first successful star identification across all scenarios
- **Performance Metrics**: Success rates, attitude RMS error with 95% CI, computation times, algorithm contribution
- **Visual Analysis**: Six publication-ready figures

## File Structure

```
celestial_navigation/
├── celestial_nav_simulation.py   # Main simulation engine
├── star_catalog.py               # Synthetic star field generator (Tycho-2-like statistical properties)
├── validation_framework.py       # Performance validation and error analysis
├── visualize.py                  # Figure generation (6 figures)
├── figures/                      # Generated visualization outputs
├── results/                      # Monte Carlo simulation reports
└── README.md
```

## Technical Implementation

### Core Algorithms

1. **Liebe's Triangle Algorithm** — Fastest method using minimal star patterns (3+ stars); primary solver in low-noise conditions
2. **Geometric Voting** — Robust approach with pattern consistency verification and 75% vote threshold; fallback under moderate noise
3. **Pyramid Algorithm** — Most reliable lost-in-space (LIS) solution with high redundancy (4+ stars); final fallback

### Operational Scenarios

| Scenario | FOV | Stars | Noise | Obscuration Prob. | Conditions |
|---|---|---|---|---|---|
| Open_Sky_Baseline | 30° | 10 | 0.8× | 0.05 | Clear sky, minimal platform noise; performance ceiling |
| Wide_FOV_Low_Noise | 20° | 7 | 1.0× | 0.10 | Open terrain, modest platform noise |
| Moderate_Obstruction_Moderate_Noise | 10° | 4 | 1.5× | 0.40 | Reduced sky access (e.g., urban canyon), moderate noise |
| Narrow_FOV_High_Noise | 12° | 5 | 2.5× | 0.30 | Reduced sky access with elevated angular noise |
| Moderate_FOV_Severe_Noise | 15° | 6 | 3.5× | 0.20 | Highest noise level tested, moderate sky access |
| Severe_Sky_Obstruction | 8° | 3 | 1.2× | 0.50 | Dense canopy or confined terrain; minimum viable star count |

These six scenarios are parameter combinations chosen to span a plausible operating range; they are not calibrated to any specific fielded sensor or named adversary system. See the paper's Simulation Parameters section for the full discussion.

### Additional Simulations

**Illustrative GPS-Denied Degradation Model (`run_gps_denied_degradation_simulation`)**
- Models positional error growth over 60 minutes after a GPS denial event
- Three modes: Dead Reckoning only, Celestial Navigation with periodic fixes, INS+Celestial hybrid
- 200 Monte Carlo trajectories per scenario; outputs mean, std, and 95th percentile error curves
- **Not a validated result.** The celestial-fix position residuals used in this model are parametric assumptions, not values computed from this simulation's own attitude-error measurements. Provided for illustrative purposes only; see the paper's Limitations section.

**Time-to-First-Identification Simulation (`run_ttfi_simulation`)**
- Measures time-to-first-identification (TTFI) after GPS loss across all scenarios
- Models sensor warmup, algorithm cascade time, and retry backoff
- 500 trials per scenario; outputs mean, median, and 95th percentile TTFI

### Simulation Parameters

- **Star Field**: 535-star synthetic catalog constructed to approximate the statistical properties of the real Tycho-2 catalog (magnitude distribution and sky density). Fifteen stars are fixed at real bright-star positions; the remaining 520 are randomly generated. No live Tycho-2 data is loaded or queried; geometric and statistical properties have not been validated against the real catalog.
- **Match Tolerance**: 15 arcseconds
- **Field of View**: 8–30 degrees (scenario-dependent)
- **Noise Levels**: 0.8× to 3.5× baseline measurement error, applied as additive uniform-distributed error to each star's right ascension and declination
- **Obscuration Probability**: 5–50% sky obstruction

## Requirements

- Python 3.7+
- `numpy`
- `scipy`
- `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Usage

### Running the Full Simulation

```bash
python celestial_nav_simulation.py
```

This runs:
- 1,000 Monte Carlo trials per scenario (6,000 total)
- Illustrative GPS degradation model (200 trajectories per scenario)
- TTFI simulation (500 trials per scenario)
- Saves a timestamped report to `results/`

### Generating All Figures

```bash
python visualize.py
```

## Output Files

### Simulation Results
- `results/simulation_results_[timestamp].txt` — Full statistical report including success rates, attitude error with 95% CI, algorithm distribution, illustrative degradation curves, and TTFI statistics

### Figures
- `fig1_success_rates.png` — Identification success rates across scenarios
- `fig2_algorithm_performance.png` — Stacked bar chart showing progressive fallback strategy contribution
- `fig3_environmental_analysis.png` — Multi-panel: noise vs. success, star availability, computation time, attitude error
- `fig4_sensor_layout.png` — Technical diagram of vehicle-mounted sensor angular FOV
- `fig5_gps_degradation_curve.png` — Illustrative, non-validated positional error growth after GPS denial: DR vs. Celestial Nav vs. INS hybrid
- `fig6_ttfi_latency.png` — Time-to-first-identification by scenario; fix success rate vs. TTFI scatter

## Key Findings

- **Progressive Strategy**: Voting and Pyramid fallbacks provide meaningful additional coverage over Liebe alone, particularly in high-noise conditions
- **Sky Access as the Binding Constraint**: The system maintains high identification success across most scenarios; the 3-star visibility case (Severe_Sky_Obstruction) is the binding constraint, not angular noise
- **Computational Efficiency**: All algorithms complete within milliseconds — suitable for real-time embedded vehicle applications
- **TTFI**: Mean time-to-first-identification is under 4 seconds in five of six scenarios; rises sharply under severe sky obstruction (mean 30.56s, 32.4% eventual fix rate)
- **Scope**: These findings describe star identification and camera-orientation recovery only. The illustrative degradation model above is not a validated finding; no position or timing fix is implemented or demonstrated by this codebase.

## Applications

This codebase evaluates the feasibility of the star-identification step only. Potential downstream applications, contingent on future work implementing the missing position/timing architecture, include:

- GPS-denied navigation for convoy and logistics operations
- Backup PNT input in signal-degraded environments
- Autonomous vehicle navigation in GPS-unreliable areas
- Navigation in geographically challenging terrain (dense forest, tunnels, mountain passes)
- Passive sensing (no RF emissions) in emissions-controlled environments

## Research Significance

This work contributes to:

- Development of robust, passive, non-RF-dependent inputs for layered ground-vehicle PNT architectures
- Quantitative assessment of star-identification performance and its limits under varying sky-visibility and noise conditions
- Algorithm optimization for real-time embedded systems
- An illustrative, non-validated exploration of how positional error might grow under GPS denial, intended to motivate future work rather than to stand as a measured result
