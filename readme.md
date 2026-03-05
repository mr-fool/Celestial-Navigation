# Celestial Navigation for Military Ground Vehicles

## Project Overview

This project simulates and analyzes the performance of celestial navigation systems for military ground vehicles in GPS-denied environments. The research develops and tests a progressive star identification algorithm strategy suitable for vehicle-mounted camera systems operating under adversary-threat conditions, including electronic warfare saturation, ISR-contested airspace, urban C2 denial, and geographically denied territory.

## Research Context

Modern military operations face increasing vulnerabilities from GPS jamming and spoofing by near-peer adversaries. This research addresses the critical need for alternative Position, Navigation, and Timing (PNT) systems by investigating celestial navigation techniques adapted for military ground vehicle platforms. Unlike prior work focused on maritime and aviation applications, this simulation targets the under-studied problem of land-based celestial navigation under realistic operational constraints.

## Key Features

- **Monte Carlo Simulation**: 1,000 trials per operational scenario (6,000 total trials)
- **Progressive Algorithm Strategy**: Three-tiered approach (Liebe → Voting → Pyramid)
- **Adversary-Threat Scenarios**: Six scenarios framed by near-peer threat vectors (Russia EW, China ISR, urban C2 denial, denied territory)
- **GPS Degradation Simulation**: Positional error growth curves comparing Dead Reckoning, Celestial Nav, and INS+Celestial hybrid over 60 minutes post-GPS denial
- **C2 Latency Simulation**: Time-to-first-fix (TTFF) measurement across all scenarios as a proxy for C2 reintegration speed
- **Performance Metrics**: Success rates, attitude RMS error with 95% CI, computation times, algorithm contribution
- **Visual Analysis**: Six publication-ready figures

## File Structure

```
celestial_navigation/
├── celestial_nav_simulation.py   # Main simulation engine
├── star_catalog.py               # Tycho-2 star catalog implementation
├── validation_framework.py       # Performance validation and error analysis
├── visualize.py                  # Figure generation for paper (6 figures)
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

| Scenario | FOV | Stars | Noise | Threat Actor |
|---|---|---|---|---|
| Baseline_Uncontested | 30° | 10 | 0.8× | None |
| EW_Degraded_Rural | 20° | 7 | 1.0× | Russia / China |
| Urban_C2_Denied | 10° | 4 | 1.5× | China |
| Russian_EW_Saturation | 12° | 5 | 2.5× | Russia |
| Chinese_ISR_Contested | 15° | 6 | 3.5× | China |
| SOCOM_Denied_Territory | 8° | 3 | 1.2× | Near-Peer / Irregular |

### Additional Simulations

**GPS-Denied Degradation (`run_gps_denied_degradation_simulation`)**
- Models positional error growth over 60 minutes after a GPS denial event
- Three modes: Dead Reckoning only, Celestial Navigation with periodic fixes, INS+Celestial hybrid
- 200 Monte Carlo trajectories per scenario; outputs mean, std, and 95th percentile error curves

**C2 Reintegration Latency (`run_jado_c2_latency_simulation`)**
- Measures time-to-first-fix (TTFF) after GPS loss across all scenarios
- Models sensor warmup, algorithm cascade time, and retry backoff
- 500 trials per scenario; outputs mean, median, and 95th percentile TTFF

### Simulation Parameters

- **Star Catalog**: Tycho-2 catalog, 535 stars
- **Match Tolerance**: 15 arcseconds
- **Field of View**: 8–30 degrees (scenario-dependent)
- **Noise Levels**: 0.8× to 3.5× baseline measurement error
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
- GPS degradation simulation (200 trajectories per scenario)
- C2 latency simulation (500 trials per scenario)
- Saves a timestamped report to `results/`

### Generating All Figures

```bash
python visualize.py
```

## Output Files

### Simulation Results
- `results/simulation_results_[timestamp].txt` — Full statistical report including success rates, attitude error with 95% CI, algorithm distribution, degradation curves, and TTFF statistics

### Figures
- `fig1_success_rates.png` — Identification success rates across adversary-threat scenarios
- `fig2_algorithm_performance.png` — Stacked bar chart showing progressive fallback strategy contribution
- `fig3_environmental_analysis.png` — Multi-panel: noise vs. success, star availability, computation time, attitude error
- `fig4_sensor_layout.png` — Technical diagram of vehicle-mounted sensor angular FOV
- `fig5_gps_degradation_curve.png` — Positional error growth after GPS denial: DR vs. Celestial Nav vs. INS hybrid
- `fig6_c2_latency.png` — Time-to-first-fix by scenario; fix success rate vs. TTFF scatter

## Key Findings

- **Progressive Strategy**: Voting and Pyramid fallbacks provide meaningful additional coverage over Liebe alone, particularly in high-noise (EW saturation) conditions
- **Environmental Resilience**: System maintains operational viability across most scenarios; SOCOM_Denied_Territory (3 visible stars) is the binding constraint
- **Computational Efficiency**: All algorithms complete within milliseconds — suitable for real-time embedded vehicle applications
- **GPS Denial Impact**: Dead reckoning error grows unbounded after GPS loss; celestial nav periodic fixes bound positional error to operationally viable levels in most scenarios
- **C2 Latency**: TTFF under 5 seconds in open/rural scenarios; rises sharply under severe sky obstruction

## Military Applications

- GPS-denied navigation for convoy and logistics operations
- Backup PNT in electronic warfare environments
- Autonomous vehicle navigation in contested areas
- Special operations in geographically denied territory
- Passive navigation (no RF emissions) in emissions-controlled environments

## Research Significance

This work contributes to:

- Development of robust passive PNT alternatives for military ground vehicles
- Quantitative assessment of celestial navigation limitations in land-based applications
- Algorithm optimization for real-time embedded systems
- Empirical modeling of positional error growth under GPS denial


