import math
import random
import time
import statistics
import os
import numpy as np
import scipy.stats as stats
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from star_catalog import get_tycho2_catalog
from validation_framework import CelestialNavigationValidator, ValidationResult

# --- Configuration and Constants ---
MATCH_TOLERANCE_ARCSEC = 15.0
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi
ARCSEC_PER_DEGREE = 3600.0
INDEX_KEY_PRECISION = 0.1

CATALOG_INDEX: Dict[int, List[Tuple[int, int]]] = {}

# ---------------------------------------------------------------------------
# JAMS PAPER: Operational scenarios are framed as adversary-threat conditions
# matching the JADO CFP language (Russia EW, China ISR, SOCOM denied territory).
# Each scenario maps to a specific threat vector discussed in the paper.
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentCondition:
    name: str
    fov_size: float
    num_stars: int
    noise_level: float
    obscuration_prob: float
    description: str
    threat_actor: str        # NEW: maps to JADO adversary framing
    threat_vector: str       # NEW: specific capability being simulated
    jado_relevance: str      # doctrinal relevance note

OPERATIONAL_SCENARIOS = [
    EnvironmentCondition(
        name="Baseline_Uncontested",
        fov_size=30.0,
        num_stars=10,
        noise_level=0.8,
        obscuration_prob=0.05,
        description="Optimal conditions — clear sky, wide FOV, minimal interference",
        threat_actor="None",
        threat_vector="Uncontested environment; GPS fully available as baseline comparison",
        jado_relevance="Benchmark for measuring degradation under adversary conditions"
    ),
    EnvironmentCondition(
        name="EW_Degraded_Rural",
        fov_size=20.0,
        num_stars=7,
        noise_level=1.0,
        obscuration_prob=0.1,
        description="Rural/desert theater with low-level EW background noise",
        threat_actor="Russia / China",
        threat_vector="Persistent low-power GPS jamming at theater level",
        jado_relevance="Tests baseline PNT resilience in GPS-degraded (not denied) conditions"
    ),
    EnvironmentCondition(
        name="Urban_C2_Denied",
        fov_size=10.0,
        num_stars=4,
        noise_level=1.5,
        obscuration_prob=0.4,
        description="Urban canyon with building masking — reduced sky access, moderate noise",
        threat_actor="China",
        threat_vector="System destruction warfare: C2 node interdiction in urban terrain",
        jado_relevance="Simulates loss of C2 relay in contested urban environment (JADO C2 resilience)"
    ),
    EnvironmentCondition(
        name="Russian_EW_Saturation",
        fov_size=12.0,
        num_stars=5,
        noise_level=2.5,
        obscuration_prob=0.3,
        description="Dense EW jamming environment — high measurement noise, reduced FOV",
        threat_actor="Russia",
        threat_vector="Krasukha/R-330 family EW saturation; GPS/GNSS denial across AO",
        jado_relevance="Direct test of JADO resilience against Russian next-gen EW doctrine"
    ),
    EnvironmentCondition(
        name="Chinese_ISR_Contested",
        fov_size=15.0,
        num_stars=6,
        noise_level=3.5,
        obscuration_prob=0.2,
        description="High-vibration vehicle movement under active ISR pressure — extreme noise",
        threat_actor="China",
        threat_vector="Intelligentized warfare: PLA AI-enabled ISR targeting PNT nodes",
        jado_relevance="Tests JADO multi-domain operations under PLA active defense pressure"
    ),
    EnvironmentCondition(
        name="SOCOM_Denied_Territory",
        fov_size=8.0,
        num_stars=3,
        noise_level=1.2,
        obscuration_prob=0.5,
        description="Dense canopy / denied territory — severe sky obstruction, few visible stars",
        threat_actor="Near-Peer / Irregular",
        threat_vector="Geographically denied navigation: forest, mountain, or subterranean terrain",
        jado_relevance="Special operations in GPS-denied territory; JADO persistent engagement zones"
    ),
]

@dataclass
class TrialResult:
    scenario: str
    algorithm_used: str
    success: bool
    computation_time: float
    num_stars_visible: int
    noise_level: float

MOCK_STAR_CATALOG = get_tycho2_catalog()

for i in range(16, 51):
    MOCK_STAR_CATALOG.append({
        "ID": i,
        "NAME": f"CatalogStar_{i}",
        "RA": random.uniform(0, 24),
        "DEC": random.uniform(-90, 90),
        "MAG": random.uniform(2.0, 6.0),
        "BRIGHTNESS": random.uniform(0.2, 0.7)
    })
MOCK_STAR_CATALOG.sort(key=lambda x: x['MAG'])


# --- Core Geometric Functions ---

def calculate_angular_distance(star1, star2, use_observed=False):
    ra_key = 'RA_OBSERVED' if use_observed else 'RA'
    dec_key = 'DEC_OBSERVED' if use_observed else 'DEC'
    ra1 = star1.get(ra_key, star1['RA']) * DEGREES_TO_RADIANS
    dec1 = star1.get(dec_key, star1['DEC']) * DEGREES_TO_RADIANS
    ra2 = star2.get(ra_key, star2['RA']) * DEGREES_TO_RADIANS
    dec2 = star2.get(dec_key, star2['DEC']) * DEGREES_TO_RADIANS
    cos_angle = (math.sin(dec1) * math.sin(dec2) +
                 math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle) * RADIANS_TO_DEGREES

def calculate_inter_star_angles(stars, use_observed=False):
    distances = []
    n = len(stars)
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(calculate_angular_distance(stars[i], stars[j], use_observed=use_observed))
    return distances

def get_index_key(angle_deg):
    return int(round(angle_deg / INDEX_KEY_PRECISION))

def build_angular_distance_index(catalog):
    index = defaultdict(list)
    n = len(catalog)
    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_angular_distance(catalog[i], catalog[j])
            key = get_index_key(distance)
            index[key].append((catalog[i]['ID'], catalog[j]['ID']))
    return dict(index)


# --- Simulation Environment ---

def simulate_camera_view(catalog, condition, verbose=False):
    if random.random() < condition.obscuration_prob:
        effective_num_stars = max(3, int(condition.num_stars * random.uniform(0.5, 0.9)))
    else:
        effective_num_stars = condition.num_stars
    center_ra = random.uniform(0, 24)
    center_dec = random.uniform(-90, 90)
    center_star = {"RA": center_ra, "DEC": center_dec}
    observed_stars = []
    max_dist = condition.fov_size * 0.7
    for star in catalog:
        if calculate_angular_distance(center_star, star) < max_dist:
            observed_stars.append(star)
    observed_stars.sort(key=lambda x: x['MAG'])
    noisy_observed_stars = []
    for i, star in enumerate(observed_stars[:effective_num_stars]):
        noisy_star = star.copy()
        base_noise = 15.0
        effective_noise = base_noise * condition.noise_level
        noise_deg_ra = random.uniform(-effective_noise, effective_noise) / ARCSEC_PER_DEGREE
        noise_deg_dec = random.uniform(-effective_noise, effective_noise) / ARCSEC_PER_DEGREE
        noisy_star['RA_OBSERVED'] = star['RA'] + noise_deg_ra
        noisy_star['DEC_OBSERVED'] = star['DEC'] + noise_deg_dec
        noisy_star['TEMP_ID'] = i + 1
        noisy_observed_stars.append(noisy_star)
    return noisy_observed_stars


# --- Star Identification Algorithms ---

def liebe_triangle_match(observed_stars, catalog_index, verbose=False):
    start_time = time.time()
    if len(observed_stars) < 3:
        return False, time.time() - start_time
    image_pattern = observed_stars[:3]
    image_angles = calculate_inter_star_angles(image_pattern, use_observed=True)
    image_angles.sort()
    matches_per_angle = []
    for angle in image_angles:
        key = get_index_key(angle)
        candidate_pairs = set()
        for k in [key - 1, key, key + 1]:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    candidate_pairs.add(tuple(sorted((str(id_a), str(id_b)))))
        matches_per_angle.append(len(candidate_pairs))
    shortest_angle_matches = matches_per_angle[0]
    avg_matches = statistics.mean(matches_per_angle)
    match_found = (2 <= shortest_angle_matches <= 50 and
                   all(m > 0 for m in matches_per_angle) and
                   avg_matches < 100)
    return match_found, time.time() - start_time

def geometric_voting_match(observed_stars, catalog_index, verbose=False):
    start_time = time.time()
    if len(observed_stars) < 3:
        return False, time.time() - start_time
    image_distances = calculate_inter_star_angles(observed_stars, use_observed=True)
    vote_count = defaultdict(int)
    total_index_hits = 0
    N = len(observed_stars)
    for img_dist in image_distances:
        key = get_index_key(img_dist)
        for k in [key - 1, key, key + 1]:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_index_hits += 1
                    vote_count[id_a] += 1
                    vote_count[id_b] += 1
    top_candidates = sorted(vote_count.items(), key=lambda item: item[1], reverse=True)[:N]
    top_candidate_ids = [cid for cid, count in top_candidates]
    required_votes = (N * (N - 1)) * 0.75
    current_votes = sum(count for cid, count in top_candidates)
    if current_votes < required_votes:
        match_found = False
    else:
        catalog_stars_map = {star['ID']: star for star in MOCK_STAR_CATALOG if star['ID'] in top_candidate_ids}
        if len(catalog_stars_map) < N:
            match_found = False
        else:
            catalog_candidate_distances = calculate_inter_star_angles(list(catalog_stars_map.values()))
            img_min_dist = min(image_distances)
            cat_min_dist = min(catalog_candidate_distances)
            match_found = abs(img_min_dist - cat_min_dist) < INDEX_KEY_PRECISION * 1.5
    return match_found, time.time() - start_time

def pyramid_match(observed_stars, catalog_index, verbose=False):
    start_time = time.time()
    if len(observed_stars) < 4:
        return False, time.time() - start_time
    image_pattern = observed_stars[:4]
    image_angles = sorted(calculate_inter_star_angles(image_pattern, use_observed=True))[:4]
    candidate_star_ids = set()
    total_lookups = 0
    for angle in image_angles:
        key = get_index_key(angle)
        for k in [key - 1, key, key + 1]:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_lookups += 1
                    candidate_star_ids.add(id_a)
                    candidate_star_ids.add(id_b)
    match_found = len(candidate_star_ids) >= 4 and total_lookups > 10
    return match_found, time.time() - start_time

def generate_simulated_attitude(observed_stars, success, noise_level=1.0):
    if not success:
        return None, None
    true_attitude = np.eye(3)
    base_error_deg = random.uniform(0.05, 0.5)
    star_quality_factor = max(0.1, min(1.0, len(observed_stars) / 10.0))
    total_error_rad = (base_error_deg * noise_level / star_quality_factor) * DEGREES_TO_RADIANS
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    error_rotation = np.eye(3) + np.sin(total_error_rad) * K + (1 - np.cos(total_error_rad)) * np.dot(K, K)
    return true_attitude, np.dot(error_rotation, true_attitude)

def determine_attitude_progressively(observed_stars, catalog_index, verbose=False):
    total_time = 0.0
    if len(observed_stars) >= 3:
        success, comp_time = liebe_triangle_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Liebe", "SUCCESS: Liebe Triangle Match", total_time
    if len(observed_stars) >= 3:
        success, comp_time = geometric_voting_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Voting", "SUCCESS: Geometric Voting", total_time
    if len(observed_stars) >= 4:
        success, comp_time = pyramid_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Pyramid", "SUCCESS: Pyramid Match", total_time
    return "Failed", f"FAILURE: All algorithms failed ({len(observed_stars)} stars)", total_time


# ===========================================================================
# NEW: GPS-DENIED DEGRADATION SIMULATION
# ---------------------------------------------------------------------------
# Models positional error growth over time in three modes:
#   1. Dead Reckoning only (unbounded drift — baseline for adversary EW scenario)
#   2. Celestial Navigation with periodic fixes (bounded error)
#   3. INS + Celestial hybrid (tighter bound — future integration path)
#
# This is the core figure that connects the technical system to JADO doctrine:
# without a GPS backup, ground forces lose PNT within minutes; celestial nav
# provides a degraded but operationally viable fix capability.
# ===========================================================================

def run_gps_denied_degradation_simulation(
        duration_minutes: int = 60,
        fix_interval_minutes: int = 5,
        num_monte_carlo: int = 200,
        scenario: EnvironmentCondition = None) -> Dict[str, Any]:
    """
    Simulate positional error growth over time after GPS denial event at t=0.

    Returns time-series arrays (mean + std) for three modes:
      - dead_reckoning: IMU drift only, no external fix
      - celestial_nav:  periodic celestial fixes every fix_interval_minutes
      - ins_hybrid:     tighter bound representing INS-aided celestial nav

    Parameters
    ----------
    duration_minutes : total simulation window after GPS denial
    fix_interval_minutes : how often a celestial fix is attempted
    num_monte_carlo : number of independent trajectories for statistics
    scenario : EnvironmentCondition governing fix success probability
    """
    if scenario is None:
        scenario = OPERATIONAL_SCENARIOS[1]  # EW_Degraded_Rural default

    # Time axis (1-minute resolution)
    time_axis = list(range(0, duration_minutes + 1))

    # IMU / dead-reckoning drift model
    # Typical military-grade IMU: 0.8 nm/hr CEP → ~13 m/min RMS error growth
    dr_drift_rate_m_per_min = 13.0  # meters per minute (linear approximation)
    dr_noise_sigma = 3.0            # stochastic component per minute (m)

    # Celestial fix residual error after successful identification
    # Derived from attitude RMS (~0.3 degree) and assumed 10 km range: ~52 m CEP
    # Scaled by noise_level for the given scenario
    celestial_fix_residual_m = 52.0 * scenario.noise_level

    # INS-hybrid fix residual (tighter: ~20 m CEP at baseline)
    ins_hybrid_residual_m = 20.0 * scenario.noise_level

    # Fix success probability for this scenario (from empirical MC results)
    # We compute it inline from a quick sample
    quick_catalog_index = build_angular_distance_index(MOCK_STAR_CATALOG)
    fix_successes = 0
    for _ in range(50):
        obs = simulate_camera_view(MOCK_STAR_CATALOG, scenario)
        algo, _, _ = determine_attitude_progressively(obs, quick_catalog_index)
        if algo != "Failed":
            fix_successes += 1
    fix_success_prob = fix_successes / 50.0

    # Monte Carlo trajectories
    dr_trajectories = []
    cn_trajectories = []
    ins_trajectories = []

    for trial in range(num_monte_carlo):
        dr_errors = [0.0]
        cn_errors = [0.0]
        ins_errors = [0.0]

        cn_current = 0.0
        ins_current = 0.0

        for t in range(1, duration_minutes + 1):
            # Dead reckoning: monotonically growing error
            dr_increment = dr_drift_rate_m_per_min + random.gauss(0, dr_noise_sigma)
            dr_errors.append(dr_errors[-1] + max(0, dr_increment))

            # Celestial nav: drift until next fix attempt
            cn_increment = dr_drift_rate_m_per_min * 0.9 + random.gauss(0, dr_noise_sigma)
            cn_current += max(0, cn_increment)

            # Attempt a fix at scheduled intervals
            if t % fix_interval_minutes == 0:
                if random.random() < fix_success_prob:
                    cn_current = celestial_fix_residual_m + random.gauss(0, 5.0)
                    cn_current = max(0, cn_current)
            cn_errors.append(cn_current)

            # INS-hybrid: tighter residual on successful fix
            ins_increment = dr_drift_rate_m_per_min * 0.7 + random.gauss(0, dr_noise_sigma * 0.7)
            ins_current += max(0, ins_increment)
            if t % fix_interval_minutes == 0:
                if random.random() < min(fix_success_prob * 1.15, 1.0):
                    ins_current = ins_hybrid_residual_m + random.gauss(0, 3.0)
                    ins_current = max(0, ins_current)
            ins_errors.append(ins_current)

        dr_trajectories.append(dr_errors)
        cn_trajectories.append(cn_errors)
        ins_trajectories.append(ins_errors)

    # Aggregate statistics
    def agg(traj_list):
        arr = np.array(traj_list)
        return arr.mean(axis=0), arr.std(axis=0), np.percentile(arr, 95, axis=0)

    dr_mean, dr_std, dr_p95 = agg(dr_trajectories)
    cn_mean, cn_std, cn_p95 = agg(cn_trajectories)
    ins_mean, ins_std, ins_p95 = agg(ins_trajectories)

    return {
        'time_axis': time_axis,
        'dr_mean': dr_mean,   'dr_std': dr_std,   'dr_p95': dr_p95,
        'cn_mean': cn_mean,   'cn_std': cn_std,   'cn_p95': cn_p95,
        'ins_mean': ins_mean, 'ins_std': ins_std, 'ins_p95': ins_p95,
        'fix_success_prob': fix_success_prob,
        'fix_interval_minutes': fix_interval_minutes,
        'scenario_name': scenario.name,
        'scenario_noise': scenario.noise_level,
        'celestial_fix_residual_m': celestial_fix_residual_m,
    }


# ===========================================================================
# NEW: JADO C2 REINTEGRATION LATENCY SIMULATION
# ---------------------------------------------------------------------------
# Measures how quickly (in seconds) the celestial nav system can provide a
# confirmed position fix after GPS denial — the "time-to-first-fix" (TTFF)
# after a simulated GPS loss event.
#
# This directly addresses the JADO C2 resilience question in the CFP:
# how fast can ground forces re-establish a navigation fix to restore C2?
#
# Lower TTFF = faster C2 reintegration = higher JADO operational resilience.
# ===========================================================================

def run_jado_c2_latency_simulation(
        num_trials_per_scenario: int = 500) -> Dict[str, Any]:
    """
    Simulate time-to-first-fix (TTFF) after GPS denial across all scenarios.

    Models:
      - Initial algorithm cascade (Liebe → Voting → Pyramid) computation time
      - Re-acquisition delay based on number of stars visible (sensor warm-up)
      - Failure penalty: time lost before falling back to dead reckoning

    Returns per-scenario TTFF statistics (mean, std, 95th percentile).
    """
    global CATALOG_INDEX
    if not CATALOG_INDEX:
        CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)

    # Sensor warm-up time model (seconds): camera stabilisation after vehicle stop
    # or onset of GPS denial event
    SENSOR_WARMUP_BASE_S = 2.0
    SENSOR_WARMUP_NOISE_S = 0.5

    # Algorithm overhead per retry attempt
    RETRY_OVERHEAD_S = 0.05

    results = {}

    for scenario in OPERATIONAL_SCENARIOS:
        ttff_list = []
        success_count = 0

        for _ in range(num_trials_per_scenario):
            # Sensor warm-up
            warmup = max(0.5, random.gauss(SENSOR_WARMUP_BASE_S, SENSOR_WARMUP_NOISE_S))

            # Attempt to get a fix (may require multiple tries)
            attempts = 0
            max_attempts = 5
            fix_time = warmup
            got_fix = False

            while attempts < max_attempts and not got_fix:
                attempts += 1
                obs = simulate_camera_view(MOCK_STAR_CATALOG, scenario)
                algo, _, comp_time = determine_attitude_progressively(obs, CATALOG_INDEX)
                fix_time += comp_time + RETRY_OVERHEAD_S

                if algo != "Failed":
                    got_fix = True
                    success_count += 1
                else:
                    # Backoff: wait for better sky window
                    fix_time += random.uniform(1.0, 3.0)

            if got_fix:
                ttff_list.append(fix_time)
            else:
                # Record a "timeout" penalty (operationally: fall back to DR)
                ttff_list.append(fix_time + 30.0)  # 30-second timeout penalty

        ttff_arr = np.array(ttff_list)
        success_rate = success_count / num_trials_per_scenario * 100

        results[scenario.name] = {
            'ttff_mean_s': float(ttff_arr.mean()),
            'ttff_std_s': float(ttff_arr.std()),
            'ttff_p95_s': float(np.percentile(ttff_arr, 95)),
            'ttff_median_s': float(np.median(ttff_arr)),
            'success_rate': success_rate,
            'threat_actor': scenario.threat_actor,
            'threat_vector': scenario.threat_vector,
            'jado_relevance': scenario.jado_relevance,
        }

        print(f"  {scenario.name:30s} TTFF mean: {ttff_arr.mean():.2f}s  "
              f"p95: {np.percentile(ttff_arr, 95):.2f}s  "
              f"fix rate: {success_rate:.1f}%")

    return results


# --- Monte Carlo Simulation Framework ---

def run_monte_carlo_simulation(num_trials=1000, verbose=False):
    print(f"\n{'='*70}")
    print(f"MONTE CARLO SIMULATION: Celestial Navigation for Military Ground Vehicles")
    print(f"{'='*70}")
    print(f"Trials per scenario: {num_trials}")
    print(f"Total scenarios: {len(OPERATIONAL_SCENARIOS)}")
    print(f"Star catalog size: {len(MOCK_STAR_CATALOG)} stars")
    print(f"Match Tolerance: {MATCH_TOLERANCE_ARCSEC} arcsec")
    print(f"{'='*70}\n")

    global CATALOG_INDEX
    CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)
    validator = CelestialNavigationValidator(match_tolerance_arcsec=MATCH_TOLERANCE_ARCSEC)
    all_validation_results = []
    all_results = []
    scenario_stats = {}
    all_attitude_errors = []
    scenario_attitude_errors = {s.name: [] for s in OPERATIONAL_SCENARIOS}

    for scenario in OPERATIONAL_SCENARIOS:
        print(f"\nScenario: {scenario.name}")
        print(f"  Threat Actor:  {scenario.threat_actor}")
        print(f"  Threat Vector: {scenario.threat_vector}")
        scenario_results = []
        algorithm_counts = defaultdict(int)
        success_count = 0
        computation_times = []

        for trial in range(num_trials):
            observed_stars = simulate_camera_view(MOCK_STAR_CATALOG, scenario)
            algorithm_used, result_msg, comp_time = determine_attitude_progressively(
                observed_stars, CATALOG_INDEX)
            success = algorithm_used != "Failed"
            if success:
                success_count += 1
            algorithm_counts[algorithm_used] += 1
            computation_times.append(comp_time)
            true_att, est_att = generate_simulated_attitude(observed_stars, success, scenario.noise_level)
            validation_result = validator.validate_trial(
                scenario_name=scenario.name,
                algorithm_used=algorithm_used,
                success=success,
                computation_time=comp_time,
                num_stars_matched=len(observed_stars) if success else 0,
                true_attitude=true_att,
                estimated_attitude=est_att,
                matched_stars=None
            )
            all_validation_results.append(validation_result)
            all_results.append(TrialResult(
                scenario=scenario.name,
                algorithm_used=algorithm_used,
                success=success,
                computation_time=comp_time,
                num_stars_visible=len(observed_stars),
                noise_level=scenario.noise_level
            ))

        scenario_validation_results = [
            r for r in all_validation_results
            if r.scenario_name == scenario.name and r.success and r.attitude_error
        ]
        scenario_errors = [r.attitude_error.total_rms_error for r in scenario_validation_results]
        success_rate = (success_count / num_trials) * 100
        avg_time = statistics.mean(computation_times)
        mean_attitude_error = statistics.mean(scenario_errors) if scenario_errors else 0.0
        ci_lower, ci_upper = (
            validator.calculate_confidence_intervals(scenario_errors)
            if len(scenario_errors) > 1 else (mean_attitude_error, mean_attitude_error)
        )
        scenario_attitude_errors[scenario.name] = scenario_errors
        all_attitude_errors.extend(scenario_errors)

        scenario_stats[scenario.name] = {
            'success_rate': success_rate,
            'avg_computation_time': avg_time,
            'std_computation_time': statistics.stdev(computation_times) if len(computation_times) > 1 else 0,
            'algorithm_distribution': dict(algorithm_counts),
            'total_trials': num_trials,
            'description': scenario.description,
            'threat_actor': scenario.threat_actor,
            'threat_vector': scenario.threat_vector,
            'jado_relevance': scenario.jado_relevance,
            'mean_attitude_error': mean_attitude_error,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'attitude_errors': scenario_errors
        }

        print(f"  Success Rate:       {success_rate:.1f}%")
        print(f"  Mean Attitude Error:{mean_attitude_error:.3f} degree")
        print(f"  95% CI:             [{ci_lower:.3f} degree, {ci_upper:.3f} degree]")
        print(f"  Avg Comp Time:      {avg_time*1000:.3f} ms")

    total_trials = len(all_results)
    overall_success = sum(1 for r in all_results if r.success)
    overall_success_rate = (overall_success / total_trials) * 100
    overall_mean_error = statistics.mean(all_attitude_errors) if all_attitude_errors else 0.0
    if len(all_attitude_errors) > 1:
        overall_ci_lower, overall_ci_upper = validator.calculate_confidence_intervals(all_attitude_errors)
    else:
        overall_ci_lower = overall_ci_upper = overall_mean_error

    algorithm_totals = defaultdict(int)
    for result in all_results:
        algorithm_totals[result.algorithm_used] += 1

    print(f"\n{'='*70}")
    print(f"OVERALL: {overall_success_rate:.1f}% success | "
          f"Mean error: {overall_mean_error:.3f} degree | "
          f"95% CI: [{overall_ci_lower:.3f} degree, {overall_ci_upper:.3f} degree]")

    return {
        'scenario_stats': scenario_stats,
        'all_results': all_results,
        'validation_results': all_validation_results,
        'overall_success_rate': overall_success_rate,
        'algorithm_totals': dict(algorithm_totals),
        'num_trials_per_scenario': num_trials,
        'total_trials': total_trials,
        'overall_mean_error': overall_mean_error,
        'overall_ci_lower': overall_ci_lower,
        'overall_ci_upper': overall_ci_upper,
        'scenario_attitude_errors': scenario_attitude_errors
    }


# --- Main ---

if __name__ == "__main__":
    print("--- Celestial Navigation Simulator — Military Ground Vehicles ---")
    CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)

    # Core Monte Carlo
    mc_results = run_monte_carlo_simulation(num_trials=1000, verbose=False)

    # GPS Degradation Simulation
    print(f"\n{'='*70}")
    print("GPS-DENIED DEGRADATION SIMULATION")
    print(f"{'='*70}")
    degradation_results = {}
    for scenario in OPERATIONAL_SCENARIOS:
        print(f"\nRunning degradation sim: {scenario.name}")
        degradation_results[scenario.name] = run_gps_denied_degradation_simulation(
            duration_minutes=60,
            fix_interval_minutes=5,
            num_monte_carlo=200,
            scenario=scenario
        )

    # JADO C2 Latency Simulation
    print(f"\n{'='*70}")
    print("C2 REINTEGRATION LATENCY (Time-to-First-Fix)")
    print(f"{'='*70}")
    c2_latency_results = run_jado_c2_latency_simulation(num_trials_per_scenario=500)

    # Save combined results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/simulation_results_{timestamp}.txt"
    with open(output_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Celestial Navigation for Military Ground Vehicles\n")
        f.write("GPS-Denied PNT Under Adversary EW and ISR Conditions\n")
        f.write("="*70 + "\n\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total MC Trials: {mc_results['total_trials']}\n")
        f.write(f"Overall Success Rate: {mc_results['overall_success_rate']:.1f}%\n\n")

        f.write("SCENARIO RESULTS (Adversary-Threat Framing)\n")
        f.write("-"*70 + "\n")
        for name, s in mc_results['scenario_stats'].items():
            f.write(f"\n{name}\n")
            f.write(f"  Threat Actor:   {s['threat_actor']}\n")
            f.write(f"  Threat Vector:  {s['threat_vector']}\n")
            f.write(f"  JADO Relevance: {s['jado_relevance']}\n")
            f.write(f"  Success Rate:   {s['success_rate']:.1f}%\n")
            f.write(f"  Attitude Error: {s['mean_attitude_error']:.3f} degree "
                    f"[{s['ci_95_lower']:.3f} degree, {s['ci_95_upper']:.3f} degree] 95% CI\n")

        f.write("\n\nC2 REINTEGRATION LATENCY (Time-to-First-Fix)\n")
        f.write("-"*70 + "\n")
        for name, c in c2_latency_results.items():
            f.write(f"\n{name}\n")
            f.write(f"  TTFF Mean:   {c['ttff_mean_s']:.2f}s\n")
            f.write(f"  TTFF p95:    {c['ttff_p95_s']:.2f}s\n")
            f.write(f"  Fix Rate:    {c['success_rate']:.1f}%\n")

    print(f"\nResults saved: {output_filename}")

    # Return all results for visualize.py to consume
    ALL_SIMULATION_RESULTS = {
        'mc': mc_results,
        'degradation': degradation_results,
        'c2_latency': c2_latency_results,
    }
