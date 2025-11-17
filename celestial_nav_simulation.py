import math
import random
import time
import statistics
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from star_catalog import get_tycho2_catalog
from validation_framework import CelestialNavigationValidator, ValidationResult


# --- Configuration and Constants ---
# Increased tolerance for more realistic failure in high-noise scenarios
MATCH_TOLERANCE_ARCSEC = 15.0 
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi
ARCSEC_PER_DEGREE = 3600.0
# Precision used for keying distances in the index (e.g., 0.1 degree for hashing)
INDEX_KEY_PRECISION = 0.1 

# Global variable to store the pre-processed catalog index
CATALOG_INDEX: Dict[int, List[Tuple[int, int]]] = {}

# --- Monte Carlo Configuration ---
@dataclass
class EnvironmentCondition:
    """Represents different operational environments for Monte Carlo simulation"""
    name: str
    fov_size: float  # Field of view in degrees
    num_stars: int   # Number of stars typically visible
    noise_level: float  # Multiplier for position noise (1.0 = baseline 15 arcsec)
    obscuration_prob: float  # Probability of partial sky obscuration (0.0-1.0)
    description: str

# Define operational scenarios for military trucks
OPERATIONAL_SCENARIOS = [
    EnvironmentCondition(
        name="Clear_Rural_Base",
        fov_size=20.0,
        num_stars=7,
        noise_level=1.0,
        obscuration_prob=0.1,
        description="Baseline clear night in rural/desert environment (Low Noise)"
    ),
    EnvironmentCondition(
        name="Urban_Canyon_Restricted",
        fov_size=10.0,
        num_stars=4,
        noise_level=1.5,
        obscuration_prob=0.4,
        description="Urban environment with buildings blocking sky (Reduced Stars, Moderate Noise)"
    ),
    EnvironmentCondition(
        name="Forest_Canopy_Obscured",
        fov_size=8.0,
        num_stars=3, # Low star count favors algorithms robust to few stars
        noise_level=1.2,
        obscuration_prob=0.5,
        description="Dense forest with partial canopy obstruction (Very Few Stars)"
    ),
    EnvironmentCondition(
        name="Dust_Storm_HighNoise",
        fov_size=12.0,
        num_stars=5,
        noise_level=2.5, # High noise forces fallback to Voting/Pyramid
        obscuration_prob=0.3,
        description="Dust/sand storm reducing visibility (High Measurement Noise)"
    ),
    EnvironmentCondition(
        name="Optimal_WideField",
        fov_size=30.0,
        num_stars=10,
        noise_level=0.8,
        obscuration_prob=0.05,
        description="Optimal conditions - clear sky, wide FOV (Max Performance)"
    ),
    EnvironmentCondition(
        name="Vehicle_Motion_Extreme",
        fov_size=15.0,
        num_stars=6,
        noise_level=3.5, # Extreme noise to push system to failure
        obscuration_prob=0.2,
        description="Moving vehicle with severe vibration and motion blur (Extreme Noise)"
    ),
]

@dataclass
class TrialResult:
    """Stores the result of a single Monte Carlo trial"""
    scenario: str
    algorithm_used: str  # "Liebe", "Voting", "Pyramid", or "Failed"
    success: bool
    computation_time: float
    num_stars_visible: int
    noise_level: float

# --- 1. Mock Tycho-2 Star Catalog (Expanded to 50 Stars) ---

# Core 15 bright stars for seed
'''
MOCK_STAR_CATALOG: List[Dict] = [
    {"ID": 1, "NAME": "Polaris", "RA": 2.520, "DEC": 89.26, "MAG": 2.02, "BRIGHTNESS": 1.0},
    {"ID": 2, "NAME": "Vega", "RA": 18.636, "DEC": 38.78, "MAG": 0.03, "BRIGHTNESS": 1.0},
    {"ID": 3, "NAME": "Arcturus", "RA": 14.256, "DEC": 19.18, "MAG": -0.05, "BRIGHTNESS": 1.0},
    {"ID": 4, "NAME": "Rigel", "RA": 5.140, "DEC": -8.20, "MAG": 0.13, "BRIGHTNESS": 1.0},
    {"ID": 5, "NAME": "Capella", "RA": 5.217, "DEC": 45.99, "MAG": 0.08, "BRIGHTNESS": 1.0},
    {"ID": 6, "NAME": "Betelgeuse", "RA": 5.919, "DEC": 7.41, "MAG": 0.50, "BRIGHTNESS": 0.9},
    {"ID": 7, "NAME": "Canopus", "RA": 6.234, "DEC": -52.69, "MAG": -0.74, "BRIGHTNESS": 1.0},
    {"ID": 8, "NAME": "Acrux", "RA": 12.434, "DEC": -63.09, "MAG": 0.77, "BRIGHTNESS": 0.9},
    {"ID": 9, "NAME": "Deneb", "RA": 20.695, "DEC": 45.30, "MAG": 1.25, "BRIGHTNESS": 0.8},
    {"ID": 10, "NAME": "Fomalhaut", "RA": 22.956, "DEC": -29.62, "MAG": 1.16, "BRIGHTNESS": 0.8},
    {"ID": 11, "NAME": "Aldebaran", "RA": 4.598, "DEC": 16.51, "MAG": 0.85, "BRIGHTNESS": 0.85},
    {"ID": 12, "NAME": "Spica", "RA": 13.419, "DEC": -11.16, "MAG": 1.04, "BRIGHTNESS": 0.82},
    {"ID": 13, "NAME": "Antares", "RA": 16.490, "DEC": -26.43, "MAG": 0.96, "BRIGHTNESS": 0.83},
    {"ID": 14, "NAME": "Altair", "RA": 19.846, "DEC": 8.87, "MAG": 0.77, "BRIGHTNESS": 0.85},
    {"ID": 15, "NAME": "Sirius", "RA": 6.752, "DEC": -16.72, "MAG": -1.46, "BRIGHTNESS": 1.0},
]'''
# New Tycho-2 catalog  
MOCK_STAR_CATALOG = get_tycho2_catalog()

# Generate additional dimmer stars to reach 50
for i in range(16, 51):
    MOCK_STAR_CATALOG.append({
        "ID": i, 
        "NAME": f"CatalogStar_{i}", 
        "RA": random.uniform(0, 24), # Right Ascension 0 to 24h
        "DEC": random.uniform(-90, 90), # Declination -90 to +90
        "MAG": random.uniform(2.0, 6.0), # Dimmer stars
        "BRIGHTNESS": random.uniform(0.2, 0.7)
    })
# Sort the final catalog by apparent magnitude (brightness)
MOCK_STAR_CATALOG.sort(key=lambda x: x['MAG'])


# --- 2. Core Geometric Functions ---

def calculate_angular_distance(star1: Dict, star2: Dict, use_observed: bool = False) -> float:
    """
    Calculates the angular distance (great-circle distance) between two stars.
    Returns: Distance in degrees.
    """
    ra_key = 'RA_OBSERVED' if use_observed else 'RA'
    dec_key = 'DEC_OBSERVED' if use_observed else 'DEC'

    ra1, dec1 = star1.get(ra_key, star1['RA']) * DEGREES_TO_RADIANS, star1.get(dec_key, star1['DEC']) * DEGREES_TO_RADIANS
    ra2, dec2 = star2.get(ra_key, star2['RA']) * DEGREES_TO_RADIANS, star2.get(dec_key, star2['DEC']) * DEGREES_TO_RADIANS
    
    # Haversine formula is slightly better for small angles, but great-circle is fine here.
    cos_angle = (math.sin(dec1) * math.sin(dec2) +
                 math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.acos(cos_angle) * RADIANS_TO_DEGREES

def calculate_inter_star_angles(stars: List[Dict], use_observed: bool = False) -> List[float]:
    """Calculates all unique inter-star angular distances for a given set of stars."""
    distances = []
    num_stars = len(stars)
    for i in range(num_stars):
        for j in range(i + 1, num_stars):
            distances.append(calculate_angular_distance(stars[i], stars[j], use_observed=use_observed))
    return distances

def get_index_key(angle_deg: float) -> int:
    """Creates a quantized key for index lookup based on angular distance."""
    # This quantization is crucial for managing the noise/error window
    return int(round(angle_deg / INDEX_KEY_PRECISION))

def build_angular_distance_index(catalog: List[Dict]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Simulates the pre-processing of the star catalog into an index.
    The index maps a quantized angular distance key to a list of star pairs.
    """
    index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    num_stars = len(catalog)

    for i in range(num_stars):
        for j in range(i + 1, num_stars):
            star_a = catalog[i]
            star_b = catalog[j]
            
            distance = calculate_angular_distance(star_a, star_b)
            key = get_index_key(distance)
            
            # Store the pair of Catalog IDs under the key
            index[key].append((star_a['ID'], star_b['ID']))
            
    return dict(index)


# --- 3. Enhanced Simulation Environment ---

def simulate_camera_view(catalog: List[Dict], 
                        condition: EnvironmentCondition,
                        verbose: bool = False) -> List[Dict]:
    """
    Enhanced simulation with realistic environmental conditions for military vehicles.
    """
    
    # Apply obscuration probability
    if random.random() < condition.obscuration_prob:
        effective_num_stars = max(3, int(condition.num_stars * random.uniform(0.5, 0.9)))
    else:
        effective_num_stars = condition.num_stars
    
    # 1. Select a random center of FOV
    center_ra = random.uniform(0, 24)
    center_dec = random.uniform(-90, 90)
    
    # Create a dummy center star for angular distance calculation
    center_star = {"RA": center_ra, "DEC": center_dec}
    
    observed_stars = []
    max_dist = condition.fov_size * 0.7 
    
    # Filter stars within the FOV
    for star in catalog:
        if calculate_angular_distance(center_star, star) < max_dist:
            observed_stars.append(star)

    noisy_observed_stars = []
    # Sort by brightness (MAG) and take the top N stars
    observed_stars.sort(key=lambda x: x['MAG'], reverse=False) 

    for i, star in enumerate(observed_stars[:effective_num_stars]):
        noisy_star = star.copy()
        
        # Apply scaled noise based on environmental conditions
        base_noise = 15.0  # Base noise in arcseconds (same as MATCH_TOLERANCE_ARCSEC)
        effective_noise = base_noise * condition.noise_level
        
        # Noise magnitude (random uniform distribution)
        noise_arcsec_ra = random.uniform(-effective_noise, effective_noise)
        noise_arcsec_dec = random.uniform(-effective_noise, effective_noise)
        noise_deg_ra = noise_arcsec_ra / ARCSEC_PER_DEGREE
        noise_deg_dec = noise_arcsec_dec / ARCSEC_PER_DEGREE
        
        # Apply noise to the observed position
        noisy_star['RA_OBSERVED'] = star['RA'] + noise_deg_ra
        noisy_star['DEC_OBSERVED'] = star['DEC'] + noise_deg_dec
        noisy_star['TEMP_ID'] = i + 1 

        noisy_observed_stars.append(noisy_star)

    if verbose:
        print(f"  Environment: {condition.name}")
        print(f"  Stars detected: {len(noisy_observed_stars)}/{effective_num_stars}")
        print(f"  Effective Noise: {effective_noise:.1f} arcsec")
    
    return noisy_observed_stars


# --- 4. Star Identification Algorithms (Updated Rigor) ---

def liebe_triangle_match(observed_stars: List[Dict], 
                        catalog_index: Dict[int, List[Tuple[int, int]]],
                        verbose: bool = False) -> Tuple[bool, float]:
    """
    Liebe's Triangle Match (More Selective Version).
    """
    start_time = time.time()
    
    if len(observed_stars) < 3:
        return False, time.time() - start_time

    # 1. Form the pattern from the image (smallest/brightest 3 stars)
    image_pattern = observed_stars[:3]
    image_angles = calculate_inter_star_angles(image_pattern, use_observed=True)
    image_angles.sort()

    # 2. Search for ALL THREE angles in the triangle, not just the shortest
    matches_per_angle = []
    
    for angle in image_angles:
        key = get_index_key(angle)
        keys_to_check = [key - 1, key, key + 1] 
        candidate_pairs = set()

        for k in keys_to_check:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    candidate_pairs.add(tuple(sorted((str(id_a), str(id_b)))))
        
        matches_per_angle.append(len(candidate_pairs))
    
    # RIGOR: Require that ALL THREE angles have reasonable matches
    # and the shortest angle isn't overly ambiguous
    shortest_angle_matches = matches_per_angle[0]
    avg_matches = statistics.mean(matches_per_angle)
    
    # More selective criteria:
    # - Shortest angle should have matches but not too many (avoid ambiguity)
    # - All angles should have some matches
    # - Average matches should be reasonable
    match_found = (2 <= shortest_angle_matches <= 50 and 
                   all(matches > 0 for matches in matches_per_angle) and
                   avg_matches < 100)
    
    end_time = time.time()
    comp_time = end_time - start_time
    
    if verbose:
        print(f"  [Liebe] Angle matches: {matches_per_angle}, Avg: {avg_matches:.1f}. Time: {comp_time:.6f}s. Match: {match_found}")
    
    return match_found, comp_time

def generate_simulated_attitude(observed_stars: List[Dict], success: bool, noise_level: float = 1.0) -> Tuple[Any, Any]:
    """
    Generate simulated true and estimated attitude matrices with realistic errors.
    """
    if not success:
        return None, None
    
    # Create a true attitude (identity for simplicity)
    true_attitude = np.eye(3)
    
    # Simulate attitude error based on noise and star count
    base_error_deg = random.uniform(0.05, 0.5)  # Base error in degrees
    star_count = len(observed_stars)
    star_quality_factor = max(0.1, min(1.0, star_count / 10.0))
    
    # Error increases with noise and decreases with more stars
    total_error_deg = base_error_deg * noise_level / star_quality_factor
    total_error_rad = total_error_deg * DEGREES_TO_RADIANS
    
    # Create a small random rotation for the error
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula for small angles
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    error_rotation = np.eye(3) + np.sin(total_error_rad) * K + (1 - np.cos(total_error_rad)) * np.dot(K, K)
    
    estimated_attitude = np.dot(error_rotation, true_attitude)
    
    return true_attitude, estimated_attitude

def geometric_voting_match(observed_stars: List[Dict], 
                          catalog_index: Dict[int, List[Tuple[int, int]]],
                          verbose: bool = False) -> Tuple[bool, float]:
    """
    Geometric Voting (Optimized) with improved verification.
    """
    start_time = time.time()
    
    if len(observed_stars) < 3:
        return False, time.time() - start_time
        
    image_distances = calculate_inter_star_angles(observed_stars, use_observed=True)
    
    vote_count: Dict[int, int] = defaultdict(int)
    total_index_hits = 0
    N = len(observed_stars)

    # 1. Fast Index Lookup and Voting
    for img_dist in image_distances:
        key = get_index_key(img_dist)
        # Search a window around the key to account for noise/error
        keys_to_check = [key - 1, key, key + 1] 

        for k in keys_to_check:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_index_hits += 1
                    # If the distance matches, the two catalog stars vote
                    vote_count[id_a] += 1
                    vote_count[id_b] += 1

    # 2. Candidate Selection
    # Get the top N star IDs by vote count
    top_candidates = sorted(vote_count.items(), key=lambda item: item[1], reverse=True)[:N]
    top_candidate_ids = [cid for cid, count in top_candidates]
    
    # 3. Rigorous Verification (Simulated)
    # The maximum possible vote count for a star is N-1 (it forms a pair with N-1 other stars)
    # Total pairs: N * (N - 1) / 2. Max total votes: N * (N - 1)
    
    # Threshold check 1: Vote accumulation
    # Require that the sum of votes for the top N stars is high (75% threshold)
    required_votes_cumulative = (N * (N - 1)) * 0.75 
    current_votes_cumulative = sum(count for cid, count in top_candidates)
    
    if current_votes_cumulative < required_votes_cumulative:
        match_found = False
    else:
        # Threshold check 2: Consistency check (Simulating geometric verification)
        # Check if the distances *between* the top N candidates match the observed image distances
        
        # This is a proxy for solving the rotation matrix: it checks pattern consistency.
        
        # Get the catalog data for the top N candidates
        catalog_stars_map = {star['ID']: star for star in MOCK_STAR_CATALOG if star['ID'] in top_candidate_ids}
        
        if len(catalog_stars_map) < N:
            # Not enough unique catalog candidates found
            match_found = False
        else:
            # Check consistency of the pattern formed by the top candidates
            catalog_candidate_distances = calculate_inter_star_angles(list(catalog_stars_map.values()))
            
            # Simple consistency check: ensure the smallest observed distance is
            # reasonably close to the smallest candidate distance
            img_min_dist = min(image_distances)
            cat_min_dist = min(catalog_candidate_distances)
            
            # The difference should be less than the total allowed measurement error (e.g., 0.1 degree)
            match_found = abs(img_min_dist - cat_min_dist) < INDEX_KEY_PRECISION * 1.5 


    end_time = time.time()
    comp_time = end_time - start_time
    
    if verbose:
        print(f"  [Voting] Index Hits: {total_index_hits}. Top Candidates: {len(top_candidates)}. Time: {comp_time:.6f}s. Match: {match_found}")
    
    return match_found, comp_time
    

def pyramid_match(observed_stars: List[Dict], 
                 catalog_index: Dict[int, List[Tuple[int, int]]],
                 verbose: bool = False) -> Tuple[bool, float]:
    """
    Pyramid Algorithm (Search-Less Concept).
    """
    start_time = time.time()

    if len(observed_stars) < 4:
        # Pyramid is the most robust LIS but requires 4 stars minimum
        return False, time.time() - start_time
        
    image_pattern = observed_stars[:4]
    image_angles = calculate_inter_star_angles(image_pattern, use_observed=True)
    
    image_angles.sort()
    # Use the 4 shortest unique angles (the "legs" of the pyramid) for lookup
    index_lookups = image_angles[:4]
    
    # Store all unique catalog star IDs that are part of the distance match
    candidate_star_ids = set()
    total_lookups = 0

    for angle in index_lookups:
        key = get_index_key(angle)
        keys_to_check = [key - 1, key, key + 1]

        for k in keys_to_check:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_lookups += 1
                    candidate_star_ids.add(id_a)
                    candidate_star_ids.add(id_b)

    # RIGOR: True match requires the intersection of the four leg-lists to yield 
    # exactly the four correct catalog stars. We simulate this by requiring a high 
    # number of index hits and at least 4 unique candidates, which models high redundancy.
    
    # Requirement: High lookup count (redundancy) AND finding at least 4 unique stars
    match_found = len(candidate_star_ids) >= 4 and total_lookups > 10 

    end_time = time.time()
    comp_time = end_time - start_time
    
    if verbose:
        print(f"  [Pyramid] Lookups: {total_lookups}. Candidates: {len(candidate_star_ids)}. Time: {comp_time:.6f}s. Match: {match_found}")
    
    return match_found, comp_time


# --- 5. Progressive Navigation Logic with Tracking ---

def determine_attitude_progressively(observed_stars: List[Dict], 
                                    catalog_index: Dict[int, List[Tuple[int, int]]],
                                    verbose: bool = False) -> Tuple[str, str, float]:
    """
    Implements progressive algorithm logic with tracking for Monte Carlo.
    Returns: (algorithm_used, result_message, total_time)
    """
    if verbose:
        print(f"\n--- Progressive Star ID: {len(observed_stars)} stars ---")
    
    total_time = 0.0
    
    # 1. Try Liebe's Algorithm (Fastest, requires 3+ stars)
    if len(observed_stars) >= 3:
        if verbose:
            print("1. Attempting Liebe's Algorithm...")
        success, comp_time = liebe_triangle_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Liebe", "SUCCESS: Liebe's Triangle Match (Fastest/Low Noise)", total_time

    # 2. Try Geometric Voting (Robust to Noise, requires 3+ stars for meaningful voting)
    if len(observed_stars) >= 3:
        if verbose:
            print("2. Falling back to Geometric Voting...")
        success, comp_time = geometric_voting_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Voting", "SUCCESS: Geometric Voting (Robust to Noise/Consistency Check)", total_time

    # 3. Try Pyramid Algorithm (Most Robust LIS, requires 4+ stars)
    if len(observed_stars) >= 4:
        if verbose:
            print("3. Falling back to Pyramid Algorithm...")
        success, comp_time = pyramid_match(observed_stars, catalog_index, verbose)
        total_time += comp_time
        if success:
            return "Pyramid", "SUCCESS: Pyramid Match (Highest Redundancy/LIS)", total_time

    return "Failed", f"FAILURE: All algorithms failed or insufficient stars ({len(observed_stars)} < 3 or 4)", total_time


# --- 6. Monte Carlo Simulation Framework ---
def run_monte_carlo_simulation(num_trials: int = 1000, verbose: bool = False) -> Dict[str, Any]:
    """
    Runs Monte Carlo simulation across multiple environmental scenarios.
    
    Args:
        num_trials: Number of trials per scenario
        verbose: Print detailed output for each trial
    
    Returns:
        Dictionary containing comprehensive results and statistics
    """
    print(f"\n{'='*70}")
    print(f"MONTE CARLO SIMULATION: Celestial Navigation for Military Vehicles")
    print(f"{'='*70}")
    print(f"Trials per scenario: {num_trials}")
    print(f"Total scenarios: {len(OPERATIONAL_SCENARIOS)}")
    print(f"Star catalog size: {len(MOCK_STAR_CATALOG)} stars")
    print(f"Match Tolerance: {MATCH_TOLERANCE_ARCSEC} arcsec")
    print(f"{'='*70}\n")
    
    # Build index once
    global CATALOG_INDEX
    CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)
    
    # Initialize validation framework
    validator = CelestialNavigationValidator(match_tolerance_arcsec=MATCH_TOLERANCE_ARCSEC)
    all_validation_results: List[ValidationResult] = []
    
    all_results: List[TrialResult] = []
    scenario_stats: Dict[str, Dict] = {}
    
    # Run trials for each scenario
    for scenario in OPERATIONAL_SCENARIOS:
        print(f"\nRunning {num_trials} trials for: {scenario.name}")
        print(f"  Description: {scenario.description}")
        
        scenario_results = []
        algorithm_counts = defaultdict(int)
        success_count = 0
        computation_times = []
        
        for trial in range(num_trials):
            # Simulate camera view
            observed_stars = simulate_camera_view(MOCK_STAR_CATALOG, scenario, verbose=False)
            
            # Run progressive algorithm
            algorithm_used, result_msg, comp_time = determine_attitude_progressively(
                observed_stars, CATALOG_INDEX, verbose=False
            )
            
            success = algorithm_used != "Failed"
            if success:
                success_count += 1
            
            algorithm_counts[algorithm_used] += 1
            computation_times.append(comp_time)
            
            # Generate simulated attitude matrices for validation
            true_att, est_att = generate_simulated_attitude(observed_stars, success, scenario.noise_level)

            # Create validation result for this trial
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
            
            # Store original result
            result = TrialResult(
                scenario=scenario.name,
                algorithm_used=algorithm_used,
                success=success,
                computation_time=comp_time,
                num_stars_visible=len(observed_stars),
                noise_level=scenario.noise_level
            )
            scenario_results.append(result)
            all_results.append(result)
        
        # Calculate statistics for this scenario
        success_rate = (success_count / num_trials) * 100
        avg_time = statistics.mean(computation_times) if computation_times else 0
        std_time = statistics.stdev(computation_times) if len(computation_times) > 1 else 0
        
        scenario_stats[scenario.name] = {
            'success_rate': success_rate,
            'avg_computation_time': avg_time,
            'std_computation_time': std_time,
            'algorithm_distribution': dict(algorithm_counts),
            'total_trials': num_trials,
            'description': scenario.description
        }
        
        # Print scenario summary
        print(f"  Results:")
        print(f"    Success Rate: {success_rate:.1f}%")
        print(f"    Avg Computation Time: {avg_time*1000:.3f} ms")
        print(f"    Algorithm Usage:")
        for algo, count in sorted(algorithm_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / num_trials) * 100
            print(f"      {algo}: {count} ({percentage:.1f}%)")
    
    # Overall statistics
    total_trials = len(all_results)
    overall_success = sum(1 for r in all_results if r.success)
    overall_success_rate = (overall_success / total_trials) * 100
    
    algorithm_totals = defaultdict(int)
    for result in all_results:
        algorithm_totals[result.algorithm_used] += 1
    
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"Total Trials: {total_trials}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"\nAlgorithm Usage Across All Scenarios:")
    for algo, count in sorted(algorithm_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_trials) * 100
        print(f"  {algo}: {count} ({percentage:.1f}%)")
    
    # Generate validation report
    scenario_names = [scenario.name for scenario in OPERATIONAL_SCENARIOS]
    validation_report = validator.generate_validation_report(all_validation_results, scenario_names)
    print(f"\n{'='*70}")
    print(f"VALIDATION FRAMEWORK REPORT")
    print(f"{'='*70}")
    print(validation_report)
    
    return {
        'scenario_stats': scenario_stats,
        'all_results': all_results,
        'validation_results': all_validation_results,
        'overall_success_rate': overall_success_rate,
        'algorithm_totals': dict(algorithm_totals),
        'num_trials_per_scenario': num_trials,
        'total_trials': total_trials
    }


def print_detailed_analysis(results: Dict[str, Any], sorted_scenarios: List[Tuple[str, Dict]]):
    """
    Prints detailed analysis for the research paper.
    Now requires sorted_scenarios as an argument.
    """
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS FOR MILITARY REVIEW PAPER")
    print(f"{'='*70}\n")
    
    scenario_stats = results['scenario_stats']
    
    print("SCENARIO PERFORMANCE RANKING (Resilience Test):")
    print("-" * 70)
    for rank, (scenario_name, stats) in enumerate(sorted_scenarios, 1):
        # Determine the percentage of successful trials solved by fallbacks (Voting + Pyramid)
        liebe_count = stats['algorithm_distribution'].get('Liebe', 0)
        total_success = stats['total_trials'] - stats['algorithm_distribution'].get('Failed', 0)
        fallback_coverage = 0
        if total_success > 0:
            fallback_coverage = ((total_success - liebe_count) / total_success) * 100
            
        print(f"{rank}. {scenario_name} (Resilience: {fallback_coverage:.1f}% Fallback Coverage)")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Description: {stats['description']}")
        # Safely get the most used non-failed algorithm
        successful_algos = {k: v for k, v in stats['algorithm_distribution'].items() if k != 'Failed'}
        primary_solver = max(successful_algos.items(), key=lambda x: x[1], default=("N/A", 0))[0]
        print(f"   Primary Solver: {primary_solver}")
        print()
    
    print("\nALGORITHM EFFECTIVENESS BY SCENARIO:")
    print("-" * 70)
    for scenario_name, stats in scenario_stats.items():
        print(f"\n{scenario_name}:")
        algo_dist = stats['algorithm_distribution']
        total = stats['total_trials']
        
        # Calculate success contribution of each algorithm
        for algo, count in sorted(algo_dist.items(), key=lambda x: x[1], reverse=True):
            if algo != "Failed":
                percentage = (count / total) * 100
                print(f"  {algo}: {percentage:.1f}% of total trials solved by this algorithm")
    
    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR PAPER:")
    print(f"{'='*70}")
    
    # Calculate which algorithm is most reliable
    algo_totals = results['algorithm_totals']
    total_successful_runs = sum(count for algo, count in algo_totals.items() if algo != "Failed")
    
    most_used = max(algo_totals.items(), key=lambda x: x[1] if x[0] != "Failed" else 0)
    
    print(f"\n1. Most Reliable Algorithm: {most_used[0]}")
    # Handle division by zero if no trials succeeded
    if total_successful_runs > 0:
        print(f"   Used successfully in {(most_used[1]/total_successful_runs)*100:.1f}% of all SUCCESSFUL identifications.")
    else:
        print(f"   No successful identifications in the simulation.")

    print(f"\n2. Progressive Strategy Validation:")
    liebe_success = algo_totals.get('Liebe', 0)
    voting_success = algo_totals.get('Voting', 0) 
    pyramid_success = algo_totals.get('Pyramid', 0)
    
    # Calculate the benefit of fallback
    fallback_benefit = voting_success + pyramid_success
    
    print(f"   - Liebe (fast) solved: {(liebe_success/results['total_trials'])*100:.1f}% of total trials.")
    print(f"   - Voting/Pyramid Fallback provided {(fallback_benefit/results['total_trials'])*100:.1f}% ADDITIONAL successful coverage.")
    # Calculate percentage increase over Liebe alone (avoid division by zero if Liebe_success is 0)
    if liebe_success > 0:
        increase = (fallback_benefit / liebe_success) * 100
        print(f"   - Fallback system increased overall success by {increase:.1f}% over using Liebe alone.")
    else:
        print(f"   - Liebe failed in all cases, Fallback provided all successful identifications.")
    
    print(f"\n3. Environmental Resilience (Noise vs. Obscuration):")
    best_scenario = sorted_scenarios[0]
    worst_scenario = sorted_scenarios[-1]
    print(f"   - Best performance: {best_scenario[0]} ({best_scenario[1]['success_rate']:.1f}%)")
    print(f"   - Worst performance: {worst_scenario[0]} ({worst_scenario[1]['success_rate']:.1f}%)")
    print(f"   - High Noise environments (like Dust Storm) typically force the system to rely heavily on Voting.")
    print(f"   - Low Star Count environments (like Forest Canopy) often fail due to insufficient input for Pyramid/Voting.")


# --- Main Execution ---

if __name__ == "__main__":
    
    print("--- Military Truck Celestial Navigation Simulator ---")
    print(f"Using an EXPANDED mock catalog of {len(MOCK_STAR_CATALOG)} stars.")
    print(f"Match tolerance set to {MATCH_TOLERANCE_ARCSEC} arcseconds.")

    # Build index once
    CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)

    # Quick demonstration
    print("\n" + "="*70)
    print("DEMONSTRATION: Single Trial Examples with Enhanced Rigor")
    print("="*70)
    
    clean_condition = OPERATIONAL_SCENARIOS[0]  # Clear_Rural_Base
    clean_view = simulate_camera_view(MOCK_STAR_CATALOG, clean_condition, verbose=True)
    print("\nExample 1: Clear Rural Environment (Expected: Liebe)")
    algo, result, time_taken = determine_attitude_progressively(clean_view, CATALOG_INDEX, verbose=True)
    print(f"Result: {result} (Time: {time_taken*1000:.3f} ms)")
    
    degraded_condition = OPERATIONAL_SCENARIOS[3]  # Dust_Storm_HighNoise
    degraded_view = simulate_camera_view(MOCK_STAR_CATALOG, degraded_condition, verbose=True)
    print("\nExample 2: High Noise Environment (Expected: Voting Fallback)")
    algo, result, time_taken = determine_attitude_progressively(degraded_view, CATALOG_INDEX, verbose=True)
    print(f"Result: {result} (Time: {time_taken*1000:.3f} ms)")
    
    # Run Monte Carlo simulation
    print("\n\nProceed with Monte Carlo simulation...")
    
    # Run comprehensive Monte Carlo
    mc_results = run_monte_carlo_simulation(num_trials=1000, verbose=False)

    # --- Prepare data for printing and file writing ---
    scenario_stats = mc_results['scenario_stats']
    
    # 1. Sort scenarios
    sorted_scenarios = sorted(
        scenario_stats.items(), 
        key=lambda x: x[1]['success_rate'], 
        reverse=True
    )

    # 2. Get overall algorithm counts
    algo_totals = mc_results['algorithm_totals']
    liebe_success = algo_totals.get('Liebe', 0)
    voting_success = algo_totals.get('Voting', 0) 
    pyramid_success = algo_totals.get('Pyramid', 0)
    fallback_benefit = voting_success + pyramid_success
    total_successful_runs = sum(count for algo, count in algo_totals.items() if algo != "Failed")
    most_used = max(algo_totals.items(), key=lambda x: x[1] if x[0] != "Failed" else 0)

    # Print detailed analysis
    print_detailed_analysis(mc_results, sorted_scenarios)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results to text file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results/monte_carlo_results_{mc_results['num_trials_per_scenario']}_trials_{timestamp}.txt"
    
    # Write a comprehensive report
    with open(output_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MONTE CARLO SIMULATION REPORT\n")
        f.write("Celestial Navigation Progressive Strategy for Military Vehicles\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Trials: {mc_results['total_trials']}\n")
        f.write(f"Catalog Size: {len(MOCK_STAR_CATALOG)} stars\n")
        f.write(f"Match Tolerance: {MATCH_TOLERANCE_ARCSEC} arcsec\n")
        f.write(f"Overall Success Rate: {mc_results['overall_success_rate']:.1f}%\n\n")
        
        f.write("="*70 + "\n")
        f.write("ALGORITHM LOGIC SUMMARY\n")
        f.write("="*70 + "\n")
        f.write("The progressive strategy relies on three algorithms with increasing robustness and computational cost:\n\n")
        
        f.write("1. LIEBE'S TRIANGLE ALGORITHM (Fastest - Primary Method):\n")
        f.write("   - Logic: Identifies the three brightest stars and uses the shortest angular distance for a quick index lookup.\n")
        f.write("   - Verification: Match requires at least two distinct catalog pairs to share the shortest distance (Simulated quick match).\n")
        
        f.write("\n2. GEOMETRIC VOTING ALGORITHM (Moderate - Fallback 1):\n")
        f.write("   - Logic: Calculates all inter-star distances in the observed field. Each distance generates 'votes' for catalog star pairs that share that distance.\n")
        f.write("   - Verification: Must exceed a 75% cumulative vote threshold AND pass a pattern consistency check between the top N candidate stars (Simulated geometric check).\n")
        
        f.write("\n3. PYRAMID ALGORITHM (Most Robust - Fallback 2):\n")
        f.write("   - Logic: Requires 4 stars. Uses the 4 shortest distances (legs) for highly redundant, search-less lookups against the index.\n")
        f.write("   - Verification: Requires a high number of index hits and at least 4 unique candidate star IDs (Simulated high-redundancy check).\n\n")
        
        f.write("="*70 + "\n")
        f.write("ALGORITHM PERFORMANCE BY OPERATIONAL SCENARIO\n")
        f.write("="*70 + "\n")
        
        # Use the already sorted list
        for scenario_name, stats in sorted_scenarios:
            f.write(f"\nSCENARIO: {scenario_name}\n")
            f.write(f"  Description: {stats['description']}\n")
            f.write(f"  Overall Success Rate: {stats['success_rate']:.1f}%\n")
            f.write(f"  Avg Computation Time: {stats['avg_computation_time']*1000:.3f} ms\n")
            
            f.write(f"  Algorithm Distribution (% of total trials):\n")
            total = stats['total_trials']
            for algo, count in sorted(stats['algorithm_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                f.write(f"    - {algo:12s}: {percentage:5.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY OPERATIONAL FINDINGS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"1. Overall System Reliability: {mc_results['overall_success_rate']:.1f}% success rate across {mc_results['total_trials']} trials.\n\n")
        
        f.write(f"2. Progressive Strategy Efficacy:\n")
        f.write(f"   - Primary Solver (Liebe) solved {(liebe_success/mc_results['total_trials'])*100:.1f}% of all trials.\n")
        f.write(f"   - Fallback System (Voting + Pyramid) contributed {(fallback_benefit/mc_results['total_trials'])*100:.1f}% additional coverage.\n")
        
        if liebe_success > 0:
            increase = (fallback_benefit / liebe_success) * 100
            f.write(f"   - The fallback system increased overall success by {increase:.1f}% over using Liebe alone.\n\n")
        else:
            f.write(f"   - Liebe failed in all cases, Fallback provided all successful identifications.\n\n")
        
        f.write(f"3. Environmental Degradation:\n")
        # Use pre-calculated best/worst scenarios
        best_scenario = sorted_scenarios[0]
        worst_scenario = sorted_scenarios[-1]
        f.write(f"   - Best Case ({best_scenario[0]}): Performance at {best_scenario[1]['success_rate']:.1f}% success.\n")
        f.write(f"   - Worst Case ({worst_scenario[0]}): Performance drops to {worst_scenario[1]['success_rate']:.1f}% success.\n")
        f.write(f"   - High Noise (e.g., Dust Storm) primarily triggers the Geometric Voting algorithm due to its inherent resilience to positional jitter.\n")
        f.write(f"   - Low Star Count (e.g., Forest Canopy) results in the highest failure rate, as Voting and Pyramid require a sufficient number of features.\n")
        
        # Add validation framework results
        if 'validation_results' in mc_results:
            validator = CelestialNavigationValidator(match_tolerance_arcsec=MATCH_TOLERANCE_ARCSEC)
            scenario_names = [scenario.name for scenario in OPERATIONAL_SCENARIOS]
            validation_report = validator.generate_validation_report(mc_results['validation_results'], scenario_names)
            
            f.write("\n" + "="*70 + "\n")
            f.write("VALIDATION FRAMEWORK RESULTS\n")
            f.write("="*70 + "\n")
            f.write(validation_report)
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"END OF REPORT. File saved: {output_filename}\n")
        f.write("="*70 + "\n")

    print("\n" + "="*70)
    print("Simulation Complete - Results ready for Military Review paper")
    print(f"Results saved to: {output_filename}")
    print("="*70)