import math
import random
import time
from typing import List, Dict, Tuple, Optional, Any

# --- Configuration and Constants ---
# Use a very small tolerance for star matching (simulating high-precision required)
MATCH_TOLERANCE_ARCSEC = 10.0 
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi
ARCSEC_PER_DEGREE = 3600.0
# Precision used for keying distances in the index (e.g., 0.1 degree for hashing)
INDEX_KEY_PRECISION = 0.1 

# Global variable to store the pre-processed catalog index
CATALOG_INDEX: Dict[int, List[Tuple[int, int]]] = {}

# --- 1. Mock Tycho-2 Star Catalog (Simplified) ---
# In a real implementation, this would be loaded from a large, external file.
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
    {"ID": 10, "NAME": "Fomalhaut", "RA": 22.956, "DEC": -29.62, "MAG": 1.16, "BRBRIGHTNESS": 0.8},
]


# --- 2. Core Geometric Functions ---

def calculate_angular_distance(star1: Dict, star2: Dict, use_observed: bool = False) -> float:
    """
    Calculates the angular distance (great-circle distance) between two stars.
    If use_observed is True, uses 'RA_OBSERVED'/'DEC_OBSERVED' keys.
    Returns: Distance in degrees.
    """
    ra_key = 'RA_OBSERVED' if use_observed else 'RA'
    dec_key = 'DEC_OBSERVED' if use_observed else 'DEC'

    ra1, dec1 = star1.get(ra_key, star1['RA']) * DEGREES_TO_RADIANS, star1.get(dec_key, star1['DEC']) * DEGREES_TO_RADIANS
    ra2, dec2 = star2.get(ra_key, star2['RA']) * DEGREES_TO_RADIANS, star2.get(dec_key, star2['DEC']) * DEGREES_TO_RADIANS
    
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
    # Example: 15.67 degrees -> key 156, using 0.1 degree precision
    return int(round(angle_deg / INDEX_KEY_PRECISION))

def build_angular_distance_index(catalog: List[Dict]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Simulates the pre-processing of the star catalog into an index (k-vector/hash map).
    
    The index maps a quantized angular distance key to a list of star pairs
    in the catalog that share that distance. This replaces the slow N^3 or N^2 search.
    """
    index: Dict[int, List[Tuple[int, int]]] = {}
    num_stars = len(catalog)

    print(f"Building angular distance index for {num_stars} stars...")

    for i in range(num_stars):
        for j in range(i + 1, num_stars):
            star_a = catalog[i]
            star_b = catalog[j]
            
            # Calculate the true angular distance
            distance = calculate_angular_distance(star_a, star_b)
            
            # Create the hash key (quantized distance)
            key = get_index_key(distance)
            
            # Store the pair of Catalog IDs under the key
            if key not in index:
                index[key] = []
            index[key].append((star_a['ID'], star_b['ID']))
            
    print(f"Index built with {len(index)} unique distance keys.")
    return index

# --- 3. Simulation Environment ---

def simulate_camera_view(catalog: List[Dict], fov_size: float = 20.0, num_stars: int = 5) -> List[Dict]:
    """
    Simulates a 'Lost-In-Space' image capture: selects a random patch of sky and adds noise.
    """
    if len(catalog) < num_stars:
        return catalog
        
    center_star = random.choice(catalog)
    observed_stars = []
    max_dist = fov_size * 0.7 

    for star in catalog:
        if calculate_angular_distance(center_star, star) < max_dist:
            observed_stars.append(star)

    noisy_observed_stars = []
    # Sort by brightness and take the brightest N stars that fit the FOV
    observed_stars.sort(key=lambda x: x['MAG'], reverse=False) 

    for i, star in enumerate(observed_stars[:num_stars]):
        noisy_star = star.copy()
        
        # Add random angular position noise (e.g., up to 10 arcseconds)
        noise_arcsec_ra = random.uniform(-10.0, 10.0)
        noise_arcsec_dec = random.uniform(-10.0, 10.0)
        noise_deg_ra = noise_arcsec_ra / ARCSEC_PER_DEGREE
        noise_deg_dec = noise_arcsec_dec / ARCSEC_PER_DEGREE
        
        # Apply noise to the observed position
        noisy_star['RA_OBSERVED'] = star['RA'] + noise_deg_ra
        noisy_star['DEC_OBSERVED'] = star['DEC'] + noise_deg_dec
        
        # Assign a temporary camera ID (not the true ID)
        noisy_star['TEMP_ID'] = i + 1 

        noisy_observed_stars.append(noisy_star)

    return noisy_observed_stars

# --- 4. Star Identification Algorithms (Optimized Search Logic) ---

def liebe_triangle_match(observed_stars: List[Dict], catalog_index: Dict[int, List[Tuple[int, int]]]) -> bool:
    """
    Liebe's Triangle Match (Optimized): Uses 3 angles and the pre-computed index for O(1) lookup.
    """
    start_time = time.time()
    
    if len(observed_stars) < 3:
        return False

    # 1. Form the pattern from the image (the first 3 stars)
    image_pattern = observed_stars[:3]
    # Calculate the 3 inter-star distances from the observed, noisy positions
    image_angles = calculate_inter_star_angles(image_pattern, use_observed=True)
    image_angles.sort() # Sort the three angles for canonical comparison

    # 2. Search the Index (O(1) lookup concept)
    candidate_triples = set()
    
    # Use the first angle to find initial candidate pairs
    key = get_index_key(image_angles[0])
    # Search a window around the key to account for noise
    keys_to_check = [key - 1, key, key + 1] 

    for k in keys_to_check:
        if k in catalog_index:
            for id_a, id_b in catalog_index[k]:
                # Start building candidate triples from this pair
                candidate_triples.add(tuple(sorted((id_a, id_b))))
    
    # This simulation skips the complex step of intersecting candidate lists
    # for the other two angles. We simply check if the smallest angle of the 
    # image pattern exists in the catalog index. If it does, we assume a match 
    # and proceed to the verification stage (O(logN) verification is omitted here).
    
    match_found = len(candidate_triples) > 0 # Simple check based on finding a matching pair
    
    end_time = time.time()
    print(f"  [Liebe-Optimized] Index hits: {len(candidate_triples)}. Time: {end_time - start_time:.6f}s. Match found: {match_found}")
    return match_found


def geometric_voting_match(observed_stars: List[Dict], catalog_index: Dict[int, List[Tuple[int, int]]]) -> bool:
    """
    Geometric Voting (Optimized): Uses all star pairs, leveraging the index for faster vote counting.
    """
    start_time = time.time()
    
    if len(observed_stars) < 2:
        return False
        
    # 1. Compute all inter-star distances in the image
    image_distances = calculate_inter_star_angles(observed_stars, use_observed=True)
    
    vote_count: Dict[int, int] = {} # Key: Catalog ID, Value: Number of votes
    total_index_hits = 0

    # 2. Fast Index Lookup and Voting
    for img_dist in image_distances:
        key = get_index_key(img_dist)
        # Search a window around the key to account for noise/error
        keys_to_check = [key - 1, key, key + 1] 

        for k in keys_to_check:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_index_hits += 1
                    # If the distance matches, the two catalog stars vote
                    vote_count[id_a] = vote_count.get(id_a, 0) + 1
                    vote_count[id_b] = vote_count.get(id_b, 0) + 1

    # 3. Check for a decisive winner
    # Identify the top N candidates (where N is the number of observed stars)
    top_votes = sorted(vote_count.values(), reverse=True)
    
    match_found = False
    if len(top_votes) >= len(observed_stars):
        # If the top N candidates receive a significant number of votes, it's a match.
        # Required votes is based on the number of possible pairs from N stars: N*(N-1)/2 
        # Since each pair yields 2 votes, required votes is roughly N*(N-1)
        N = len(observed_stars)
        required_votes_cumulative = (N * (N - 1)) / 2 * 2 * 0.75 # 75% of max possible votes
        
        # Check if the sum of votes for the top N candidates meets the threshold
        if sum(top_votes[:N]) >= required_votes_cumulative:
             match_found = True

    end_time = time.time()
    print(f"  [Voting-Optimized] Index Hits: {total_index_hits}. Time: {end_time - start_time:.6f}s. Match found: {match_found}")
    return match_found
    
def pyramid_match(observed_stars: List[Dict], catalog_index: Dict[int, List[Tuple[int, int]]]) -> bool:
    """
    Pyramid Algorithm (Search-Less Concept): Uses four stars and their 6 distances.
    The primary strength is that it's "search-less" after index creation.
    """
    start_time = time.time()

    if len(observed_stars) < 4:
        # High computation penalty is removed since we are now simulating the lookup speed
        return False
        
    # 1. Form the 4-star pattern (Pyramid)
    image_pattern = observed_stars[:4]
    image_angles = calculate_inter_star_angles(image_pattern, use_observed=True)
    
    # 2. Implement the Search-less Index Lookup
    # A real Pyramid algorithm generates 4 specific "legs" which are used as keys.
    # We will simulate this by using the 4 shortest unique angles from the 6 total angles.

    image_angles.sort()
    # Use the 4 shortest angles to perform the candidate lookups
    index_lookups = image_angles[:4]
    
    # Set to store all unique catalog star IDs that are part of the distance match
    candidate_star_ids = set()
    total_lookups = 0

    for angle in index_lookups:
        key = get_index_key(angle)
        keys_to_check = [key - 1, key, key + 1] # Window for noise

        for k in keys_to_check:
            if k in catalog_index:
                for id_a, id_b in catalog_index[k]:
                    total_lookups += 1
                    candidate_star_ids.add(id_a)
                    candidate_star_ids.add(id_b)

    # 3. Verification
    # The true match is found if the intersection of the candidate lists for all 4 legs 
    # results in exactly the four correct catalog stars.
    
    # Since we are simulating, we check if we found *at least* four unique,
    # highly-voted candidate IDs (a proxy for a successful intersection).
    
    match_found = len(candidate_star_ids) >= 4 and total_lookups > 5 

    end_time = time.time()
    print(f"  [Pyramid-Optimized] Index Lookups: {total_lookups}. Candidate IDs: {len(candidate_star_ids)}. Time: {end_time - start_time:.6f}s. Match found: {match_found}")
    return match_found


# --- 5. Progressive Navigation Logic ---

def determine_attitude_progressively(observed_stars: List[Dict], catalog_index: Dict[int, List[Tuple[int, int]]]) -> str:
    """
    Implements your proposed progressive algorithm logic (fast -> robust).
    """
    print(f"\n--- Starting Progressive Star ID for {len(observed_stars)} Observed Stars ---")
    
    # 1. Try Liebe's Algorithm (Fastest, relies on simple pattern matching)
    print("1. Attempting Liebe's Algorithm (Fast & Low Computation)...")
    if liebe_triangle_match(observed_stars, catalog_index):
        return "SUCCESS: Attitude determined by Liebe's Triangle Match (Fastest/Low Noise)."

    # 2. If Liebe's fails (due to lost focus, noise, or LIS), try Geometric Voting
    print("2. Liebe's failed. Falling back to Geometric Voting (Robust to Noise)...")
    if geometric_voting_match(observed_stars, catalog_index):
        return "SUCCESS: Attitude determined by Geometric Voting (Moderate Robustness/Decisive Match)."

    # 3. If Voting fails (due to extreme noise or LIS), use Pyramid Algorithm
    print("3. Voting failed. Falling back to Pyramid Algorithm (Most Robust LIS)...")
    if pyramid_match(observed_stars, catalog_index):
        return "SUCCESS: Attitude determined by Pyramid Match (Highest Robustness/True LIS)."

    return "FAILURE: All star identification attempts failed (GNSS/CELNAV fully degraded)."

# --- Main Execution ---

if __name__ == "__main__":
    
    print("--- Military Truck Celestial Navigation Simulator ---")
    print(f"Using a mock catalog of {len(MOCK_STAR_CATALOG)} stars.")
    print(f"Match tolerance set to {MATCH_TOLERANCE_ARCSEC} arcseconds.")

    # BUILD THE INDEX ONCE (This is the critical pre-processing step)
    CATALOG_INDEX = build_angular_distance_index(MOCK_STAR_CATALOG)

    # --- Scenario 1: Clean View (Should be solved quickly by Liebe's) ---
    clean_view = simulate_camera_view(MOCK_STAR_CATALOG, fov_size=15.0, num_stars=5)
    print("\n=======================================================")
    print("SCENARIO 1: Typical Clean View (5 Stars Visible, Low Noise)")
    print("=======================================================")
    result_clean = determine_attitude_progressively(clean_view, CATALOG_INDEX)
    print(f"\nFinal Result 1: {result_clean}")
    
    # --- Scenario 2: Degraded View (Should force fallback) ---
    # We use fewer stars (3) to ensure Liebe's and Pyramid fail the star count check.
    degraded_view = simulate_camera_view(MOCK_STAR_CATALOG, fov_size=5.0, num_stars=3)
    print("\n=======================================================")
    print("SCENARIO 2: Degraded View (3 Stars Visible)")
    print("=======================================================")
    result_degraded = determine_attitude_progressively(degraded_view, CATALOG_INDEX)
    print(f"\nFinal Result 2: {result_degraded}")

    # --- Scenario 3: Max Visibility View (Should be quick, testing Voting/Pyramid with max stars) ---
    max_view = simulate_camera_view(MOCK_STAR_CATALOG, fov_size=90.0, num_stars=10)
    print("\n=======================================================")
    print("SCENARIO 3: Wide Field View (10 Stars Visible, High Redundancy)")
    print("=======================================================")
    result_max = determine_attitude_progressively(max_view, CATALOG_INDEX)
    print(f"\nFinal Result 3: {result_max}")