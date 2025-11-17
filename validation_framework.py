import math
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AttitudeError:
    """Represents attitude estimation error metrics"""
    roll_error: float  # degrees
    pitch_error: float  # degrees
    yaw_error: float  # degrees
    total_rms_error: float  # degrees
    confidence_interval_95: Tuple[float, float]  # 95% CI for total error

@dataclass
class ValidationResult:
    """Comprehensive validation results for a single trial"""
    scenario_name: str
    algorithm_used: str
    success: bool
    computation_time: float
    num_stars_matched: int
    attitude_error: Optional[AttitudeError]
    confidence_score: float  # 0-1 scale

class CelestialNavigationValidator:
    """
    Framework for validating celestial navigation algorithms with statistical rigor
    """
    
    def __init__(self, match_tolerance_arcsec: float = 15.0):
        self.match_tolerance_arcsec = match_tolerance_arcsec
        self.degrees_to_radians = math.pi / 180.0
        self.radians_to_degrees = 180.0 / math.pi
        
    def calculate_attitude_error(self, 
                               true_attitude: np.ndarray,
                               estimated_attitude: np.ndarray) -> AttitudeError:
        """
        Calculate the error between true and estimated attitude matrices.
        
        Args:
            true_attitude: 3x3 rotation matrix (true orientation)
            estimated_attitude: 3x3 rotation matrix (estimated orientation)
            
        Returns:
            AttitudeError object with roll, pitch, yaw errors and RMS
        """
        # Calculate the error rotation matrix
        error_matrix = estimated_attitude @ true_attitude.T
        
        # Extract Euler angles from error matrix
        # Roll (x-axis rotation)
        roll_error = math.atan2(error_matrix[2, 1], error_matrix[2, 2])
        
        # Pitch (y-axis rotation)
        pitch_error = math.asin(-error_matrix[2, 0])
        
        # Yaw (z-axis rotation)  
        yaw_error = math.atan2(error_matrix[1, 0], error_matrix[0, 0])
        
        # Convert to degrees
        roll_error_deg = abs(roll_error * self.radians_to_degrees)
        pitch_error_deg = abs(pitch_error * self.radians_to_degrees)
        yaw_error_deg = abs(yaw_error * self.radians_to_degrees)
        
        # Calculate RMS error
        rms_error = math.sqrt(
            (roll_error_deg**2 + pitch_error_deg**2 + yaw_error_deg**2) / 3
        )
        
        return AttitudeError(
            roll_error=roll_error_deg,
            pitch_error=pitch_error_deg,
            yaw_error=yaw_error_deg,
            total_rms_error=rms_error,
            confidence_interval_95=(0, 0)  # Will be calculated from multiple trials
        )
    
    def calculate_confidence_intervals(self, 
                                     errors: List[float],
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for attitude errors using t-distribution.
        
        Args:
            errors: List of RMS attitude errors from multiple trials
            confidence_level: Desired confidence level (0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if len(errors) < 2:
            return (0, 0)
            
        n = len(errors)
        mean_error = statistics.mean(errors)
        stdev_error = statistics.stdev(errors) if n > 1 else 0
        
        # For small samples, use t-distribution
        # For 95% CI and n-1 degrees of freedom
        if n <= 30:
            # Approximate t-value for 95% CI
            # For exact values, you could use scipy.stats.t.ppf
            t_value = 2.045 if n-1 >= 20 else (
                2.086 if n-1 >= 10 else
                2.262 if n-1 >= 5 else
                3.182  # n-1 = 2
            )
        else:
            # For large samples, use z-value
            t_value = 1.96
            
        margin_of_error = t_value * (stdev_error / math.sqrt(n))
        
        return (mean_error - margin_of_error, mean_error + margin_of_error)
    
    def calculate_star_position_error(self,
                                    matched_stars: List[Tuple[Dict, Dict]]) -> float:
        """
        Calculate the average angular position error for matched stars.
        
        Args:
            matched_stars: List of (observed_star, catalog_star) tuples
            
        Returns:
            Average angular error in arcseconds
        """
        if not matched_stars:
            return float('inf')
            
        errors = []
        for observed_star, catalog_star in matched_stars:
            # Calculate angular distance between observed and catalog positions
            ra1 = observed_star.get('RA_OBSERVED', observed_star['RA']) * self.degrees_to_radians
            dec1 = observed_star.get('DEC_OBSERVED', observed_star['DEC']) * self.degrees_to_radians
            ra2 = catalog_star['RA'] * self.degrees_to_radians
            dec2 = catalog_star['DEC'] * self.degrees_to_radians
            
            cos_angle = (math.sin(dec1) * math.sin(dec2) +
                        math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            angular_error_deg = math.acos(cos_angle) * self.radians_to_degrees
            angular_error_arcsec = angular_error_deg * 3600
            
            errors.append(angular_error_arcsec)
        
        return statistics.mean(errors)
    
    def calculate_confidence_score(self,
                                 num_stars_matched: int,
                                 star_position_error: float,
                                 computation_time: float) -> float:
        """
        Calculate a composite confidence score (0-1) for the attitude solution.
        
        Args:
            num_stars_matched: Number of successfully matched stars
            star_position_error: Average angular error in arcseconds
            computation_time: Time taken for computation in seconds
            
        Returns:
            Confidence score between 0 (no confidence) and 1 (high confidence)
        """
        # Normalize each factor to 0-1 scale
        star_count_score = min(num_stars_matched / 6.0, 1.0)  # Max at 6+ stars
        
        # Error score: 0 arcsec = 1.0, tolerance = 0.5, 2x tolerance = 0.0
        error_score = max(0, 1.0 - (star_position_error / (2 * self.match_tolerance_arcsec)))
        
        # Time score: faster is better, but not critical
        time_score = max(0, 1.0 - (computation_time / 0.1))  # Penalize over 100ms
        
        # Weighted combination
        confidence = (0.5 * star_count_score + 
                     0.4 * error_score + 
                     0.1 * time_score)
        
        return max(0, min(1, confidence))
    
    def validate_trial(self,
                      scenario_name: str,
                      algorithm_used: str,
                      success: bool,
                      computation_time: float,
                      num_stars_matched: int,
                      true_attitude: Optional[np.ndarray] = None,
                      estimated_attitude: Optional[np.ndarray] = None,
                      matched_stars: Optional[List[Tuple[Dict, Dict]]] = None) -> ValidationResult:
        """
        Comprehensive validation for a single trial.
        
        Args:
            scenario_name: Name of the operational scenario
            algorithm_used: Which algorithm was used
            success: Whether identification was successful
            computation_time: Time taken for computation
            num_stars_matched: Number of stars successfully matched
            true_attitude: True attitude matrix (if available)
            estimated_attitude: Estimated attitude matrix (if available)
            matched_stars: List of matched star pairs (if available)
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        attitude_error = None
        confidence_score = 0.0
        
        if success and true_attitude is not None and estimated_attitude is not None:
            # Calculate attitude error
            attitude_error = self.calculate_attitude_error(true_attitude, estimated_attitude)
            
            # Calculate star position error if matched stars available
            star_position_error = 0.0
            if matched_stars:
                star_position_error = self.calculate_star_position_error(matched_stars)
            else:
                # Estimate based on algorithm and conditions
                star_position_error = self.match_tolerance_arcsec * 0.8  # Conservative estimate
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(
                num_stars_matched, star_position_error, computation_time
            )
        elif success:
            # Successful but no attitude matrices - estimate confidence
            star_position_error = self.match_tolerance_arcsec * 1.0  # Typical error
            confidence_score = self.calculate_confidence_score(
                num_stars_matched, star_position_error, computation_time
            )
        else:
            # Failed identification
            confidence_score = 0.0
        
        return ValidationResult(
            scenario_name=scenario_name,
            algorithm_used=algorithm_used,
            success=success,
            computation_time=computation_time,
            num_stars_matched=num_stars_matched,
            attitude_error=attitude_error,
            confidence_score=confidence_score
        )
    
    def analyze_scenario_performance(self,
                                   validation_results: List[ValidationResult]) -> Dict[str, any]:
        """
        Analyze performance metrics for a specific scenario.
        
        Args:
            validation_results: List of validation results for the scenario
            
        Returns:
            Dictionary with comprehensive performance statistics
        """
        successful_trials = [r for r in validation_results if r.success]
        failed_trials = [r for r in validation_results if not r.success]
        
        # Basic statistics
        success_rate = len(successful_trials) / len(validation_results) * 100
        
        # Attitude error statistics
        attitude_errors = []
        if successful_trials and successful_trials[0].attitude_error:
            attitude_errors = [r.attitude_error.total_rms_error for r in successful_trials]
            mean_attitude_error = statistics.mean(attitude_errors) if attitude_errors else 0
            std_attitude_error = statistics.stdev(attitude_errors) if len(attitude_errors) > 1 else 0
        else:
            mean_attitude_error = 0
            std_attitude_error = 0
        
        # Confidence intervals for attitude error
        error_ci_95 = self.calculate_confidence_intervals(attitude_errors)
        
        # Computation time statistics
        comp_times = [r.computation_time for r in validation_results]
        mean_comp_time = statistics.mean(comp_times) if comp_times else 0
        
        # Confidence score statistics
        confidence_scores = [r.confidence_score for r in successful_trials]
        mean_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'total_trials': len(validation_results),
            'success_rate_percent': success_rate,
            'mean_attitude_error_deg': mean_attitude_error,
            'std_attitude_error_deg': std_attitude_error,
            'attitude_error_ci_95_deg': error_ci_95,
            'mean_computation_time_ms': mean_comp_time * 1000,
            'mean_confidence_score': mean_confidence,
            'successful_trials': len(successful_trials),
            'failed_trials': len(failed_trials),
            'algorithm_distribution': self._get_algorithm_distribution(validation_results)
        }
    
    def _get_algorithm_distribution(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Get distribution of algorithms used in successful trials."""
        distribution = {}
        for result in results:
            if result.success:
                distribution[result.algorithm_used] = distribution.get(result.algorithm_used, 0) + 1
        return distribution
    
    def generate_validation_report(self,
                                 all_results: List[ValidationResult],
                                 scenarios: List[str]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            all_results: All validation results across scenarios
            scenarios: List of scenario names
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("CELESTIAL NAVIGATION VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Total Trials: {len(all_results)}")
        report.append(f"Scenarios: {', '.join(scenarios)}")
        report.append("")
        
        # Overall statistics
        successful_trials = [r for r in all_results if r.success]
        overall_success_rate = len(successful_trials) / len(all_results) * 100
        
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Success Rate: {overall_success_rate:.1f}%")
        report.append(f"Successful Trials: {len(successful_trials)}")
        report.append(f"Failed Trials: {len(all_results) - len(successful_trials)}")
        
        if successful_trials:
            attitude_errors = [r.attitude_error.total_rms_error for r in successful_trials 
                             if r.attitude_error]
            if attitude_errors:
                mean_error = statistics.mean(attitude_errors)
                error_ci = self.calculate_confidence_intervals(attitude_errors)
                report.append(f"Mean Attitude Error: {mean_error:.3f}°")
                report.append(f"95% CI Attitude Error: [{error_ci[0]:.3f}°, {error_ci[1]:.3f}°]")
        
        # Scenario-wise breakdown
        report.append("")
        report.append("SCENARIO-WISE PERFORMANCE")
        report.append("-" * 30)
        
        for scenario in scenarios:
            scenario_results = [r for r in all_results if r.scenario_name == scenario]
            if scenario_results:
                scenario_stats = self.analyze_scenario_performance(scenario_results)
                report.append(f"\n{scenario}:")
                report.append(f"  Success Rate: {scenario_stats['success_rate_percent']:.1f}%")
                report.append(f"  Mean Attitude Error: {scenario_stats['mean_attitude_error_deg']:.3f}°")
                report.append(f"  Mean Computation Time: {scenario_stats['mean_computation_time_ms']:.2f} ms")
                report.append(f"  Mean Confidence: {scenario_stats['mean_confidence_score']:.3f}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Example test of the validation framework
    validator = CelestialNavigationValidator()
    
    # Example attitude matrices (identity = no error)
    true_attitude = np.eye(3)
    estimated_attitude = np.eye(3)  # Perfect estimation
    
    # Calculate error
    error = validator.calculate_attitude_error(true_attitude, estimated_attitude)
    print(f"Perfect estimation error: {error.total_rms_error:.6f}°")
    
    # Example with small error
    small_error_matrix = np.array([
        [0.9998, -0.0175, 0.0087],
        [0.0175, 0.9998, -0.0015],
        [-0.0087, 0.0015, 0.9999]
    ])
    error = validator.calculate_attitude_error(true_attitude, small_error_matrix)
    print(f"Small error estimation: {error.total_rms_error:.3f}°")
    
    # Test confidence intervals
    sample_errors = [0.1, 0.15, 0.12, 0.18, 0.11]
    ci = validator.calculate_confidence_intervals(sample_errors)
    print(f"95% CI for sample errors: [{ci[0]:.3f}, {ci[1]:.3f}]")