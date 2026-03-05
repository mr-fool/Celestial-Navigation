import math
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AttitudeError:
    roll_error: float
    pitch_error: float
    yaw_error: float
    total_rms_error: float
    confidence_interval_95: Tuple[float, float]

@dataclass
class ValidationResult:
    scenario_name: str
    algorithm_used: str
    success: bool
    computation_time: float
    num_stars_matched: int
    attitude_error: Optional[AttitudeError]
    confidence_score: float

class CelestialNavigationValidator:
    def __init__(self, match_tolerance_arcsec=15.0):
        self.match_tolerance_arcsec = match_tolerance_arcsec
        self.degrees_to_radians = math.pi / 180.0
        self.radians_to_degrees = 180.0 / math.pi
        
    def calculate_attitude_error(self, true_attitude, estimated_attitude):
        error_matrix = estimated_attitude @ true_attitude.T
        roll_error = math.atan2(error_matrix[2, 1], error_matrix[2, 2])
        pitch_error = math.asin(max(-1.0, min(1.0, -error_matrix[2, 0])))
        yaw_error = math.atan2(error_matrix[1, 0], error_matrix[0, 0])
        roll_error_deg = abs(roll_error * self.radians_to_degrees)
        pitch_error_deg = abs(pitch_error * self.radians_to_degrees)
        yaw_error_deg = abs(yaw_error * self.radians_to_degrees)
        rms_error = math.sqrt((roll_error_deg**2 + pitch_error_deg**2 + yaw_error_deg**2) / 3)
        return AttitudeError(roll_error_deg, pitch_error_deg, yaw_error_deg, rms_error, (0, 0))
    
    def calculate_confidence_intervals(self, errors, confidence_level=0.95):
        if len(errors) < 2:
            return (0, 0)
        n = len(errors)
        mean_error = statistics.mean(errors)
        stdev_error = statistics.stdev(errors) if n > 1 else 0
        t_value = 2.045 if n-1 >= 20 else (2.086 if n-1 >= 10 else (2.262 if n-1 >= 5 else 3.182))
        if n > 30:
            t_value = 1.96
        margin_of_error = t_value * (stdev_error / math.sqrt(n))
        return (mean_error - margin_of_error, mean_error + margin_of_error)
    
    def calculate_confidence_score(self, num_stars_matched, star_position_error, computation_time):
        star_count_score = min(num_stars_matched / 6.0, 1.0)
        error_score = max(0, 1.0 - (star_position_error / (2 * self.match_tolerance_arcsec)))
        time_score = max(0, 1.0 - (computation_time / 0.1))
        confidence = (0.5 * star_count_score + 0.4 * error_score + 0.1 * time_score)
        return max(0, min(1, confidence))
    
    def validate_trial(self, scenario_name, algorithm_used, success, computation_time,
                      num_stars_matched, true_attitude=None, estimated_attitude=None, matched_stars=None):
        attitude_error = None
        confidence_score = 0.0
        if success and true_attitude is not None and estimated_attitude is not None:
            attitude_error = self.calculate_attitude_error(true_attitude, estimated_attitude)
            star_position_error = self.match_tolerance_arcsec * 0.8
            confidence_score = self.calculate_confidence_score(num_stars_matched, star_position_error, computation_time)
        elif success:
            star_position_error = self.match_tolerance_arcsec * 1.0
            confidence_score = self.calculate_confidence_score(num_stars_matched, star_position_error, computation_time)
        return ValidationResult(scenario_name, algorithm_used, success, computation_time,
                               num_stars_matched, attitude_error, confidence_score)
    
    def analyze_scenario_performance(self, validation_results):
        successful_trials = [r for r in validation_results if r.success]
        success_rate = len(successful_trials) / len(validation_results) * 100
        attitude_errors = [r.attitude_error.total_rms_error for r in successful_trials if r.attitude_error]
        mean_attitude_error = statistics.mean(attitude_errors) if attitude_errors else 0
        error_ci_95 = self.calculate_confidence_intervals(attitude_errors)
        comp_times = [r.computation_time for r in validation_results]
        mean_comp_time = statistics.mean(comp_times) if comp_times else 0
        confidence_scores = [r.confidence_score for r in successful_trials]
        mean_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        return {
            'total_trials': len(validation_results),
            'success_rate_percent': success_rate,
            'mean_attitude_error_deg': mean_attitude_error,
            'attitude_error_ci_95_deg': error_ci_95,
            'mean_computation_time_ms': mean_comp_time * 1000,
            'mean_confidence_score': mean_confidence,
            'successful_trials': len(successful_trials),
            'failed_trials': len(validation_results) - len(successful_trials),
        }
    
    def generate_validation_report(self, all_results, scenarios):
        report = []
        report.append("CELESTIAL NAVIGATION VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Total Trials: {len(all_results)}")
        successful_trials = [r for r in all_results if r.success]
        overall_success_rate = len(successful_trials) / len(all_results) * 100
        report.append(f"Success Rate: {overall_success_rate:.1f}%")
        attitude_errors = [r.attitude_error.total_rms_error for r in successful_trials if r.attitude_error]
        if attitude_errors:
            mean_error = statistics.mean(attitude_errors)
            error_ci = self.calculate_confidence_intervals(attitude_errors)
            report.append(f"Mean Attitude Error: {mean_error:.3f} degree")
            report.append(f"95% CI: [{error_ci[0]:.3f}, {error_ci[1]:.3f}] degree")
        report.append("\nSCENARIO-WISE PERFORMANCE")
        for scenario in scenarios:
            scenario_results = [r for r in all_results if r.scenario_name == scenario]
            if scenario_results:
                stats = self.analyze_scenario_performance(scenario_results)
                report.append(f"\n{scenario}:")
                report.append(f"  Success Rate: {stats['success_rate_percent']:.1f}%")
                report.append(f"  Mean Attitude Error: {stats['mean_attitude_error_deg']:.3f} degree")
                report.append(f"  Mean Computation Time: {stats['mean_computation_time_ms']:.2f} ms")
        return "\n".join(report)
