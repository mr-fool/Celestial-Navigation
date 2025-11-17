"""
Script to generate 4 diagrams for the Celestial Navigation Military Review Paper:
- Figure 1: Monte Carlo Success Rates by Operational Scenario
- Figure 2: Algorithm Performance and Fallback Strategy Analysis
- Figure 3: Environmental Degradation Impact Analysis
- Figure 4: Celestial Navigation Sensor Layout Diagram
"""

import os
import matplotlib
import sys
import importlib.util
import numpy as np
from typing import Dict, Any, List, Tuple

# Use 'Agg' backend for non-interactive environments
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

# Function to import simulation module
def import_simulation_module(filepath):
    """Import the celestial navigation simulation module dynamically"""
    spec = importlib.util.spec_from_file_location("celestial_nav", filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        print(f"Error: Simulation module not found at {filepath}")
        return None

def run_simulation_for_visualization():
    """Run the simulation to get data for visualization"""
    simulation_file = "celestial_nav_simulation.py"
    sim_module = import_simulation_module(simulation_file)
    if sim_module is None:
        return None
    
    print("Running Monte Carlo simulation for visualization data...")
    results = sim_module.run_monte_carlo_simulation(num_trials=1000, verbose=False)
    return results

def generate_success_rates_chart(scenario_stats: Dict[str, Dict], output_dir: str = 'figures') -> str:
    """
    Generate Figure 1: Monte Carlo Success Rates by Operational Scenario
    """
    if not MATPLOTLIB_AVAILABLE:
        return "Skipped (Matplotlib missing)"

    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    scenarios = []
    success_rates = []
    colors = []
    descriptions = []
    
    color_map = {
        'Optimal_WideField': '#2ecc71',      # Green - best performance
        'Clear_Rural_Base': '#3498db',       # Blue - good performance  
        'Urban_Canyon_Restricted': '#f39c12', # Orange - moderate
        'Forest_Canopy_Obscured': '#e74c3c',  # Red - poor
        'Dust_Storm_HighNoise': '#9b59b6',    # Purple - challenging
        'Vehicle_Motion_Extreme': '#34495e'   # Dark gray - worst
    }
    
    # Sort scenarios by success rate
    sorted_scenarios = sorted(
        scenario_stats.items(), 
        key=lambda x: x[1]['success_rate'], 
        reverse=True
    )
    
    for scenario_name, stats in sorted_scenarios:
        scenarios.append(scenario_name.replace('_', '\n'))
        success_rates.append(stats['success_rate'])
        colors.append(color_map.get(scenario_name, '#95a5a6'))
        descriptions.append(stats['description'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bars
    bars = ax.bar(scenarios, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Customize the chart
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Operational Scenario', fontsize=14, fontweight='bold')
    ax.set_title('Figure 1: Star Identification Success Rates by Military Operational Scenario\n(1000 Trials per Scenario)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis limit
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add scenario descriptions as annotations
    for i, (scenario, desc) in enumerate(zip(scenarios, descriptions)):
        short_desc = desc.split('(')[0].strip()
        ax.text(i, -15, short_desc, ha='center', va='top', fontsize=9, 
                style='italic', color='gray', rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, "fig1_success_rates.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def generate_algorithm_performance_chart(scenario_stats: Dict[str, Dict], output_dir: str = 'figures') -> str:
    """
    Generate Figure 2: Algorithm Performance and Fallback Strategy Analysis
    """
    if not MATPLOTLIB_AVAILABLE:
        return "Skipped (Matplotlib missing)"

    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for stacked bar chart
    scenarios = []
    liebe_data = []
    voting_data = []
    pyramid_data = []
    failed_data = []
    
    # Sort scenarios by overall success rate
    sorted_scenarios = sorted(
        scenario_stats.items(), 
        key=lambda x: x[1]['success_rate'], 
        reverse=True
    )
    
    for scenario_name, stats in sorted_scenarios:
        scenarios.append(scenario_name.replace('_', '\n'))
        algo_dist = stats['algorithm_distribution']
        total = stats['total_trials']
        
        liebe_pct = (algo_dist.get('Liebe', 0) / total) * 100
        voting_pct = (algo_dist.get('Voting', 0) / total) * 100
        pyramid_pct = (algo_dist.get('Pyramid', 0) / total) * 100
        failed_pct = (algo_dist.get('Failed', 0) / total) * 100
        
        liebe_data.append(liebe_pct)
        voting_data.append(voting_pct)
        pyramid_data.append(pyramid_pct)
        failed_data.append(failed_pct)
    
    # Create figure with more vertical space
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased height for title space
    
    # Create stacked bars
    bar_width = 0.7
    x_pos = np.arange(len(scenarios))
    
    bars_liebe = ax.bar(x_pos, liebe_data, bar_width, label='Liebe (Fast)', 
                       color='#2ecc71', alpha=0.8, edgecolor='black')
    bars_voting = ax.bar(x_pos, voting_data, bar_width, bottom=liebe_data, 
                        label='Voting (Robust)', color='#3498db', alpha=0.8, edgecolor='black')
    bars_pyramid = ax.bar(x_pos, pyramid_data, bar_width, 
                         bottom=[i+j for i,j in zip(liebe_data, voting_data)], 
                         label='Pyramid (LIS)', color='#9b59b6', alpha=0.8, edgecolor='black')
    bars_failed = ax.bar(x_pos, failed_data, bar_width, 
                        bottom=[i+j+k for i,j,k in zip(liebe_data, voting_data, pyramid_data)], 
                        label='Failed', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    # Add value annotations
    for i, (liebe, vote, pyramid, fail) in enumerate(zip(liebe_data, voting_data, pyramid_data, failed_data)):
        total_success = liebe + vote + pyramid
        ax.text(i, total_success + 2, f'{total_success:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Customize chart
    ax.set_ylabel('Percentage of Trials (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Operational Scenario', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2: Algorithm Performance and Progressive Fallback Strategy\n(1000 Trials per Scenario)', 
                 fontsize=16, fontweight='bold', pad=30)  # Increased pad for title spacing
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 110)
    
    # Move legend to below the chart to avoid covering title
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=4, framealpha=0.9, fontsize=11)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add strategy explanation (moved to side to avoid legend conflict)
    explanation_text = (
        "Progressive Strategy:\n"
        "1. Liebe Algorithm (Fastest)\n"
        "2. Geometric Voting (Robust)\n" 
        "3. Pyramid (Most Reliable LIS)"
    )
    ax.text(0.02, 0.85, explanation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legend
    
    # Save figure
    filepath = os.path.join(output_dir, "fig2_algorithm_performance.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def generate_environmental_analysis_chart(scenario_stats: Dict[str, Dict], output_dir: str = 'figures') -> str:
    """
    Generate Figure 3: Environmental Degradation Impact Analysis
    """
    if not MATPLOTLIB_AVAILABLE:
        return "Skipped (Matplotlib missing)"

    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for bubble chart
    scenarios = []
    noise_levels = []
    star_counts = []
    success_rates = []
    bubble_sizes = []
    colors = []
    
    scenario_params = {
        'Optimal_WideField': {'noise': 0.8, 'stars': 10, 'color': '#2ecc71'},
        'Clear_Rural_Base': {'noise': 1.0, 'stars': 7, 'color': '#3498db'},
        'Urban_Canyon_Restricted': {'noise': 1.5, 'stars': 4, 'color': '#f39c12'},
        'Forest_Canopy_Obscured': {'noise': 1.2, 'stars': 3, 'color': '#e74c3c'},
        'Dust_Storm_HighNoise': {'noise': 2.5, 'stars': 5, 'color': '#9b59b6'},
        'Vehicle_Motion_Extreme': {'noise': 3.5, 'stars': 6, 'color': '#34495e'}
    }
    
    for scenario_name, stats in scenario_stats.items():
        scenarios.append(scenario_name)
        params = scenario_params[scenario_name]
        noise_levels.append(params['noise'])
        star_counts.append(params['stars'])
        success_rates.append(stats['success_rate'])
        # Use meaningful bubble sizes based on computation time (avoid zeros)
        comp_time = max(stats['avg_computation_time'], 0.0001)  # Avoid zero
        bubble_sizes.append(comp_time * 8000)  # Adjusted scale for better visibility
        colors.append(params['color'])
    
    # Create figure with subplots - simplified layout
    fig = plt.figure(figsize=(16, 12))  # Larger figure for better readability
    
    # Subplot 1: Noise vs Success Rate (main analysis)
    ax1 = fig.add_subplot(2, 2, 1)
    scatter1 = ax1.scatter(noise_levels, success_rates, s=bubble_sizes, c=colors, 
                          alpha=0.7, edgecolors='black')
    
    # Add scenario labels with better positioning to avoid overlaps
    label_offsets = {
        'Optimal_WideField': (15, 15),
        'Clear_Rural_Base': (15, -25),  # Moved further down
        'Urban_Canyon_Restricted': (-25, 15),  # Moved further left
        'Forest_Canopy_Obscured': (15, -30),   # Moved further down
        'Dust_Storm_HighNoise': (-30, 15),     # Moved further left  
        'Vehicle_Motion_Extreme': (15, -35)    # Moved further down
    }
    
    for i, scenario in enumerate(scenarios):
        offset = label_offsets.get(scenario, (10, 10))
        ax1.annotate(scenario.replace('_', '\n'), 
                    (noise_levels[i], success_rates[i]),
                    xytext=offset, textcoords='offset points',
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1))
    
    ax1.set_xlabel('Noise Level (Multiplier of Baseline)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Noise Impact on Performance\n(Bubble size = Computation Time)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 4.0)
    ax1.set_ylim(0, 105)
    
    # Subplot 2: Star Count vs Success Rate
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(star_counts, success_rates, s=120, c=colors, alpha=0.7, edgecolors='black')
    
    # Custom offsets for star availability plot to fix "Clear" label overlap
    star_label_offsets = {
        'Optimal_WideField': (10, 10),
        'Clear_Rural_Base': (10, -25),  # Moved down to avoid overlap
        'Urban_Canyon_Restricted': (-25, 10),  # Moved left
        'Forest_Canopy_Obscured': (10, -25),   # Moved down
        'Dust_Storm_HighNoise': (-25, 10),     # Moved left
        'Vehicle_Motion_Extreme': (10, -25)    # Moved down
    }
    
    for i, scenario in enumerate(scenarios):
        offset = star_label_offsets.get(scenario, (10, 10))
        ax2.annotate(scenario.replace('_', '\n'), 
                    (star_counts[i], success_rates[i]),
                    xytext=offset, textcoords='offset points',
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1))
    
    ax2.set_xlabel('Visible Stars in FOV', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Star Availability Impact', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 11)
    ax2.set_ylim(0, 105)
    
    # Subplot 3: Computation Time by Scenario (fixed to show actual values)
    ax3 = fig.add_subplot(2, 2, 3)
    computation_times = [max(stats['avg_computation_time'] * 1000, 0.01) for stats in scenario_stats.values()]  # Convert to ms, ensure minimum
    scenario_names_short = [name.replace('_', '\n') for name in scenarios]
    
    bars = ax3.bar(scenario_names_short, computation_times, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels with proper formatting
    for bar, time_val in zip(bars, computation_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.3f} ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax3.set_ylabel('Computation Time (milliseconds)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Operational Scenario', fontsize=12, fontweight='bold')
    ax3.set_title('Algorithm Processing Time', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    # Set reasonable y-limit based on data
    max_time = max(computation_times) * 1.2
    ax3.set_ylim(0, max_time)
    
    # Subplot 4: Success rate by scenario for quick reference
    ax4 = fig.add_subplot(2, 2, 4)
    success_rates_values = [stats['success_rate'] for stats in scenario_stats.values()]
    
    bars2 = ax4.bar(scenario_names_short, success_rates_values, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, rate in zip(bars2, success_rates_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Operational Scenario', fontsize=12, fontweight='bold')
    ax4.set_title('Success Rate Overview', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Main title
    fig.suptitle('Figure 3: Environmental Degradation Impact Analysis\nMilitary Vehicle Celestial Navigation Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, "fig3_environmental_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def generate_sensor_layout(fov_size: float, output_dir: str = 'figures') -> str:
    """
    Generate Figure 4: Celestial Navigation Sensor Layout Diagram
    """
    if not MATPLOTLIB_AVAILABLE:
        return "Skipped (Matplotlib missing)"

    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    radius = fov_size / 2.0
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')

    # 1. Draw the Sensor Field of View (FOV)
    fov_circle = patches.Circle((0, 0), radius, 
                                edgecolor='#1f77b4', facecolor='lightblue', 
                                linestyle='-', linewidth=3, alpha=0.3, 
                                label=f'Sensor FOV ({fov_size}° Diameter)')
    ax.add_patch(fov_circle)

    # 2. Draw Center Point (Bore Sight)
    ax.plot(0, 0, 'o', color='red', markersize=12, markeredgewidth=2, 
            label='Vehicle Bore Sight (0, 0)')
    ax.text(0, 0.5, 'Bore Sight (Reference)', ha='center', va='bottom', 
            fontsize=12, color='red', fontweight='bold')

    # 3. Draw Axes and Labels
    # Horizontal Axis (Azimuthal Component)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.text(radius + 0.5, 0, 'Azimuthal Component →', ha='left', va='center', fontsize=12)
    ax.text(-radius - 0.5, 0, '← Azimuthal Component', ha='right', va='center', fontsize=12)

    # Vertical Axis (Elevation Component)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.text(0, radius + 0.5, 'Elevation Component ↑', ha='center', va='bottom', fontsize=12)
    ax.text(0, -radius - 0.5, 'Elevation Component ↓', ha='center', va='top', fontsize=12)
    
    # 4. Add Sample Star Features (Multiple stars to show pattern recognition)
    # Adjusted positions to avoid text overlap
    star_positions = [
        (radius * 0.7, radius * 0.4),   # Primary star
        (radius * -0.5, radius * 0.6),  # Secondary star
        (radius * 0.3, radius * -0.5),  # Tertiary star
        (radius * -0.2, radius * -0.7), # Quaternary star (moved down)
    ]
    
    star_labels = ['Primary Star', 'Secondary Star', 'Tertiary Star', 'Quaternary Star']
    label_offsets = [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, -0.4)]  # Quaternary label moved down
    
    for i, ((x, y), label, offset) in enumerate(zip(star_positions, star_labels, label_offsets)):
        ax.plot(x, y, '*', color='gold', markersize=15, markeredgewidth=1, 
                markeredgecolor='black', label=f'Detected Star {i+1}' if i == 0 else "")
        ax.text(x + offset[0], y + offset[1], label, ha='left', va='bottom', fontsize=10)
        
        # Add coordinate lines
        ax.plot([x, x], [0, y], ':', color='green', linewidth=1, alpha=0.7)
        ax.plot([0, x], [y, y], ':', color='green', linewidth=1, alpha=0.7)

    # 5. Add sensor mounting illustration (moved further down to avoid text overlap)
    sensor_box = FancyBboxPatch((-radius*0.1, -radius*0.3), radius*0.2, radius*0.2,
                               boxstyle="round,pad=0.02", linewidth=2,
                               edgecolor='#2c3e50', facecolor='#ecf0f1', alpha=0.8)
    ax.add_patch(sensor_box)
    # Moved sensor text further down
    ax.text(0, -radius*0.5, 'Sensor\nAssembly', ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 6. Final Touches
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_xlabel('Angular Position (Degrees)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Angular Position (Degrees)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4: Celestial Navigation Sensor Layout (Angular Space)\nMilitary Truck Roof-Mounted Configuration', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set ticks
    tick_interval = 5.0
    major_ticks = np.arange(-fov_size/2, fov_size/2 + tick_interval, tick_interval)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add technical specifications
    specs_text = (
        f"Technical Specifications:\n"
        f"• FOV Diameter: {fov_size}°\n"
        f"• Angular Resolution: <15 arcsec\n"
        f"• Mounting: Truck roof\n"
        f"• Operation: GPS-denied environments"
    )
    ax.text(0.02, 0.98, specs_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Create the filename
    filepath = os.path.join(output_dir, "fig4_sensor_layout.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def generate_all_figures():
    """Generate all 4 figures for the Military Review paper"""
    print("Starting visualization generation for Military Review paper...")
    
    # Run simulation to get data
    results = run_simulation_for_visualization()
    if results is None:
        print("ERROR: Could not run simulation. Please check celestial_nav_simulation.py")
        return
    
    scenario_stats = results['scenario_stats']
    
    # Get FOV size for sensor layout
    simulation_file = "celestial_nav_simulation.py"
    sim_module = import_simulation_module(simulation_file)
    if sim_module and sim_module.OPERATIONAL_SCENARIOS:
        fov_size = sim_module.OPERATIONAL_SCENARIOS[0].fov_size
    else:
        fov_size = 20.0  # Default
    
    print(f"\nGenerating 4 figures with FOV size: {fov_size}°")
    print("="*70)
    
    # Generate all figures
    figures = []
    
    try:
        print("1. Generating Figure 1: Success Rates...")
        fig1_path = generate_success_rates_chart(scenario_stats)
        figures.append(("Figure 1: Success Rates", fig1_path))
        
        print("2. Generating Figure 2: Algorithm Performance...")
        fig2_path = generate_algorithm_performance_chart(scenario_stats)
        figures.append(("Figure 2: Algorithm Performance", fig2_path))
        
        print("3. Generating Figure 3: Environmental Analysis...")
        fig3_path = generate_environmental_analysis_chart(scenario_stats)
        figures.append(("Figure 3: Environmental Analysis", fig3_path))
        
        print("4. Generating Figure 4: Sensor Layout...")
        fig4_path = generate_sensor_layout(fov_size)
        figures.append(("Figure 4: Sensor Layout", fig4_path))
        
        # Print summary
        print("\n" + "="*70)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*70)
        print("Generated figures for Military Review paper:")
        for name, path in figures:
            print(f"  {name:30} -> {path}")
            
        print(f"\nTotal trials analyzed: {results['total_trials']:,}")
        print(f"Overall success rate: {results['overall_success_rate']:.1f}%")
        
    except Exception as e:
        print(f"\nERROR: Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()
        return

# --- Main Execution ---

if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: Required packages not available.")
        print("Please install: pip install matplotlib numpy")
        sys.exit(1)
    
    generate_all_figures()