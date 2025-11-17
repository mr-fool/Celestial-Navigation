"""
Visualization Script for Celestial Navigation Monte Carlo Results
Generates publication-ready figures for Military Review paper
ASCII characters only for cross-platform compatibility
"""

import sys
import os
from datetime import datetime
import importlib.util

# Import the simulation module dynamically
def import_simulation_module(filepath):
    """Import the celestial navigation simulation module"""
    spec = importlib.util.spec_from_file_location("celestial_nav", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not available. Install with: pip install matplotlib")
    print("Continuing with ASCII-based visualizations only...")

import numpy as np


class CelestialNavVisualizer:
    """Generate visualizations for celestial navigation simulation results"""
    
    def __init__(self, results, output_dir='figures'):
        """
        Initialize visualizer with simulation results
        
        Args:
            results: Dictionary containing Monte Carlo simulation results
            output_dir: Directory to save figures
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for this visualization session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_all_figures(self):
        """Generate all figures for the paper"""
        print("\n" + "="*70)
        print("GENERATING FIGURES FOR MILITARY REVIEW PAPER")
        print("="*70)
        
        figures_generated = []
        
        # ASCII visualizations (always available)
        print("\nGenerating ASCII visualizations...")
        ascii_file = self.generate_ascii_summary()
        figures_generated.append(ascii_file)
        
        # Matplotlib visualizations (if available)
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating matplotlib figures...")
            
            # Figure 1: Success Rate by Scenario (Bar Chart)
            fig1 = self.plot_success_rates_by_scenario()
            figures_generated.append(fig1)
            
            # Figure 2: Algorithm Distribution (Stacked Bar Chart)
            fig2 = self.plot_algorithm_distribution()
            figures_generated.append(fig2)
            
            # Figure 3: Computation Time Analysis (Box Plot)
            fig3 = self.plot_computation_time_analysis()
            figures_generated.append(fig3)
            
            # Figure 4: Progressive Strategy Flow (Sankey-style)
            fig4 = self.plot_progressive_strategy_flow()
            figures_generated.append(fig4)
            
            # Figure 5: Environmental Resilience Heatmap
            fig5 = self.plot_environmental_resilience()
            figures_generated.append(fig5)
            
            # Figure 6: Algorithm Effectiveness Matrix
            fig6 = self.plot_algorithm_effectiveness_matrix()
            figures_generated.append(fig6)
            
        else:
            print("\nSkipping matplotlib figures (library not available)")
        
        print("\n" + "="*70)
        print("FIGURE GENERATION COMPLETE")
        print("="*70)
        print(f"\nFigures saved to: {self.output_dir}/")
        print("\nGenerated files:")
        for fig in figures_generated:
            print(f"  - {fig}")
        
        return figures_generated
    
    def generate_ascii_summary(self):
        """Generate ASCII-based visualization summary"""
        filename = f"{self.output_dir}/ascii_summary_{self.timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CELESTIAL NAVIGATION SIMULATION - ASCII VISUALIZATION\n")
            f.write("="*70 + "\n\n")
            
            # Add algorithm explanations section
            f.write("ALGORITHM EXPLANATIONS (Simple Examples)\n")
            f.write("-"*70 + "\n\n")
            
            f.write("LIEBE'S TRIANGLE ALGORITHM (Fast - Primary Method):\n")
            f.write("  Uses: 3 stars form a triangle pattern\n")
            f.write("  Example: Camera sees stars A, B, C\n")
            f.write("    Distance A-B = 15.3 deg, B-C = 22.7 deg, A-C = 31.2 deg\n")
            f.write("    Pattern [15.3, 22.7, 31.2] is looked up in catalog index\n")
            f.write("    If exact match found -> Identification success\n")
            f.write("  Best for: Clean conditions, low noise\n\n")
            
            f.write("GEOMETRIC VOTING ALGORITHM (Moderate - Fallback 1):\n")
            f.write("  Uses: All star pairs 'vote' for catalog stars\n")
            f.write("  Example: Same 3 stars (A, B, C)\n")
            f.write("    Distance 15.3 deg matches Star#42 & Star#67 -> 1 vote each\n")
            f.write("    Distance 22.7 deg matches Star#67 & Star#89 -> 1 vote each\n")
            f.write("    Distance 31.2 deg matches Star#42 & Star#89 -> 1 vote each\n")
            f.write("    Vote Tally: Star#42=2, Star#67=2, Star#89=2\n")
            f.write("    High vote count = Correct identification\n")
            f.write("  Best for: Noisy conditions, partial sky obstruction\n\n")
            
            f.write("PYRAMID ALGORITHM (Most Robust - Fallback 2):\n")
            f.write("  Uses: 4 stars with 6 distance measurements\n")
            f.write("  Example: Stars A, B, C, D create 6 distances\n")
            f.write("    Uses 4 shortest distances as unique signature\n")
            f.write("    High redundancy makes it very noise-tolerant\n")
            f.write("  Best for: Lost-in-space scenarios (requires 4+ stars)\n\n")
            
            f.write("="*70 + "\n\n")
            
            # Success Rate Bar Chart (ASCII)
            f.write("FIGURE 1: Success Rate by Scenario\n")
            f.write("-"*70 + "\n\n")
            
            scenario_stats = self.results['scenario_stats']
            sorted_scenarios = sorted(scenario_stats.items(), 
                                     key=lambda x: x[1]['success_rate'], 
                                     reverse=True)
            
            max_name_len = max(len(name) for name, _ in sorted_scenarios)
            
            for name, stats in sorted_scenarios:
                success_rate = stats['success_rate']
                bar_length = int(success_rate / 2)  # Scale to fit 50 chars max
                bar = '#' * bar_length
                f.write(f"{name:<{max_name_len}} | {bar} {success_rate:5.1f}%\n")
            
            f.write("\n" + "0%" + " "*43 + "50%" + " "*43 + "100%\n\n")
            
            # Algorithm Success Rate Table
            f.write("\nFIGURE 2: Algorithm Success Rates by Scenario\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"{'Scenario':<20} {'Liebe':<15} {'Voting':<15} {'Pyramid':<15} {'Failed':<15}\n")
            f.write(f"{'':20} {'(% success)':<15} {'(% success)':<15} {'(% success)':<15} {'(% failed)':<15}\n")
            f.write("-"*70 + "\n")
            
            for name, stats in scenario_stats.items():
                dist = stats['algorithm_distribution']
                total = stats['total_trials']
                
                liebe_pct = (dist.get('Liebe', 0) / total) * 100
                voting_pct = (dist.get('Voting', 0) / total) * 100
                pyramid_pct = (dist.get('Pyramid', 0) / total) * 100
                failed_pct = (dist.get('Failed', 0) / total) * 100
                
                f.write(f"{name:<20} {liebe_pct:6.1f}%         {voting_pct:6.1f}%         "
                       f"{pyramid_pct:6.1f}%         {failed_pct:6.1f}%\n")
            
            # Overall Statistics
            f.write("\n\nFIGURE 3: Overall Algorithm Performance\n")
            f.write("-"*70 + "\n\n")
            
            algo_totals = self.results['algorithm_totals']
            total_trials = self.results['total_trials']
            
            for algo, count in sorted(algo_totals.items(), 
                                     key=lambda x: x[1], reverse=True):
                percentage = (count / total_trials) * 100
                bar_length = int(percentage / 2)
                bar = '*' * bar_length
                f.write(f"{algo:<12} | {bar} {percentage:5.1f}% ({count} trials)\n")
            
            # Computation Time Summary
            f.write("\n\nFIGURE 4: Computation Time Summary\n")
            f.write("-"*70 + "\n\n")
            
            f.write(f"{'Scenario':<20} {'Avg Time (ms)':<15} {'Std Dev (ms)':<15}\n")
            f.write("-"*70 + "\n")
            
            for name, stats in scenario_stats.items():
                avg_time = stats['avg_computation_time'] * 1000
                std_time = stats['std_computation_time'] * 1000
                f.write(f"{name:<20} {avg_time:>12.3f}    {std_time:>12.3f}\n")
            
            # Progressive Strategy Effectiveness
            f.write("\n\nFIGURE 5: Progressive Strategy Effectiveness\n")
            f.write("-"*70 + "\n\n")
            
            liebe_count = algo_totals.get('Liebe', 0)
            voting_count = algo_totals.get('Voting', 0)
            pyramid_count = algo_totals.get('Pyramid', 0)
            
            total_success = liebe_count + voting_count + pyramid_count
            
            if total_success > 0:
                f.write(f"Total Successful Identifications: {total_success}\n\n")
                f.write(f"Primary (Liebe):    {liebe_count:5d} ({liebe_count/total_success*100:5.1f}% of successes)\n")
                f.write(f"Fallback 1 (Voting): {voting_count:5d} ({voting_count/total_success*100:5.1f}% of successes)\n")
                f.write(f"Fallback 2 (Pyramid): {pyramid_count:5d} ({pyramid_count/total_success*100:5.1f}% of successes)\n\n")
                
                fallback_value = voting_count + pyramid_count
                f.write(f"Fallback algorithms recovered {fallback_value} cases ({fallback_value/total_success*100:.1f}%)\n")
                f.write(f"that Liebe alone could not solve.\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF ASCII VISUALIZATION\n")
            f.write("="*70 + "\n")
        
        print(f"  ASCII summary saved: {filename}")
        return filename
    
    def plot_success_rates_by_scenario(self):
        """Figure 1: Bar chart of success rates by scenario"""
        filename = f"{self.output_dir}/fig1_success_rates_{self.timestamp}.png"
        
        scenario_stats = self.results['scenario_stats']
        sorted_scenarios = sorted(scenario_stats.items(), 
                                 key=lambda x: x[1]['success_rate'], 
                                 reverse=True)
        
        names = [name for name, _ in sorted_scenarios]
        success_rates = [stats['success_rate'] for _, stats in sorted_scenarios]
        
        # Color code by performance
        colors = []
        for rate in success_rates:
            if rate >= 80:
                colors.append('#2ecc71')  # Green - Excellent
            elif rate >= 50:
                colors.append('#f39c12')  # Orange - Moderate
            elif rate >= 20:
                colors.append('#e74c3c')  # Red - Poor
            else:
                colors.append('#95a5a6')  # Gray - Failed
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, success_rates, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Operational Scenario', fontsize=12, fontweight='bold')
        ax.set_title('Celestial Navigation Success Rate by Environment\n' + 
                     f'({self.results["num_trials_per_scenario"]} trials per scenario)',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            ax.text(rate + 2, i, f'{rate:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add legend for color coding
        legend_elements = [
            mpatches.Patch(color='#2ecc71', label='Excellent (>=80%)'),
            mpatches.Patch(color='#f39c12', label='Moderate (50-79%)'),
            mpatches.Patch(color='#e74c3c', label='Poor (20-49%)'),
            mpatches.Patch(color='#95a5a6', label='Failed (<20%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 1 saved: {filename}")
        return filename
    
    def plot_algorithm_distribution(self):
        """Figure 2: Stacked bar chart showing algorithm distribution per scenario"""
        filename = f"{self.output_dir}/fig2_algorithm_distribution_{self.timestamp}.png"
        
        scenario_stats = self.results['scenario_stats']
        scenarios = list(scenario_stats.keys())
        
        # Extract data
        liebe_pcts = []
        voting_pcts = []
        pyramid_pcts = []
        failed_pcts = []
        
        for scenario in scenarios:
            dist = scenario_stats[scenario]['algorithm_distribution']
            total = scenario_stats[scenario]['total_trials']
            
            liebe_pcts.append((dist.get('Liebe', 0) / total) * 100)
            voting_pcts.append((dist.get('Voting', 0) / total) * 100)
            pyramid_pcts.append((dist.get('Pyramid', 0) / total) * 100)
            failed_pcts.append((dist.get('Failed', 0) / total) * 100)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(scenarios))
        width = 0.6
        
        p1 = ax.bar(x, liebe_pcts, width, label='Liebe (Fast)', color='#3498db')
        p2 = ax.bar(x, voting_pcts, width, bottom=liebe_pcts, 
                    label='Voting (Moderate)', color='#2ecc71')
        p3 = ax.bar(x, pyramid_pcts, width, 
                    bottom=np.array(liebe_pcts) + np.array(voting_pcts),
                    label='Pyramid (Robust)', color='#9b59b6')
        p4 = ax.bar(x, failed_pcts, width,
                    bottom=np.array(liebe_pcts) + np.array(voting_pcts) + np.array(pyramid_pcts),
                    label='Failed', color='#e74c3c')
        
        ax.set_ylabel('Percentage of Trials (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Operational Scenario', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Success Rates by Scenario\n' +
                     'Progressive Fallback Strategy Performance',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 2 saved: {filename}")
        return filename
    
    def plot_computation_time_analysis(self):
        """Figure 3: Computation time comparison"""
        filename = f"{self.output_dir}/fig3_computation_time_{self.timestamp}.png"
        
        scenario_stats = self.results['scenario_stats']
        scenarios = list(scenario_stats.keys())
        
        avg_times = [stats['avg_computation_time'] * 1000 for stats in scenario_stats.values()]
        std_times = [stats['std_computation_time'] * 1000 for stats in scenario_stats.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(scenarios))
        bars = ax.bar(x, avg_times, yerr=std_times, capsize=5, 
                      color='#3498db', edgecolor='black', linewidth=1.2,
                      error_kw={'linewidth': 2, 'ecolor': '#e74c3c'})
        
        ax.set_ylabel('Computation Time (milliseconds)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Operational Scenario', fontsize=12, fontweight='bold')
        ax.set_title('Average Computation Time per Scenario\n' +
                     '(Error bars show standard deviation)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (bar, avg, std) in enumerate(zip(bars, avg_times, std_times)):
            ax.text(i, avg + std + 0.0005, f'{avg:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 3 saved: {filename}")
        return filename
    
    def plot_progressive_strategy_flow(self):
        """Figure 4: Visualization of progressive strategy flow"""
        filename = f"{self.output_dir}/fig4_progressive_flow_{self.timestamp}.png"
        
        algo_totals = self.results['algorithm_totals']
        total_trials = self.results['total_trials']
        
        liebe_count = algo_totals.get('Liebe', 0)
        voting_count = algo_totals.get('Voting', 0)
        pyramid_count = algo_totals.get('Pyramid', 0)
        failed_count = algo_totals.get('Failed', 0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define boxes for each stage
        stages = [
            {'name': 'Total Trials', 'count': total_trials, 'y': 4, 'color': '#95a5a6'},
            {'name': 'Liebe Algorithm\n(Fast)', 'count': liebe_count, 'y': 3, 'color': '#3498db'},
            {'name': 'Geometric Voting\n(Fallback 1)', 'count': voting_count, 'y': 2, 'color': '#2ecc71'},
            {'name': 'Pyramid Algorithm\n(Fallback 2)', 'count': pyramid_count, 'y': 1, 'color': '#9b59b6'},
            {'name': 'Failed', 'count': failed_count, 'y': 0, 'color': '#e74c3c'}
        ]
        
        # Draw boxes
        for stage in stages:
            width = (stage['count'] / total_trials) * 8
            x_center = 0
            
            rect = mpatches.Rectangle((x_center - width/2, stage['y'] - 0.3), 
                                     width, 0.6, 
                                     linewidth=2, 
                                     edgecolor='black',
                                     facecolor=stage['color'],
                                     alpha=0.7)
            ax.add_patch(rect)
            
            # Add text
            percentage = (stage['count'] / total_trials) * 100
            ax.text(x_center, stage['y'], 
                   f"{stage['name']}\n{stage['count']} ({percentage:.1f}%)",
                   ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # Total to Liebe
        ax.annotate('', xy=(0, 3.3), xytext=(0, 3.7), arrowprops=arrow_props)
        
        # Liebe to Voting (failures)
        if voting_count > 0:
            ax.annotate('', xy=(0, 2.3), xytext=(0, 2.7), arrowprops=arrow_props)
            ax.text(0.5, 2.5, 'Liebe\nFailed', ha='left', va='center', fontsize=9, style='italic')
        
        # Voting to Pyramid (failures)
        if pyramid_count > 0:
            ax.annotate('', xy=(0, 1.3), xytext=(0, 1.7), arrowprops=arrow_props)
            ax.text(0.5, 1.5, 'Voting\nFailed', ha='left', va='center', fontsize=9, style='italic')
        
        # To Failed
        if failed_count > 0:
            ax.annotate('', xy=(0, 0.3), xytext=(0, 0.7), arrowprops=arrow_props)
            ax.text(0.5, 0.5, 'All\nFailed', ha='left', va='center', fontsize=9, style='italic')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.5, 4.5)
        ax.axis('off')
        
        ax.set_title('Progressive Algorithm Strategy Flow\n' +
                     f'Total Trials: {total_trials}',
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 4 saved: {filename}")
        return filename
    
    def plot_environmental_resilience(self):
        """Figure 5: Environmental factors vs success rate"""
        filename = f"{self.output_dir}/fig5_environmental_resilience_{self.timestamp}.png"
        
        # This would require access to the OPERATIONAL_SCENARIOS
        # For now, create a simplified version showing success vs noise
        
        scenario_stats = self.results['scenario_stats']
        
        # Extract data (using scenario names as proxy for conditions)
        scenarios = []
        success_rates = []
        
        for name, stats in scenario_stats.items():
            scenarios.append(name)
            success_rates.append(stats['success_rate'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if rate >= 80 else '#f39c12' if rate >= 50 
                 else '#e74c3c' if rate >= 20 else '#95a5a6' 
                 for rate in success_rates]
        
        ax.scatter(range(len(scenarios)), success_rates, 
                  s=500, c=colors, edgecolors='black', linewidth=2, alpha=0.7)
        
        # Add scenario labels
        for i, (scenario, rate) in enumerate(zip(scenarios, success_rates)):
            ax.text(i, rate + 5, scenario, ha='center', va='bottom', 
                   fontsize=9, rotation=45)
        
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Operational Scenarios (ordered as defined)', fontsize=12, fontweight='bold')
        ax.set_title('Environmental Impact on Navigation Success\n' +
                     'Scenario Performance Comparison',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks([])
        
        # Add horizontal reference lines
        ax.axhline(y=80, color='#2ecc71', linestyle='--', alpha=0.5, label='Excellent (80%)')
        ax.axhline(y=50, color='#f39c12', linestyle='--', alpha=0.5, label='Moderate (50%)')
        ax.axhline(y=20, color='#e74c3c', linestyle='--', alpha=0.5, label='Poor (20%)')
        ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 5 saved: {filename}")
        return filename
    
    def plot_algorithm_effectiveness_matrix(self):
        """Figure 6: Heat map showing which algorithm works best in each scenario"""
        filename = f"{self.output_dir}/fig6_algorithm_effectiveness_{self.timestamp}.png"
        
        scenario_stats = self.results['scenario_stats']
        scenarios = list(scenario_stats.keys())
        algorithms = ['Liebe', 'Voting', 'Pyramid']
        
        # Create matrix
        matrix = []
        for scenario in scenarios:
            row = []
            dist = scenario_stats[scenario]['algorithm_distribution']
            total = scenario_stats[scenario]['total_trials']
            
            for algo in algorithms:
                percentage = (dist.get(algo, 0) / total) * 100
                row.append(percentage)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks
        ax.set_xticks(np.arange(len(algorithms)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels(algorithms, fontsize=11, fontweight='bold')
        ax.set_yticklabels(scenarios, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values in cells
        for i in range(len(scenarios)):
            for j in range(len(algorithms)):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", 
                             fontsize=10, fontweight='bold')
        
        ax.set_title('Algorithm Effectiveness Matrix\n' +
                     '(Percentage of trials solved by each algorithm)',
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Percentage (%)', rotation=270, labelpad=20, 
                      fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Figure 6 saved: {filename}")
        return filename


def main():
    """Main execution function"""
    print("="*70)
    print("CELESTIAL NAVIGATION VISUALIZATION GENERATOR")
    print("="*70)
    
    # Check for simulation module
    sim_file = "celestial_nav_simulation.py"
    if not os.path.exists(sim_file):
        print(f"\nERROR: Simulation file '{sim_file}' not found!")
        print("Please ensure celestial_nav_simulation.py is in the same directory.")
        return 1
    
    print(f"\nImporting simulation module: {sim_file}")
    
    try:
        sim_module = import_simulation_module(sim_file)
    except Exception as e:
        print(f"\nERROR: Failed to import simulation module: {e}")
        return 1
    
    # Run simulation
    print("\nRunning Monte Carlo simulation...")
    print("(This may take a few minutes with 1000 trials per scenario)")
    
    try:
        # Build catalog index
        sim_module.CATALOG_INDEX = sim_module.build_angular_distance_index(
            sim_module.MOCK_STAR_CATALOG
        )
        
        # Run simulation with configurable trials
        num_trials = 1000  # Can be changed via command line argument
        if len(sys.argv) > 1:
            num_trials = int(sys.argv[1])
        
        results = sim_module.run_monte_carlo_simulation(
            num_trials=num_trials, 
            verbose=False
        )
        
    except Exception as e:
        print(f"\nERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nSimulation complete!")
    print(f"Total trials: {results['total_trials']}")
    print(f"Overall success rate: {results['overall_success_rate']:.1f}%")
    
    # Generate visualizations
    visualizer = CelestialNavVisualizer(results)
    figures = visualizer.generate_all_figures()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated {len(figures)} figure(s)")
    print("\nReady for Military Review paper submission!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
