"""
Visualization for JAMS Paper:
  Fig 1: Monte Carlo Success Rates by Operational Scenario
  Fig 2: Algorithm Performance and Fallback Strategy
  Fig 3: Environmental Degradation Impact Analysis
  Fig 4: Celestial Navigation Sensor Layout Diagram
  Fig 5: GPS-Denied Positional Error Degradation Curve  [NEW — JAMS key figure]
  Fig 6: C2 Reintegration Latency (Time-to-First-Fix)
"""

import os
import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-whitegrid')

FIGURES_DIR = 'figures'

# Consistent color palette tied to adversary framing
SCENARIO_COLORS = {
    'Baseline_Uncontested':    '#2ecc71',  # Green  — no threat
    'EW_Degraded_Rural':       '#3498db',  # Blue   — low threat
    'Urban_C2_Denied':         '#f39c12',  # Orange — moderate (China)
    'SOCOM_Denied_Territory':  '#e74c3c',  # Red    — denied territory
    'Russian_EW_Saturation':   '#9b59b6',  # Purple — Russia EW
    'Chinese_ISR_Contested':   '#34495e',  # Dark   — China ISR
}

THREAT_LABELS = {
    'Baseline_Uncontested':    'Uncontested',
    'EW_Degraded_Rural':       'EW Degraded\n(Russia/China)',
    'Urban_C2_Denied':         'Urban C2\nDenied (China)',
    'SOCOM_Denied_Territory':  'SOCOM\nDenied Territory',
    'Russian_EW_Saturation':   'Russian EW\nSaturation',
    'Chinese_ISR_Contested':   'Chinese ISR\nContested',
}

# ---------------------------------------------------------------------------
# Helper: import simulation module
# ---------------------------------------------------------------------------
def import_simulation_module(filepath="celestial_nav_simulation.py"):
    spec = importlib.util.spec_from_file_location("celestial_nav", filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing simulation module: {e}")
        return None

def run_all_simulations():
    sim = import_simulation_module()
    if sim is None:
        return None

    print("Running Monte Carlo simulation...")
    mc_results = sim.run_monte_carlo_simulation(num_trials=1000, verbose=False)

    print("\nRunning GPS degradation simulations...")
    degradation_results = {}
    for scenario in sim.OPERATIONAL_SCENARIOS:
        print(f"  {scenario.name}")
        degradation_results[scenario.name] = sim.run_gps_denied_degradation_simulation(
            duration_minutes=60,
            fix_interval_minutes=5,
            num_monte_carlo=200,
            scenario=scenario
        )

    print("\nRunning C2 latency simulation...")
    c2_latency_results = sim.run_jado_c2_latency_simulation(num_trials_per_scenario=500)

    return {
        'mc': mc_results,
        'degradation': degradation_results,
        'c2_latency': c2_latency_results,
        'scenarios': sim.OPERATIONAL_SCENARIOS,
    }


# ---------------------------------------------------------------------------
# Figure 1: Success Rates
# ---------------------------------------------------------------------------
def generate_success_rates_chart(scenario_stats, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    sorted_scenarios = sorted(scenario_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    labels = [THREAT_LABELS.get(n, n) for n, _ in sorted_scenarios]
    rates  = [s['success_rate'] for _, s in sorted_scenarios]
    colors = [SCENARIO_COLORS.get(n, '#95a5a6') for n, _ in sorted_scenarios]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Identification Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Operational Scenario (Adversary Threat Framing)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 1: Star Identification Success Rates Across Adversary Threat Scenarios\n'
                 '(1,000 Monte Carlo Trials per Scenario, Progressive Liebe → Voting → Pyramid Strategy)',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_ylim(0, 108)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(5.5, 71.5, 'Operational viability threshold (70%)', color='red', fontsize=9, ha='right')
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig1_success_rates.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Figure 2: Algorithm Performance (stacked bar)
# ---------------------------------------------------------------------------
def generate_algorithm_performance_chart(scenario_stats, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    sorted_scenarios = sorted(scenario_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    labels = [THREAT_LABELS.get(n, n) for n, _ in sorted_scenarios]
    liebe_d, voting_d, pyramid_d, failed_d = [], [], [], []
    for name, stats in sorted_scenarios:
        total = stats['total_trials']
        liebe_d.append(stats['algorithm_distribution'].get('Liebe', 0) / total * 100)
        voting_d.append(stats['algorithm_distribution'].get('Voting', 0) / total * 100)
        pyramid_d.append(stats['algorithm_distribution'].get('Pyramid', 0) / total * 100)
        failed_d.append(stats['algorithm_distribution'].get('Failed', 0) / total * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 8))
    w = 0.65
    b1 = ax.bar(x, liebe_d, w, label='Liebe (Fastest)', color='#2ecc71', alpha=0.85, edgecolor='black')
    b2 = ax.bar(x, voting_d, w, bottom=liebe_d, label='Geometric Voting (Robust)', color='#3498db', alpha=0.85, edgecolor='black')
    b3 = ax.bar(x, pyramid_d, w, bottom=[a+b for a,b in zip(liebe_d, voting_d)],
                label='Pyramid (LIS, Most Robust)', color='#9b59b6', alpha=0.85, edgecolor='black')
    b4 = ax.bar(x, failed_d, w, bottom=[a+b+c for a,b,c in zip(liebe_d, voting_d, pyramid_d)],
                label='Failed (Operationally Lost)', color='#e74c3c', alpha=0.85, edgecolor='black')

    for i, (l, v, p, f) in enumerate(zip(liebe_d, voting_d, pyramid_d, failed_d)):
        total_success = l + v + p
        if total_success > 0:
            ax.text(i, total_success + 1.5, f'{total_success:.0f}%',
                    ha='center', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Percentage of Trials (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Operational Scenario (Adversary Threat)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 2: Algorithm Contribution and Progressive Fallback Strategy\n'
                 '(Liebe → Voting → Pyramid; stacked = cumulative success)',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_ylim(0, 112)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=11, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.text(0.01, 0.97,
            "Progressive strategy: Liebe (fast) → Voting (noise-robust) → Pyramid (lost-in-space)\n"
            "Fallback algorithms recover identifications that would otherwise be lost.",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fp = os.path.join(output_dir, "fig2_algorithm_performance.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Figure 3: Environmental / Noise Analysis
# ---------------------------------------------------------------------------
def generate_environmental_analysis_chart(scenario_stats, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    scenario_params = {
        'Baseline_Uncontested':   {'noise': 0.8, 'stars': 10},
        'EW_Degraded_Rural':      {'noise': 1.0, 'stars': 7},
        'Urban_C2_Denied':        {'noise': 1.5, 'stars': 4},
        'SOCOM_Denied_Territory': {'noise': 1.2, 'stars': 3},
        'Russian_EW_Saturation':  {'noise': 2.5, 'stars': 5},
        'Chinese_ISR_Contested':  {'noise': 3.5, 'stars': 6},
    }
    names  = list(scenario_stats.keys())
    noise  = [scenario_params[n]['noise'] for n in names]
    stars  = [scenario_params[n]['stars'] for n in names]
    rates  = [scenario_stats[n]['success_rate'] for n in names]
    colors = [SCENARIO_COLORS.get(n, '#95a5a6') for n in names]
    times  = [max(scenario_stats[n]['avg_computation_time'] * 1000, 0.01) for n in names]
    labels = [THREAT_LABELS.get(n, n) for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 3: Environmental & Threat-Condition Impact Analysis\n'
                 'Military Vehicle Celestial Navigation Performance',
                 fontsize=15, fontweight='bold', y=0.99)

    # 3a: Noise vs success (bubble = comp time)
    ax = axes[0, 0]
    bubble = [max(t * 6000, 80) for t in times]
    ax.scatter(noise, rates, s=bubble, c=colors, alpha=0.75, edgecolors='black', zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (noise[i], rates[i]), xytext=(8, 8), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.set_xlabel('EW/Noise Level (×baseline 15 arcsec)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Identification Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Noise Severity vs. Success Rate\n(bubble = computation time)', fontsize=12, fontweight='bold')
    ax.set_xlim(0.5, 4.0); ax.set_ylim(0, 105)
    ax.axhline(70, color='red', linestyle='--', alpha=0.4); ax.grid(True, alpha=0.3)

    # 3b: Star count vs success
    ax = axes[0, 1]
    ax.scatter(stars, rates, s=120, c=colors, alpha=0.75, edgecolors='black', zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (stars[i], rates[i]), xytext=(8, 8), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.set_xlabel('Stars Visible in FOV', fontsize=11, fontweight='bold')
    ax.set_ylabel('Identification Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Star Availability vs. Success Rate', fontsize=12, fontweight='bold')
    ax.set_xlim(2, 11); ax.set_ylim(0, 105)
    ax.axvline(3, color='orange', linestyle='--', alpha=0.5)
    ax.text(3.1, 5, 'Min for Pyramid', color='orange', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3c: Computation time bar
    ax = axes[1, 0]
    short_labels = [n.replace('_', '\n') for n in names]
    bars = ax.bar(short_labels, times, color=colors, alpha=0.8, edgecolor='black')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{t:.3f}ms', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Mean Computation Time (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Algorithm Processing Time by Scenario\n(all < 1 ms → real-time capable)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 3d: Attitude error bar
    ax = axes[1, 1]
    errors = [scenario_stats[n]['mean_attitude_error'] for n in names]
    ci_lo  = [scenario_stats[n]['ci_95_lower'] for n in names]
    ci_hi  = [scenario_stats[n]['ci_95_upper'] for n in names]
    err_low  = [e - l for e, l in zip(errors, ci_lo)]
    err_high = [h - e for e, h in zip(errors, ci_hi)]
    bars = ax.bar(short_labels, errors, color=colors, alpha=0.8, edgecolor='black',
                  yerr=[err_low, err_high], capsize=5, error_kw={'linewidth': 1.5})
    ax.set_ylabel('Mean Attitude RMS Error (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Attitude Estimation Error by Scenario\n(error bars = 95% CI)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(output_dir, "fig3_environmental_analysis.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Figure 4: Sensor Layout
# ---------------------------------------------------------------------------
def generate_sensor_layout(fov_size=20.0, output_dir=FIGURES_DIR):
    os.makedirs(output_dir, exist_ok=True)
    radius = fov_size / 2.0
    fig, ax = plt.subplots(figsize=(10, 10))
    fov_circle = patches.Circle((0, 0), radius, edgecolor='#1f77b4', facecolor='lightblue',
                                 linewidth=3, alpha=0.3, label=f'Sensor FOV ({fov_size}° Diameter)')
    ax.add_patch(fov_circle)
    ax.plot(0, 0, 'o', color='red', markersize=12, label='Vehicle Bore Sight')
    ax.text(0, 0.5, 'Bore Sight', ha='center', va='bottom', fontsize=11, color='red', fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.text(radius + 0.5, 0, 'Azimuth →', ha='left', va='center', fontsize=11)
    ax.text(0, radius + 0.5, 'Elevation ↑', ha='center', va='bottom', fontsize=11)
    star_positions = [(radius*0.7, radius*0.4), (radius*-0.5, radius*0.6),
                      (radius*0.3, radius*-0.5), (radius*-0.2, radius*-0.7)]
    for i, (x, y) in enumerate(star_positions):
        ax.plot(x, y, '*', color='gold', markersize=16, markeredgewidth=1, markeredgecolor='black')
        ax.text(x + 0.3, y + 0.3, f'Star {i+1}', fontsize=10)
        ax.plot([x, x], [0, y], ':', color='green', linewidth=1, alpha=0.6)
        ax.plot([0, x], [y, y], ':', color='green', linewidth=1, alpha=0.6)
    sensor_box = FancyBboxPatch((-radius*0.1, -radius*0.3), radius*0.2, radius*0.2,
                                boxstyle="round,pad=0.02", linewidth=2,
                                edgecolor='#2c3e50', facecolor='#ecf0f1', alpha=0.8)
    ax.add_patch(sensor_box)
    ax.set_xlim(-radius-1, radius+1); ax.set_ylim(-radius-1, radius+1)
    ax.set_xlabel('Angular Position (Degrees)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Angular Position (Degrees)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 4: Vehicle-Mounted Celestial Navigation Sensor — Angular FOV\n'
                 'Roof-Mounted Configuration for Military Ground Vehicle (GPS-Denied Operations)',
                 fontsize=14, fontweight='bold', pad=16)
    ticks = np.arange(-fov_size/2, fov_size/2 + 5, 5.0)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(0.02, 0.98,
            f"Specifications:\n• FOV: {fov_size}°\n• Resolution: <15 arcsec\n"
            "• Platform: Military truck\n• Mode: GPS-denied navigation",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))
    plt.tight_layout()
    fp = os.path.join(output_dir, "fig4_sensor_layout.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Figure 5: GPS-Denied Positional Error Degradation Curve  [NEW — JAMS key figure]
# ---------------------------------------------------------------------------
def generate_gps_degradation_figure(degradation_results, output_dir=FIGURES_DIR):
    """
    Shows positional error growth over 60 minutes after GPS denial for each
    adversary scenario, comparing Dead Reckoning vs Celestial Nav vs INS+Celestial.
    This is the paper's central empirical argument for why celestial nav matters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Subplot layout: top = comparison of all scenarios for celestial nav;
    #                 bottom = detailed DR vs CN vs INS for worst + best case
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel A: Celestial Nav mean error across all scenarios ──────────────
    ax_a = fig.add_subplot(gs[0, :])
    for name, res in degradation_results.items():
        t = res['time_axis']
        m = res['cn_mean']
        s = res['cn_std']
        color = SCENARIO_COLORS.get(name, '#95a5a6')
        label = THREAT_LABELS.get(name, name).replace('\n', ' ')
        ax_a.plot(t, m, color=color, linewidth=2.0, label=label)
        ax_a.fill_between(t, m - s, m + s, color=color, alpha=0.12)

    ax_a.set_xlabel('Time After GPS Denial (minutes)', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('Position Error (meters, CEP)', fontsize=12, fontweight='bold')
    ax_a.set_title('Panel A — Celestial Nav Position Error After GPS Denial by Adversary Scenario\n'
                   '(shaded band = ±1 std, 200 Monte Carlo trajectories per scenario)',
                   fontsize=12, fontweight='bold')
    ax_a.legend(loc='upper left', ncol=3, fontsize=9, framealpha=0.9)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, 60)

    # Add fix-interval tick marks
    fix_interval = list(degradation_results.values())[0]['fix_interval_minutes']
    for t in range(fix_interval, 61, fix_interval):
        ax_a.axvline(t, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax_a.text(fix_interval + 0.3, ax_a.get_ylim()[1] * 0.94,
              f'↕ fix\nattempt\nevery {fix_interval}min', fontsize=8, color='gray')

    # ── Panel B: DR vs CN vs INS — Best case (Baseline_Uncontested) ─────────
    best_name = 'Baseline_Uncontested'
    if best_name in degradation_results:
        ax_b = fig.add_subplot(gs[1, 0])
        res = degradation_results[best_name]
        t = res['time_axis']
        ax_b.plot(t, res['dr_mean'],  color='#e74c3c', lw=2.0, label='Dead Reckoning only')
        ax_b.fill_between(t, res['dr_mean']-res['dr_std'], res['dr_mean']+res['dr_std'],
                          color='#e74c3c', alpha=0.15)
        ax_b.plot(t, res['cn_mean'],  color='#3498db', lw=2.0, label='Celestial Nav')
        ax_b.fill_between(t, res['cn_mean']-res['cn_std'], res['cn_mean']+res['cn_std'],
                          color='#3498db', alpha=0.15)
        ax_b.plot(t, res['ins_mean'], color='#2ecc71', lw=2.0, linestyle='--',
                  label='INS + Celestial (future)')
        ax_b.set_title(f'Panel B — {THREAT_LABELS[best_name].replace(chr(10)," ")}\n'
                       'DR vs Celestial Nav vs INS Hybrid', fontsize=11, fontweight='bold')
        ax_b.set_xlabel('Time After GPS Denial (minutes)', fontsize=11)
        ax_b.set_ylabel('Position Error (meters)', fontsize=11)
        ax_b.legend(fontsize=9, framealpha=0.9)
        ax_b.grid(True, alpha=0.3); ax_b.set_xlim(0, 60)
        ax_b.text(0.03, 0.97,
                  f"Fix success: {res['fix_success_prob']*100:.0f}%\n"
                  f"Fix residual: ~{res['celestial_fix_residual_m']:.0f}m CEP",
                  transform=ax_b.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # ── Panel C: DR vs CN vs INS — Worst case (Russian_EW_Saturation) ───────
    worst_name = 'Russian_EW_Saturation'
    if worst_name in degradation_results:
        ax_c = fig.add_subplot(gs[1, 1])
        res = degradation_results[worst_name]
        t = res['time_axis']
        ax_c.plot(t, res['dr_mean'],  color='#e74c3c', lw=2.0, label='Dead Reckoning only')
        ax_c.fill_between(t, res['dr_mean']-res['dr_std'], res['dr_mean']+res['dr_std'],
                          color='#e74c3c', alpha=0.15)
        ax_c.plot(t, res['cn_mean'],  color='#3498db', lw=2.0, label='Celestial Nav')
        ax_c.fill_between(t, res['cn_mean']-res['cn_std'], res['cn_mean']+res['cn_std'],
                          color='#3498db', alpha=0.15)
        ax_c.plot(t, res['ins_mean'], color='#9b59b6', lw=2.0, linestyle='--',
                  label='INS + Celestial (future)')
        ax_c.set_title(f'Panel C — {THREAT_LABELS[worst_name].replace(chr(10)," ")}\n'
                       'DR vs Celestial Nav vs INS Hybrid', fontsize=11, fontweight='bold')
        ax_c.set_xlabel('Time After GPS Denial (minutes)', fontsize=11)
        ax_c.set_ylabel('Position Error (meters)', fontsize=11)
        ax_c.legend(fontsize=9, framealpha=0.9)
        ax_c.grid(True, alpha=0.3); ax_c.set_xlim(0, 60)
        ax_c.text(0.03, 0.97,
                  f"Fix success: {res['fix_success_prob']*100:.0f}%\n"
                  f"Fix residual: ~{res['celestial_fix_residual_m']:.0f}m CEP\n"
                  "Note: higher noise → larger fix residual",
                  transform=ax_c.transAxes, fontsize=9, va='top',
                  bbox=dict(boxstyle="round", facecolor='#fde8e8', alpha=0.85))

    fig.suptitle('Figure 2: Positional Error Growth After GPS Denial Event\n'
                 'Comparison of Dead Reckoning, Celestial Navigation, and INS+Celestial Hybrid',
                 fontsize=15, fontweight='bold', y=1.01)
    fp = os.path.join(output_dir, "fig5_gps_degradation_curve.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Figure 6: C2 Reintegration Latency (Time-to-First-Fix)
# ---------------------------------------------------------------------------
def generate_jado_c2_latency_figure(c2_latency_results, output_dir=FIGURES_DIR):
    """
    Shows time-to-first-fix (TTFF) after GPS denial across all adversary scenarios.
    Lower TTFF = faster C2 re-establishment.
    This quantifies C2 reintegration speed after GPS denial.
    """
    os.makedirs(output_dir, exist_ok=True)

    names   = list(c2_latency_results.keys())
    means   = [c2_latency_results[n]['ttff_mean_s'] for n in names]
    p95s    = [c2_latency_results[n]['ttff_p95_s'] for n in names]
    medians = [c2_latency_results[n]['ttff_median_s'] for n in names]
    fix_rates = [c2_latency_results[n]['success_rate'] for n in names]
    colors  = [SCENARIO_COLORS.get(n, '#95a5a6') for n in names]
    labels  = [THREAT_LABELS.get(n, n) for n in names]

    # Sort by mean TTFF ascending
    order = np.argsort(means)
    names_s   = [names[i] for i in order]
    means_s   = [means[i] for i in order]
    p95s_s    = [p95s[i] for i in order]
    medians_s = [medians[i] for i in order]
    fix_s     = [fix_rates[i] for i in order]
    colors_s  = [colors[i] for i in order]
    labels_s  = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Figure 3: C2 Reintegration Latency — Time-to-First-Fix (TTFF) After GPS Denial\n'
                 '(Lower TTFF = faster C2 re-establishment; 500 trials per scenario)',
                 fontsize=14, fontweight='bold', y=1.02)

    # ── Bar chart: TTFF mean and p95 by scenario ─────────────────────────────
    x = np.arange(len(labels_s))
    w = 0.35
    bars_mean   = ax.bar(x - w/2, means_s,   w, color=colors_s, alpha=0.85, edgecolor='black', label='Mean TTFF')
    bars_p95    = ax.bar(x + w/2, p95s_s,    w, color=colors_s, alpha=0.45, edgecolor='black',
                         hatch='//', label='95th Percentile TTFF')

    for i, (m, p) in enumerate(zip(means_s, p95s_s)):
        ax.text(i - w/2, m + 0.3, f'{m:.1f}s', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + w/2, p + 0.3, f'{p:.1f}s', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_s, fontsize=9)
    ax.set_ylabel('Time-to-First-Fix (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operational Scenario (Adversary Threat)', fontsize=12, fontweight='bold')
    ax.set_title('TTFF by Adversary Scenario\n(mean vs 95th percentile)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    # Add C2 threshold annotation
    # Notional threshold: <15s for tactical C2 reintegration
    ax.axhline(15, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(len(labels_s) - 0.5, 15.5, 'Notional C2\nlatency threshold\n(15s)', color='red',
            fontsize=9, ha='right')

    plt.tight_layout()
    fp = os.path.join(output_dir, "fig6_c2_latency.png")
    plt.savefig(fp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fp


# ---------------------------------------------------------------------------
# Master: generate all figures
# ---------------------------------------------------------------------------
def generate_all_figures():
    print("=" * 70)
    print("Military Ground Vehicle Celestial Navigation — Visualization")
    print("=" * 70)

    all_data = run_all_simulations()
    if all_data is None:
        print("ERROR: Simulation failed.")
        return

    scenario_stats = all_data['mc']['scenario_stats']
    fov_size = all_data['scenarios'][0].fov_size if all_data['scenarios'] else 20.0

    os.makedirs(FIGURES_DIR, exist_ok=True)
    figures = []

    print("\n1. Figure 1: Success Rates...")
    figures.append(("Fig 1 — Success Rates",        generate_success_rates_chart(scenario_stats)))
    print("2. Figure 2: Algorithm Performance...")
    figures.append(("Fig 2 — Algorithm Performance", generate_algorithm_performance_chart(scenario_stats)))
    print("3. Figure 3: Environmental Analysis...")
    figures.append(("Fig 3 — Environmental Analysis",generate_environmental_analysis_chart(scenario_stats)))
    print("4. Figure 4: Sensor Layout...")
    figures.append(("Fig 4 — Sensor Layout",          generate_sensor_layout(fov_size)))
    print("5. Figure 5: GPS Degradation Curve...")
    figures.append(("Fig 5 — GPS Degradation Curve",  generate_gps_degradation_figure(all_data['degradation'])))
    print("6. Figure 6: C2 Latency...")
    figures.append(("Fig 6 — C2 Latency",        generate_jado_c2_latency_figure(all_data['c2_latency'])))

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    for name, path in figures:
        print(f"  {name:35} → {path}")

    print(f"\nOverall MC success rate: {all_data['mc']['overall_success_rate']:.1f}%")
    print(f"Total trials: {all_data['mc']['total_trials']:,}")


if __name__ == "__main__":
    generate_all_figures()
