#!/usr/bin/env python3
"""
Create publication-quality graphs from group testing simulation results.
"""

import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Set seaborn style for pretty plots
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_ablation_data(db_path: str, ablated_param: str) -> pd.DataFrame:
    """Load ablation study data from SQLite database."""
    conn = sqlite3.connect(db_path)

    # Map ablated_param to the actual column name
    param_column_map = {
        'defect_rate': 'param_defect_rate',
        'flake_rate': 'param_flake_rate',
        'C': 'param_c',
        'M': 'param_m',
        'K': 'param_k',
        'change_arrival_rate': 'param_change_arrival_rate',
    }

    query = f"""
    SELECT * FROM simulation_results
    WHERE ablated_param = ?
    ORDER BY {param_column_map.get(ablated_param, 'param_' + ablated_param.lower())}
    """

    df = pd.read_sql_query(query, conn, params=(ablated_param,))
    conn.close()

    return df


def get_param_display_name(param: str) -> str:
    """Get human-readable display name for parameter."""
    display_names = {
        'defect_rate': 'Defect Rate',
        'flake_rate': 'Flake Rate',
        'C': 'Changes per Train (C)',
        'M': 'Number of Minibatches (M)',
        'K': 'Column Weight (K)',
        'change_arrival_rate': 'Change Arrival Rate (changes/h)',
    }
    return display_names.get(param, param)


def get_param_column(param: str) -> str:
    """Get the database column name for a parameter."""
    return f'param_{param.lower()}'


def plot_metric_vs_param(
    df: pd.DataFrame,
    param: str,
    metric: str,
    metric_label: str,
    ax: Optional[plt.Axes] = None,
    color: str = None,
    non_negative: bool = True,
) -> plt.Axes:
    """Plot a single metric vs ablated parameter with error bars."""
    if ax is None:
        fig, ax = plt.subplots()

    param_col = get_param_column(param)
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_stddev'

    x = df[param_col]
    y = df[mean_col]
    std = df[std_col]

    # For non-negative metrics, clip lower error bar at 0
    if non_negative:
        yerr_lower = np.minimum(std, y)  # Can't go below 0
        yerr_upper = std
        yerr = [yerr_lower, yerr_upper]
    else:
        yerr = std

    # Use errorbar for mean ± stddev
    if color:
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, capthick=1.5,
                   linewidth=2, markersize=6, color=color, label=metric_label)
    else:
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, capthick=1.5,
                   linewidth=2, markersize=6, label=metric_label)

    ax.set_xlabel(get_param_display_name(param))
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)

    return ax


def plot_ablation_study(
    db_path: str,
    ablated_param: str,
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """
    Create all plots for an ablation study.

    Returns list of paths to created plot files.
    """
    df = load_ablation_data(db_path, ablated_param)

    if df.empty:
        print(f"No data found for ablated_param='{ablated_param}'")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    created_files = []
    param_display = get_param_display_name(ablated_param)

    # Define metrics to plot
    metrics = [
        ('false_rejection_rate', 'False Rejection Rate'),
        ('submit_latency', 'Submit Latency (hours)'),
        ('capacity_cost_ratio', 'Capacity Cost Ratio'),
        ('e2e_cost', 'E2E Cost (SWEh/change)'),
    ]

    # Individual plots for each metric
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric_vs_param(df, ablated_param, metric, label, ax)
        ax.set_title(f'{label} vs {param_display}')

        # Add sample count annotation
        avg_samples = df['sample_count'].mean()
        ax.annotate(f'Avg samples: {avg_samples:.0f}',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        filename = f'{ablated_param}_{metric}.png'
        filepath = output_path / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        created_files.append(str(filepath))

        if show:
            plt.show()
        plt.close(fig)

    # Combined 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = sns.color_palette("deep", 4)

    for idx, ((metric, label), ax) in enumerate(zip(metrics, axes.flat)):
        plot_metric_vs_param(df, ablated_param, metric, label, ax, color=colors[idx])
        ax.set_title(f'{label}')

    fig.suptitle(f'Ablation Study: {param_display}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f'{ablated_param}_combined.png'
    filepath = output_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    created_files.append(str(filepath))

    if show:
        plt.show()
    plt.close(fig)

    print(f"Created {len(created_files)} plots for {ablated_param}")
    return created_files


def plot_all_ablations(
    db_path: str,
    output_dir: str = "plots",
    show: bool = False,
) -> dict[str, list[str]]:
    """
    Create plots for all ablation studies in the database.

    Returns dict mapping ablated_param to list of created files.
    """
    conn = sqlite3.connect(db_path)
    params = pd.read_sql_query(
        "SELECT DISTINCT ablated_param FROM simulation_results WHERE ablated_param != 'single'",
        conn
    )['ablated_param'].tolist()
    conn.close()

    results = {}
    for param in params:
        results[param] = plot_ablation_study(db_path, param, output_dir, show)

    return results


def plot_comparison(
    db_path: str,
    metric: str,
    metric_label: str,
    params_to_compare: list[str],
    output_dir: str = "plots",
    show: bool = False,
) -> str:
    """
    Create a comparison plot showing one metric across multiple ablation studies.
    Normalizes x-axis to [0, 1] for comparison.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("husl", len(params_to_compare))

    for param, color in zip(params_to_compare, colors):
        df = load_ablation_data(db_path, param)
        if df.empty:
            continue

        param_col = get_param_column(param)
        mean_col = f'{metric}_mean'

        x = df[param_col]
        y = df[mean_col]

        # Normalize x to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min())

        ax.plot(x_norm, y, 'o-', linewidth=2, markersize=6,
               color=color, label=get_param_display_name(param))

    ax.set_xlabel('Normalized Parameter Value')
    ax.set_ylabel(metric_label)
    ax.set_title(f'{metric_label} Sensitivity to Different Parameters')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'comparison_{metric}.png'
    filepath = output_path / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)

    return str(filepath)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create graphs from simulation results')
    parser.add_argument('--db', default='simulation_results.db', help='Path to SQLite database')
    parser.add_argument('--param', help='Specific ablated parameter to plot')
    parser.add_argument('--output', default='plots', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--all', action='store_true', help='Plot all ablation studies')

    args = parser.parse_args()

    if args.all:
        results = plot_all_ablations(args.db, args.output, args.show)
        print(f"\nCreated plots for {len(results)} ablation studies")
    elif args.param:
        files = plot_ablation_study(args.db, args.param, args.output, args.show)
        print(f"\nCreated {len(files)} plots")
    else:
        parser.print_help()
