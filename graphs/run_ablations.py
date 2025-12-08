#!/usr/bin/env python3
"""
Automate running ablation studies and creating plots.
"""

import subprocess
import argparse
import sys
from pathlib import Path
from make_graph import plot_ablation_study, plot_all_ablations, plot_comparison


# All parameters that can be ablated
ABLATION_PARAMS = [
    'defect_rate',
    'flake_rate',
    'C',
    'M',
    'K',
    'change_arrival_rate',
]


def run_ablation(
    param: str,
    db_path: str = 'simulation_results.db',
    samples: int = 10000,
    workers: int = 4,
    go_binary: str = './group_testing_sim',
) -> bool:
    """
    Run a single ablation study using the Go simulation.

    Returns True if successful, False otherwise.
    """
    cmd = [
        go_binary,
        '-ablate', param,
        '-db', db_path,
        '-samples', str(samples),
        '-workers', str(workers),
    ]

    print(f"\n{'='*60}")
    print(f"Running ablation study for: {param}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running ablation for {param}: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Go binary not found at {go_binary}")
        print("Please build it first with: go build -o group_testing_sim group_testing_sim.go")
        return False


def run_all_ablations(
    db_path: str = 'simulation_results.db',
    samples: int = 10000,
    workers: int = 4,
    go_binary: str = './group_testing_sim',
    params: list[str] = None,
) -> dict[str, bool]:
    """
    Run ablation studies for all (or specified) parameters.

    Returns dict mapping param to success status.
    """
    if params is None:
        params = ABLATION_PARAMS

    results = {}
    for param in params:
        results[param] = run_ablation(param, db_path, samples, workers, go_binary)

    return results


def create_all_plots(
    db_path: str = 'simulation_results.db',
    output_dir: str = 'plots',
    params: list[str] = None,
) -> dict[str, list[str]]:
    """
    Create plots for all (or specified) ablation studies.

    Returns dict mapping param to list of created files.
    """
    if params is None:
        return plot_all_ablations(db_path, output_dir)

    results = {}
    for param in params:
        results[param] = plot_ablation_study(db_path, param, output_dir)

    return results


def create_comparison_plots(
    db_path: str = 'simulation_results.db',
    output_dir: str = 'plots',
) -> list[str]:
    """Create comparison plots across all ablation studies."""
    metrics = [
        ('false_rejection_rate', 'False Rejection Rate'),
        ('submit_latency', 'Submit Latency (hours)'),
        ('capacity_cost_ratio', 'Capacity Cost Ratio'),
        ('e2e_cost', 'E2E Cost (SWEh/change)'),
    ]

    created_files = []
    for metric, label in metrics:
        filepath = plot_comparison(db_path, metric, label, ABLATION_PARAMS, output_dir)
        created_files.append(filepath)
        print(f"Created comparison plot: {filepath}")

    return created_files


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation studies and create plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablation studies with default settings
  python run_ablations.py --run

  # Run specific ablation studies
  python run_ablations.py --run --params defect_rate flake_rate

  # Just create plots from existing data
  python run_ablations.py --plot

  # Run ablations and create plots
  python run_ablations.py --run --plot

  # Fast run with fewer samples for testing
  python run_ablations.py --run --plot --samples 1000
        """
    )

    parser.add_argument('--run', action='store_true',
                       help='Run ablation studies')
    parser.add_argument('--plot', action='store_true',
                       help='Create plots from results')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison plots across all parameters')
    parser.add_argument('--params', nargs='+', choices=ABLATION_PARAMS,
                       help='Specific parameters to ablate (default: all)')
    parser.add_argument('--db', default='simulation_results.db',
                       help='Path to SQLite database')
    parser.add_argument('--output', default='plots',
                       help='Output directory for plots')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Maximum samples per grid point')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--binary', default='./group_testing_sim',
                       help='Path to Go binary')

    args = parser.parse_args()

    if not args.run and not args.plot and not args.compare:
        parser.print_help()
        print("\nError: Please specify --run, --plot, or --compare")
        sys.exit(1)

    # Ensure output directory exists
    Path(args.output).mkdir(exist_ok=True)

    # Run ablation studies
    if args.run:
        print("\n" + "="*60)
        print("RUNNING ABLATION STUDIES")
        print("="*60)

        run_results = run_all_ablations(
            db_path=args.db,
            samples=args.samples,
            workers=args.workers,
            go_binary=args.binary,
            params=args.params,
        )

        # Summary
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        for param, success in run_results.items():
            status = "OK" if success else "FAILED"
            print(f"  {param}: {status}")

        failed = [p for p, s in run_results.items() if not s]
        if failed:
            print(f"\nWarning: {len(failed)} ablation(s) failed: {failed}")

    # Create plots
    if args.plot:
        print("\n" + "="*60)
        print("CREATING PLOTS")
        print("="*60)

        plot_results = create_all_plots(
            db_path=args.db,
            output_dir=args.output,
            params=args.params,
        )

        total_plots = sum(len(files) for files in plot_results.values())
        print(f"\nCreated {total_plots} plots across {len(plot_results)} ablation studies")

    # Create comparison plots
    if args.compare:
        print("\n" + "="*60)
        print("CREATING COMPARISON PLOTS")
        print("="*60)

        comparison_files = create_comparison_plots(
            db_path=args.db,
            output_dir=args.output,
        )
        print(f"\nCreated {len(comparison_files)} comparison plots")

    print("\nDone!")


if __name__ == '__main__':
    main()
