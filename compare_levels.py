#!/usr/bin/env python3
"""
Compare Level 1 vs Level 2 optimization results.

Shows the improvement from adding implicit hyperparameters.
"""

import pickle
import sys


def compare_studies(level1_path, level2_path):
    """Compare two optimization studies."""

    # Load studies
    with open(level1_path, 'rb') as f:
        level1 = pickle.load(f)

    with open(level2_path, 'rb') as f:
        level2 = pickle.load(f)

    print("=" * 80)
    print("LEVEL 1 vs LEVEL 2 COMPARISON")
    print("=" * 80)
    print()

    # Basic stats
    print("OPTIMIZATION STATISTICS")
    print("-" * 80)
    print(f"{'':30s} {'Level 1':>20s} {'Level 2':>20s}")
    print("-" * 80)
    print(f"{'Parameters optimized':30s} {len(level1.best_params):>20d} {len(level2.best_params):>20d}")
    print(f"{'Trials completed':30s} {len(level1.trials):>20d} {len(level2.trials):>20d}")
    print(f"{'Best objective':30s} {level1.best_value:>20.2f} {level2.best_value:>20.2f}")

    improvement = level2.best_value - level1.best_value
    improvement_pct = (improvement / abs(level1.best_value)) * 100
    print(f"{'Improvement':30s} {improvement:>20.2f} ({improvement_pct:+.2f}%)")
    print()

    # Parameter comparison
    print("BEST PARAMETERS")
    print("-" * 80)

    # Level 1 parameters
    level1_params = {'resources', 'maxbatch', 'maxk', 'kdiv', 'flaketol'}
    print("\nLevel 1 (Explicit) Parameters:")
    print(f"{'Parameter':25s} {'Level 1':>15s} {'Level 2':>15s} {'Change':>15s}")
    print("-" * 70)

    for param in sorted(level1_params):
        if param in level1.best_params and param in level2.best_params:
            v1 = level1.best_params[param]
            v2 = level2.best_params[param]

            if isinstance(v1, float):
                change = v2 - v1
                change_pct = (change / v1) * 100 if v1 != 0 else 0
                print(f"{param:25s} {v1:15.4f} {v2:15.4f} {change_pct:+14.1f}%")
            else:
                change = v2 - v1
                change_pct = (change / v1) * 100 if v1 != 0 else 0
                print(f"{param:25s} {v1:15d} {v2:15d} {change_pct:+14.1f}%")

    # Level 2 parameters (only in level2)
    level2_only = set(level2.best_params.keys()) - level1_params

    if level2_only:
        print("\nLevel 2 (Implicit) Parameters (NEW):")
        print(f"{'Parameter':25s} {'Value':>15s} {'Default':>15s}")
        print("-" * 55)

        defaults = {
            'verify_latency': 2,
            'fix_delay': 60,
            'verify_resource_mult': 16,
            'bp_threshold_1': 200,
            'bp_threshold_2': 400,
            'bp_threshold_3': 800,
        }

        for param in sorted(level2_only):
            value = level2.best_params[param]
            default = defaults.get(param, '?')

            if isinstance(value, float):
                print(f"{param:25s} {value:15.4f} {default:>15}")
            else:
                diff = value - default if isinstance(default, int) else '?'
                diff_str = f"({diff:+d})" if isinstance(diff, int) else ""
                print(f"{param:25s} {value:15d} {default:>15} {diff_str}")

    print()

    # Key insights
    print("KEY INSIGHTS")
    print("-" * 80)

    # Check which Level 1 params changed significantly
    significant_changes = []
    for param in level1_params:
        if param in level1.best_params and param in level2.best_params:
            v1 = level1.best_params[param]
            v2 = level2.best_params[param]
            change_pct = ((v2 - v1) / v1) * 100 if v1 != 0 else 0

            if abs(change_pct) > 10:
                direction = "increased" if change_pct > 0 else "decreased"
                significant_changes.append(f"  • {param} {direction} by {abs(change_pct):.1f}%")

    if significant_changes:
        print("\nLevel 1 parameters that changed significantly:")
        for change in significant_changes:
            print(change)

    # Check Level 2 params vs defaults
    level2_changes = []
    for param in sorted(level2_only):
        value = level2.best_params[param]
        default = defaults.get(param)

        if default and isinstance(value, int):
            change_pct = ((value - default) / default) * 100
            if abs(change_pct) > 10:
                direction = "increased" if change_pct > 0 else "decreased"
                level2_changes.append(f"  • {param}: {value} (default: {default}, {direction} {abs(change_pct):.1f}%)")

    if level2_changes:
        print("\nLevel 2 parameters that differ from defaults:")
        for change in level2_changes:
            print(change)
    else:
        print("\nLevel 2 parameters: All near defaults (Level 1 params were more important)")

    if improvement > 0:
        print(f"\n✅ Level 2 improved objective by {abs(improvement):.2f} ({abs(improvement_pct):.2f}%)")
        print(f"   The implicit parameters helped find a better configuration!")
    else:
        print(f"\n⚠️  Level 2 was {abs(improvement):.2f} worse ({abs(improvement_pct):.2f}%)")
        print(f"   This suggests Level 1 params are more important, or need more trials")

    print()
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_levels.py <level1_study.pkl> <level2_study.pkl>")
        sys.exit(1)

    compare_studies(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
