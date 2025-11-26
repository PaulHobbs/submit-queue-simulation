#!/usr/bin/env python3
"""
Analyze parameter sensitivity from Optuna optimization results.

Compares the importance of Level 1 vs Level 2 parameters and identifies
which parameters have the biggest impact on the objective function.
"""

import optuna
import numpy as np
from typing import Dict, List, Tuple


def load_study_from_trials(trials_data: List[Dict]) -> optuna.Study:
    """Reconstruct Optuna study from trial data."""
    study = optuna.create_study(direction='maximize')

    for trial_dict in trials_data:
        trial = optuna.trial.create_trial(
            params=trial_dict['params'],
            distributions={
                k: optuna.distributions.IntDistribution(low=0, high=10000)
                if isinstance(v, int) else
                optuna.distributions.FloatDistribution(low=0.0, high=1.0)
                for k, v in trial_dict['params'].items()
            },
            values=[trial_dict['value']],
        )
        study.add_trial(trial)

    return study


def analyze_parameter_importance(study: optuna.Study) -> Dict[str, float]:
    """Calculate parameter importance using fANOVA."""
    try:
        importance = optuna.importance.get_param_importances(study)
        return importance
    except Exception as e:
        print(f"Could not compute fANOVA importance: {e}")
        return {}


def correlation_analysis(study: optuna.Study) -> Dict[str, float]:
    """Compute correlation between each parameter and objective."""
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

    if len(trials) < 2:
        return {}

    # Get all parameter names
    param_names = list(trials[0].params.keys())

    correlations = {}
    for param_name in param_names:
        param_values = []
        objectives = []

        for trial in trials:
            if param_name in trial.params:
                param_values.append(trial.params[param_name])
                objectives.append(trial.value)

        if len(param_values) > 1:
            correlation = np.corrcoef(param_values, objectives)[0, 1]
            correlations[param_name] = correlation

    return correlations


def sensitivity_by_range(study: optuna.Study) -> Dict[str, Tuple[float, float, float]]:
    """
    For each parameter, compute sensitivity as the range of objectives
    when the parameter varies.

    Returns: {param_name: (min_obj, max_obj, range)}
    """
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

    if len(trials) < 2:
        return {}

    param_names = list(trials[0].params.keys())

    sensitivities = {}
    for param_name in param_names:
        objectives_for_param = {}

        for trial in trials:
            if param_name in trial.params:
                param_val = trial.params[param_name]
                if param_val not in objectives_for_param:
                    objectives_for_param[param_val] = []
                objectives_for_param[param_val].append(trial.value)

        # Average objectives for each parameter value
        avg_objectives = [np.mean(objs) for objs in objectives_for_param.values()]

        if len(avg_objectives) > 1:
            min_obj = min(avg_objectives)
            max_obj = max(avg_objectives)
            range_obj = max_obj - min_obj
            sensitivities[param_name] = (min_obj, max_obj, range_obj)

    return sensitivities


def categorize_parameters(param_names: List[str]) -> Dict[str, List[str]]:
    """Categorize parameters into Level 1 and Level 2."""
    level1 = ['resources', 'maxbatch', 'maxk', 'kdiv', 'flaketol']
    level2 = ['verify_latency', 'fix_delay', 'verify_resource_mult',
              'bp_threshold_1', 'bp_threshold_2', 'bp_threshold_3']

    categories = {
        'Level 1 (Explicit)': [p for p in param_names if p in level1],
        'Level 2 (Implicit)': [p for p in param_names if p in level2],
    }

    return categories


def print_analysis(study: optuna.Study):
    """Print comprehensive parameter sensitivity analysis."""
    print("=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()

    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    print(f"Total trials analyzed: {len(trials)}")
    print(f"Best objective: {study.best_value:.2f}")
    print()

    # Get all parameter names
    if not trials:
        print("No completed trials to analyze!")
        return

    param_names = list(trials[0].params.keys())
    categories = categorize_parameters(param_names)

    # 1. Parameter Importance (fANOVA)
    print("1. PARAMETER IMPORTANCE (fANOVA)")
    print("-" * 80)
    importance = analyze_parameter_importance(study)

    if importance:
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for level, params in categories.items():
            print(f"\n{level}:")
            level_importance = [(p, imp) for p, imp in sorted_importance if p in params]
            for param, imp in level_importance:
                bar = "█" * int(imp * 50)
                print(f"  {param:25s} {imp:6.1%} {bar}")

        # Summary by level
        print("\nImportance by Level:")
        for level, params in categories.items():
            level_imp = sum(imp for p, imp in importance.items() if p in params)
            print(f"  {level:25s} {level_imp:6.1%}")
    else:
        print("  (Could not compute - not enough trials or variance)")

    print()

    # 2. Correlation Analysis
    print("2. CORRELATION WITH OBJECTIVE")
    print("-" * 80)
    correlations = correlation_analysis(study)

    if correlations:
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        for level, params in categories.items():
            print(f"\n{level}:")
            level_corr = [(p, corr) for p, corr in sorted_corr if p in params]
            for param, corr in level_corr:
                direction = "↑" if corr > 0 else "↓"
                bar = "█" * int(abs(corr) * 30)
                print(f"  {param:25s} {corr:+7.3f} {direction} {bar}")

        print("\nInterpretation:")
        print("  Positive correlation: higher value → better objective")
        print("  Negative correlation: lower value → better objective")

    print()

    # 3. Sensitivity by Range
    print("3. OBJECTIVE RANGE SENSITIVITY")
    print("-" * 80)
    sensitivities = sensitivity_by_range(study)

    if sensitivities:
        sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1][2], reverse=True)

        for level, params in categories.items():
            print(f"\n{level}:")
            level_sens = [(p, s) for p, s in sorted_sens if p in params]
            for param, (min_obj, max_obj, range_obj) in level_sens:
                print(f"  {param:25s} range: {range_obj:10.2f}  (min: {min_obj:8.2f}, max: {max_obj:8.2f})")

        print("\nInterpretation:")
        print("  Larger range = more sensitive parameter")
        print("  This shows how much the objective changes when the parameter varies")

    print()

    # 4. Best Configuration
    print("4. BEST CONFIGURATION")
    print("-" * 80)

    for level, params in categories.items():
        print(f"\n{level}:")
        for param in params:
            if param in study.best_params:
                value = study.best_params[param]
                if isinstance(value, float):
                    print(f"  {param:25s} {value:.4f}")
                else:
                    print(f"  {param:25s} {value}")

    print()
    print("=" * 80)


def main():
    """Load and analyze optimization results."""
    import sys
    import pickle

    if len(sys.argv) < 2:
        print("Usage: python analyze_sensitivity.py <study.pkl>")
        print()
        print("The study file is created by running:")
        print("  python optimizer_robust.py --save-study study.pkl")
        sys.exit(1)

    study_path = sys.argv[1]

    # Load study
    print(f"Loading study from {study_path}...")
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print(f"Loaded {len(study.trials)} trials")
        print()
    except Exception as e:
        print(f"Error loading study: {e}")
        sys.exit(1)

    # Analyze
    print_analysis(study)


if __name__ == '__main__':
    main()
