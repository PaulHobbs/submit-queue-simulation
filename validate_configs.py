#!/usr/bin/env python3
"""
Independent validation of optimized configurations.

Evaluates each configuration with high sample counts to get unbiased
performance estimates with confidence intervals.
"""

import subprocess
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys


@dataclass
class Scenario:
    """Test scenario definition."""
    name: str
    weight: float
    traffic: int
    culprit_prob: float
    test_stability: float
    bp_thresholds: Tuple[int, int, int]
    bp_jitter: float = 0.25

    def sample_bp_thresholds(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        """Sample backpressure thresholds with jitter."""
        if self.bp_jitter == 0:
            return self.bp_thresholds

        jittered = []
        for threshold in self.bp_thresholds:
            low = int(threshold * (1 - self.bp_jitter))
            high = int(threshold * (1 + self.bp_jitter))
            jittered.append(rng.integers(low, high + 1))

        return tuple(jittered)


# Same 8 scenarios as corrected Level 2
SCENARIOS = [
    Scenario('normal-std', weight=0.25, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(200, 400, 800)),
    Scenario('normal-aggressive', weight=0.10, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(300, 600, 1200)),
    Scenario('normal-conservative', weight=0.05, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(100, 200, 400)),
    Scenario('spike-std', weight=0.20, traffic=16, culprit_prob=0.04, test_stability=0.98,
             bp_thresholds=(200, 400, 800)),
    Scenario('spike-conservative', weight=0.10, traffic=16, culprit_prob=0.04, test_stability=0.98,
             bp_thresholds=(100, 200, 400)),
    Scenario('low', weight=0.10, traffic=4, culprit_prob=0.02, test_stability=0.99,
             bp_thresholds=(200, 400, 800)),
    Scenario('flaky', weight=0.10, traffic=8, culprit_prob=0.03, test_stability=0.90,
             bp_thresholds=(200, 400, 800)),
    Scenario('crisis', weight=0.10, traffic=20, culprit_prob=0.08, test_stability=0.92,
             bp_thresholds=(200, 400, 800)),
]


# Best configurations from each optimization
CONFIGS = {
    'Level 1': {
        'resources': 19,
        'maxbatch': 3702,
        'maxk': 13,
        'kdiv': 5,
        'flaketol': 0.096,
        'ntests': 32,
        'optimized': False,
        # Use defaults for Level 2 params
        'verify_latency': 2,
        'fix_delay': 60,
        'verify_resource_mult': 16,
    },
    'Incorrect Level 2': {
        'resources': 58,
        'maxbatch': 1364,
        'maxk': 15,
        'kdiv': 4,
        'flaketol': 0.070,
        'ntests': 32,
        'optimized': False,
        'verify_latency': 1,
        'fix_delay': 89,
        'verify_resource_mult': 19,
    },
    'Corrected Level 2': {
        'resources': 81,
        'maxbatch': 843,
        'maxk': 11,
        'kdiv': 5,
        'flaketol': 0.0688,
        'ntests': 32,
        'optimized': False,
        'verify_latency': 9,
        'fix_delay': 103,
        'verify_resource_mult': 24,
    },
}


# Objective function weights (same as optimizer)
WEIGHTS = {
    'verification_cost': 2.0,
    'escaped_culprit_cost': 20.0,
    'resource_alpha': 0.05,
    'test_quality_gamma': 100.0,
    'latency_beta': 1.0,
    'target_wait_time': 20.0,
}


def compute_objective(metrics: Dict) -> float:
    """Compute developer productivity score."""
    slowdown = max(metrics['slowdown'], 0.1)
    throughput_score = 1000.0 / slowdown

    innocent_flagged = metrics['innocent_flagged']
    culprits_escaped = max(0, metrics['culprits_created'] - metrics['culprits_caught'])

    wasted_debug_cost = (
        innocent_flagged * WEIGHTS['verification_cost'] +
        culprits_escaped * WEIGHTS['escaped_culprit_cost']
    )

    resource_cost = (
        WEIGHTS['resource_alpha'] *
        metrics['config']['Resources'] *
        metrics['batch_utilization']
    )

    ntests = metrics['config']['NTests']
    active_tests = metrics['active_tests']
    demoted_tests = ntests - active_tests
    test_quality_cost = WEIGHTS['test_quality_gamma'] * demoted_tests

    wait_p95 = metrics['wait_time_p95']
    target_wait = WEIGHTS['target_wait_time']
    latency_penalty = WEIGHTS['latency_beta'] * (wait_p95 / target_wait) ** 2

    objective = (
        throughput_score
        - wasted_debug_cost
        - resource_cost
        - test_quality_cost
        - latency_penalty
    )

    return objective


def run_simulation(config: Dict, scenario: Scenario, rng: np.random.Generator,
                   binary_path='./submit_queue') -> Dict:
    """Run single simulation."""
    bp1, bp2, bp3 = scenario.sample_bp_thresholds(rng)

    args = [
        binary_path,
        '-json',
        '-resources', str(config['resources']),
        '-traffic', str(scenario.traffic),
        '-ntests', str(config['ntests']),
        '-maxbatch', str(config['maxbatch']),
        '-maxk', str(config['maxk']),
        '-kdiv', str(config['kdiv']),
        '-flaketol', str(config['flaketol']),
        '-optimized', str(config.get('optimized', False)).lower(),
        '-culprit-prob', str(scenario.culprit_prob),
        '-test-stability', str(scenario.test_stability),
        '-bp-threshold-1', str(bp1),
        '-bp-threshold-2', str(bp2),
        '-bp-threshold-3', str(bp3),
        '-verify-latency', str(config['verify_latency']),
        '-fix-delay', str(config['fix_delay']),
        '-verify-resource-mult', str(config['verify_resource_mult']),
    ]

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=90,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed: {result.stderr}")

    return json.loads(result.stdout)


def validate_config(config_name: str, config: Dict, scenarios: List[Scenario],
                    samples_per_scenario: int = 20, seed: int = 42) -> Dict:
    """
    Validate configuration with independent high-sample evaluation.

    Returns:
        {
            'scenario_results': {scenario_name: [objectives]},
            'weighted_mean': float,
            'std_error': float,
            'confidence_interval': (lower, upper),
        }
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*80}")
    print(f"Validating: {config_name}")
    print(f"{'='*80}")
    print(f"Samples per scenario: {samples_per_scenario}")
    print(f"Total simulations: {len(scenarios) * samples_per_scenario}")
    print()

    scenario_results = {}
    all_objectives = []

    for scenario in scenarios:
        print(f"Testing {scenario.name:20s} ", end='', flush=True)
        objectives = []

        for i in range(samples_per_scenario):
            if i % 5 == 0:
                print('.', end='', flush=True)

            try:
                metrics = run_simulation(config, scenario, rng)
                objective = compute_objective(metrics)
                objectives.append(objective)
            except Exception as e:
                print(f"\n  Error on sample {i}: {e}")
                objectives.append(-1e6)  # Penalty for failure

        scenario_mean = np.mean(objectives)
        scenario_std = np.std(objectives)
        print(f" mean: {scenario_mean:8.2f} ± {scenario_std:6.2f}")

        scenario_results[scenario.name] = objectives

        # Add to overall results weighted by scenario importance
        all_objectives.extend([scenario_mean] * int(scenario.weight * 100))

    # Compute weighted statistics
    weighted_objectives = []
    for scenario in scenarios:
        scenario_mean = np.mean(scenario_results[scenario.name])
        # Add weighted copies
        weighted_objectives.extend([scenario_mean] * int(scenario.weight * 100))

    weighted_mean = np.mean(weighted_objectives)
    weighted_std = np.std(weighted_objectives)

    # Compute standard error of the mean
    # Use per-scenario means as the samples for conservative estimate
    scenario_means = [np.mean(scenario_results[s.name]) for s in scenarios]
    std_error = np.std(scenario_means) / np.sqrt(len(scenarios))

    # 95% confidence interval (approximately ±2 std errors)
    ci_lower = weighted_mean - 2 * std_error
    ci_upper = weighted_mean + 2 * std_error

    print()
    print(f"{'─'*80}")
    print(f"Weighted Mean Objective: {weighted_mean:.2f}")
    print(f"Standard Error:          {std_error:.2f}")
    print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"{'─'*80}")

    return {
        'config_name': config_name,
        'scenario_results': scenario_results,
        'weighted_mean': weighted_mean,
        'std_error': std_error,
        'confidence_interval': (ci_lower, ci_upper),
    }


def compare_configs(results: Dict[str, Dict]):
    """Compare configurations and test statistical significance."""
    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*80}\n")

    # Sort by weighted mean
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['weighted_mean'], reverse=True)

    print("Rank | Configuration       | Weighted Mean | 95% CI             | vs Best")
    print("-----|---------------------|---------------|--------------------|---------")

    best_mean = sorted_configs[0][1]['weighted_mean']

    for rank, (name, result) in enumerate(sorted_configs, 1):
        mean = result['weighted_mean']
        ci_low, ci_high = result['confidence_interval']
        ci_width = ci_high - ci_low

        improvement = ((mean - best_mean) / abs(best_mean)) * 100 if mean != best_mean else 0.0

        print(f"{rank:4d} | {name:19s} | {mean:13.2f} | "
              f"±{ci_width/2:7.2f} ({ci_width:6.2f}) | {improvement:+6.2f}%")

    print()

    # Statistical significance tests
    print("Pairwise Statistical Significance (Welch's t-test):")
    print()

    from scipy import stats

    config_names = list(results.keys())
    for i, name1 in enumerate(config_names):
        for name2 in config_names[i+1:]:
            # Get all individual samples for each config
            samples1 = []
            samples2 = []

            for scenario in SCENARIOS:
                samples1.extend(results[name1]['scenario_results'][scenario.name])
                samples2.extend(results[name2]['scenario_results'][scenario.name])

            # Welch's t-test (doesn't assume equal variance)
            t_stat, p_value = stats.ttest_ind(samples1, samples2, equal_var=False)

            mean1 = results[name1]['weighted_mean']
            mean2 = results[name2]['weighted_mean']
            diff = mean1 - mean2

            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "n.s."

            print(f"{name1:20s} vs {name2:20s}: "
                  f"diff={diff:8.2f}, p={p_value:.4f} {significance}")

    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate optimized configurations')
    parser.add_argument('--samples', type=int, default=20,
                       help='Samples per scenario (default: 20)')
    parser.add_argument('--configs', nargs='+',
                       choices=list(CONFIGS.keys()),
                       default=list(CONFIGS.keys()),
                       help='Configurations to validate')
    parser.add_argument('--binary', type=str, default='./submit_queue',
                       help='Path to simulation binary')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Check for scipy
    try:
        import scipy.stats
    except ImportError:
        print("Warning: scipy not installed. Statistical tests will be skipped.")
        print("Install with: pip install scipy")

    # Validate each configuration
    results = {}

    start_time = time.time()

    for config_name in args.configs:
        config = CONFIGS[config_name]
        result = validate_config(
            config_name,
            config,
            SCENARIOS,
            samples_per_scenario=args.samples,
            seed=args.seed,
        )
        results[config_name] = result

    elapsed = time.time() - start_time

    # Compare configurations
    compare_configs(results)

    # Summary
    print(f"{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Configurations tested: {len(results)}")
    print(f"Samples per config: {args.samples * len(SCENARIOS)}")
    print(f"Total simulations: {len(results) * args.samples * len(SCENARIOS)}")
    print()


if __name__ == '__main__':
    main()
