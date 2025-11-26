#!/usr/bin/env python3
"""
Empirical validation of top candidate configurations.

Extracts top candidates by different criteria (observed value, GP posterior mean,
GP lower confidence bound) and validates each with high sample counts to get
unbiased performance estimates.
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import subprocess
import json
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
import optuna


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


# Same 8 scenarios as optimization
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


WEIGHTS = {
    'verification_cost': 2.0,
    'escaped_culprit_cost': 20.0,
    'resource_alpha': 0.05,
    'test_quality_gamma': 100.0,
    'latency_beta': 1.0,
    'target_wait_time': 20.0,
}

# Integer parameters (need rounding)
INTEGER_PARAMS = {
    'resources', 'maxbatch', 'maxk', 'kdiv',
    'verify_latency', 'fix_delay', 'verify_resource_mult', 'ntests'
}


def normalize_config(config: Dict) -> Dict:
    """Normalize config to have correct types (int vs float)."""
    normalized = {}
    for key, value in config.items():
        if key in INTEGER_PARAMS:
            normalized[key] = int(round(value))
        else:
            normalized[key] = value
    return normalized


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
        '-ntests', str(config.get('ntests', 32)),
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
        '-verify-latency', str(config.get('verify_latency', 2)),
        '-fix-delay', str(config.get('fix_delay', 60)),
        '-verify-resource-mult', str(config.get('verify_resource_mult', 16)),
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


def extract_top_candidates(
    study: optuna.Study,
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    param_names: List[str],
    k: int = 10,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Extract top K candidates by different criteria.

    Returns:
        List of (criterion_name, config) tuples
    """
    candidates = []

    # 1. Top K by observed value
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)

    print(f"Extracting top {k} candidates by different criteria...")
    print()

    print(f"Top {k} by Observed Value:")
    for i, trial in enumerate(sorted_trials[:k], 1):
        config = trial.params.copy()
        config['ntests'] = 32
        config['optimized'] = False
        candidates.append((f'observed_rank_{i}', config))
        print(f"  {i:2d}. Observed: {trial.value:8.2f} | {config}")

    print()

    # 2. Sample many configs and rank by GP posterior mean
    print(f"Top {k} by GP Posterior Mean:")
    n_samples = 5000
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
    samples_unit = sampler.random(n=n_samples)

    # Get bounds from trials
    param_bounds = {}
    for param_name in param_names:
        values = [t.params[param_name] for t in trials]
        param_bounds[param_name] = (min(values), max(values))

    # Scale samples to bounds
    samples = np.zeros_like(samples_unit)
    for i, param in enumerate(param_names):
        low, high = param_bounds[param]
        samples[:, i] = low + samples_unit[:, i] * (high - low)

    # Predict with GP
    samples_scaled = scaler.transform(samples)
    means, stds = gp.predict(samples_scaled, return_std=True)

    # Top K by posterior mean
    top_indices = np.argsort(means)[-k:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        config = {name: samples[idx, i] for i, name in enumerate(param_names)}
        config['ntests'] = 32
        config['optimized'] = False
        config = normalize_config(config)  # Round integers
        candidates.append((f'gp_mean_rank_{rank}', config))
        print(f"  {rank:2d}. GP Mean: {means[idx]:8.2f} ± {stds[idx]:6.2f} | {config}")

    print()

    # 3. Top K by GP lower confidence bound (mean - 2*std)
    print(f"Top {k} by GP Lower Confidence Bound (mean - 2*std):")
    lcb = means - 2 * stds
    top_indices = np.argsort(lcb)[-k:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        config = {name: samples[idx, i] for i, name in enumerate(param_names)}
        config['ntests'] = 32
        config['optimized'] = False
        config = normalize_config(config)  # Round integers
        candidates.append((f'gp_lcb_rank_{rank}', config))
        print(f"  {rank:2d}. LCB: {lcb[idx]:8.2f} (mean: {means[idx]:8.2f}, std: {stds[idx]:6.2f})")

    print()

    return candidates


def config_to_key(config: Dict) -> str:
    """Convert config to hashable key for deduplication."""
    # Round floats to avoid tiny differences
    key_parts = []
    for k in sorted(config.keys()):
        v = config[k]
        if isinstance(v, float):
            key_parts.append(f"{k}={v:.6f}")
        else:
            key_parts.append(f"{k}={v}")
    return "|".join(key_parts)


def deduplicate_candidates(
    candidates: List[Tuple[str, Dict]]
) -> List[Tuple[List[str], Dict]]:
    """
    Deduplicate candidates, keeping track of all criteria that selected each.

    Returns:
        List of (criteria_list, config) tuples
    """
    config_map = {}  # key -> (criteria_list, config)

    for criterion, config in candidates:
        key = config_to_key(config)
        if key not in config_map:
            config_map[key] = ([criterion], config)
        else:
            config_map[key][0].append(criterion)

    # Sort by number of criteria (most popular first)
    deduplicated = sorted(config_map.values(), key=lambda x: len(x[0]), reverse=True)

    return deduplicated


def validate_candidate(
    name: str,
    config: Dict,
    scenarios: List[Scenario],
    samples_per_scenario: int,
    rng: np.random.Generator,
) -> Dict:
    """
    Validate a single candidate with high sample counts.

    Returns:
        {
            'name': str,
            'config': Dict,
            'scenario_results': {scenario_name: [objectives]},
            'weighted_mean': float,
            'std_error': float,
            'confidence_interval': (lower, upper),
        }
    """
    print(f"\nValidating: {name}")
    print(f"  Config: {config}")
    print(f"  Samples per scenario: {samples_per_scenario}")

    scenario_results = {}

    for scenario in scenarios:
        print(f"    {scenario.name:20s} ", end='', flush=True)
        objectives = []

        for i in range(samples_per_scenario):
            if i % 5 == 0 and i > 0:
                print('.', end='', flush=True)

            try:
                metrics = run_simulation(config, scenario, rng)
                objective = compute_objective(metrics)
                objectives.append(objective)
            except Exception as e:
                print(f"\n      Error on sample {i}: {e}")
                objectives.append(-1e6)

        scenario_mean = np.mean(objectives)
        scenario_std = np.std(objectives)
        print(f" mean: {scenario_mean:8.2f} ± {scenario_std:6.2f}")

        scenario_results[scenario.name] = objectives

    # Compute weighted statistics
    scenario_means = [np.mean(scenario_results[s.name]) for s in scenarios]
    scenario_weights = [s.weight for s in scenarios]

    weighted_mean = sum(m * w for m, w in zip(scenario_means, scenario_weights))
    std_error = np.std(scenario_means) / np.sqrt(len(scenarios))

    ci_lower = weighted_mean - 2 * std_error
    ci_upper = weighted_mean + 2 * std_error

    print(f"  → Weighted Mean: {weighted_mean:8.2f} ± {std_error:6.2f}")
    print(f"     95% CI: [{ci_lower:8.2f}, {ci_upper:8.2f}]")

    return {
        'name': name,
        'config': config,
        'scenario_results': scenario_results,
        'weighted_mean': weighted_mean,
        'std_error': std_error,
        'confidence_interval': (ci_lower, ci_upper),
    }


def compare_results(results: List[Dict]):
    """Compare all validated configurations."""
    print(f"\n{'='*80}")
    print("EMPIRICAL VALIDATION RESULTS")
    print(f"{'='*80}\n")

    # Sort by weighted mean
    sorted_results = sorted(results, key=lambda x: x['weighted_mean'], reverse=True)

    print("Rank | Name                           | Weighted Mean | 95% CI      | vs Best")
    print("-----|--------------------------------|---------------|-------------|----------")

    best_mean = sorted_results[0]['weighted_mean']

    for rank, result in enumerate(sorted_results, 1):
        name = result['name']
        mean = result['weighted_mean']
        stderr = result['std_error']
        ci_low, ci_high = result['confidence_interval']

        improvement_pct = ((mean - best_mean) / abs(best_mean) * 100) if mean != best_mean else 0.0

        print(f"{rank:4d} | {name:30s} | {mean:13.2f} | ±{stderr:6.2f}    | {improvement_pct:+7.2f}%")

    print()

    # Statistical significance tests
    print("Pairwise Statistical Significance (Welch's t-test):")
    print()

    from scipy import stats

    for i in range(min(5, len(sorted_results))):
        for j in range(i+1, min(5, len(sorted_results))):
            result1 = sorted_results[i]
            result2 = sorted_results[j]

            # Get all samples
            samples1 = []
            samples2 = []
            for scenario in SCENARIOS:
                samples1.extend(result1['scenario_results'][scenario.name])
                samples2.extend(result2['scenario_results'][scenario.name])

            t_stat, p_value = stats.ttest_ind(samples1, samples2, equal_var=False)

            mean1 = result1['weighted_mean']
            mean2 = result2['weighted_mean']
            diff = mean1 - mean2

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

            print(f"  #{i+1} vs #{j+1}: diff={diff:8.2f}, p={p_value:.4f} {sig}")

    print()
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Empirical validation of top candidates')
    parser.add_argument('study_path', help='Path to Optuna study pickle file')
    parser.add_argument('--samples', type=int, default=50,
                       help='Samples per scenario for validation (default: 50)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top candidates per criterion (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, help='Output JSON file for results')

    args = parser.parse_args()

    # Load study
    print(f"Loading study from {args.study_path}...")
    with open(args.study_path, 'rb') as f:
        study = pickle.load(f)

    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    print(f"Loaded {len(trials)} completed trials")
    print(f"Best observed value: {study.best_value:.2f}")
    print()

    # Fit GP
    print("Fitting Gaussian Process to trial data...")
    X = []
    y = []
    param_names = sorted(trials[0].params.keys())

    for trial in trials:
        X.append([trial.params[name] for name in param_names])
        y.append(trial.value)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True,
    )

    gp.fit(X_scaled, y)
    print(f"GP fitted. Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.2f}")
    print()

    # Extract candidates
    candidates = extract_top_candidates(study, gp, scaler, param_names, k=args.top_k)

    # Deduplicate
    deduplicated = deduplicate_candidates(candidates)

    print(f"{'='*80}")
    print(f"Found {len(deduplicated)} unique candidates (from {len(candidates)} total)")
    print(f"{'='*80}\n")

    for i, (criteria, config) in enumerate(deduplicated[:10], 1):
        print(f"{i:2d}. Selected by {len(criteria)} criteria: {', '.join(criteria[:3])}")
        if len(criteria) > 3:
            print(f"    (and {len(criteria)-3} more)")

    print()

    # Validate each candidate
    rng = np.random.default_rng(args.seed)
    results = []

    start_time = time.time()

    for criteria, config in deduplicated:
        name = f"{criteria[0]}" + (f" (+{len(criteria)-1})" if len(criteria) > 1 else "")
        result = validate_candidate(name, config, SCENARIOS, args.samples, rng)
        results.append(result)

    elapsed = time.time() - start_time

    # Compare results
    compare_results(results)

    print(f"{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Candidates validated: {len(results)}")
    print(f"Samples per candidate: {args.samples * len(SCENARIOS)}")
    print(f"Total simulations: {len(results) * args.samples * len(SCENARIOS)}")
    print()

    # Save results
    if args.output:
        import json
        output_data = {
            'candidates': [
                {
                    'name': r['name'],
                    'config': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                              for k, v in r['config'].items()},
                    'weighted_mean': float(r['weighted_mean']),
                    'std_error': float(r['std_error']),
                    'confidence_interval': [float(r['confidence_interval'][0]),
                                           float(r['confidence_interval'][1])],
                }
                for r in results
            ],
            'best_candidate': {
                'name': results[0]['name'],
                'config': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                          for k, v in results[0]['config'].items()},
                'weighted_mean': float(results[0]['weighted_mean']),
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
