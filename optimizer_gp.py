#!/usr/bin/env python3
"""
GP-based hyperparameter optimizer with multi-scenario evaluation.

Uses Gaussian Process sampler for Bayesian optimization to ensure the search
strategy is aligned with the posterior distribution used for final selection.
Tests configurations across different scenarios for robustness.
"""

import subprocess
import json
import time
import numpy as np
import optuna
from typing import Dict, Tuple, Optional, List
import argparse
import sys
from dataclasses import dataclass

# Create RNG for jitter
_rng = np.random.default_rng()

# Objective function weights
DEFAULT_WEIGHTS = {
    'verification_cost': 2.0,
    'escaped_culprit_cost': 20.0,
    'resource_alpha': 0.05,
    'test_quality_gamma': 100.0,
    'latency_beta': 1.0,
    'target_wait_time': 20.0,
}

MAX_COST = -1e6
TIMEOUT_SECONDS = 90


@dataclass
class Scenario:
    """Defines a test scenario with specific conditions."""
    name: str
    weight: float  # Importance weight (sum should be 1.0)
    traffic: int
    culprit_prob: float
    test_stability: float
    bp_thresholds: Tuple[int, int, int]  # Backpressure thresholds (environmental parameter)
    bp_jitter: float = 0.25  # Jitter fraction for backpressure (±25% by default)

    def describe(self):
        return f"{self.name} (traffic={self.traffic}, culprits={self.culprit_prob:.3f}, stability={self.test_stability:.2f}, bp={self.bp_thresholds})"

    def sample_bp_thresholds(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        """Sample backpressure thresholds with jitter."""
        if self.bp_jitter == 0:
            return self.bp_thresholds

        jittered = []
        for threshold in self.bp_thresholds:
            # Add uniform jitter ±bp_jitter fraction
            low = int(threshold * (1 - self.bp_jitter))
            high = int(threshold * (1 + self.bp_jitter))
            jittered.append(rng.integers(low, high + 1))

        return tuple(jittered)


# Define scenarios for robust optimization
# Now includes explicit developer behavior variations (backpressure thresholds)
SCENARIOS = [
    # Normal operation with standard developer behavior - most common
    Scenario('normal-std', weight=0.25, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(200, 400, 800)),  # Standard developer response

    # Normal operation with aggressive developers - submit despite queue
    Scenario('normal-aggressive', weight=0.10, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(300, 600, 1200)),  # Less deterred by queues

    # Normal operation with conservative developers - very queue-sensitive
    Scenario('normal-conservative', weight=0.05, traffic=8, culprit_prob=0.03, test_stability=1.0,
             bp_thresholds=(100, 200, 400)),  # More deterred by queues

    # Traffic spike with standard behavior
    Scenario('spike-std', weight=0.20, traffic=16, culprit_prob=0.04, test_stability=0.98,
             bp_thresholds=(200, 400, 800)),

    # Traffic spike with conservative developers (queue grows faster)
    Scenario('spike-conservative', weight=0.10, traffic=16, culprit_prob=0.04, test_stability=0.98,
             bp_thresholds=(100, 200, 400)),

    # Low traffic - common during off-hours
    Scenario('low', weight=0.10, traffic=4, culprit_prob=0.02, test_stability=0.99,
             bp_thresholds=(200, 400, 800)),

    # Flaky tests with standard behavior - occasional problem
    Scenario('flaky', weight=0.10, traffic=8, culprit_prob=0.03, test_stability=0.90,
             bp_thresholds=(200, 400, 800)),

    # Crisis mode - rare but critical
    Scenario('crisis', weight=0.10, traffic=20, culprit_prob=0.08, test_stability=0.92,
             bp_thresholds=(200, 400, 800)),
]


class SimulationRunner:
    """Runs Go simulation with scenario parameters."""

    def __init__(self, binary_path='./submit_queue'):
        self.binary_path = binary_path
        self.run_count = 0
        self.timeout_count = 0

    def run(self, config: Dict, scenario: Scenario, seed: Optional[int] = None) -> Optional[Dict]:
        """Run simulation with given configuration and scenario."""
        if seed is None:
            seed = int(time.time() * 1000000) + self.run_count

        self.run_count += 1

        # Sample backpressure thresholds with jitter (environmental parameter)
        bp1, bp2, bp3 = scenario.sample_bp_thresholds(_rng)

        args = [
            self.binary_path,
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
            '-seed', str(seed),
            # Backpressure thresholds (environmental, not optimized)
            '-bp-threshold-1', str(bp1),
            '-bp-threshold-2', str(bp2),
            '-bp-threshold-3', str(bp3),
        ]

        # Add Level 2 design parameters if present (controllable)
        if 'verify_latency' in config:
            args.extend(['-verify-latency', str(config['verify_latency'])])
        if 'fix_delay' in config:
            args.extend(['-fix-delay', str(config['fix_delay'])])
        if 'verify_resource_mult' in config:
            args.extend(['-verify-resource-mult', str(config['verify_resource_mult'])])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                return None

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            self.timeout_count += 1
            return None

        except Exception as e:
            return None


class ObjectiveFunction:
    """Computes developer productivity across scenarios."""

    def __init__(self, weights: Dict = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def compute(self, metrics: Dict) -> float:
        """Compute developer productivity score."""
        slowdown = max(metrics['slowdown'], 0.1)
        throughput_score = 1000.0 / slowdown

        innocent_flagged = metrics['innocent_flagged']
        culprits_escaped = max(0, metrics['culprits_created'] - metrics['culprits_caught'])

        wasted_debug_cost = (
            innocent_flagged * self.weights['verification_cost'] +
            culprits_escaped * self.weights['escaped_culprit_cost']
        )

        resource_cost = (
            self.weights['resource_alpha'] *
            metrics['config']['Resources'] *
            metrics['batch_utilization']
        )

        ntests = metrics['config']['NTests']
        active_tests = metrics['active_tests']
        demoted_tests = ntests - active_tests
        test_quality_cost = self.weights['test_quality_gamma'] * demoted_tests

        wait_p95 = metrics['wait_time_p95']
        target_wait = self.weights['target_wait_time']
        latency_penalty = self.weights['latency_beta'] * (wait_p95 / target_wait) ** 2

        objective = (
            throughput_score
            - wasted_debug_cost
            - resource_cost
            - test_quality_cost
            - latency_penalty
        )

        return objective

    def check_pathological(self, metrics: Dict) -> bool:
        """Check if configuration is pathological."""
        if metrics['avg_queue_size'] > 1000:
            return True
        if metrics['slowdown'] > 100:
            return True
        if metrics['max_queue_depth'] > 5000:
            return True
        return False


class MultiScenarioSampler:
    """Evaluates configurations across multiple scenarios."""

    def __init__(
        self,
        runner: SimulationRunner,
        objective_fn: ObjectiveFunction,
        scenarios: List[Scenario],
        samples_per_scenario: int = 3,
        progressive_fidelity: bool = True,
    ):
        self.runner = runner
        self.objective_fn = objective_fn
        self.scenarios = scenarios
        self.samples_per_scenario = samples_per_scenario
        self.progressive_fidelity = progressive_fidelity
        self.trial_count = 0

    def evaluate(self, config: Dict) -> Tuple[float, float, int]:
        """
        Evaluate configuration across scenarios.

        Returns:
            (weighted_mean_objective, std_error, total_samples)
        """
        self.trial_count += 1

        # Progressive fidelity: use fewer scenarios early in optimization
        if self.progressive_fidelity:
            if self.trial_count < 10:
                # First 10 trials: just normal-std scenario
                active_scenarios = [s for s in self.scenarios if s.name == 'normal-std']
            elif self.trial_count < 25:
                # Next 15 trials: all normal variants + spike-std
                active_scenarios = [s for s in self.scenarios if 'normal' in s.name or s.name == 'spike-std']
            else:
                # After trial 25: all scenarios (8 scenarios with developer behavior variations)
                active_scenarios = self.scenarios
        else:
            active_scenarios = self.scenarios

        # Normalize weights for active scenarios
        total_weight = sum(s.weight for s in active_scenarios)

        scenario_objectives = []
        scenario_weights = []
        total_samples = 0

        for scenario in active_scenarios:
            # Sample this scenario
            samples = []
            for _ in range(self.samples_per_scenario):
                metrics = self.runner.run(config, scenario)

                if metrics is None:
                    # Failed or timeout
                    return MAX_COST, 0.0, total_samples + 1

                if self.objective_fn.check_pathological(metrics):
                    return MAX_COST, 0.0, total_samples + 1

                objective = self.objective_fn.compute(metrics)
                samples.append(objective)
                total_samples += 1

            scenario_mean = np.mean(samples)
            scenario_objectives.append(scenario_mean)
            scenario_weights.append(scenario.weight / total_weight)

        # Compute weighted average across scenarios
        weighted_mean = sum(obj * w for obj, w in zip(scenario_objectives, scenario_weights))

        # Standard error estimate (rough approximation)
        std_err = np.std(scenario_objectives) / np.sqrt(len(scenario_objectives))

        return weighted_mean, std_err, total_samples


class Optimizer:
    """Bayesian optimizer with multi-scenario robustness."""

    def __init__(
        self,
        sampler: MultiScenarioSampler,
        fixed_params: Dict = None,
    ):
        self.sampler = sampler
        self.fixed_params = fixed_params or {}
        self.best_value = -np.inf
        self.best_config = None

    def suggest_config(self, trial: optuna.Trial) -> Dict:
        """Sample configuration from parameter space."""
        config = {
            'ntests': 32,
            'optimized': False,
        }

        config.update(self.fixed_params)

        # Level 1: Explicit system parameters (always optimized)
        config['resources'] = trial.suggest_int('resources', 16, 128, log=True)
        config['maxbatch'] = trial.suggest_int('maxbatch', 512, 4096, log=True)
        config['maxk'] = trial.suggest_int('maxk', 4, 24)
        config['kdiv'] = trial.suggest_int('kdiv', 2, 10)
        config['flaketol'] = trial.suggest_float('flaketol', 0.01, 0.25)

        # Level 2: Controllable design parameters (optimize these)
        config['verify_latency'] = trial.suggest_int('verify_latency', 1, 10)
        config['fix_delay'] = trial.suggest_int('fix_delay', 20, 120)
        config['verify_resource_mult'] = trial.suggest_int('verify_resource_mult', 8, 32)

        # NOTE: Backpressure thresholds are NOT optimized
        # They are environmental parameters (developer behavior) that we test robustness against
        # Values are sampled from scenarios with jitter in SimulationRunner.run()

        return config

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        config = self.suggest_config(trial)

        # Evaluate across scenarios
        mean_obj, std_err, num_samples = self.sampler.evaluate(config)

        # Store metadata
        trial.set_user_attr('std_err', std_err)
        trial.set_user_attr('num_samples', num_samples)
        trial.set_user_attr('config', config)

        # Track best
        if mean_obj > self.best_value:
            self.best_value = mean_obj
            self.best_config = config
            print(f"\n🎯 New best! Objective: {mean_obj:.2f} ± {std_err:.2f}")
            print(f"   Config: {config}")

        return mean_obj

    def optimize(self, n_trials: int = 100) -> optuna.Study:
        """Run optimization."""
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.GPSampler(
                n_startup_trials=15,
                deterministic_objective=False,  # Account for stochastic evaluations
                seed=42,
            ),
        )

        print("="*80)
        print("GP-BASED OPTIMIZATION - Multi-Scenario Evaluation")
        print("="*80)
        print("Using Gaussian Process sampler for Bayesian optimization")
        print("(ensures search and posterior selection are aligned)")
        print("="*80)
        print(f"\nScenarios:")
        for scenario in self.sampler.scenarios:
            print(f"  [{scenario.weight*100:5.1f}%] {scenario.describe()}")

        print(f"\nOptimization settings:")
        print(f"  Trials: {n_trials}")
        print(f"  Samples per scenario: {self.sampler.samples_per_scenario}")
        print(f"  Progressive fidelity: {self.sampler.progressive_fidelity}")
        print(f"  Fixed params: {self.fixed_params}")
        print()

        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[self.progress_callback],
        )

        return study

    def progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback for progress reporting."""
        if trial.number % 10 == 9:
            print(f"\n📊 Progress Report (Trial {trial.number + 1})")
            print(f"   Best value: {study.best_value:.2f}")
            print(f"   Best params: {study.best_params}")
            print(f"   Total simulations: {self.sampler.runner.run_count}")
            print(f"   Timeouts: {self.sampler.runner.timeout_count}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Robust multi-scenario optimization')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--samples-per-scenario', type=int, default=3, help='Samples per scenario')
    parser.add_argument('--no-progressive', action='store_true', help='Disable progressive fidelity')
    parser.add_argument('--binary', type=str, default='./submit_queue', help='Path to simulation binary')
    parser.add_argument('--save-study', type=str, default='study.pkl', help='Path to save study object')

    args = parser.parse_args()

    # Setup
    runner = SimulationRunner(args.binary)
    objective_fn = ObjectiveFunction()
    sampler = MultiScenarioSampler(
        runner,
        objective_fn,
        SCENARIOS,
        samples_per_scenario=args.samples_per_scenario,
        progressive_fidelity=not args.no_progressive,
    )

    optimizer = Optimizer(sampler)

    # Run optimization
    start_time = time.time()
    study = optimizer.optimize(n_trials=args.trials)
    elapsed = time.time() - start_time

    # Print final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Total trials: {len(study.trials)}")
    print(f"Total simulations: {runner.run_count}")
    print(f"Timeouts: {runner.timeout_count}")
    print(f"\nBest robust objective: {study.best_value:.2f}")
    print(f"\nBest parameters (robust across scenarios):")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    if study.best_trial.user_attrs:
        print(f"\nBest trial metadata:")
        print(f"  Std error: {study.best_trial.user_attrs['std_err']:.2f}")
        print(f"  Total samples: {study.best_trial.user_attrs['num_samples']}")
        print(f"  Full config: {study.best_trial.user_attrs['config']}")

    print("="*80)

    # Save study for analysis
    if args.save_study:
        import pickle
        with open(args.save_study, 'wb') as f:
            pickle.dump(study, f)
        print(f"\nStudy saved to {args.save_study}")
        print("Run: python analyze_sensitivity.py {args.save_study}")


if __name__ == '__main__':
    main()
