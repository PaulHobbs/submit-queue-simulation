#!/usr/bin/env python3
"""
Hyperparameter optimizer for submit queue simulation.

Uses Bayesian optimization (Optuna) with adaptive sampling to find
optimal configurations that maximize developer productivity.
"""

import subprocess
import json
import time
import numpy as np
import optuna
from typing import Dict, Tuple, Optional
import argparse
import sys

# Objective function weights (tunable based on business priorities)
DEFAULT_WEIGHTS = {
    'verification_cost': 2.0,          # Hours to verify a suspect
    'escaped_culprit_cost': 20.0,      # Hours to debug an escaped culprit
    'resource_alpha': 0.05,            # Cost per resource-tick
    'test_quality_gamma': 100.0,       # Cost per demoted test
    'latency_beta': 1.0,               # Weight for latency penalty
    'target_wait_time': 20.0,          # Target P95 wait time
}

# Constants for pathological detection
MAX_COST = -1e6
TIMEOUT_SECONDS = 60
MAX_QUEUE_SIZE = 1000
MAX_SLOWDOWN = 100


class SimulationRunner:
    """Runs Go simulation and parses results."""

    def __init__(self, binary_path='./submit_queue', csv_file=None):
        self.binary_path = binary_path
        self.csv_file = csv_file
        self.run_count = 0
        self.timeout_count = 0

    def run(self, config: Dict, seed: Optional[int] = None) -> Optional[Dict]:
        """
        Run simulation with given configuration.

        Returns:
            Dict with metrics, or None if failed/timeout
        """
        if seed is None:
            seed = int(time.time() * 1000000) + self.run_count

        self.run_count += 1

        args = [
            self.binary_path,
            '-json',
        ]

        # Add CSV file if provided
        if self.csv_file:
            args.extend(['-csv', self.csv_file])

        args.extend([
            '-resources', str(config['resources']),
            '-traffic', str(config['traffic']),
            '-ntests', str(config['ntests']),
            '-maxbatch', str(config['maxbatch']),
            '-maxk', str(config['maxk']),
            '-kdiv', str(config['kdiv']),
            '-flaketol', str(config['flaketol']),
            '-optimized', str(config.get('optimized', False)).lower(),
            '-seed', str(seed),
        ])

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                print(f"⚠️  Simulation failed: {result.stderr}", file=sys.stderr)
                return None

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            self.timeout_count += 1
            print(f"⏱️  Simulation timeout (#{self.timeout_count}): {config}", file=sys.stderr)
            return None

        except Exception as e:
            print(f"❌ Error running simulation: {e}", file=sys.stderr)
            return None


class ObjectiveFunction:
    """Computes the developer productivity objective."""

    def __init__(self, weights: Dict = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def compute(self, metrics: Dict) -> float:
        """
        Compute developer productivity score.

        Higher is better.
        """
        # Throughput score (higher is better)
        slowdown = max(metrics['slowdown'], 0.1)  # Avoid division by zero
        throughput_score = 1000.0 / slowdown

        # Wasted debug cost (lower is better)
        innocent_flagged = metrics['innocent_flagged']
        culprits_escaped = max(0, metrics['culprits_created'] - metrics['culprits_caught'])

        wasted_debug_cost = (
            innocent_flagged * self.weights['verification_cost'] +
            culprits_escaped * self.weights['escaped_culprit_cost']
        )

        # Resource cost (lower is better)
        resource_cost = (
            self.weights['resource_alpha'] *
            metrics['config']['Resources'] *
            metrics['batch_utilization']
        )

        # Test quality cost (lower is better)
        ntests = metrics['config']['NTests']
        active_tests = metrics['active_tests']
        demoted_tests = ntests - active_tests
        test_quality_cost = self.weights['test_quality_gamma'] * demoted_tests

        # Latency penalty (lower is better)
        wait_p95 = metrics['wait_time_p95']
        target_wait = self.weights['target_wait_time']
        latency_penalty = self.weights['latency_beta'] * (wait_p95 / target_wait) ** 2

        # Combine into single score
        objective = (
            throughput_score
            - wasted_debug_cost
            - resource_cost
            - test_quality_cost
            - latency_penalty
        )

        return objective

    def check_pathological(self, metrics: Dict) -> bool:
        """Check if configuration is pathological (queue collapse)."""
        if metrics['avg_queue_size'] > MAX_QUEUE_SIZE:
            return True
        if metrics['slowdown'] > MAX_SLOWDOWN:
            return True
        if metrics['max_queue_depth'] > 5000:
            return True
        return False


class AdaptiveSampler:
    """Adaptively samples configurations based on uncertainty."""

    def __init__(
        self,
        runner: SimulationRunner,
        objective_fn: ObjectiveFunction,
        min_samples: int = 3,
        max_samples: int = 20,
        target_uncertainty: float = 0.05,
    ):
        self.runner = runner
        self.objective_fn = objective_fn
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.target_uncertainty = target_uncertainty

    def evaluate(self, config: Dict) -> Tuple[float, float, int]:
        """
        Evaluate configuration with adaptive sampling.

        Returns:
            (mean_objective, std_error, num_samples)
        """
        samples = []

        # Initial samples
        for i in range(self.min_samples):
            metrics = self.runner.run(config)

            if metrics is None:
                # Failed or timeout - pathological config
                return MAX_COST, 0.0, i + 1

            if self.objective_fn.check_pathological(metrics):
                # Pathological config detected
                return MAX_COST, 0.0, i + 1

            objective = self.objective_fn.compute(metrics)
            samples.append(objective)

        # Adaptive sampling
        while len(samples) < self.max_samples:
            mean = np.mean(samples)
            std_err = np.std(samples, ddof=1) / np.sqrt(len(samples))

            # Check if uncertainty is acceptable
            rel_uncertainty = std_err / (abs(mean) + 1e-6)
            if rel_uncertainty < self.target_uncertainty:
                break

            # Add another sample
            metrics = self.runner.run(config)

            if metrics is None:
                # Use current best estimate
                break

            if self.objective_fn.check_pathological(metrics):
                # Found pathological - return max cost
                return MAX_COST, std_err, len(samples)

            objective = self.objective_fn.compute(metrics)
            samples.append(objective)

        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / np.sqrt(len(samples)) if len(samples) > 1 else 0

        return mean, std_err, len(samples)


class Optimizer:
    """Bayesian optimizer for submit queue hyperparameters."""

    def __init__(
        self,
        sampler: AdaptiveSampler,
        fixed_params: Dict = None,
    ):
        self.sampler = sampler
        self.fixed_params = fixed_params or {}
        self.best_value = -np.inf
        self.best_config = None

    def suggest_config(self, trial: optuna.Trial) -> Dict:
        """Sample configuration from parameter space."""
        config = {
            'traffic': 8,  # Fixed for now
            'ntests': 32,  # Fixed for now
            'optimized': False,  # Phase 1: optimize without matrix optimization
        }

        # Add fixed parameters
        config.update(self.fixed_params)

        # Phase 1 parameters to optimize
        config['resources'] = trial.suggest_int('resources', 16, 128, log=True)
        config['maxbatch'] = trial.suggest_int('maxbatch', 512, 4096, log=True)
        config['maxk'] = trial.suggest_int('maxk', 4, 24)
        config['kdiv'] = trial.suggest_int('kdiv', 2, 10)
        config['flaketol'] = trial.suggest_float('flaketol', 0.01, 0.25)

        return config

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        config = self.suggest_config(trial)

        # Evaluate with adaptive sampling
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
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True,
            ),
        )

        print(f"Starting optimization with {n_trials} trials...")
        print(f"Fixed params: {self.fixed_params}")
        print(f"Weights: {self.sampler.objective_fn.weights}")
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
            print(f"   Timeouts: {self.sampler.runner.timeout_count}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Optimize submit queue hyperparameters')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--min-samples', type=int, default=3, help='Min samples per config')
    parser.add_argument('--max-samples', type=int, default=20, help='Max samples per config')
    parser.add_argument('--binary', type=str, default='./submit_queue', help='Path to simulation binary')
    parser.add_argument('--fix-resources', type=int, help='Fix resources instead of optimizing')
    parser.add_argument('--csv', type=str, help='CSV file with test history')

    args = parser.parse_args()

    # Setup
    runner = SimulationRunner(args.binary, csv_file=args.csv)
    objective_fn = ObjectiveFunction()
    sampler = AdaptiveSampler(
        runner,
        objective_fn,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
    )

    fixed_params = {}
    if args.fix_resources:
        fixed_params['resources'] = args.fix_resources

    optimizer = Optimizer(sampler, fixed_params)

    # Run optimization
    start_time = time.time()
    study = optimizer.optimize(n_trials=args.trials)
    elapsed = time.time() - start_time

    # Print final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Total trials: {len(study.trials)}")
    print(f"Timeouts: {runner.timeout_count}")
    print(f"\nBest objective: {study.best_value:.2f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    if study.best_trial.user_attrs:
        print(f"\nBest trial metadata:")
        print(f"  Std error: {study.best_trial.user_attrs['std_err']:.2f}")
        print(f"  Samples: {study.best_trial.user_attrs['num_samples']}")
        print(f"  Full config: {study.best_trial.user_attrs['config']}")

    print("="*80)


if __name__ == '__main__':
    main()
