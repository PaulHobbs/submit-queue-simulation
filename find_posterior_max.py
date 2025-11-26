#!/usr/bin/env python3
"""
Find the configuration with maximum posterior mean from Bayesian optimization.

Instead of picking the trial with best observed objective (which is biased by noise),
we fit a Gaussian Process to all trial data and find the configuration with the
highest expected (posterior mean) objective.
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import optuna


def load_study(study_path: str) -> optuna.Study:
    """Load Optuna study from pickle file."""
    with open(study_path, 'rb') as f:
        return pickle.load(f)


def extract_training_data(study: optuna.Study) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract parameter values and objectives from completed trials.

    Returns:
        X: (n_trials, n_params) array of parameter values
        y: (n_trials,) array of objective values
        param_names: list of parameter names in order
    """
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])

    if not trials:
        raise ValueError("No completed trials found")

    # Get parameter names (ordered)
    param_names = sorted(trials[0].params.keys())

    # Extract data
    X = []
    y = []

    for trial in trials:
        # Get parameter values in consistent order
        x = [trial.params[name] for name in param_names]
        X.append(x)
        y.append(trial.value)

    return np.array(X), np.array(y), param_names


def fit_gaussian_process(X: np.ndarray, y: np.ndarray) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """
    Fit Gaussian Process to trial data.

    Returns:
        gp: Fitted Gaussian Process
        scaler: StandardScaler for parameter values
    """
    # Normalize inputs for better GP fitting
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define kernel
    # Use Matern kernel (more flexible than RBF, good for hyperparameter optimization)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5  # Smoothness parameter
    )

    # Fit GP
    print("Fitting Gaussian Process to trial data...")
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,  # Noise level (assume some measurement noise)
        normalize_y=True,
    )

    gp.fit(X_scaled, y)

    print(f"GP fitted. Kernel: {gp.kernel_}")
    print(f"Log marginal likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.2f}")

    return gp, scaler


def find_posterior_maximum(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    param_names: List[str],
    param_bounds: Dict[str, Tuple[float, float]],
    n_random_starts: int = 100,
) -> Tuple[Dict, float, float]:
    """
    Find configuration with maximum posterior mean using global optimization.

    Returns:
        best_config: Dictionary of parameter values
        posterior_mean: Expected objective value
        posterior_std: Uncertainty in objective value
    """
    # Convert bounds to format for differential_evolution
    bounds = [param_bounds[name] for name in param_names]

    def objective(x):
        """Negative posterior mean (for minimization)."""
        x_scaled = scaler.transform(x.reshape(1, -1))
        mean, std = gp.predict(x_scaled, return_std=True)
        return -mean[0]  # Negative because we're minimizing

    print(f"\nSearching for posterior maximum...")
    print(f"Parameter space bounds:")
    for name, (low, high) in zip(param_names, bounds):
        print(f"  {name:25s} [{low}, {high}]")

    # Global optimization
    result = differential_evolution(
        objective,
        bounds,
        maxiter=1000,
        popsize=15,
        seed=42,
        workers=1,
        polish=True,  # Local refinement at the end
        atol=1e-3,
        tol=1e-3,
    )

    # Extract best configuration
    best_x = result.x
    best_x_scaled = scaler.transform(best_x.reshape(1, -1))
    posterior_mean, posterior_std = gp.predict(best_x_scaled, return_std=True)

    best_config = {name: value for name, value in zip(param_names, best_x)}

    return best_config, posterior_mean[0], posterior_std[0]


def compare_with_best_trial(
    study: optuna.Study,
    posterior_config: Dict,
    posterior_mean: float,
    posterior_std: float,
):
    """Compare posterior maximum with best observed trial."""
    print(f"\n{'='*80}")
    print("COMPARISON: Posterior Maximum vs Best Observed Trial")
    print(f"{'='*80}\n")

    # Best observed trial
    best_trial = study.best_trial
    best_observed_value = best_trial.value
    best_observed_params = best_trial.params

    print("Best Observed Trial (selection bias from noise):")
    print(f"  Objective: {best_observed_value:.2f}")
    print(f"  Parameters:")
    for param, value in sorted(best_observed_params.items()):
        if isinstance(value, float):
            print(f"    {param:25s} {value:.4f}")
        else:
            print(f"    {param:25s} {value}")

    print("\nPosterior Maximum (expected best, accounting for noise):")
    print(f"  Posterior Mean:  {posterior_mean:.2f}")
    print(f"  Posterior Std:   {posterior_std:.2f}")
    print(f"  95% CI:          [{posterior_mean - 2*posterior_std:.2f}, {posterior_mean + 2*posterior_std:.2f}]")
    print(f"  Parameters:")
    for param, value in sorted(posterior_config.items()):
        if isinstance(value, float):
            print(f"    {param:25s} {value:.4f}")
        else:
            print(f"    {param:25s} {value}")

    print("\nParameter Differences:")
    print(f"  {'Parameter':25s} {'Observed Best':>15s} {'Posterior Max':>15s} {'Difference':>15s}")
    print(f"  {'-'*75}")
    for param in sorted(posterior_config.keys()):
        obs_val = best_observed_params[param]
        post_val = posterior_config[param]

        if isinstance(obs_val, float):
            diff = post_val - obs_val
            diff_pct = (diff / obs_val * 100) if obs_val != 0 else 0
            print(f"  {param:25s} {obs_val:15.4f} {post_val:15.4f} {diff:+15.4f} ({diff_pct:+6.1f}%)")
        else:
            diff = post_val - obs_val
            diff_pct = (diff / obs_val * 100) if obs_val != 0 else 0
            print(f"  {param:25s} {obs_val:15d} {post_val:15.0f} {diff:+15.0f} ({diff_pct:+6.1f}%)")

    print(f"\n{'='*80}\n")


def analyze_posterior_landscape(
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    param_names: List[str],
    posterior_config: Dict,
    n_samples: int = 1000,
):
    """
    Sample the posterior to understand the landscape around the maximum.
    """
    print("Analyzing posterior landscape...")

    # Sample random configurations
    from scipy.stats import qmc

    # Use Latin Hypercube Sampling for better coverage
    sampler = qmc.LatinHypercube(d=len(param_names), seed=42)

    # Need to know bounds - use reasonable ranges based on posterior max
    # This is approximate
    samples_unit = sampler.random(n=n_samples)

    # Scale to reasonable range around posterior max (±50%)
    samples = np.zeros_like(samples_unit)
    for i, param in enumerate(param_names):
        center = posterior_config[param]
        # Use ±50% range around posterior max
        low = center * 0.5
        high = center * 1.5
        samples[:, i] = low + samples_unit[:, i] * (high - low)

    # Predict on samples
    samples_scaled = scaler.transform(samples)
    means, stds = gp.predict(samples_scaled, return_std=True)

    # Find best sampled points
    top_k = 10
    top_indices = np.argsort(means)[-top_k:][::-1]

    print(f"\nTop {top_k} sampled configurations (from {n_samples} random samples):")
    print(f"  Rank | Posterior Mean | Posterior Std |")
    print(f"  -----|----------------|---------------|")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:4d} | {means[idx]:14.2f} | {stds[idx]:13.2f} |")

    print(f"\nPosterior Maximum (from optimization):")
    post_mean = gp.predict(scaler.transform(np.array(list(posterior_config.values())).reshape(1, -1)))[0]
    print(f"  Posterior Mean: {post_mean:.2f}")

    print()


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Find posterior maximum from Bayesian optimization study'
    )
    parser.add_argument('study_path', help='Path to Optuna study pickle file')
    parser.add_argument('--n-random-starts', type=int, default=100,
                       help='Random starts for global optimization')
    parser.add_argument('--output', type=str, help='Output file for posterior config (JSON)')

    args = parser.parse_args()

    # Check for sklearn
    try:
        import sklearn
    except ImportError:
        print("Error: scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        sys.exit(1)

    # Load study
    print(f"Loading study from {args.study_path}...")
    study = load_study(args.study_path)
    print(f"Loaded {len(study.trials)} trials")
    print(f"Best observed value: {study.best_value:.2f}")
    print()

    # Extract training data
    X, y, param_names = extract_training_data(study)
    print(f"Training data: {X.shape[0]} trials, {X.shape[1]} parameters")
    print(f"Objective range: [{y.min():.2f}, {y.max():.2f}]")
    print()

    # Fit GP
    gp, scaler = fit_gaussian_process(X, y)
    print()

    # Define parameter bounds
    # Get from study's parameter distributions
    param_bounds = {}
    for param_name in param_names:
        # Get min/max from observed trials
        values = [trial.params[param_name] for trial in study.trials
                 if trial.state == optuna.trial.TrialState.COMPLETE]
        param_bounds[param_name] = (min(values), max(values))

    # Find posterior maximum
    posterior_config, posterior_mean, posterior_std = find_posterior_maximum(
        gp, scaler, param_names, param_bounds, args.n_random_starts
    )

    print(f"\n{'='*80}")
    print("POSTERIOR MAXIMUM FOUND")
    print(f"{'='*80}")
    print(f"Posterior Mean:  {posterior_mean:.2f}")
    print(f"Posterior Std:   {posterior_std:.2f}")
    print(f"95% CI:          [{posterior_mean - 2*posterior_std:.2f}, "
          f"{posterior_mean + 2*posterior_std:.2f}]")
    print()

    # Compare with best observed
    compare_with_best_trial(study, posterior_config, posterior_mean, posterior_std)

    # Analyze landscape
    analyze_posterior_landscape(gp, scaler, param_names, posterior_config)

    # Save posterior config if requested
    if args.output:
        import json
        output_data = {
            'posterior_config': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                               for k, v in posterior_config.items()},
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std),
            'best_observed_value': float(study.best_value),
            'best_observed_params': study.best_params,
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Posterior maximum saved to {args.output}")


if __name__ == '__main__':
    main()
