# Submit Queue Hyperparameter Optimization Design

## Executive Summary

This document outlines a design for gradient-free optimization of submit queue hyperparameters. The system will use Bayesian optimization with adaptive sampling to find optimal configurations that maximize developer productivity while minimizing wasted debug time.

### Key Design Principles

1. **Resources & FlakeTolerance are "Cheat Codes"**
   - Both always improve velocity when adjusted permissively
   - **CRITICAL**: Must include strong cost terms in objective:
     - Resource cost (α) prevents infinite resource allocation
     - Test quality cost (γ) prevents demoting all tests
   - Without these, optimizer will find trivial "solutions"

2. **Pathological Configurations Exist**
   - Some configs cause queue collapse (e.g., Resources=1)
   - These timeout or take extremely long to simulate
   - **Strategy**: Detect and assign MAX_COST penalty
   - Prevents wasting compute on hopeless configurations

3. **Optimize Beyond Exposed Parameters**
   - **Level 1**: Explicit parameters (Resources, MaxK, etc.)
   - **Level 2**: Implicit parameters (hardcoded constants in code)
   - **Level 3**: Algorithmic strategies (formulas and approaches)
   - Example: Don't just tune `KDivisor`, consider optimizing the entire `K(N, history)` function
   - Start with Level 1, expand to Level 2, then explore Level 3

4. **Phased Optimization Strategy**
   - Phase 1: Optimize non-resource params (Resources fixed, UseOptimizedMatrix=false)
   - Phase 2: Add Resources with proper cost weighting
   - Phase 3: Add implicit parameters
   - Phase 4: Re-optimize with UseOptimizedMatrix=true
   - Phase 5: Explore algorithmic alternatives

5. **Stochastic Evaluation with Adaptive Sampling**
   - Each config evaluation is noisy (random simulation)
   - Use adaptive sampling: more samples when uncertain
   - Balance exploration (few samples) vs exploitation (many samples)

## Current System Analysis

### Hyperparameters to Optimize

Based on the current implementation, we have several tunable parameters at different levels:

#### Level 1: Explicit Parameters (Already Exposed)

1. **Resources** (`N`): Number of parallel test batches
   - Currently: `4 * traffic` (hardcoded multiplier)
   - ⚠️ **WARNING**: This is a "cheat code" - more resources always improve KPIs except cost
   - **CRITICAL**: Must have strong resource cost in objective to prevent unbounded growth
   - Trade-off: Faster throughput but higher infrastructure cost

2. **MaxMinibatchSize**: Maximum number of CLs to test together
   - Currently: 2048 (fixed)
   - Trade-off: Larger batches = higher throughput but more suspects on failure

3. **MaxK**: Maximum sparsity for matrix assignment
   - Currently: 12 (fixed)
   - Trade-off: Higher K = better culprit isolation but more test runs per CL

4. **KDivisor**: Divisor for calculating K from N
   - Currently: 5 (fixed)
   - ⚠️ **NOTE**: This is an artifact of the simple formula `K = min(MaxK, N/KDivisor)`
   - Consider optimizing the *formula itself* rather than just this parameter

5. **FlakeTolerance**: Threshold for demoting flaky tests
   - Currently: 0.05, 0.10, 0.15 (swept)
   - ⚠️ **WARNING**: Lower values always improve velocity at the cost of test quality
   - **CRITICAL**: Must have test quality KPI to prevent setting this to 0
   - Trade-off: Higher tolerance = fewer active tests = faster but riskier

6. **UseOptimizedMatrix**: Whether to use matrix optimization
   - Currently: true (fixed)
   - Note: Optimized matrices are strictly better but increase simulation cost
   - **Strategy**: Optimize other params with this=false, then re-tune with this=true

#### Level 2: Implicit Parameters (Hardcoded in Code)

These are constants embedded in the code that are actually hyperparameters:

7. **VerificationLatency**: Time to verify a suspect (line 656: `currentTick + 2`)
   - Currently: 2 ticks (hardcoded)
   - Trade-off: Longer verification = more accurate but slower recovery

8. **FixDelay**: Time to fix a culprit (line 675: `currentTick + 60`)
   - Currently: 60 ticks (hardcoded)
   - Trade-off: Models developer fix time

9. **VerificationResourceMultiplier**: Resource budget for verification (line 654: `sq.ResourceBudget*16`)
   - Currently: 16x the main resource budget (hardcoded)
   - Trade-off: More verification resources = faster suspect clearing

10. **BackpressureThresholds**: Queue sizes that trigger backpressure (lines 1230-1237)
    - Currently: [200, 400, 800] with divisors [2, 4, 8]
    - Trade-off: When and how aggressively to apply backpressure

11. **DynamicResourceScaling**: Formula for N from queue size (line 729: `N = limit/2`)
    - Currently: N scales with queue size
    - Trade-off: How much to scale resources with load

12. **FlakeFixProbability**: Probability a flake auto-fixes (line 924: `1 - passRate^(n/84)`)
    - Currently: Magic number 84 controls fix rate
    - Trade-off: How quickly flakes naturally resolve

#### Level 3: Algorithmic Strategies (Code Structure)

These are even deeper - optimizing which *algorithm* to use:

13. **K Selection Strategy**: Currently uses `K = min(MaxK, N/KDivisor)`
    - Alternative: `K(N, historical_success_rate)` - vary K based on recent success
    - Alternative: Per-CL K based on expected failure probability
    - Alternative: Adaptive K that increases when suspects are found

14. **Batch Assignment Strategy**: Currently uses random sparse matrix
    - Alternative: Assign high-risk CLs to more batches
    - Alternative: Group CLs by author/area for better isolation
    - Alternative: Temporal batching (CLs from same time window)

15. **Verification Strategy**: Currently verifies all suspects individually
    - Alternative: Binary search among suspects
    - Alternative: Batch suspects in smaller groups first
    - Alternative: Risk-based prioritization

### Recommendation: Phased Optimization

Given the complexity, optimize in phases:

**Phase 1**: Optimize Level 1 parameters (except Resources initially)
- Fix Resources at reasonable value (e.g., 4*traffic)
- Fix UseOptimizedMatrix = false (for speed)
- Optimize: MaxBatch, MaxK, KDivisor, FlakeTolerance
- ~50-100 evaluations

**Phase 2**: Add Resources with proper cost weighting
- Include Resources in optimization space
- Ensure resource cost weight is calibrated
- ~100-200 evaluations

**Phase 3**: Optimize Level 2 implicit parameters
- Add verification latency, fix delay, etc.
- ~100-200 evaluations

**Phase 4**: Re-optimize with UseOptimizedMatrix = true
- Use Phase 3 results as starting point
- Fine-tune all parameters
- ~50-100 evaluations

**Phase 5** (Future): Explore algorithmic strategies
- Requires code changes to support alternatives
- A/B test different strategies

### Current Metrics

The simulation tracks numerous metrics, but we need to synthesize them into actionable objectives:

**Throughput Metrics:**
- `Slowdown`: Ratio of ideal to actual throughput (lower is better)
- `AvgSubmitTime`: Average time for a CL to land (lower is better)
- `AvgQueueSize`: Average queue depth (lower is better)

**Quality Metrics:**
- `FalseNegativeRate`: Culprits that escape detection (lower is better)
- `TruePositiveRate`: Accuracy of culprit detection (higher is better)
- `VictimRate`: Innocent CLs wrongly flagged (lower is better)

**Cost Metrics:**
- `AvgRunsPerSubmitted`: Test runs per CL (lower is better, but affects detection)
- `Resources * BatchUtilization`: Actual resource usage
- `ActiveTests`: Number of tests running (affects cost)

**Latency Metrics:**
- `WaitTimeP50/P95/P99`: Distribution of wait times
- `MaxQueueDepth`: Peak queue depth (indicates instability)

## Optimization Objectives

### Option 1: Single Composite Metric (Recommended for Initial Implementation)

Define a **Developer Productivity Score** that combines velocity and quality:

```
DeveloperProductivity =
    ThroughputScore
  - WastedDebugCost
  - ResourceCost
  - TestQualityCost
  - LatencyPenalty

where:
  ThroughputScore = 1000 / Slowdown

  WastedDebugCost =
      InnocentFlagged * VerificationCost
    + CulpritsEscaped * EscapedCulpritCost

  ResourceCost = α * (Resources * BatchUtilization)

  TestQualityCost = γ * (NTests - ActiveTests) * LostCoverageValue

  LatencyPenalty = β * (WaitTimeP95 / TargetWaitTime)^2
```

**Tunable weights:**
- `VerificationCost`: Time cost of verifying a suspect (e.g., 2 hours)
- `EscapedCulpritCost`: Cost of a culprit escaping (e.g., 20 hours for debugging)
- `α`: **CRITICAL** - Cost per unit of resource (must be large enough to prevent infinite resources)
  - Example: If 1 resource-tick costs $0.01 and developer time is $50/hr, then α ≈ 0.02-0.05
  - **Calibration**: Run with α=0 and see resource usage, then set α to make resources cost-neutral
- `γ`: Cost of reduced test coverage
  - Example: If each demoted test represents 1% loss in bug detection, γ ≈ 50-100
  - **CRITICAL**: Without this, FlakeTolerance will be set to 0 (maximum velocity, zero quality)
- `β`: Weight for latency penalty
- `TargetWaitTime`: Desired P95 wait time
- `LostCoverageValue`: Value of each test in terms of developer time saved catching bugs

**Critical Constraints:**

1. **Resource Cost Must Be Substantial**: Without proper α, optimizer will max out resources
   - Start with α = 0.05 (resources are expensive)
   - Tune based on what resource budget you actually want

2. **Test Quality Must Be Protected**: Without γ, FlakeTolerance → 0
   - Start with γ = 100 (each lost test is valuable)
   - Balance against velocity gains

**Advantages:**
- Single scalar objective simplifies optimization
- Weights encode business priorities
- Easy to explain and tune
- Can be optimized with standard algorithms

**Disadvantages:**
- Requires choosing weights upfront
- May miss interesting trade-offs
- Weight calibration is critical

### Option 2: Multi-Objective (Pareto Front)

Optimize for 2-3 independent objectives:

1. **Primary Objective: Developer Hours Saved**
   ```
   DevHoursSaved =
       (IdealThroughput - ActualThroughput) * AvgDevTime
     + InnocentFlagged * VerificationTime
     + CulpritsEscaped * DebugTime
   ```
   (minimize)

2. **Secondary Objective: Resource Cost**
   ```
   ResourceCost = Resources * BatchUtilization * CostPerUnit
   ```
   (minimize)

3. **Tertiary Objective: P95 Latency**
   ```
   WaitTimeP95
   ```
   (minimize)

**Advantages:**
- Reveals trade-off frontier
- No need to choose weights upfront
- User can select from Pareto-optimal solutions

**Disadvantages:**
- More complex to implement
- Requires Pareto-aware optimizer (NSGA-II, MOEA/D)
- Harder to automate without human judgment

### Recommendation

Start with **Option 1** (single composite metric) because:
1. Simpler to implement and debug
2. Standard optimizers work well
3. Can iterate on weights based on results
4. Can switch to multi-objective later if needed

## Optimization Algorithm

### Choice: Bayesian Optimization with Gaussian Processes

**Why Bayesian Optimization:**
1. **Sample Efficient**: Critical since each evaluation takes ~seconds
2. **Handles Noise**: Naturally models stochastic objectives
3. **No Gradients**: Works with black-box simulations
4. **Adaptive**: Balances exploration vs. exploitation
5. **Uncertainty Quantification**: Knows when to sample more

**Implementation Options:**
1. **BoTorch** (PyTorch-based, recommended)
   - State-of-the-art implementation
   - Good support for noisy objectives
   - Can handle mixed continuous/discrete parameters
   - Built-in support for constraints

2. **Scikit-Optimize**
   - Simpler, pure Python
   - Good for prototyping
   - Less flexible for advanced features

3. **Optuna** (Alternative)
   - Easy to use
   - Good for mixed parameter types
   - Built-in pruning for failed runs
   - Strong visualization tools

**Recommended:** Start with **Optuna** for speed of implementation, migrate to **BoTorch** if we need advanced features.

### Handling Stochastic Objectives

The simulation output is noisy due to randomness. We need to handle this:

#### Adaptive Sampling Strategy

```python
def evaluate_with_adaptive_sampling(config, min_samples=3, max_samples=50):
    """
    Adaptively determine number of samples based on uncertainty.

    Strategy:
    1. Start with min_samples
    2. Compute mean and standard error
    3. If relative uncertainty < threshold, stop
    4. Otherwise, add more samples
    5. Cap at max_samples
    """
    samples = []
    for i in range(min_samples):
        result = run_simulation(config)
        samples.append(result.metric)

    while len(samples) < max_samples:
        mean = np.mean(samples)
        std_error = np.std(samples) / np.sqrt(len(samples))
        relative_uncertainty = std_error / (abs(mean) + 1e-6)

        if relative_uncertainty < target_uncertainty:
            break

        # Add more samples
        result = run_simulation(config)
        samples.append(result.metric)

    return np.mean(samples), np.std(samples) / np.sqrt(len(samples))
```

**Key Parameters:**
- `min_samples`: Start with 3-5 for quick evaluation
- `max_samples`: Cap at 20-50 to limit cost
- `target_uncertainty`: Aim for 5-10% relative uncertainty
- Early in optimization: Use fewer samples (explore broadly)
- Near optimum: Use more samples (exploit precisely)

#### Alternative: Expected Improvement with Noise

Use acquisition functions designed for noisy objectives:
- **Noisy Expected Improvement**: Accounts for observation noise
- **Confidence Bound**: UCB/LCB with noise term
- **Knowledge Gradient**: Explicitly models value of information

### Handling Pathological Configurations (Queue Collapse)

Some configurations will cause the queue to grow unboundedly, leading to:
1. **Extremely long simulation times** (each step processes more and more CLs)
2. **Timeouts** (simulation exceeds reasonable time budget)
3. **Memory issues** (queue grows to millions of items)

**Example Pathological Config:**
- `Resources = 1` (severely under-resourced)
- `FlakeTolerance = 0.30` (many tests demoted)
- Result: Queue grows indefinitely, no CLs land

**Strategy: Treat as Maximum Cost**

```python
MAX_COST = -1e6  # Very negative objective value

def evaluate_single_run(config, timeout_seconds=60):
    """
    Run simulation with timeout protection.
    Returns objective value or MAX_COST if failed.
    """
    try:
        # Run with timeout
        result = run_simulation_with_timeout(config, timeout_seconds)

        # Check for queue collapse indicators
        if result.avg_queue_size > 1000:
            # Queue is exploding - pathological config
            return MAX_COST

        if result.slowdown > 100:
            # Throughput collapsed
            return MAX_COST

        if result.max_queue_depth > 5000:
            # Hit pathological growth
            return MAX_COST

        # Compute normal objective
        return compute_objective(result)

    except TimeoutError:
        # Simulation took too long - pathological config
        print(f"⚠️  Configuration timed out: {config}")
        return MAX_COST

    except Exception as e:
        # Other failures (OOM, etc.)
        print(f"❌ Configuration failed: {config}, error: {e}")
        return MAX_COST
```

**Benefits:**
1. Optimizer learns to avoid pathological regions
2. No wasted compute on hopeless configs
3. Exploration continues in feasible space

**Implementation Notes:**
- Use `signal.alarm()` or `subprocess.run(timeout=...)` for timeouts
- Set timeout based on typical simulation time (e.g., 3x median)
- Log all timeout/failure cases for analysis
- Consider adding "soft" penalties before hard MAX_COST:
  ```python
  if result.avg_queue_size > 500:
      penalty = (result.avg_queue_size - 500) * 10
      objective -= penalty
  ```

## Parameter Space Definition

### Continuous Parameters

```python
parameter_space = {
    # Resource allocation
    'resource_multiplier': {
        'type': 'float',
        'range': [1.0, 10.0],
        'default': 4.0,
        'description': 'Multiplier for resources (N = multiplier * traffic)'
    },

    # Batch configuration
    'max_batch_size': {
        'type': 'int',
        'range': [256, 4096],
        'default': 2048,
        'log_scale': True,  # Search on log scale
    },

    # Matrix sparsity
    'max_k': {
        'type': 'int',
        'range': [4, 32],
        'default': 12,
    },

    'k_divisor': {
        'type': 'float',
        'range': [2.0, 10.0],
        'default': 5.0,
    },

    # Test health
    'flake_tolerance': {
        'type': 'float',
        'range': [0.01, 0.30],
        'default': 0.10,
    },
}
```

### Discrete/Categorical Parameters

```python
categorical_parameters = {
    'use_optimized_matrix': {
        'type': 'categorical',
        'choices': [True, False],
        'default': True,
    }
}
```

### Conditional Parameters

Some parameters may only be relevant if others are set:
```python
# Only optimize matrix if use_optimized_matrix = True
conditional_params = {
    'matrix_optimization_iterations': {
        'type': 'int',
        'range': [1, 100],
        'condition': lambda cfg: cfg['use_optimized_matrix'],
    }
}
```

### Constraints

Define constraints to avoid invalid/wasteful configurations:

```python
constraints = [
    # K should not exceed N
    lambda cfg: cfg['max_k'] <= cfg['resource_multiplier'] * traffic,

    # MaxBatch should be reasonable vs resources
    lambda cfg: cfg['max_batch_size'] >= cfg['resource_multiplier'] * traffic,

    # Don't over-resource low traffic scenarios
    lambda cfg: cfg['resource_multiplier'] <= 2.0 or traffic > 2,
]
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**1.1 Objective Function Module** (`objective.go`)
- Implement composite metric calculation
- Add configurable weights
- Unit tests for metric computation

**1.2 Optimizer Interface** (`optimizer.py`)
- Python wrapper around Go simulation
- CGo bindings OR subprocess interface
- Result parsing and metric extraction

**1.3 Basic Optimizer** (`bayesian_optimizer.py`)
- Optuna integration
- Parameter space definition
- Basic adaptive sampling

### Phase 2: Optimization Driver (Week 1-2)

**2.1 CLI Interface** (`main.go` additions)
```bash
# Run optimization
go run . --optimize \
  --objective "developer_productivity" \
  --budget 1000 \
  --weights "verification_cost=2,escaped_cost=20,resource_alpha=0.1"

# Continue optimization from checkpoint
go run . --optimize \
  --resume checkpoint.json \
  --budget +500

# Run with continuous reporting
go run . --optimize \
  --continuous \
  --report-interval 5m
```

**2.2 Progress Tracking**
- Checkpointing best configs
- Real-time visualization of optimization progress
- Logging all evaluations for analysis

**2.3 Result Reporting**
```
Optimization Summary:
  Best Configuration Found (iteration 145):
    Resources: 6.2 * traffic
    MaxBatch: 3072
    MaxK: 18
    KDivisor: 4.2
    FlakeTolerance: 0.08

  Metrics:
    Developer Productivity: 847.3 ± 12.1
    Slowdown: 1.08 ± 0.03
    WaitTimeP95: 24 ticks
    FalseNegativeRate: 0.02%

  Improvement over baseline:
    +23.4% productivity
    -15% resource usage
    -30% P95 latency
```

### Phase 3: Advanced Features (Week 2-3)

**3.1 Multi-Objective Optimization**
- Implement Pareto front tracking
- Visualization of trade-off curves
- Interactive selection of preferred point

**3.2 Contextual Optimization**
- Optimize separately for different traffic patterns
- Detect regime changes
- Recommend different configs for different scenarios

**3.3 Sensitivity Analysis**
- Which parameters matter most?
- Robustness to parameter perturbations
- Recommended ranges for manual tuning

**3.4 Transfer Learning**
- Use results from one traffic level to warm-start another
- Multi-task Bayesian optimization
- Faster convergence

## Driver API Design

### Go Interface

```go
// optimizer.go
type OptimizerConfig struct {
    Objective        string          // "developer_productivity" | "pareto"
    Budget           int             // Max evaluations
    ContinuousMode   bool            // Run continuously
    ReportInterval   time.Duration   // How often to report progress
    MinSamples       int             // Min samples per config
    MaxSamples       int             // Max samples per config
    TargetUncertainty float64        // Stop sampling when uncertainty < this
    Checkpoint       string          // Path to save/load state
    Weights          map[string]float64 // Objective function weights
}

type OptimizerResult struct {
    BestConfig       SimConfig
    BestMetric       float64
    BestMetricStdErr float64
    AllEvaluations   []Evaluation
    OptimizationTime time.Duration
    TotalEvaluations int
}

type Evaluation struct {
    Config       SimConfig
    Metric       float64
    StdErr       float64
    NumSamples   int
    Timestamp    time.Time
}

func RunOptimization(config OptimizerConfig) OptimizerResult {
    // Main optimization loop
}
```

### Python Optimizer

```python
# bayesian_optimizer.py
import optuna
from typing import Dict, Callable
import numpy as np

class SubmitQueueOptimizer:
    def __init__(
        self,
        objective_fn: Callable,
        param_space: Dict,
        constraints: List[Callable] = None,
        min_samples: int = 3,
        max_samples: int = 30,
        target_uncertainty: float = 0.05,
    ):
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.constraints = constraints or []
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.target_uncertainty = target_uncertainty

    def suggest_config(self, trial: optuna.Trial) -> Dict:
        """Sample configuration from parameter space."""
        config = {}
        for name, spec in self.param_space.items():
            if spec['type'] == 'float':
                if spec.get('log_scale'):
                    config[name] = trial.suggest_float(
                        name, spec['range'][0], spec['range'][1], log=True
                    )
                else:
                    config[name] = trial.suggest_float(
                        name, spec['range'][0], spec['range'][1]
                    )
            elif spec['type'] == 'int':
                config[name] = trial.suggest_int(
                    name, spec['range'][0], spec['range'][1]
                )
            elif spec['type'] == 'categorical':
                config[name] = trial.suggest_categorical(
                    name, spec['choices']
                )
        return config

    def evaluate_with_adaptive_sampling(
        self,
        config: Dict
    ) -> Tuple[float, float]:
        """
        Evaluate configuration with adaptive sampling.
        Returns: (mean_metric, std_error)
        """
        samples = []

        # Initial samples
        for _ in range(self.min_samples):
            result = self.objective_fn(config)
            samples.append(result)

        # Adaptive sampling
        while len(samples) < self.max_samples:
            mean = np.mean(samples)
            std_err = np.std(samples, ddof=1) / np.sqrt(len(samples))

            # Check if uncertainty is acceptable
            rel_uncertainty = std_err / (abs(mean) + 1e-6)
            if rel_uncertainty < self.target_uncertainty:
                break

            # Add another sample
            result = self.objective_fn(config)
            samples.append(result)

        return np.mean(samples), np.std(samples, ddof=1) / np.sqrt(len(samples))

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        config = self.suggest_config(trial)

        # Check constraints
        for constraint in self.constraints:
            if not constraint(config):
                raise optuna.TrialPruned()

        # Evaluate with adaptive sampling
        mean_metric, std_err = self.evaluate_with_adaptive_sampling(config)

        # Store standard error for later analysis
        trial.set_user_attr('std_err', std_err)
        trial.set_user_attr('num_samples', len(samples))

        return mean_metric

    def optimize(
        self,
        n_trials: int,
        checkpoint_path: str = None,
    ) -> optuna.Study:
        """Run optimization."""
        # Create study with appropriate sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random exploration
            multivariate=True,     # Model parameter interactions
        )

        study = optuna.create_study(
            direction='maximize',  # Maximize developer productivity
            sampler=sampler,
            study_name='submit_queue_optimization',
            storage=f'sqlite:///{checkpoint_path}' if checkpoint_path else None,
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                self._progress_callback,
            ],
        )

        return study

    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback for progress reporting."""
        if trial.number % 10 == 0:
            best_value = study.best_value
            best_params = study.best_params
            print(f"\nTrial {trial.number}")
            print(f"  Best value: {best_value:.2f}")
            print(f"  Best params: {best_params}")
```

### Integration Layer

```python
# go_interface.py
import subprocess
import json
import os
from typing import Dict

class GoSimulationRunner:
    """Interface to Go simulation."""

    def __init__(self, binary_path: str = "./submit_queue"):
        self.binary_path = binary_path

    def run_simulation(self, config: Dict) -> Dict:
        """
        Run Go simulation with given config.
        Returns metrics dictionary.
        """
        # Convert config to command-line arguments
        args = [
            self.binary_path,
            "--json-mode",  # Output results as JSON
            "--resources", str(int(config['resource_multiplier'] * config['traffic'])),
            "--max-batch", str(config['max_batch_size']),
            "--max-k", str(config['max_k']),
            "--k-divisor", str(config['k_divisor']),
            "--flake-tolerance", str(config['flake_tolerance']),
            "--use-optimized-matrix", str(config['use_optimized_matrix']),
            "--traffic", str(config['traffic']),
            "--n-tests", str(config['n_tests']),
            "--seed", str(config.get('seed', 0)),
        ]

        # Run simulation
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")

        # Parse JSON output
        return json.loads(result.stdout)

    def compute_objective(
        self,
        config: Dict,
        weights: Dict[str, float],
    ) -> float:
        """
        Run simulation and compute objective function.
        """
        metrics = self.run_simulation(config)

        # Compute composite metric
        throughput_score = 1000.0 / max(metrics['slowdown'], 0.1)

        wasted_debug_cost = (
            metrics['innocent_flagged'] * weights['verification_cost'] +
            (metrics['culprits_created'] - metrics['culprits_caught']) *
                weights['escaped_culprit_cost']
        )

        resource_cost = (
            weights['resource_alpha'] *
            metrics['resources'] *
            metrics['batch_utilization']
        )

        target_wait = weights.get('target_wait_time', 20.0)
        latency_penalty = (
            weights['latency_beta'] *
            (metrics['wait_time_p95'] / target_wait) ** 2
        )

        objective = (
            throughput_score
            - wasted_debug_cost
            - resource_cost
            - latency_penalty
        )

        return objective
```

## Continuous Optimization Mode

For long-running optimization with periodic reporting:

```python
# continuous_optimizer.py
import time
from collections import deque
from datetime import datetime, timedelta

class ContinuousOptimizer:
    """
    Runs optimization continuously with progress reporting.
    """

    def __init__(
        self,
        optimizer: SubmitQueueOptimizer,
        report_interval: timedelta = timedelta(minutes=5),
    ):
        self.optimizer = optimizer
        self.report_interval = report_interval
        self.start_time = None
        self.last_report_time = None
        self.recent_improvements = deque(maxlen=20)

    def run(self, checkpoint_path: str = 'optimization.db'):
        """Run optimization continuously."""
        self.start_time = datetime.now()
        self.last_report_time = self.start_time

        print(f"Starting continuous optimization at {self.start_time}")
        print(f"Results will be saved to {checkpoint_path}")
        print(f"Progress reports every {self.report_interval}")
        print("\nPress Ctrl+C to stop gracefully\n")

        trial_number = 0
        try:
            while True:
                # Run batch of trials
                study = self.optimizer.optimize(
                    n_trials=trial_number + 10,
                    checkpoint_path=checkpoint_path,
                )
                trial_number += 10

                # Check if it's time to report
                now = datetime.now()
                if now - self.last_report_time >= self.report_interval:
                    self._report_progress(study, now)
                    self.last_report_time = now

        except KeyboardInterrupt:
            print("\n\nOptimization stopped by user")
            self._final_report(study)

    def _report_progress(self, study: optuna.Study, now: datetime):
        """Print progress report."""
        elapsed = now - self.start_time

        best_value = study.best_value
        best_params = study.best_params
        n_trials = len(study.trials)

        # Calculate improvement rate
        recent_trials = study.trials[-20:]
        if len(recent_trials) > 1:
            improvements = [
                t.value for t in recent_trials
                if t.value and t.value > study.best_value * 0.95
            ]
            improvement_rate = len(improvements) / len(recent_trials)
        else:
            improvement_rate = 0.0

        print(f"\n{'='*80}")
        print(f"Progress Report - {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Elapsed Time: {elapsed}")
        print(f"Trials Completed: {n_trials}")
        print(f"Trials per Minute: {n_trials / (elapsed.total_seconds() / 60):.1f}")
        print(f"\nBest Configuration:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.3f}" if isinstance(value, float) else f"  {param}: {value}")
        print(f"\nBest Objective Value: {best_value:.2f}")
        print(f"Recent Improvement Rate: {improvement_rate*100:.1f}%")

        # Recommendation on whether to continue
        if improvement_rate < 0.05:
            print("\n⚠️  Few recent improvements - consider stopping soon")
        else:
            print("\n✓  Still finding improvements - continue running")
        print(f"{'='*80}\n")

    def _final_report(self, study: optuna.Study):
        """Print final optimization summary."""
        print(f"\n{'='*80}")
        print("Final Optimization Summary")
        print(f"{'='*80}")

        best_trial = study.best_trial
        print(f"Best Trial: {best_trial.number}")
        print(f"Best Value: {best_trial.value:.2f}")
        print(f"Best Parameters:")
        for param, value in best_trial.params.items():
            print(f"  {param}: {value}")

        if 'std_err' in best_trial.user_attrs:
            print(f"\nStandard Error: {best_trial.user_attrs['std_err']:.2f}")
            print(f"Samples Used: {best_trial.user_attrs['num_samples']}")

        print(f"\nTotal Trials: {len(study.trials)}")
        print(f"Total Time: {datetime.now() - self.start_time}")
        print(f"{'='*80}\n")
```

## Testing & Validation

### 1. Sanity Checks
- Verify optimizer finds known good configurations
- Test with synthetic objective (known optimum)
- Ensure constraints are respected

### 2. Ablation Studies
- Test with single parameter optimization
- Verify improvement over random search
- Compare with grid search on small space

### 3. Robustness
- Test with different random seeds
- Verify results are reproducible
- Check sensitivity to weights

### 4. Performance
- Benchmark evaluation time
- Optimize simulation speed (profiling)
- Parallelize independent trials

## Open Questions & Future Work

1. **Traffic-Dependent Optimization**
   - Should we optimize separately for each traffic level?
   - Can we learn a function: `optimal_params(traffic)`?

2. **Online Optimization**
   - Can we adapt parameters in real-time based on live metrics?
   - How to detect when re-optimization is needed?

3. **Multi-Fidelity Optimization**
   - Use cheaper approximations (fewer samples, shorter simulation) for exploration
   - Full evaluation only for promising candidates

4. **Ensemble Models**
   - Train regression model to predict metrics from config
   - Use for fast "what-if" analysis
   - Identify parameter importance

5. **Cost-Aware Optimization**
   - Resource costs may change over time
   - How to optimize for financial costs vs. developer time?

## Success Metrics

The optimization system should achieve:

1. **Efficiency**: Find near-optimal solution in <200 evaluations
2. **Reliability**: Results reproducible within 5% across runs
3. **Speed**: Each evaluation completes in <30 seconds
4. **Improvement**: >15% better than current hardcoded values
5. **Interpretability**: Clear explanation of why configuration is optimal

## Timeline

- **Week 1**: Core infrastructure + basic Bayesian optimization
- **Week 2**: Full driver implementation + adaptive sampling
- **Week 3**: Advanced features + documentation
- **Week 4**: Testing, validation, and tuning

## Conclusion

This design provides a comprehensive framework for optimizing submit queue hyperparameters. The key innovations are:

1. **Composite metric** that captures developer productivity holistically
2. **Adaptive sampling** that efficiently handles stochastic objectives
3. **Bayesian optimization** for sample-efficient search
4. **Continuous mode** for long-running optimization with progress tracking
5. **Extensibility** to multi-objective optimization if needed

The system will enable data-driven configuration of the submit queue, potentially improving developer productivity by 15-30% while reducing resource costs.
