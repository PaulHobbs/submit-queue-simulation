# Adaptive CI/CD Submit Queue Simulation

A high-performance simulation framework for testing sophisticated "Submit Queue" (also known as Merge Queue or Commit Queue) systems. This project models the advanced queue algorithms used in hyperscale software development environments (Google, Meta, GitHub) and employs **Sparse Bernoulli Group-Based Testing** for resilient culprit identification in the presence of flaky tests.

## Quick Start

```bash
# Build the project
go build submit_queue.go

# Run a basic simulation with default parameters
go run submit_queue.go

# Run with weighted scoring (accounts for test reliability)
go run submit_queue.go -weighted-scoring=true

# Run with real build history data
python3 generate_build_data.py  # Generate synthetic build history
go run submit_queue.go -csv build_history.csv

# Optimize hyperparameters using Bayesian optimization
python3 optimizer.py
```

## Project Overview

This project simulates a production CI/CD submit queue system with:
- **Real-world failure modeling** using hierarchical culprit detection
- **Sparse Bernoulli group-based testing** for efficient batch testing
- **Flaky test handling** with optional weighted scoring
- **Support for real build history** via CSV mode
- **Comprehensive metrics** tracking queue health, performance, and culprit detection accuracy
- **Hyperparameter optimization** tools using Bayesian methods

## Key Features

### 1. Hierarchical Culprit Modeling

Realistic multi-level failure modeling:
- **CL-Level Decision:** ~3% probability that a Change List is "bad"
- **Test-Level Conditional:** Bad CLs determine which specific tests they break
- **Realistic Scaling:** Adding more tests increases failure resolution without linearly increasing bad CLs
- **Diminishing Returns:** Models the natural plateau in test effectiveness

### 2. Sparse Bernoulli Group-Based Testing

Core algorithm using Random k-set design:
- **Encoding:** Each CL assigned to exactly `k` random minibatches (weight k)
- **Parallel Batches:** T parallel minibatches scale logarithmically with queue depth (T ≈ 10 × log₁₀(N))
- **Suspicion Scoring:** CL suspicion = ratio of weighted failing batches to total k batches
- **Threshold Detection:** CLs with suspicion > 0.75 identified as culprits and rejected

**Benefits:**
- Resilient to flaky tests (single flake affects only k batches)
- Efficient culprit identification with multiple failures
- Score-based robust decoding mechanism
- Sparsity prevents "poisoning" of too many tests

### 3. Weighted Scoring for Flaky Tests

Optional reliability-aware scoring:
- **Standard Mode:** Equal weight to all test failures
- **Weighted Mode:** Failures in stable tests contribute more to suspicion scores than flaky tests
- **Adaptive:** Reduces false positives from unreliable tests
- **Configurable:** Flake tolerance threshold controls test demotion

### 4. Real Build History Support (CSV Mode)

Load and simulate with actual build data:
- Real timestamps and change metadata
- Multiple build targets per change
- Time-bucketed simulation for realistic traffic patterns
- 26,000+ synthetic build history records included
- Customizable test scenarios (normal, spike, crisis, flaky, etc.)

## Configuration Parameters

### Core Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resources` | 74 | Number of parallel batch slots available |
| `maxbatch` | 684 | Maximum minibatch size |
| `maxk` | 12 | Maximum sparsity weight (CLs per minibatch) |
| `kdiv` | 5 | K divisor for dynamic weight calculation |
| `flaketol` | 0.0767 | Flake tolerance threshold (demote tests above this) |

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `traffic` | 100 | Average CLs per hour |
| `ntests` | 100 | Number of tests in the suite |
| `samples` | 1000 | Number of simulation samples/runs |
| `csv` | - | Load real data from CSV file (CSV mode) |
| `weighted-scoring` | false | Enable reliability-weighted scoring |

## Usage Examples

### Standard Simulation (Synthetic Data)

```bash
# Run with defaults
go run submit_queue.go

# Run with custom parameters
go run submit_queue.go -resources=80 -maxbatch=700 -maxk=10 -traffic=150 -ntests=200

# Run with weighted scoring
go run submit_queue.go -weighted-scoring=true -traffic=100

# Run high-sample count for stable metrics
go run submit_queue.go -samples=5000
```

### CSV Mode (Real Build History)

```bash
# Generate synthetic build history with realistic data
python3 generate_build_data.py
# Creates: build_history.csv (26k+ records)

# Run simulation with real data
go run submit_queue.go -csv build_history.csv

# Run with custom parameters
go run submit_queue.go -csv build_history.csv -resources=80 -maxk=10 -flaketol=0.1
```

### Hyperparameter Optimization

The project includes multiple optimization approaches:

```bash
# Bayesian optimization using Optuna (fastest)
python3 optimizer.py
# Finds near-optimal config in ~50 iterations
# Output: optimization_results.txt, level2_study.pkl

# Gaussian Process refinement (more thorough)
python3 optimizer_gp.py
# Fits GP model to exploration space
# Computes posterior means and confidence bounds

# Multi-scenario robust optimization (realistic)
python3 optimizer_robust.py
# Tests across 8 scenarios: normal, spike, crisis, flaky, etc.
# Weighted average of metrics
# Output: robust_results.txt

# Empirical validation with high samples (rigorous)
python3 empirical_validation.py
# Validates top candidates with 1000+ samples each
# Computes confidence intervals
# Identifies robust configurations
```

### Analysis Tools

```bash
# Sensitivity analysis - identify important parameters
python3 analyze_sensitivity.py

# Compare optimization across levels/phases
python3 compare_levels.py

# Validate configurations under stress
python3 validate_configs.py
```

## Output Metrics

The simulation tracks comprehensive metrics across several categories:

### Culprit Detection Metrics
- **Culprits Created:** Total culprits (bad CLs) introduced
- **Culprits Caught:** True positives (actual culprits detected)
- **Innocents Flagged:** False positives (innocent CLs rejected)
- **Escape Rate:** Culprits that made it through undetected

### Performance Metrics
- **Slowdown:** Relative throughput vs. ideal (1.0 = perfect)
- **Queue Depth:** Average and maximum queue size
- **Pass Rate:** Percentage of minibatches that passed all tests
- **Victim Rate:** Percentage of verified CLs that actually passed
- **Runs per CL:** Average test runs per change
- **Time to Submission:** Average ticks for successful CL submission

### Queue Health
- **Max Queue Depth:** Highest number of pending CLs
- **Max Verify Queue:** Highest verification load
- **Batch Utilization:** Resource efficiency percentage

### Test Health
- **Active Tests:** Tests not demoted due to flakiness
- **Demoted Tests:** Tests exceeding flake tolerance
- **Test Stability:** Overall suite reliability

## Project Structure

```
submit-queue-simulation/
├── submit_queue.go                  # Main simulation engine (2,337 lines)
├── go.mod                           # Go module definition
├── README.md                        # This file
│
├── optimizer.py                     # Bayesian optimization (Optuna)
├── optimizer_gp.py                  # Gaussian Process refinement
├── optimizer_robust.py              # Multi-scenario optimization
├── empirical_validation.py          # High-sample validation
├── analyze_sensitivity.py           # Parameter sensitivity analysis
├── compare_levels.py                # Compare optimization phases
├── generate_build_data.py           # Generate CSV build history
├── validate_configs.py              # Configuration stress testing
│
├── build_history.csv                # 26,500+ realistic records
├── test_data.csv                    # Small test dataset
│
├── CSV_MODE_README.md               # CSV mode usage guide
├── CSV_IMPLEMENTATION_SUMMARY.md    # CSV implementation details
├── OPTIMIZER.md                     # Optimization framework design
├── OPTIMIZATIONS.md                 # Performance optimizations
│
├── out/                             # Output directory
│   ├── simulation_output.txt        # Standard results
│   ├── output_weighted.txt          # Weighted scoring results
│   └── output_unweighted.txt        # Unweighted results
│
├── .venv/                           # Python virtual environment
└── .git/                            # Git repository
```

## Technology Stack

### Core Simulation
- **Go 1.25.4** - High-performance simulation engine
- **Standard Library Only** - No external dependencies
- **Features:**
  - XORShift+ RNG for fast random number generation
  - Bitset-based sparse matrix for efficient group testing
  - State machine for change tracking
  - Hierarchical failure modeling

### Analysis & Optimization
- **Python 3.12** - Data analysis and optimization
- **Key Libraries:**
  - **Optuna 4.6.0** - Bayesian hyperparameter optimization
  - **scikit-learn** - Gaussian Process regression
  - **NumPy 2.3.5** - Numerical computing
  - **SciPy 1.16.3** - Scientific algorithms
  - Standard: pandas, json, csv, pickle, subprocess

### Performance
- **PGO Profiling:** CPU profiling data included (cpu.prof, cpu2.prof)
- **Matrix Optimization:** Cache-aware bitset quantization (~5% granularity)
- **Parallel Processing:** Python optimization tools support parallel evaluation

## Building & Setup

### Prerequisites
- Go 1.18+ (for building)
- Python 3.10+ (for analysis/optimization)

### Build
```bash
# Compile the Go simulation
go build submit_queue.go
# Creates: submit_queue (3.3MB binary)

# Or run directly
go run submit_queue.go
```

### Python Setup
```bash
# Virtual environment already configured in .venv/
source .venv/bin/activate

# Or install dependencies manually
pip install optuna numpy scipy scikit-learn pandas
```

## Algorithm Details

### Sparse Matrix Encoding
The core algorithm uses a Random k-set design:
1. Each CL assigned to k random minibatches (out of T total)
2. T batches run in parallel, scaled by queue depth
3. Test failures recorded per batch
4. Suspicion score: ratio of failing batches to k

### State Machine (CL Lifecycle)
```
Queued → InBatch → Suspect → Verifying → [Fixing] → Submitted
                      ↓
                  (if culprit)
                  [verification]
                      ↓
                   (if confirmed)
                  [fix applied]
```

### Scoring Mechanism
**Standard:** `Suspicion = (failing k's) / k`
**Weighted:** `Suspicion = Σ(weight[test] × failure[test]) / k`

Where `weight[test]` is inverse of test's historical flake rate.

## Recent Improvements

- **CSV Mode:** Real build history support with multiple targets
- **Weighted Scoring:** Reliability-aware culprit detection
- **Multi-level Optimization:** Exposed parameters + implicit hardcoded tuning
- **Robust Optimization:** Multi-scenario evaluation for realistic conditions
- **Matrix Optimization:** Bitset-based efficiency improvements
- **Parallel Processing:** Concurrent sample collection in Python tools

## Documentation

- **CSV_MODE_README.md** - Comprehensive CSV mode usage guide
- **OPTIMIZER.md** - Detailed optimization framework design
- **OPTIMIZATIONS.md** - Performance optimization techniques
- **CORRECTED_LEVEL2_RESULTS.md** - Multi-level optimization results

## Contributing

For bug reports or improvements, please refer to the project repository issues.

## References

This project implements algorithms and concepts from research into:
- Group testing and fault diagnosis
- Sparse matrices and efficient encoding
- CI/CD pipeline optimization
- Flaky test handling in continuous integration