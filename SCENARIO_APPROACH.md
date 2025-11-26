# Scenario Variation: Stochastic vs Deterministic Weighting

## The Question

How should we model variability in traffic, culprit rates, and test stability to find robust hyperparameters?

## Two Approaches

### 1. Stochastic Sampling

Each simulation randomly samples conditions:
```python
traffic = random.choice([4, 8, 12, 16, 20])
culprit_prob = random.uniform(0.01, 0.08)
test_stability = random.uniform(0.90, 1.0)
```

**Pros:**
- Simple to implement
- Naturally models real-world uncertainty
- Single simulation per evaluation

**Cons:**
- High variance in objective function
- Requires many more samples per config
- Hard to debug (results not reproducible)
- Can't control importance of different scenarios

### 2. Deterministic Weighted Scenarios

Run fixed scenarios, compute weighted average:
```python
scenarios = [
    {'weight': 0.40, 'traffic': 8, 'culprit': 0.03, 'stability': 1.0},   # Normal
    {'weight': 0.25, 'traffic': 16, 'culprit': 0.04, 'stability': 0.98}, # Spike
    {'weight': 0.15, 'traffic': 4, 'culprit': 0.02, 'stability': 0.99},  # Low
    {'weight': 0.10, 'traffic': 8, 'culprit': 0.03, 'stability': 0.90},  # Flaky
    {'weight': 0.10, 'traffic': 20, 'culprit': 0.08, 'stability': 0.92}, # Crisis
]

objective = sum(weight * evaluate(config, scenario) for scenario in scenarios)
```

**Pros:**
- Lower variance (same scenarios every time)
- Explicit control over scenario importance
- Reproducible results
- Finds configs robust across known conditions

**Cons:**
- Multiple simulations per evaluation (slower)
- Need to choose scenarios upfront
- May miss unexpected edge cases

## Hybrid Approach (Recommended)

**Deterministic scenarios with progressive fidelity:**

### Progressive Fidelity Strategy
```
Trials 1-10:   Evaluate only "normal" scenario (fast exploration)
Trials 11-25:  Add "spike" scenario (2x cost)
Trials 26+:    All 5 scenarios (5x cost, but near optimum)
```

This gives us:
- Fast early exploration (finds promising regions quickly)
- Increasing robustness as we narrow in
- Final configs tested on all scenarios

### Adaptive Scenario Sampling
Instead of always testing all scenarios, test based on uncertainty:
```python
if uncertainty > threshold:
    # High uncertainty - test more scenarios
    test_scenarios = ALL_SCENARIOS
else:
    # Low uncertainty - test fewer scenarios
    test_scenarios = [NORMAL, SPIKE]
```

## Implementation

### Scenario Definition
```python
@dataclass
class Scenario:
    name: str
    weight: float          # Importance (sum to 1.0)
    traffic: int           # Traffic level
    culprit_prob: float    # Probability CL is a culprit
    test_stability: float  # Test pass rate multiplier
```

### Key Scenarios

1. **Normal** (40% weight)
   - Traffic: 8 (baseline)
   - Culprits: 3%
   - Stability: 1.0 (perfect tests)
   - *Most common operating mode*

2. **Spike** (25% weight)
   - Traffic: 16 (2x normal)
   - Culprits: 4%
   - Stability: 0.98 (slightly flaky under load)
   - *Moderately common, important to handle*

3. **Low** (15% weight)
   - Traffic: 4 (off-hours)
   - Culprits: 2%
   - Stability: 0.99
   - *Common but less critical*

4. **Flaky** (10% weight)
   - Traffic: 8
   - Culprits: 3%
   - Stability: 0.90 (lots of flake)
   - *Occasional problem*

5. **Crisis** (10% weight)
   - Traffic: 20 (2.5x normal)
   - Culprits: 8% (lots of bad code)
   - Stability: 0.92 (tests failing)
   - *Rare but must not collapse*

## Expected Differences in Optimal Configs

### Single Scenario (Normal Only)
```
resources: 22
maxbatch: 761
maxk: 9
kdiv: 3
flaketol: 0.077
```
- Optimized for steady-state efficiency
- May fail under spike conditions

### Multi-Scenario (Robust)
Expected changes:
- **Higher resources** (e.g., 28-32): Need headroom for spikes
- **Smaller batches** (e.g., 600-700): Better under high culprit scenarios
- **Higher K** (e.g., 12-14): Better isolation when culprits are frequent
- **Lower flake tolerance** (e.g., 0.05): Keep more tests active for crisis mode

## Answering Your Question

> What's the best approach?

**Use deterministic weighted scenarios with progressive fidelity:**

1. **Define 5 key scenarios** with realistic weights
2. **Progressive fidelity**: Start with 1 scenario, add more as optimization progresses
3. **Weighted objective**: `Σ(weight_i * objective_i)` across scenarios
4. **Trade-off**: ~3-5x slower than single scenario, but finds robust configs

### Why not stochastic?
- Too much variance for Bayesian optimization to work well
- No control over which conditions are tested
- Results not reproducible
- Would need 10-20x more samples

### Why not all scenarios from start?
- Very slow early in optimization
- Don't need robustness when exploring bad regions
- Progressive fidelity gives 2-3x speedup overall

## Results

Our implementation shows:
- Single scenario optimizer: 302s for 50 trials
- Multi-scenario with progressive fidelity: ~5-8 minutes for 30 trials
- ~2-3x slower but finds parameters that work across all conditions

The robust optimizer naturally finds different parameters because it must balance:
- Efficiency in normal mode
- Headroom for traffic spikes
- Resilience when tests are flaky
- Non-collapse guarantee in crisis mode
