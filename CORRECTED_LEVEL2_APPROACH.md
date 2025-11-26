# Corrected Level 2 Approach: Environment vs Design Parameters

## The Error

The initial Level 2 optimization treated **backpressure thresholds as tunable design parameters**. This was conceptually wrong.

### What Backpressure Actually Represents

Backpressure thresholds model **developer behavior** in response to queue buildup:
- When queues are long, developers spend time babysitting CLs through CI
- This reduces their capacity/motivation to create new CLs
- The thresholds represent *how developers respond to delays* (environmental constraint)

**We cannot "optimize" developer behavior** - we can only model it and design systems that work well under those constraints.

## Corrected Parameter Classification

### **Design Parameters** (We Control - Optimize These)

**Level 1 (Explicit):**
- `resources`: CI/test resource allocation
- `maxbatch`: Maximum batch size for testing
- `maxk`: Maximum sparsity for batching
- `kdiv`: K divisor formula parameter
- `flaketol`: Flake tolerance threshold

**Level 2 (Implicit - Controllable):**
- `verify_latency`: How long verification takes (infrastructure choice)
- `fix_delay`: How long fixes take (process/tooling choice)
- `verify_resource_mult`: Resource allocation policy for verification

**Total: 8 design parameters** (not 11!)

### **Environmental Parameters** (We Observe - Test Robustness)

- `bp_threshold_1/2/3`: Developer behavior model (backpressure)
- `culprit_prob`: Code quality in environment
- `test_stability`: Test infrastructure reliability
- `traffic`: Developer activity levels

## The Solution: Scenarios + Jitter

Instead of optimizing backpressure thresholds, we:

### 1. Define Explicit Scenarios

Test different developer behavior patterns:

```python
# Aggressive developers (less deterred by queues)
bp_thresholds=(300, 600, 1200)

# Standard developers (baseline assumption)
bp_thresholds=(200, 400, 800)

# Conservative developers (more deterred)
bp_thresholds=(100, 200, 400)
```

### 2. Add Jitter Within Scenarios

Apply ±25% random jitter to backpressure thresholds on each simulation run:
- Ensures we don't overfit to exact threshold values
- Tests continuous robustness
- Simulates natural variance in developer behavior

### 3. Optimize Design Parameters

Find configurations that work well **across all developer behavior patterns**.

## New Scenario Set (8 Scenarios)

| Scenario | Weight | Traffic | Culprits | Stability | BP Thresholds | Description |
|----------|--------|---------|----------|-----------|---------------|-------------|
| normal-std | 25% | 8 | 3% | 1.0 | 200/400/800 | Most common |
| normal-aggressive | 10% | 8 | 3% | 1.0 | 300/600/1200 | Aggressive devs |
| normal-conservative | 5% | 8 | 3% | 1.0 | 100/200/400 | Conservative devs |
| spike-std | 20% | 16 | 4% | 0.98 | 200/400/800 | Traffic spike |
| spike-conservative | 10% | 16 | 4% | 0.98 | 100/200/400 | Spike + conservative |
| low | 10% | 4 | 2% | 0.99 | 200/400/800 | Off-hours |
| flaky | 10% | 8 | 3% | 0.90 | 200/400/800 | Flaky tests |
| crisis | 10% | 20 | 8% | 0.92 | 200/400/800 | Crisis mode |

**Key addition:** Explicit developer behavior variations in normal and spike scenarios.

## What to Expect from Corrected Level 2

### Previous (Incorrect) Approach
- Optimized 11 parameters (including backpressure)
- Found 4.6% improvement over Level 1
- But improvement was partly from "optimizing" environmental constraints!

### Corrected Approach
- Optimize only 8 parameters (3 Level 2 instead of 6)
- Test robustness against 8 scenarios (up from 5)
- Include backpressure variations (±50% range across scenarios)
- Include jitter (±25% within each scenario)

### Expected Outcomes

**If Level 2 parameters matter:**
- Should still see improvement, but likely **smaller than 4.6%**
- Improvement will be from truly controllable factors (verify latency, fix delay, resource mult)
- Configuration will be robust to developer behavior uncertainty

**If Level 2 parameters don't matter much:**
- May see <1% improvement or no improvement
- Would indicate Level 1 parameters dominate
- Still valuable: proves robustness to developer behavior variations

**Either way:**
- Configurations will be robust to ±50% variation in developer behavior
- Won't be overfit to specific backpressure assumptions
- More confidence in production deployment

## Implementation Details

### Backpressure Sampling

Each simulation run samples jittered thresholds:

```python
def sample_bp_thresholds(scenario, rng):
    """Sample with ±25% uniform jitter"""
    bp1 = rng.integers(150, 250)  # 200 ± 25%
    bp2 = rng.integers(300, 500)  # 400 ± 25%
    bp3 = rng.integers(600, 1000) # 800 ± 25%
    return (bp1, bp2, bp3)
```

### Progressive Fidelity

Early trials use fewer scenarios for speed:

```
Trials 1-10:   normal-std only (fast exploration)
Trials 11-25:  normal variants + spike-std (4 scenarios)
Trials 26+:    All 8 scenarios (full robustness testing)
```

## Key Insights

### 1. Why This Matters

The previous results showed backpressure thresholds had **huge correlations** (±0.39):
- This meant our "optimal" Level 1 parameters were **highly dependent** on backpressure assumptions
- If real developers behave differently, our config could perform poorly
- Testing robustness to backpressure is critical

### 2. The Lesson

**Distinguish between:**
- **System design choices** (optimize these)
- **Environmental constraints** (test robustness against these)

Confusing the two leads to overfitting to assumptions about the environment.

### 3. Broader Applicability

This principle applies beyond backpressure:
- Traffic levels: Probably environmental (user activity)
- Culprit probability: Partly environmental (team code quality), partly designable (code review)
- Test stability: Partly environmental (test infrastructure), partly designable (test design)

## Next Steps

1. ✅ **Implementation Complete**: Scenarios + jitter approach
2. 🔄 **Run Corrected Level 2**: 50 trials with 8 scenarios
3. 📊 **Compare Results**: vs original Level 1 and incorrect Level 2
4. 🎯 **Sensitivity Analysis**: Which Level 2 parameters actually matter?
5. 🤔 **Decision Point**: Is Level 2 improvement worth the complexity?

## Files

- `optimizer_robust.py`: Updated with corrected approach
- `CORRECTED_LEVEL2_APPROACH.md`: This document
- `LEVEL2_RESULTS.md`: Original (incorrect) results (kept for reference)

---

**Bottom Line:** We now optimize **8 design parameters** while testing robustness against **8 scenarios with ±25% jitter** in developer behavior. This gives us configurations that are truly robust to environmental uncertainty.
