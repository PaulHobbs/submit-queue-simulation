# Adaptive CI/CD Submit Queue Simulation

This project is a high-performance simulation of a sophisticated "Submit Queue" (also known as a Merge Queue or Commit Queue), typical of hyperscale software development environments (like Google, Meta, or heavily utilized GitHub repositories).

It models a system that dynamically adapts its strategy—switching between **speculative pipelining** and **parallel isolation**—based on the real-time health of the repository.

## Key Simulation Features

### 1. Hierarchical Culprit Modeling (Realistic Failure Scaling)
Unlike simple simulations that assume every test failure is independent, this model uses a hierarchical approach to better reflect reality:
*   **Culprit Decision:** A Change List (CL) first determines if it is fundamentally "bad" (approx. 3% probability).
*   **Affected Tests:** If a CL is "bad," it then determines *which* specific tests it breaks based on conditional probabilities.
*   **Impact:** This ensures that **adding more tests does not linearly increase the number of bad CLs**. Instead, adding tests increases the "resolution" of failures, modeling diminishing returns in catching bugs.

### 2. Adaptive Resource Allocation
The system operates with a fixed **Resource Budget** (e.g., 8 execution slots). It constantly monitors the **Failure Rate Estimate (EMA)** and automatically reshapes the pipeline:

| Failure Rate | Mode | Configuration | Strategy |
| :--- | :--- | :--- | :--- |
| **< 5%** | **Deep Pipeline** | 1 Lane, 8 Deep | Maximize batching and lookahead speculation. Fastest when green. |
| **5% - 20%** | **Split Pipeline** | 2 Lanes, 4 Deep | Ignores one bad lane while continuing to submit on the other. |
| **20% - 60%** | **Wide Pipeline** | 4 Lanes, 2 Deep | High chance of failure; spreads bets to find a passing combination. |
| **> 60%** | **Full Parallel** | 8 Lanes, 1 Deep | Maximum isolation. Assumes almost everything fails; hunting for the single passing CL. |

### 3. Flaky Tests & Fixes
*   Tests have a base probability of flaking (false negatives).
*   The simulation models "fixing" these flakes: if a test causes too many rejections, it is identified and its pass rate is artificially improved, simulating developer intervention.

## Usage

No external dependencies are required.

```bash
go run submit_queue.go
```