# Adaptive CI/CD Submit Queue Simulation

This project is a high-performance simulation of a sophisticated "Submit Queue" (also known as a Merge Queue or Commit Queue), typical of hyperscale software development environments (like Google, Meta, or heavily utilized GitHub repositories).

It now models a system utilizing **Sparse Bernoulli Group-Based Testing**, a method specifically designed for resilience against flaky tests and efficient culprit identification in the presence of noise, replacing traditional adaptive pipeline strategies.

## Key Simulation Features

### 1. Hierarchical Culprit Modeling (Realistic Failure Scaling)
Unlike simple simulations that assume every test failure is independent, this model uses a hierarchical approach to better reflect reality:
*   **Culprit Decision:** A Change List (CL) first determines if it is fundamentally "bad" (approx. 3% probability).
*   **Affected Tests:** If a CL is "bad," it then determines *which* specific tests it breaks based on conditional probabilities.
*   **Impact:** This ensures that **adding more tests does not linearly increase the number of bad CLs**. Instead, adding tests increases the "resolution" of failures, modeling diminishing returns in catching bugs.

### 2. Sparse Bernoulli Group-Based Testing
This simulation now employs a **Sparse Bernoulli Matrix** (also known as a Random k-set design) for its group-based testing algorithm. This approach replaces the previous adaptive pipelining strategy, providing superior resilience to flaky tests and robust culprit identification.

The core idea is to encode each Change List (CL) into a small, fixed number of randomly selected minibatches. This sparsity prevents a single culprit from "poisoning" too many tests and allows for a robust, score-based decoding mechanism.

**Key Concepts:**
*   **Encoding Matrix (Implicit):** Each CL is assigned to exactly `k` randomly chosen minibatches out of a total of `T` available minibatches.
*   **Batch Count (T):** The total number of minibatches run in parallel. This scales logarithmically with the number of pending CLs (`N`), typically `T ≈ 10 * log10(N)`, ensuring enough parallel batches to distinguish culprits effectively.
*   **Weight (k):** The number of minibatches a single CL participates in. This value is optimized to be small (e.g., 4-6) to provide sufficient signal for detection without overly saturating the matrix.
*   **Suspicion Scoring:** For each CL, a "Suspicion Score" is calculated as the ratio of weighted failing batches it participated in to the total `k` batches it was assigned. CLs with scores above a predefined threshold (e.g., > 0.75) are identified as culprits and rejected.

**Benefits:**
*   **Resilience to Flakiness:** A single flaky test only affects a small fraction of a CL's assigned batches, allowing the scoring mechanism to filter out noise.
*   **Efficient Culprit Identification:** Even with multiple culprits or flaky tests, the sparse design and score-based decoding make it highly probable to identify the true offenders.

### 3. Weighted Scoring for Flaky Tests (Optional)
The simulation includes an optional "Weighted Scoring" mechanism to further enhance culprit detection by incorporating the historical reliability of individual tests.

*   **Standard Scoring (Default):** Treats all test failures equally, regardless of the test's historical flake rate.
*   **Weighted Scoring (Optional):** When enabled, the suspicion score of a CL is adjusted based on the reliability of the *specific tests that failed* within its assigned minibatches. A failure in a historically stable test contributes significantly more to a CL's suspicion score than a failure in a known flaky test. This allows the system to prioritize and react more strongly to failures in critical, reliable tests.

This feature helps the simulator make more informed decisions, reducing the chance of rejecting innocent CLs due to highly flaky tests.

## Usage

No external dependencies are required.

```bash
# Run with default (unweighted) scoring
go run submit_queue.go

# Run with weighted scoring enabled
go run submit_queue.go -weighted-scoring=true
```