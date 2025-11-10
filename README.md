
# CI/CD Submit Queue Simulation

This project is a high-performance, parallelized simulation of a "Submit Queue" (often called a Merge Queue or Commit Queue) typical in large-scale software development environments.

It models a pipelined, speculative CI system that attempts to increase throughput by running multiple batches of changes in parallel, falling back to safer, slower modes when failures (flaky tests or actual breakages) occur.

## Overview

The simulation explores how varying levels of **Pipeline Depth**, **Traffic Load**, and **Test Suite Instability** impact the overall productivity of developers trying to merge code.

It is written in Go and utilizes goroutines to run dozens of simulation scenarios simultaneously while maintaining a deterministic, ordered output format for easy reading.

### Key Simulation Concepts

*   **Speculative Pipelining**: The queue assumes the current batch at the head of the queue will pass, and immediately starts testing subsequent batches. If the head fails, speculative work might need to be discarded.
*   **Parallel Fallback Mode**: When speculation fails too often (due to high failure rates), the queue switches to a "Parallel" mode, testing independent batches and submitting the first one that passes, rather than trying to maintain strict ordering.
*   **Flaky Tests**: Tests have a base probability of failing even on good code. The simulation models "fixing" these flakes over time if they cause too many rejections.
*   **Hard Breakages & Culprit Finding**: Some changes naturally break tests (0% pass rate). These require simulated manual intervention (quarantine/fixes), slowing down the batch they are in.
*   **Adaptive Developer Load**: The simulation includes a feedback loop where simulated developers slow down their submission rate if the queue size gets too large.

## Usage

No external dependencies are required beyond a standard Go installation.

```bash
go run main.go
```

*Note: The simulation performs heavy computation. It runs 30,000 ticks for every permutation of parameters. On a standard laptop, a full run will take 30-180 seconds, utilizing all available CPU cores.*

## Configuration & Parameters

The simulation automatically iterates through several key dimensions:

| Parameter | Values Tested | Description |
| :--- | :--- | :--- |
| **Pipeline Depth** | 1, 2, 4, 8 | How many concurrent batches the queue attempts to run. A depth of 1 is a simple serial queue. |
| **Traffic Load** | 1x, 2x, 3x, 4x | Multiplier on the "ideal" throughput of 25 changes/hour. 4x represents 100 changes/hour. |
| **N Tests** | 16, 32, 64, 128 | The number of distinct tests in the suite. More tests = higher chance of flakes/conflicts. |
| **Max Batch** | 32, 64, 128 | The maximum number of changes allowed in a single test run. |


### Column Definitions

*   **Productivity \* 1/x (Slowdown Factor)**: The ratio of Ideal Throughput to Actual Throughput.
    *   `1.0` = Perfect standard (queue is keeping up perfectly with demand).
    *   `2.0` = The system is running twice as slow as needed to keep up with incoming traffic.
*   **Avg Queue Size**: The average number of changes waiting to be tested. High numbers indicate the system is overloaded.
*   **Pass Rate**: The percentage of minibatches that passed all tests on the first try. Low pass rates cripple pipelined queues.
*   **Avg Time to Submit (h)**: The average wall-clock time (in simulated hours) from a developer uploading a change to it being merged.

## Technical Implementation Details

*   **Parallel Execution w/ Ordered Output**: To utilize multi-core CPUs, simulations run in a pool of goroutines. A dedicated printer goroutine uses a sequence ID buffer to hold results that finish early, ensuring the final printed table remains perfectly sorted by configuration parameters.
*   **Thread-Safe Fast RNG**: Because `math/rand` global functions use locks, they become a bottleneck in highly parallel simulations. Each simulation goroutine is allocated its own lock-free `FastRNG` instance pre-seeded with a large buffer of random numbers.

## Interpreting Output

The output is printed incrementally as simulations finish, ordered by the parameters above.

```text
Pipeline depth/parallelism: 1

Ideal throughput: 25 CLs/hour = Productivity 1/1x 
Max Batch  | Productivity * 1/x     | Avg Queue Size | Pass Rate | Avg Time to Submit (h)
-------------------------------------------------------------------------------------------------------------
n_tests: 16
32         | 1.38                   | 193            | 0.61      | 12.61
64         | 1.00                   | 7              | 0.92      | 2.28
128        | 1.00                   | 10             | 0.75      | 2.42
n_tests: 32
32         | 1.45                   | 203            | 0.54      | 13.79
64         | 1.00                   | 0              | 1.00      | 2.00
128        | 1.16                   | 109            | 0.42      | 7.05
n_tests: 64
32         | 2.93                   | 413            | 0.27      | 50.37
64         | 3.08                   | 474            | 0.13      | 60.22
128        | 1.11                   | 91             | 0.36      | 6.03
n_tests: 128
32         | 4.51                   | 697            | 0.17      | 127.38
64         | 3.33                   | 502            | 0.12      | 68.81
128        | 1.20                   | 140            | 0.26      | 8.71

Ideal throughput: 50 CLs/hour = Productivity 1/1x 
Max Batch  | Productivity * 1/x     | Avg Queue Size | Pass Rate | Avg Time to Submit (h)
-------------------------------------------------------------------------------------------------------------
n_tests: 16
32         | 1.69                   | 262            | 0.93      | 10.83
64         | 1.34                   | 119            | 0.79      | 5.20
128        | 1.06                   | 45             | 0.76      | 2.96
n_tests: 32
32         | 6.91                   | 1238           | 0.23      | 171.95
64         | 1.97                   | 309            | 0.40      | 14.17
128        | 2.44                   | 431            | 0.18      | 22.98
n_tests: 64
32         | 6.24                   | 912            | 0.25      | 115.69
64         | 1.26                   | 74             | 1.00      | 3.86
128        | 1.67                   | 254            | 0.27      | 10.50
n_tests: 128
32         | 7.34                   | 1204           | 0.21      | 178.14
64         | 7.03                   | 1077           | 0.11      | 152.29
128        | 41.89                  | 59708          | 0.01      | 1085.15

... etc
```