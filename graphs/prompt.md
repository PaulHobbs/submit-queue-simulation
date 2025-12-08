Please make a go simulation of a non-adaptive group testing *submit queue* for CI where T=16 build targets (think phone device images for different platforms) and 16 boot tests are run on a set of candidate C changes.

- The minibatch / group testing phase uses SC-LDPC (Spatially Coupled Low-Density Parity Codes) matrices with B blocks and M minibatches.
  - B blocks, M minibatches, K batches per change are strategy parameters which can be ablated. Default to M=C/3 minibatches and B=M/4 blocks by default, rounded to the nearest integer. Make a reasonable assumption for choosing K, maybe M/3?
- Any change which has any minibatch pass on a given build/test is innocent for that build/test. If a change is innocent
 for all builds/tests, it is innocent and submitted.
- Any *ambiguous* culprits found by the minibatch phase (use definite defectives) will go through an exoneration phase where we use individualized testing on each ambiguous culprit on any build / build+test combo that it was a possible culprit for. This will let us determine which among the ambiguous culprits was a true culprit, modulo some rare flakes. Assume A=2 attempts to retry boot test failures, if the boot test is flaky (fails with flake P=0.01 probability)

Purpose: We'll use the go simulation to grab ~10k samples at each point along a dimension of hyperparameter design under certain assumptions, and then use python to plot those results. This will help drive system design decisions.

Performance: You can use early stopping if the empirical variance is low after enough samples (~1000).

Metrics we may be interested in plotting (y-axis):
- False rejection rate. What percent of innocent change are rejected due to flaky results causing them to be either definite defectives or ambiguous culprits which are then rejected in exoneration phase due to boot test flake?
- Submit delay latency per-change (train / batch pickup latency (time to pick up a batch), train execution latency (based on sampled execution time), possible exoneration phase latency (if necessary))
- Capacity cost as a fraction of individual testing cost. Assume individual testing costs running those T builds and tests on each change an expected 1 / (1-flake) times. Capacity cost is `T * 16 * B + (number of ambiguous culprits * (1/(1-flake))`.
- E2E: 0.5 SWEh/h waiting + 100 SWEh per false rejection + 0.05 SWEh per (build target + test) combo execution, or T times 0.05 SWEh capacity per minibatch. Assume builds and test take (1h +/- 30m) per build and (30m +/- 10m) per test when
passing, and 1/2 as long if the result is a fail.

Assumptions to ablate, or hyperparameters to tweak (x-axis):
- Parameters: B, M, C
- Defect rate. Assume defects happen with probability `defect_rate=0.03` by default, but tweaking this will result in different system outcomes. If a change is defective, it has changeEffects which is a vector (or index list) representing of the set of builds and tests it causes failures in. To make these correlated, let's assume a lognormal number of builds and tests are set to 1 in the vector, with min(ceil(lognormal(u=2, sigma=2)), 16+16) broken by the change. If a build is broken by a change, all tests for that minibatch are canceled with no results (equivalent to a fail). 
- Flake rate: Instead of assuming a uniform 0.01 flake rate, assume a lognnormal flake rate drawn from (mean=p, stddev=p), clipped to p=1.0 of course. We can ablate the results of the simulation by flake rate. Flake rate is drawn once per 10 batches, to simulate a changing flake rate over time.
- change arriving rate. Assuming by default 100 change/h, but allow ablating this.

Assumptions to not ablate:
- Resource constraints: all builds and tests run in parallel with no caps on parallelism.

The go simulation can update a sqlite database of sample results with an appropriate schema, given an x-axis parameter to ablate, and optionally its grid search range. Each row is a sample group, collecting mean & stddev for the 10k samples
 collected, (or fewer if the variance is low due to early stopping).
- Schema: each row has columns specifying the full set of parameters chosen, all metrics collected and their stddev for that sample group, and which metric was ablated

Later, we'll add a python script which can use seaborn to plot the results in a pretty way.

Before we start, please ask clarifying questions.

