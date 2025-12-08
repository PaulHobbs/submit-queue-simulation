package main

import (
	"database/sql"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Parameters holds all simulation parameters
type Parameters struct {
	// Train size
	C int // Number of changes per train

	// SC-LDPC parameters
	M int // Number of minibatches
	B int // Number of blocks (for SC-LDPC structure)
	K int // Column weight (minibatches per change)
	W int // Coupling width for SC-LDPC

	// System parameters
	T             int     // Number of build targets (each with 1 boot test)
	A             int     // Number of attempts for exoneration (1 = no retry)
	DDExoneration bool    // Whether to exonerate definite defectives

	// Rates
	DefectRate       float64 // Probability a change is defective
	FlakeRate        float64 // Base flake rate for tests
	ChangeArrivalRate float64 // Changes per hour

	// Timing (in hours)
	BuildPassMean   float64
	BuildPassStd    float64
	BuildFailMean   float64
	BuildFailStd    float64
	TestPassMean    float64
	TestPassStd     float64
	TestFailMean    float64
	TestFailStd     float64
}

// DefaultParameters returns the default simulation parameters
func DefaultParameters() Parameters {
	C := 60
	M := C / 3         // 20 minibatches
	B := M / 4         // 5 blocks
	K := M / 3         // ~7 minibatches per change
	if K < 1 {
		K = 1
	}

	return Parameters{
		C:                 C,
		M:                 M,
		B:                 B,
		K:                 K,
		W:                 2, // Coupling width
		T:                 16,
		A:                 2, // 1 retry on fail
		DDExoneration:     false,
		DefectRate:        0.03,
		FlakeRate:         0.01,
		ChangeArrivalRate: 100.0,
		BuildPassMean:     1.0,
		BuildPassStd:      0.5,
		BuildFailMean:     0.5,
		BuildFailStd:      0.25,
		TestPassMean:      0.5,
		TestPassStd:       1.0 / 6.0, // ~10 min std
		TestFailMean:      0.25,
		TestFailStd:       1.0 / 12.0,
	}
}

// Change represents a submitted change
type Change struct {
	ID           int
	IsDefective  bool
	AffectedPairs []int // Which build+test pairs this change breaks (0 to T-1)
	ArrivalTime  float64
}

// MinibatchResult holds results for one minibatch
type MinibatchResult struct {
	MinibatchID   int
	Changes       []int              // Change IDs in this minibatch
	Results       map[int]bool       // build+test pair -> passed (true) or failed (false)
	PairDurations map[int]float64    // build+test pair -> duration for that pair
	Duration      float64            // Max duration across all pairs (for this minibatch)
}

// TrainResult holds aggregate results for a train
type TrainResult struct {
	Changes            []Change
	InnocentChanges    []int // Change IDs that are innocent
	DefiniteDefectives []int // Change IDs that are definite defectives
	AmbiguousCulprits  []int // Change IDs that are ambiguous

	// Per-change tracking of which pairs they weren't exonerated on
	UnexoneratedPairs map[int][]int // changeID -> list of build+test pairs

	FalseRejections     int     // Innocent changes rejected due to flakes
	TrueRejections      int     // Actual defectives rejected
	MinibatchDuration   float64 // Max duration of minibatch phase (parallel)
	ExonerationDuration float64
	PickupLatency       float64 // Time waiting for batch to form

	// Per-change latencies (incremental exoneration model)
	InnocentLatencies []float64 // Latency for each innocent change (pickup + time to full exoneration)
	CulpritLatencies  []float64 // Latency for culprits (pickup + minibatch + exoneration)

	MinibatchExecutions   int // Number of build executions in minibatch phase
	ExonerationExecutions int // Number of build executions in exoneration phase
}

// Metrics holds computed metrics for a simulation run
type Metrics struct {
	FalseRejectionRate float64
	SubmitLatency      float64 // Average per-change latency
	CapacityCostRatio  float64 // Relative to individual testing
	E2ECost            float64 // Total SWEh cost
}

// SCLDPCMatrix generates an SC-LDPC-like assignment matrix
// Returns a matrix where matrix[m][c] = true if change c is in minibatch m
func GenerateSCLDPCMatrix(params Parameters) [][]bool {
	matrix := make([][]bool, params.M)
	for m := range matrix {
		matrix[m] = make([]bool, params.C)
	}

	// SC-LDPC structure: divide into B blocks with coupling width W
	// Each block spans M/B minibatches and C/B changes
	minibatchesPerBlock := params.M / params.B
	changesPerBlock := params.C / params.B

	if minibatchesPerBlock < 1 {
		minibatchesPerBlock = 1
	}
	if changesPerBlock < 1 {
		changesPerBlock = 1
	}

	// For each change, assign to K minibatches using SC-LDPC structure
	for c := 0; c < params.C; c++ {
		// Determine which block this change belongs to
		changeBlock := c / changesPerBlock
		if changeBlock >= params.B {
			changeBlock = params.B - 1
		}

		// Collect candidate minibatches (from current block and W neighboring blocks)
		candidates := []int{}
		for b := changeBlock; b <= changeBlock+params.W && b < params.B; b++ {
			startM := b * minibatchesPerBlock
			endM := startM + minibatchesPerBlock
			if endM > params.M {
				endM = params.M
			}
			for m := startM; m < endM; m++ {
				candidates = append(candidates, m)
			}
		}

		// Randomly select K minibatches from candidates
		if len(candidates) > params.K {
			rand.Shuffle(len(candidates), func(i, j int) {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			})
			candidates = candidates[:params.K]
		}

		for _, m := range candidates {
			matrix[m][c] = true
		}
	}

	return matrix
}

// GenerateChanges creates a batch of C changes with defects
func GenerateChanges(params Parameters, startTime float64, rng *rand.Rand) []Change {
	changes := make([]Change, params.C)

	// Calculate inter-arrival times
	meanInterArrival := 1.0 / params.ChangeArrivalRate // hours per change
	currentTime := startTime

	for i := 0; i < params.C; i++ {
		// Exponential inter-arrival time
		interArrival := rng.ExpFloat64() * meanInterArrival
		currentTime += interArrival

		change := Change{
			ID:          i,
			ArrivalTime: currentTime,
		}

		// Determine if defective
		if rng.Float64() < params.DefectRate {
			change.IsDefective = true

			// Lognormal number of affected pairs
			// lognormal(mu=2, sigma=2) has median = e^2 ≈ 7.4
			numAffected := int(math.Ceil(math.Exp(2.0 + 2.0*rng.NormFloat64())))
			if numAffected < 1 {
				numAffected = 1
			}
			if numAffected > params.T {
				numAffected = params.T
			}

			// Randomly select which pairs are affected
			pairs := rng.Perm(params.T)[:numAffected]
			change.AffectedPairs = pairs
		}

		changes[i] = change
	}

	return changes
}

// SimulateMinibatch runs a single minibatch and returns results
func SimulateMinibatch(
	minibatchID int,
	changeIDs []int,
	changes []Change,
	params Parameters,
	flakeRate float64,
	rng *rand.Rand,
) MinibatchResult {
	result := MinibatchResult{
		MinibatchID:   minibatchID,
		Changes:       changeIDs,
		Results:       make(map[int]bool),
		PairDurations: make(map[int]float64),
		Duration:      0,
	}

	// For each build+test pair
	for pair := 0; pair < params.T; pair++ {
		// Check if any change in this minibatch breaks this pair
		broken := false
		for _, cid := range changeIDs {
			for _, affected := range changes[cid].AffectedPairs {
				if affected == pair {
					broken = true
					break
				}
			}
			if broken {
				break
			}
		}

		// Simulate flake (only causes failures, not passes)
		flaked := rng.Float64() < flakeRate

		// Result: passes if not broken and not flaked
		passed := !broken && !flaked
		result.Results[pair] = passed

		// Calculate duration for this pair
		var buildDuration, testDuration float64
		if broken || flaked {
			// Failed - shorter duration
			buildDuration = math.Max(0.1, params.BuildFailMean+params.BuildFailStd*rng.NormFloat64())
			testDuration = math.Max(0.05, params.TestFailMean+params.TestFailStd*rng.NormFloat64())
		} else {
			// Passed
			buildDuration = math.Max(0.1, params.BuildPassMean+params.BuildPassStd*rng.NormFloat64())
			testDuration = math.Max(0.05, params.TestPassMean+params.TestPassStd*rng.NormFloat64())
		}

		pairDuration := buildDuration + testDuration
		result.PairDurations[pair] = pairDuration

		// All pairs run in parallel, so take max for overall minibatch duration
		if pairDuration > result.Duration {
			result.Duration = pairDuration
		}
	}

	return result
}

// AnalyzeMinibatchResults determines innocent, DD, and ambiguous changes
func AnalyzeMinibatchResults(
	changes []Change,
	results []MinibatchResult,
	matrix [][]bool,
	params Parameters,
) (innocent []int, definiteDefectives []int, ambiguous []int, unexponeratedPairs map[int][]int) {
	// Track which changes passed each pair at least once
	passedPair := make(map[int]map[int]bool) // changeID -> pair -> passed
	for i := 0; i < params.C; i++ {
		passedPair[i] = make(map[int]bool)
	}

	// Track failures: which minibatches failed each pair
	failedMinibatches := make(map[int][]int) // pair -> list of minibatch IDs that failed

	for _, res := range results {
		for pair, passed := range res.Results {
			if passed {
				// All changes in this minibatch are exonerated for this pair
				for _, cid := range res.Changes {
					passedPair[cid][pair] = true
				}
			} else {
				failedMinibatches[pair] = append(failedMinibatches[pair], res.MinibatchID)
			}
		}
	}

	// Determine innocent changes (passed all pairs)
	innocentSet := make(map[int]bool)
	for cid := 0; cid < params.C; cid++ {
		if len(passedPair[cid]) == params.T {
			innocentSet[cid] = true
			innocent = append(innocent, cid)
		}
	}

	// For each failed pair, find candidate culprits
	// Candidates = changes in failed minibatches that weren't exonerated for this pair
	candidatesPerFailure := make(map[int]map[int]map[int]bool) // pair -> minibatchID -> set of candidate changeIDs

	for pair, failedMBs := range failedMinibatches {
		candidatesPerFailure[pair] = make(map[int]map[int]bool)
		for _, mbID := range failedMBs {
			candidates := make(map[int]bool)
			for _, cid := range results[mbID].Changes {
				if !innocentSet[cid] && !passedPair[cid][pair] {
					candidates[cid] = true
				}
			}
			candidatesPerFailure[pair][mbID] = candidates
		}
	}

	// Find definite defectives: changes that are the sole candidate for some failure
	ddSet := make(map[int]bool)
	for pair, mbCandidates := range candidatesPerFailure {
		for _, candidates := range mbCandidates {
			if len(candidates) == 1 {
				for cid := range candidates {
					ddSet[cid] = true
					_ = pair // used implicitly
				}
			}
		}
	}

	for cid := range ddSet {
		definiteDefectives = append(definiteDefectives, cid)
	}

	// Ambiguous = not innocent, not DD
	unexponeratedPairs = make(map[int][]int)
	for cid := 0; cid < params.C; cid++ {
		if !innocentSet[cid] && !ddSet[cid] {
			ambiguous = append(ambiguous, cid)

			// Track which pairs this change wasn't exonerated on
			for pair := 0; pair < params.T; pair++ {
				if !passedPair[cid][pair] {
					unexponeratedPairs[cid] = append(unexponeratedPairs[cid], pair)
				}
			}
		}
	}

	return
}

// CalculateInnocentLatencies computes latency for each innocent change using incremental exoneration
// For each change: latency = max over all pairs of (min over K minibatches of first pass time)
func CalculateInnocentLatencies(
	innocentChanges []int,
	results []MinibatchResult,
	matrix [][]bool,
	params Parameters,
) []float64 {
	latencies := make([]float64, len(innocentChanges))

	for i, cid := range innocentChanges {
		// For each pair, find the earliest passing minibatch that contains this change
		maxFirstPassTime := 0.0

		for pair := 0; pair < params.T; pair++ {
			firstPassTime := math.Inf(1)

			// Check all minibatches this change is in
			for m := 0; m < params.M; m++ {
				if !matrix[m][cid] {
					continue // Change not in this minibatch
				}

				// Did this minibatch pass for this pair?
				if results[m].Results[pair] {
					// This minibatch passed - record its duration for this pair
					duration := results[m].PairDurations[pair]
					if duration < firstPassTime {
						firstPassTime = duration
					}
				}
			}

			// Track the slowest pair to get its first pass
			if firstPassTime > maxFirstPassTime {
				maxFirstPassTime = firstPassTime
			}
		}

		latencies[i] = maxFirstPassTime
	}

	return latencies
}

// SimulateExoneration runs individual tests for ambiguous culprits
func SimulateExoneration(
	ambiguous []int,
	unexponeratedPairs map[int][]int,
	changes []Change,
	params Parameters,
	flakeRate float64,
	rng *rand.Rand,
) (rejected []int, exonerationDuration float64, executions int) {
	maxChangeDuration := 0.0

	for _, cid := range ambiguous {
		pairs := unexponeratedPairs[cid]
		if len(pairs) == 0 {
			continue
		}

		executions += len(pairs)

		// Test each pair with up to A attempts
		// All pairs for a change run in parallel, so track max duration
		allPassed := true
		maxPairDuration := 0.0

		for _, pair := range pairs {
			pairPassed := false
			pairDuration := 0.0

			// Check if this change actually breaks this pair
			actuallyBreaks := false
			for _, affected := range changes[cid].AffectedPairs {
				if affected == pair {
					actuallyBreaks = true
					break
				}
			}

			for attempt := 0; attempt < params.A; attempt++ {
				flaked := rng.Float64() < flakeRate

				var buildDuration, testDuration float64

				if actuallyBreaks {
					// Actually broken - will fail
					buildDuration = math.Max(0.1, params.BuildFailMean+params.BuildFailStd*rng.NormFloat64())
					testDuration = math.Max(0.05, params.TestFailMean+params.TestFailStd*rng.NormFloat64())
					// Test fails, no retry for actual defects
					pairDuration += buildDuration + testDuration
					break
				} else if flaked {
					// Not broken but flaked - will retry
					buildDuration = math.Max(0.1, params.BuildFailMean+params.BuildFailStd*rng.NormFloat64())
					testDuration = math.Max(0.05, params.TestFailMean+params.TestFailStd*rng.NormFloat64())
					pairDuration += buildDuration + testDuration
					// Continue to next attempt (retry)
				} else {
					// Not broken, not flaked - passes!
					buildDuration = math.Max(0.1, params.BuildPassMean+params.BuildPassStd*rng.NormFloat64())
					testDuration = math.Max(0.05, params.TestPassMean+params.TestPassStd*rng.NormFloat64())
					pairDuration += buildDuration + testDuration
					pairPassed = true
					break
				}
			}

			if !pairPassed {
				allPassed = false
			}

			// Pairs run in parallel, so take max
			if pairDuration > maxPairDuration {
				maxPairDuration = pairDuration
			}
		}

		if !allPassed {
			rejected = append(rejected, cid)
		}

		// Changes run in parallel, so take max across changes
		if maxPairDuration > maxChangeDuration {
			maxChangeDuration = maxPairDuration
		}
	}

	exonerationDuration = maxChangeDuration
	return
}

// SimulateTrain runs a complete train simulation
func SimulateTrain(params Parameters, rng *rand.Rand, trainStartTime float64) TrainResult {
	return SimulateTrainDebug(params, rng, trainStartTime, false)
}

// SimulateTrainDebug runs a train with optional debug output
func SimulateTrainDebug(params Parameters, rng *rand.Rand, trainStartTime float64, debug bool) TrainResult {
	result := TrainResult{
		UnexoneratedPairs: make(map[int][]int),
	}

	// Generate flake rate for this batch (lognormal with mean=p, stddev=p)
	// Using lognormal parameterization: if X ~ N(mu, sigma), then exp(X) ~ Lognormal
	// For mean=p, stddev=p: we need mu and sigma such that:
	// E[exp(X)] = exp(mu + sigma^2/2) = p
	// Var[exp(X)] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2) = p^2
	// This gives sigma^2 = ln(2) ≈ 0.693, mu = ln(p) - sigma^2/2
	baseFR := params.FlakeRate
	sigma := math.Sqrt(math.Log(2))
	mu := math.Log(baseFR) - sigma*sigma/2
	flakeRate := math.Min(1.0, math.Exp(mu+sigma*rng.NormFloat64()))

	// Generate changes
	changes := GenerateChanges(params, trainStartTime, rng)
	result.Changes = changes

	// Calculate pickup latency (time for last change to arrive after first)
	if len(changes) > 0 {
		result.PickupLatency = changes[len(changes)-1].ArrivalTime - changes[0].ArrivalTime
		// Cap at 1 hour (we wait for C changes or 1 hour)
		if result.PickupLatency > 1.0 {
			result.PickupLatency = 1.0
		}
	}

	// Generate SC-LDPC matrix
	matrix := GenerateSCLDPCMatrix(params)

	// Run minibatch phase
	minibatchResults := make([]MinibatchResult, params.M)
	maxMinibatchDuration := 0.0
	result.MinibatchExecutions = params.T * params.M // T builds per minibatch

	for m := 0; m < params.M; m++ {
		// Get changes in this minibatch
		var changeIDs []int
		for c := 0; c < params.C; c++ {
			if matrix[m][c] {
				changeIDs = append(changeIDs, c)
			}
		}

		mbResult := SimulateMinibatch(m, changeIDs, changes, params, flakeRate, rng)
		minibatchResults[m] = mbResult

		if mbResult.Duration > maxMinibatchDuration {
			maxMinibatchDuration = mbResult.Duration
		}
	}
	result.MinibatchDuration = maxMinibatchDuration

	// Analyze results
	innocent, definiteDefectives, ambiguous, unexponeratedPairs := AnalyzeMinibatchResults(
		changes, minibatchResults, matrix, params,
	)

	result.InnocentChanges = innocent
	result.DefiniteDefectives = definiteDefectives
	result.AmbiguousCulprits = ambiguous
	result.UnexoneratedPairs = unexponeratedPairs

	// Calculate innocent change latencies using incremental exoneration model
	innocentLatencies := CalculateInnocentLatencies(innocent, minibatchResults, matrix, params)
	result.InnocentLatencies = innocentLatencies

	if debug {
		defectCount := 0
		for _, c := range changes {
			if c.IsDefective {
				defectCount++
			}
		}
		totalUnexonPairs := 0
		for _, pairs := range unexponeratedPairs {
			totalUnexonPairs += len(pairs)
		}
		avgInnocentLat := 0.0
		if len(innocentLatencies) > 0 {
			for _, lat := range innocentLatencies {
				avgInnocentLat += lat
			}
			avgInnocentLat /= float64(len(innocentLatencies))
		}
		fmt.Printf("Debug: defects=%d, innocent=%d, DD=%d, ambiguous=%d, avgUnexonPairs=%.1f\n",
			defectCount, len(innocent), len(definiteDefectives), len(ambiguous),
			float64(totalUnexonPairs)/math.Max(1, float64(len(ambiguous))))
		fmt.Printf("Debug: pickup=%.2fh, minibatchMax=%.2fh, avgInnocentLat=%.2fh, flakeRate=%.4f\n",
			result.PickupLatency, result.MinibatchDuration, avgInnocentLat, flakeRate)
	}

	// Handle definite defectives
	if params.DDExoneration {
		// Run exoneration on DDs too
		ddUnexonerated := make(map[int][]int)
		for _, cid := range definiteDefectives {
			// DDs need testing on all pairs they weren't exonerated on
			for pair := 0; pair < params.T; pair++ {
				ddUnexonerated[cid] = append(ddUnexonerated[cid], pair)
			}
		}
		ddRejected, ddDuration, ddExec := SimulateExoneration(
			definiteDefectives, ddUnexonerated, changes, params, flakeRate, rng,
		)
		result.ExonerationExecutions += ddExec

		for _, cid := range ddRejected {
			if !changes[cid].IsDefective {
				result.FalseRejections++
			} else {
				result.TrueRejections++
			}
		}
		if ddDuration > result.ExonerationDuration {
			result.ExonerationDuration = ddDuration
		}
	} else {
		// DDs are rejected without exoneration
		for _, cid := range definiteDefectives {
			if !changes[cid].IsDefective {
				result.FalseRejections++
			} else {
				result.TrueRejections++
			}
		}
	}

	// Run exoneration on ambiguous culprits
	ambigRejected, ambigDuration, ambigExec := SimulateExoneration(
		ambiguous, unexponeratedPairs, changes, params, flakeRate, rng,
	)
	result.ExonerationExecutions += ambigExec

	for _, cid := range ambigRejected {
		if !changes[cid].IsDefective {
			result.FalseRejections++
		} else {
			result.TrueRejections++
		}
	}
	if ambigDuration > result.ExonerationDuration {
		result.ExonerationDuration = ambigDuration
	}

	if debug {
		fmt.Printf("Debug: exoneration=%.2fh, exonExec=%d, falseRej=%d, trueRej=%d\n",
			result.ExonerationDuration, result.ExonerationExecutions,
			result.FalseRejections, result.TrueRejections)
	}

	return result
}

// ComputeMetrics calculates metrics from a train result
func ComputeMetrics(result TrainResult, params Parameters) Metrics {
	metrics := Metrics{}

	// Count innocent changes
	innocentCount := 0
	for _, c := range result.Changes {
		if !c.IsDefective {
			innocentCount++
		}
	}

	// False rejection rate
	if innocentCount > 0 {
		metrics.FalseRejectionRate = float64(result.FalseRejections) / float64(innocentCount)
	}

	// Submit latency per change using incremental exoneration model
	// - Innocent changes: pickup + time to get all pairs exonerated (min across K minibatches per pair)
	// - Culprits (DD + ambiguous): pickup + full minibatch duration + exoneration duration
	totalWaitingHours := 0.0
	numChangesForLatency := 0

	// Innocent changes use incremental latencies
	for _, lat := range result.InnocentLatencies {
		totalWaitingHours += result.PickupLatency + lat
		numChangesForLatency++
	}

	// Culprits wait for full minibatch + exoneration
	numCulprits := len(result.DefiniteDefectives) + len(result.AmbiguousCulprits)
	culpritLatency := result.PickupLatency + result.MinibatchDuration + result.ExonerationDuration
	totalWaitingHours += culpritLatency * float64(numCulprits)
	numChangesForLatency += numCulprits

	if numChangesForLatency > 0 {
		metrics.SubmitLatency = totalWaitingHours / float64(numChangesForLatency)
	}

	// Capacity cost ratio
	// Individual testing cost = T builds * C changes * (1/(1-flake)) expected attempts
	individualCost := float64(params.T) * float64(params.C) * (1.0 / (1.0 - params.FlakeRate))

	// Group testing cost = T builds * M minibatches + exoneration executions
	groupCost := float64(result.MinibatchExecutions) + float64(result.ExonerationExecutions)

	if individualCost > 0 {
		metrics.CapacityCostRatio = groupCost / individualCost
	}

	// E2E Cost per change
	// = 0.5 SWEh/h waiting * avg latency per change
	// + 100 SWEh * false rejection rate
	// + 0.05 SWEh per build execution / C changes
	waitingCostPerChange := 0.5 * metrics.SubmitLatency
	rejectionCostPerChange := 100.0 * metrics.FalseRejectionRate
	executionCostPerChange := 0.05 * float64(result.MinibatchExecutions+result.ExonerationExecutions) / float64(params.C)

	metrics.E2ECost = waitingCostPerChange + rejectionCostPerChange + executionCostPerChange

	return metrics
}

// OnlineStats tracks running mean and variance
type OnlineStats struct {
	n      int
	mean   float64
	m2     float64 // Sum of squared differences from mean
}

func (s *OnlineStats) Add(x float64) {
	s.n++
	delta := x - s.mean
	s.mean += delta / float64(s.n)
	delta2 := x - s.mean
	s.m2 += delta * delta2
}

func (s *OnlineStats) Mean() float64 {
	return s.mean
}

func (s *OnlineStats) Variance() float64 {
	if s.n < 2 {
		return 0
	}
	return s.m2 / float64(s.n-1)
}

func (s *OnlineStats) StdDev() float64 {
	return math.Sqrt(s.Variance())
}

func (s *OnlineStats) StdErr() float64 {
	if s.n < 2 {
		return math.Inf(1)
	}
	return s.StdDev() / math.Sqrt(float64(s.n))
}

// SimulationResult holds the results of a full simulation run
type SimulationResult struct {
	Params Parameters

	FalseRejectionRateMean   float64
	FalseRejectionRateStdDev float64

	SubmitLatencyMean   float64
	SubmitLatencyStdDev float64

	CapacityCostRatioMean   float64
	CapacityCostRatioStdDev float64

	E2ECostMean   float64
	E2ECostStdDev float64

	SampleCount int
	AblatedParam string
}

// RunSimulation runs the full simulation with early stopping
func RunSimulation(params Parameters, maxSamples int, ablatedParam string) SimulationResult {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	var (
		frStats       OnlineStats
		latencyStats  OnlineStats
		capacityStats OnlineStats
		e2eStats      OnlineStats
	)

	minSamples := 1000
	targetRelStdErr := 0.01 // 1% relative standard error for early stopping

	trainTime := 0.0

	for i := 0; i < maxSamples; i++ {
		result := SimulateTrain(params, rng, trainTime)
		metrics := ComputeMetrics(result, params)

		frStats.Add(metrics.FalseRejectionRate)
		latencyStats.Add(metrics.SubmitLatency)
		capacityStats.Add(metrics.CapacityCostRatio)
		e2eStats.Add(metrics.E2ECost)

		// Update train time (flake rate changes every 10 trains)
		trainTime += result.PickupLatency + result.MinibatchDuration + result.ExonerationDuration

		// Check for early stopping after minSamples
		if i >= minSamples && i%100 == 0 {
			// Check if all metrics have low relative standard error
			allLow := true

			if frStats.Mean() > 0 && frStats.StdErr()/frStats.Mean() > targetRelStdErr {
				allLow = false
			}
			if latencyStats.Mean() > 0 && latencyStats.StdErr()/latencyStats.Mean() > targetRelStdErr {
				allLow = false
			}
			if capacityStats.Mean() > 0 && capacityStats.StdErr()/capacityStats.Mean() > targetRelStdErr {
				allLow = false
			}
			if e2eStats.Mean() > 0 && e2eStats.StdErr()/e2eStats.Mean() > targetRelStdErr {
				allLow = false
			}

			if allLow {
				break
			}
		}
	}

	return SimulationResult{
		Params:                   params,
		FalseRejectionRateMean:   frStats.Mean(),
		FalseRejectionRateStdDev: frStats.StdDev(),
		SubmitLatencyMean:        latencyStats.Mean(),
		SubmitLatencyStdDev:      latencyStats.StdDev(),
		CapacityCostRatioMean:    capacityStats.Mean(),
		CapacityCostRatioStdDev:  capacityStats.StdDev(),
		E2ECostMean:              e2eStats.Mean(),
		E2ECostStdDev:            e2eStats.StdDev(),
		SampleCount:              frStats.n,
		AblatedParam:             ablatedParam,
	}
}

// Database functions
func InitDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	createTable := `
	CREATE TABLE IF NOT EXISTS simulation_results (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

		-- Parameters
		param_c INTEGER,
		param_m INTEGER,
		param_b INTEGER,
		param_k INTEGER,
		param_w INTEGER,
		param_t INTEGER,
		param_a INTEGER,
		param_dd_exoneration BOOLEAN,
		param_defect_rate REAL,
		param_flake_rate REAL,
		param_change_arrival_rate REAL,

		-- Metrics
		false_rejection_rate_mean REAL,
		false_rejection_rate_stddev REAL,
		submit_latency_mean REAL,
		submit_latency_stddev REAL,
		capacity_cost_ratio_mean REAL,
		capacity_cost_ratio_stddev REAL,
		e2e_cost_mean REAL,
		e2e_cost_stddev REAL,

		-- Meta
		sample_count INTEGER,
		ablated_param TEXT
	);
	`

	_, err = db.Exec(createTable)
	if err != nil {
		return nil, err
	}

	return db, nil
}

func SaveResult(db *sql.DB, result SimulationResult) error {
	insert := `
	INSERT INTO simulation_results (
		param_c, param_m, param_b, param_k, param_w, param_t, param_a, param_dd_exoneration,
		param_defect_rate, param_flake_rate, param_change_arrival_rate,
		false_rejection_rate_mean, false_rejection_rate_stddev,
		submit_latency_mean, submit_latency_stddev,
		capacity_cost_ratio_mean, capacity_cost_ratio_stddev,
		e2e_cost_mean, e2e_cost_stddev,
		sample_count, ablated_param
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
	`

	_, err := db.Exec(insert,
		result.Params.C, result.Params.M, result.Params.B, result.Params.K, result.Params.W,
		result.Params.T, result.Params.A, result.Params.DDExoneration,
		result.Params.DefectRate, result.Params.FlakeRate, result.Params.ChangeArrivalRate,
		result.FalseRejectionRateMean, result.FalseRejectionRateStdDev,
		result.SubmitLatencyMean, result.SubmitLatencyStdDev,
		result.CapacityCostRatioMean, result.CapacityCostRatioStdDev,
		result.E2ECostMean, result.E2ECostStdDev,
		result.SampleCount, result.AblatedParam,
	)

	return err
}

// Grid search ranges for each parameter
type AblationConfig struct {
	ParamName string
	Values    []float64
}

func GetAblationConfig(paramName string) AblationConfig {
	switch paramName {
	case "defect_rate":
		// 0.005 to 0.10, 20 points
		values := make([]float64, 20)
		for i := range values {
			values[i] = 0.005 + float64(i)*(0.10-0.005)/19.0
		}
		return AblationConfig{ParamName: paramName, Values: values}

	case "flake_rate":
		// 0.001 to 0.05, 20 points
		values := make([]float64, 20)
		for i := range values {
			values[i] = 0.001 + float64(i)*(0.05-0.001)/19.0
		}
		return AblationConfig{ParamName: paramName, Values: values}

	case "C":
		// 20 to 120, 20 points
		values := make([]float64, 20)
		for i := range values {
			values[i] = 20 + float64(i)*(120-20)/19.0
		}
		return AblationConfig{ParamName: paramName, Values: values}

	case "M":
		// 5 to 40, 20 points (keeping C fixed)
		values := make([]float64, 20)
		for i := range values {
			values[i] = 5 + float64(i)*(40-5)/19.0
		}
		return AblationConfig{ParamName: paramName, Values: values}

	case "K":
		// 2 to 15, 14 points
		values := make([]float64, 14)
		for i := range values {
			values[i] = float64(2 + i)
		}
		return AblationConfig{ParamName: paramName, Values: values}

	case "change_arrival_rate":
		// 20 to 200, 20 points
		values := make([]float64, 20)
		for i := range values {
			values[i] = 20 + float64(i)*(200-20)/19.0
		}
		return AblationConfig{ParamName: paramName, Values: values}

	default:
		return AblationConfig{}
	}
}

func ApplyAblation(params Parameters, paramName string, value float64) Parameters {
	switch paramName {
	case "defect_rate":
		params.DefectRate = value
	case "flake_rate":
		params.FlakeRate = value
	case "C":
		params.C = int(math.Round(value))
		// Recalculate derived parameters
		params.M = params.C / 3
		if params.M < 1 {
			params.M = 1
		}
		params.B = params.M / 4
		if params.B < 1 {
			params.B = 1
		}
		params.K = params.M / 3
		if params.K < 1 {
			params.K = 1
		}
	case "M":
		params.M = int(math.Round(value))
		if params.M < 1 {
			params.M = 1
		}
		params.B = params.M / 4
		if params.B < 1 {
			params.B = 1
		}
		params.K = params.M / 3
		if params.K < 1 {
			params.K = 1
		}
	case "K":
		params.K = int(math.Round(value))
		if params.K < 1 {
			params.K = 1
		}
		if params.K > params.M {
			params.K = params.M
		}
	case "change_arrival_rate":
		params.ChangeArrivalRate = value
	}
	return params
}

func main() {
	// Command line flags
	dbPath := flag.String("db", "simulation_results.db", "Path to SQLite database")
	ablateParam := flag.String("ablate", "", "Parameter to ablate (defect_rate, flake_rate, C, M, K, change_arrival_rate)")
	maxSamples := flag.Int("samples", 10000, "Maximum samples per grid point")
	workers := flag.Int("workers", 4, "Number of parallel workers")
	singleRun := flag.Bool("single", false, "Run a single simulation with default parameters")
	debugRun := flag.Bool("debug", false, "Run a few trains with debug output")

	flag.Parse()

	// Initialize database
	db, err := InitDB(*dbPath)
	if err != nil {
		fmt.Printf("Error initializing database: %v\n", err)
		return
	}
	defer db.Close()

	if *debugRun {
		// Run a few trains with debug output
		params := DefaultParameters()
		fmt.Println("Debug mode: Running 5 trains with detailed output")
		fmt.Printf("C=%d, M=%d, B=%d, K=%d, T=%d, defectRate=%.2f, flakeRate=%.2f\n",
			params.C, params.M, params.B, params.K, params.T, params.DefectRate, params.FlakeRate)
		fmt.Println()

		rng := rand.New(rand.NewSource(42)) // Fixed seed for reproducibility
		for i := 0; i < 5; i++ {
			fmt.Printf("=== Train %d ===\n", i+1)
			result := SimulateTrainDebug(params, rng, 0, true)
			metrics := ComputeMetrics(result, params)
			fmt.Printf("Total latency: %.2fh, E2E cost: %.2f SWEh\n\n",
				metrics.SubmitLatency, metrics.E2ECost)
		}
		return
	}

	if *singleRun {
		// Run single simulation with default parameters
		params := DefaultParameters()
		fmt.Println("Running single simulation with default parameters...")
		fmt.Printf("C=%d, M=%d, B=%d, K=%d, T=%d\n", params.C, params.M, params.B, params.K, params.T)

		result := RunSimulation(params, *maxSamples, "single")

		fmt.Printf("\nResults (n=%d samples):\n", result.SampleCount)
		fmt.Printf("  False Rejection Rate: %.4f ± %.4f\n", result.FalseRejectionRateMean, result.FalseRejectionRateStdDev)
		fmt.Printf("  Submit Latency: %.2f ± %.2f hours\n", result.SubmitLatencyMean, result.SubmitLatencyStdDev)
		fmt.Printf("  Capacity Cost Ratio: %.4f ± %.4f\n", result.CapacityCostRatioMean, result.CapacityCostRatioStdDev)
		fmt.Printf("  E2E Cost: %.2f ± %.2f SWEh\n", result.E2ECostMean, result.E2ECostStdDev)

		err = SaveResult(db, result)
		if err != nil {
			fmt.Printf("Error saving result: %v\n", err)
		}
		return
	}

	if *ablateParam == "" {
		fmt.Println("Please specify a parameter to ablate with -ablate")
		fmt.Println("Options: defect_rate, flake_rate, C, M, K, change_arrival_rate")
		return
	}

	config := GetAblationConfig(*ablateParam)
	if len(config.Values) == 0 {
		fmt.Printf("Unknown ablation parameter: %s\n", *ablateParam)
		return
	}

	fmt.Printf("Running ablation study for %s with %d grid points\n", *ablateParam, len(config.Values))
	fmt.Printf("Using %d workers, up to %d samples per point\n", *workers, *maxSamples)

	// Run ablation in parallel
	var wg sync.WaitGroup
	results := make(chan SimulationResult, len(config.Values))
	semaphore := make(chan struct{}, *workers)

	for _, value := range config.Values {
		wg.Add(1)
		go func(v float64) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			params := DefaultParameters()
			params = ApplyAblation(params, config.ParamName, v)

			fmt.Printf("  Running %s=%.4f (C=%d, M=%d, K=%d)...\n", config.ParamName, v, params.C, params.M, params.K)
			result := RunSimulation(params, *maxSamples, config.ParamName)
			fmt.Printf("  Done %s=%.4f (n=%d samples)\n", config.ParamName, v, result.SampleCount)

			results <- result
		}(value)
	}

	// Close results channel when all workers done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Save results as they come in
	count := 0
	for result := range results {
		err := SaveResult(db, result)
		if err != nil {
			fmt.Printf("Error saving result: %v\n", err)
		}
		count++
	}

	fmt.Printf("\nCompleted %d ablation points. Results saved to %s\n", count, *dbPath)
}
