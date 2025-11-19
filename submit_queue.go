package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// FastRNG is a per-goroutine RNG state to avoid locking.
type FastRNG struct {
	state uint64
}

func NewFastRNG(seed int64) *FastRNG {
	return &FastRNG{state: uint64(seed)}
}

func (r *FastRNG) Float64() float64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return float64(z^(z>>31)) / float64(1<<64-1)
}

// --- Data Structures ---

// DistEntry is a single row in a cumulative distribution
type DistEntry struct {
	Limit float64
	Value float64
}

// Distribution is a cumulative distribution
type Distribution []DistEntry

// TestDefinition keeps track of a test's tendency to be affected by CLs and its
// historical regression pass rate distribution.
type TestDefinition struct {
	ID        int
	PAffected float64
	PassRates Distribution
}

// sample takes a random sample of a Distribution
func sample(d Distribution, rng *FastRNG) float64 {
	s := rng.Float64()
	for _, e := range d {
		if s < e.Limit {
			return e.Value
		}
	}
	return d[len(d)-1].Value
}

// Change keeps track of which tests a Change affects.
type Change struct {
	ID           int
	CreationTick int             // Track creation time for submit stats
	Effects      map[int]float64 // Map from Test ID to pass rate effect
}

func NewChange(id, tick int, testDefs []TestDefinition, rng *FastRNG) *Change {
	c := &Change{
		ID:           id,
		CreationTick: tick,
		Effects:      make(map[int]float64, len(testDefs)/10),
	}
	for _, td := range testDefs {
		if rng.Float64() < td.PAffected {
			c.Effects[td.ID] = sample(td.PassRates, rng)
		}
	}
	return c
}

// IsHardBreak is true only if the test has 100% fail rate.
// These tests can be debugged and fixed by culprit finding / developers after
// quarantine.
func (c *Change) IsHardBreak() bool {
	for _, e := range c.Effects {
		if e == 0.0 {
			return true
		}
	}
	return false
}

// FixHardBreaks simulates a developer fixing a breakage.
// We assume a fix restores functionality but might still result in a flaky test.
// We exclude the probability of re-breaking the test (0.0)
// by normalizing the random sample to the range of non-zero outcomes.
func (c *Change) FixHardBreaks(testDefMap map[int]*TestDefinition, rng *FastRNG) {
	for tid, effect := range c.Effects {
		if effect == 0.0 {
			tDef := testDefMap[tid]

			// 1. Find the upper bound (Limit) of the "safe zone".
			// We scan the cumulative distribution to find the highest Limit
			// associated with a non-zero Value.
			safeLimit := 0.0
			for _, entry := range tDef.PassRates {
				if entry.Value > 0.0 {
					safeLimit = entry.Limit
				}
			}

			// Edge case: If the test definition is 100% breakage (safeLimit 0),
			// we simply force a fix to 1.0 to avoid infinite loops or stuck states.
			if safeLimit == 0.0 {
				c.Effects[tid] = 1.0
				continue
			}

			// 2. Generate a random number scaled strictly to the safe zone.
			// This effectively renormalizes the CDF to exclude the failure tail.
			s := rng.Float64() * safeLimit

			// 3. Sample the distribution using the scaled random number.
			newEffect := 1.0 // Default fallback
			for _, e := range tDef.PassRates {
				if s < e.Limit {
					newEffect = e.Value
					break
				}
			}

			// 4. Apply the fix
			c.Effects[tid] = newEffect
		}
	}
}

// --- Minibatch Logic ---

type Minibatch struct {
	Changes    []*Change // Cumulative changes in this batch
	NewChanges []*Change // Only the new changes added at this specific stage
	LaneID     int       // Which parallel lane is this?
	DepthID    int       // How deep in the pipeline?
}

func (mb *Minibatch) Evaluate(repoBasePPass map[int]float64, allTestIDs []int, rng *FastRNG) (bool, bool) {
	passed := true
	hardFailure := false

	for _, tid := range allTestIDs {
		effP := 1.0
		if base, ok := repoBasePPass[tid]; ok {
			effP = base
		}
		for _, cl := range mb.Changes {
			if eff, ok := cl.Effects[tid]; ok {
				if eff < effP {
					effP = eff
				}
			}
		}
		if effP == 0.0 {
			hardFailure = true
		}
		if effP < 1.0 && rng.Float64() >= effP {
			passed = false
			if hardFailure {
				break
			}
		}
	}
	return passed, hardFailure
}

// --- Adaptive Submit Queue ---

type PipelinedSubmitQueue struct {
	// Configuration
	TestDefs         []TestDefinition
	TestDefM         map[int]*TestDefinition
	AllTestIDs       []int
	ResourceBudget   int // Total slots (Width * Depth)
	MaxMinibatchSize int

	// State
	RepoBasePPass   map[int]float64
	PendingChanges  []*Change
	ChangeIDCounter int
	rng             *FastRNG

	// Adaptive State
	FailureRateEst float64 // Exponential Moving Average of batch failure rate (0.0 to 1.0)

	// Statistics
	TotalMinibatches  int
	PassedMinibatches int
	TotalSubmitted    int
	TotalWaitTicks    int // Sum of (SubmitTick - CreationTick) for all submitted CLs
}

func NewPipelinedSubmitQueue(testDefs []TestDefinition, resources, maxMB int, rng *FastRNG) *PipelinedSubmitQueue {
	testDefMap := make(map[int]*TestDefinition, len(testDefs))
	for i := range testDefs {
		testDefMap[testDefs[i].ID] = &testDefs[i]
	}

	sq := &PipelinedSubmitQueue{
		TestDefs:         testDefs,
		TestDefM:         testDefMap,
		AllTestIDs:       make([]int, len(testDefs)),
		RepoBasePPass:    make(map[int]float64),
		ResourceBudget:   resources, // e.g., 8
		MaxMinibatchSize: maxMB,
		PendingChanges:   make([]*Change, 0, 1024),
		rng:              rng,
		FailureRateEst:   0.0, // Start assuming smooth sailing
	}
	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
	}
	return sq
}

func (sq *PipelinedSubmitQueue) ResetStats() {
	sq.TotalMinibatches = 0
	sq.PassedMinibatches = 0
	sq.TotalSubmitted = 0
	sq.TotalWaitTicks = 0
	// We do NOT reset FailureRateEst, as knowledge carries over
}

func (sq *PipelinedSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		sq.PendingChanges = append(sq.PendingChanges, NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs, sq.rng))
		sq.ChangeIDCounter++
	}
}

// getDimensions returns (width, depth) based on failure rate estimate.
// The product width*depth will always be <= ResourceBudget.
func (sq *PipelinedSubmitQueue) getDimensions() (int, int) {
	// Mode 1: Extremely Low Failure (< 5%) -> Deep Pipeline
	// All resources in 1 lane. Maximize batching and lookahead.
	if sq.FailureRateEst < 0.05 {
		return 1, sq.ResourceBudget
	}

	// Mode 2: Low Failure (5% - 20%) -> Split Pipeline
	// e.g., if Budget=8, 2 Lanes of 4 Depth.
	// Allows ignoring 1 bad lane while still submitting fast on the other.
	if sq.FailureRateEst < 0.20 {
		w := 2
		d := sq.ResourceBudget / w
		return w, d
	}

	// Mode 3: Medium Failure (20% - 60%) -> Wide Pipeline
	// e.g., if Budget=8, 4 Lanes of 2 Depth.
	// High chance of failure, so we spread bets.
	if sq.FailureRateEst < 0.60 {
		w := 4
		d := sq.ResourceBudget / w
		return w, d
	}

	// Mode 4: High Failure (> 60%) -> Full Parallel
	// e.g., if Budget=8, 8 Lanes of 1 Depth.
	// Maximum isolation.
	return sq.ResourceBudget, 1
}

func (sq *PipelinedSubmitQueue) Step(currentTick int) int {
	if len(sq.PendingChanges) == 0 {
		return 0
	}

	width, depth := sq.getDimensions()
	return sq.stepAdaptive(currentTick, width, depth)
}

func (sq *PipelinedSubmitQueue) stepAdaptive(currentTick, width, depth int) int {
	pendingCount := len(sq.PendingChanges)

	// 1. Determine Batch Sizing
	// We need to fill 'width' lanes.
	totalSlots := width // Effectively we are pulling 'width' independent chunks

	rawSizePerLane := int(math.Ceil(float64(pendingCount) / float64(totalSlots)))

	// Inside each lane, we split that chunk into 'depth' pieces
	changesPerStage := int(math.Ceil(float64(rawSizePerLane) / float64(depth)))

	if changesPerStage < 1 {
		changesPerStage = 1
	}
	if changesPerStage > sq.MaxMinibatchSize {
		changesPerStage = sq.MaxMinibatchSize
	}

	// 2. Build Batches
	// We construct a 2D grid: batches[lane][depth]
	minibatches := make([]Minibatch, 0, width*depth)

	// We keep track of which changes belong to which lane/stage to reconstruct state later
	grid := make([][]*Minibatch, width)

	cursor := 0
	for w := 0; w < width; w++ {
		grid[w] = make([]*Minibatch, depth)

		// Accumulator for this lane (speculative pipeline)
		var laneAccumulator []*Change

		for d := 0; d < depth; d++ {
			// Grab changes for this specific stage
			end := cursor + changesPerStage
			if cursor >= pendingCount {
				// No more changes, stop building this lane
				break
			}
			if end > pendingCount {
				end = pendingCount
			}

			newChanges := sq.PendingChanges[cursor:end]
			cursor = end

			// Add to accumulator
			laneAccumulator = append(laneAccumulator, newChanges...)

			// Snapshot accumulator for this batch
			mbChanges := make([]*Change, len(laneAccumulator))
			copy(mbChanges, laneAccumulator)

			mb := Minibatch{
				Changes:    mbChanges,
				NewChanges: newChanges,
				LaneID:     w,
				DepthID:    d,
			}

			minibatches = append(minibatches, mb)
			// Store pointer in grid for easy lookup
			grid[w][d] = &minibatches[len(minibatches)-1]
		}
	}

	// 3. Evaluate All (Parallel Execution)
	failuresInTick := 0
	totalInTick := 0

	results := make(map[*Minibatch]struct{ passed, hard bool })

	for i := range minibatches {
		// Using pointer to the slice element we just appended
		mb := &minibatches[i]
		passed, hard := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)
		results[mb] = struct{ passed, hard bool }{passed, hard}

		sq.TotalMinibatches++
		totalInTick++
		if passed {
			sq.PassedMinibatches++
		} else {
			failuresInTick++
		}
	}

	// 4. Update Failure Rate Estimate (EMA)
	currentRate := 0.0
	if totalInTick > 0 {
		currentRate = float64(failuresInTick) / float64(totalInTick)
	}
	// Alpha = 0.2 (Slowly moving average, keeps memory of recent chaos)
	sq.FailureRateEst = 0.8*sq.FailureRateEst + 0.2*currentRate

	// 5. Select Winner (Submission Logic)
	// We prefer the "Queue Head" (Lane 0).
	// If Lane 0 partially passes, we take the deepest pass.
	// If Lane 0 totally fails, we check Lane 1, etc.
	// However, if we pick Lane 1, we are effectively skipping Lane 0's changes.
	// In a submit queue, this is valid (Lane 0 is quarantined), but we must
	// be careful not to submit Lane 1 if it inherently depended on Lane 0.
	// In this simulation, parallel lanes are independent.

	submittedCount := 0
	var winningLaneIndex = -1
	var winningDepthIndex = -1

	for w := 0; w < width; w++ {
		// Scan from Deepest -> Shallowest
		for d := depth - 1; d >= 0; d-- {
			mb := grid[w][d]
			if mb == nil {
				continue
			} // Slot might be empty if low on pending changes

			res := results[mb]
			if res.passed {
				winningLaneIndex = w
				winningDepthIndex = d
				goto FoundWinner // Break out of both loops
			}
		}

		// If we are here, this lane failed completely.
		// We check the next lane (Parallel fallback).
	}

FoundWinner:
	if winningLaneIndex != -1 {
		// Submit the winner
		winner := grid[winningLaneIndex][winningDepthIndex]

		// Apply effects of ALL changes in the cumulative batch
		for _, cl := range winner.Changes {
			sq.applyEffect(cl, currentTick)
		}
		submittedCount = len(winner.Changes)
		sq.TotalSubmitted += submittedCount

		// Remove submitted changes from Pending
		// We must identify exactly which CLs were submitted.
		// Since 'PendingChanges' is linear, and our lanes sliced it linearly:
		// We need to remove the submitted ones.
		// The simplest way in this simulation is to rebuild PendingChanges.

		// Identify the set of IDs submitted for fast lookup
		submittedIDs := make(map[int]bool)
		for _, cl := range winner.Changes {
			submittedIDs[cl.ID] = true
		}

		newPending := make([]*Change, 0, len(sq.PendingChanges)-submittedCount)
		for _, cl := range sq.PendingChanges {
			if !submittedIDs[cl.ID] {
				newPending = append(newPending, cl)
			}
		}
		sq.PendingChanges = newPending

		sq.ApplyFlakyFixes(submittedCount)
	}

	// 6. Hard Fail Fixes (Culprit Finding)
	// We apply fixes to ANY Hard Failures found in ANY batch executed this tick.
	// Even if we didn't submit them, the "Run" happened, so the "Logs" exist.
	for i := range minibatches {
		mb := &minibatches[i]
		res := results[mb]
		if res.hard {
			// Fix hard breaks in the *new* changes of this batch.
			// (Cumulative changes from previous stages were already fixed or are being fixed by their own stage's run)
			for _, cl := range mb.NewChanges {
				if cl.IsHardBreak() {
					cl.FixHardBreaks(sq.TestDefM, sq.rng)
				}
			}
		}
	}

	return submittedCount
}

func (sq *PipelinedSubmitQueue) ApplyFlakyFixes(n int) {
	for t, passRate := range sq.RepoBasePPass {
		if sq.rng.Float64() > math.Pow(passRate, float64(n)/84.0) {
			sq.RepoBasePPass[t] = 1
		}
	}
}

func (sq *PipelinedSubmitQueue) applyEffect(cl *Change, currentTick int) {
	for tid, effect := range cl.Effects {
		current := sq.RepoBasePPass[tid]
		if _, ok := sq.RepoBasePPass[tid]; !ok {
			current = 1.0
		}
		if effect < current {
			sq.RepoBasePPass[tid] = effect
		}
	}
	sq.TotalWaitTicks += (currentTick - cl.CreationTick)
}

// --- Simulation & Reporting ---

// Android presubmit runs take about 2h. This might actually underestimate how
// long a SQ batch would take to run since it needs tip-of-tree builds instead
// of LKGB.
var nChangesPer2Hour = []int{5, 5, 5, 5, 60, 60, 60, 60, 10, 10, 10, 10}

const idealThroughput = 25 // per 2hour, 12.5 per h
const nSamples = 100

type SimConfig struct {
	SeqID     int
	Resources int // Budget (e.g. 8)
	Traffic   int
	NTests    int
	MaxBatch  int
}

type SimResult struct {
	Config        SimConfig
	Slowdown      float64
	AvgQueueSize  float64
	MBPassRate    float64
	AvgSubmitTime float64
	FinalFailEst  float64 // To see if it adapted correctly
}

func runSimulation(cfg SimConfig, seed int64) SimResult {
	const primingIter = 3 * 6 * 7 // 3w
	const nIter = 30 * 6 * 7      // 30w
	// Use provided seed to ensure different seeds for different configs and samples
	rng := NewFastRNG(seed)

	testDefs := make([]TestDefinition, cfg.NTests)
	for i := 0; i < cfg.NTests; i++ {
		testDefs[i] = TestDefinition{
			ID: i,
			// Only 0.5% of CLs affect a test's pass rate post-smoketest (TreeHugger)
			// This means 0.25% of CLs are "bad" (increase a test's failure rate).
			PAffected: 0.005,
			// Distribution of new pass rates after a transition
			PassRates: []DistEntry{
				// Allow a 50% chance for flake to randomly get fixed,
				// in addition to the chance to purposely get fixed.
				// See ApplyFlakyFixes for how flake gets purposely fixed.
				{0.5, 1.0},
				// 10% chance of a "low flake rate" between 0.5% and 5%
				{0.55, 0.995},
				{0.59, 0.98},
				{0.60, 0.95},
				// 8% chance of a "high flake rate" between 20% and 80%
				{0.64, 0.80},
				{0.68, 0.20},
				// Some failures are breakages which can be
				// culprit-found and quarantined. These are probably
				// mid-air collisions between LKGB and TOT
				{1.0, 0.0},
			},
		}
	}

	sq := NewPipelinedSubmitQueue(testDefs, cfg.Resources, cfg.MaxBatch, rng)
	// Initial load
	sq.AddChanges(cfg.Traffic*nChangesPer2Hour[len(nChangesPer2Hour)-1], 0)

	totalQ := 0
	submittedTotal := 0

	// Run both priming and main simulation in one loop to maintain continuity
	// of load patterns (nChangesPer2Hour[i%12]).
	for i := 0; i < primingIter+nIter; i++ {
		// Once priming is complete, reset statistics to start fresh for nIter.
		if i == primingIter {
			sq.ResetStats()
		}

		submitted := sq.Step(i)

		// Only collect aggregate statistics after priming phase.
		if i >= primingIter {
			submittedTotal += submitted
			totalQ += len(sq.PendingChanges)
		}

		qSize := len(sq.PendingChanges)
		changesToAdd := cfg.Traffic * nChangesPer2Hour[i%12]

		// Backpressure Simulation (Throttle ingress if queue explodes)
		if qSize >= 200 && qSize < 400 {
			changesToAdd /= 2
		}
		if qSize >= 400 && qSize < 800 {
			changesToAdd /= 4
		}
		if qSize >= 800 {
			changesToAdd /= 8
		}

		if changesToAdd > 0 {
			sq.AddChanges(changesToAdd, i)
		}
	}

	mbPassRate := 0.0
	if sq.TotalMinibatches > 0 {
		mbPassRate = float64(sq.PassedMinibatches) / float64(sq.TotalMinibatches)
	}

	avgSubmitTime := 0.0
	if sq.TotalSubmitted > 0 {
		avgSubmitTime = 2.0 + 2*float64(sq.TotalWaitTicks)/float64(sq.TotalSubmitted)
	}

	throughput := 0.0
	if nIter > 0 {
		throughput = float64(submittedTotal) / float64(nIter)
	}

	slowdown := math.Inf(1)
	if throughput > 0 {
		slowdown = float64(idealThroughput*cfg.Traffic) / throughput
	}

	return SimResult{
		Config:        cfg,
		Slowdown:      slowdown,
		AvgQueueSize:  float64(totalQ) / float64(nIter),
		MBPassRate:    mbPassRate,
		AvgSubmitTime: avgSubmitTime,
		FinalFailEst:  sq.FailureRateEst,
	}
}

// runAveragedSimulation performs nSamples of the simulation and returns averaged results.
func runAveragedSimulation(cfg SimConfig) SimResult {
	var sumSlowdown, sumQSize, sumMBPassRate, sumSubmitTime, sumFailEst float64

	for i := 0; i < nSamples; i++ {
		seed := int64((cfg.SeqID*nSamples + i) * 997)
		res := runSimulation(cfg, seed)

		sumSlowdown += res.Slowdown
		sumQSize += res.AvgQueueSize
		sumMBPassRate += res.MBPassRate
		sumSubmitTime += res.AvgSubmitTime
		sumFailEst += res.FinalFailEst
	}

	return SimResult{
		Config:        cfg,
		Slowdown:      sumSlowdown / float64(nSamples),
		AvgQueueSize:  sumQSize / float64(nSamples),
		MBPassRate:    sumMBPassRate / float64(nSamples),
		AvgSubmitTime: sumSubmitTime / float64(nSamples),
		FinalFailEst:  sumFailEst / float64(nSamples),
	}
}

// printIncremental handles the stateful header printing and the result row.
func printIncremental(res SimResult, lastCfg *SimConfig) {
	cfg := res.Config
	if lastCfg == nil || cfg.Resources != lastCfg.Resources {
		fmt.Printf("Resource Budget (Execution Slots): %d\n", cfg.Resources)
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic {
		fmt.Printf("\nIdeal throughput: %d CLs/2hour\n", idealThroughput*cfg.Traffic)
		fmt.Printf("%-10s | %-12s | %-14s | %-9s | %-10s | %s\n",
			"Max Batch", "Slowdown", "Avg Queue", "Pass Rate", "Fail Est", "Avg Time (h)")
		fmt.Println("-----------------------------------------------------------------------------------------")
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic || cfg.NTests != lastCfg.NTests {
		fmt.Printf("n_tests: %d\n", cfg.NTests)
	}

	fmt.Printf("%-10d | %-12.2f | %-14.0f | %-9.2f | %-10.2f | %.2f\n",
		cfg.MaxBatch, res.Slowdown, res.AvgQueueSize, res.MBPassRate, res.FinalFailEst, res.AvgSubmitTime)
}

func main() {
	resultsCh := make(chan SimResult, 100)
	var wg sync.WaitGroup
	start := time.Now()

	// 1. Start the ordered printer goroutine
	printDone := make(chan struct{})
	go func() {
		defer close(printDone)
		buffer := make(map[int]SimResult)
		nextExpectedID := 0
		var lastCfg *SimConfig

		for res := range resultsCh {
			buffer[res.Config.SeqID] = res

			// Try to print as many contiguous results as possible from the buffer
			for {
				if nextRes, ok := buffer[nextExpectedID]; ok {
					printIncremental(nextRes, lastCfg)
					// keep a copy of the config for stateful header printing
					cfgCopy := nextRes.Config
					lastCfg = &cfgCopy
					delete(buffer, nextExpectedID)
					nextExpectedID++
				} else {
					break
				}
			}
		}
	}()

	// 2. Spawn all simulations
	seqCounter := 0
	// We test with a Resource Budget of 8 (simulating 8 machines/executors)
	// This matches your description of switching between 1x8, 2x4, 4x2, 8x1
	resources := 8

	for _, traffic := range []int{1, 2} {
		for _, nTests := range []int{16, 32, 64} {
			for _, maxBatch := range []int{32, 64, 128} {
				wg.Add(1)
				cfg := SimConfig{
					SeqID:     seqCounter,
					Resources: resources,
					Traffic:   traffic,
					NTests:    nTests,
					MaxBatch:  maxBatch,
				}
				go func(c SimConfig) {
					defer wg.Done()
					resultsCh <- runAveragedSimulation(c)
				}(cfg)
				seqCounter++
			}
		}
	}

	// 3. Wait for simulations to finish, then close channels
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	<-printDone
	fmt.Printf("\nAll simulations complete in %v.\n", time.Since(start))
}
