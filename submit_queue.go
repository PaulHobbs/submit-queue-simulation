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

	// Hierarchical logic
	// 1. A CL decides if it is a "Culprit" (Bad CL) based on a fixed rate (3%).
	// 2. If it is a Culprit, it decides which tests it breaks.
	//    We calculate the conditional probability needed to maintain the marginal PAffected rate.
	//    P(TestAffected) = P(Culprit) * P(Affected | Culprit)
	//    -> P(Affected | Culprit) = P(TestAffected) / P(Culprit)

	const ProbCulprit = 0.03

	if rng.Float64() < ProbCulprit {
		for _, td := range testDefs {
			// Calculate the probability that *this* specific test catches *this* culprit.
			// This scales the test's sensitivity relative to the global culprit rate.
			pCatchGivenCulprit := td.PAffected / ProbCulprit
			if pCatchGivenCulprit > 1.0 {
				pCatchGivenCulprit = 1.0
			}

			if rng.Float64() < pCatchGivenCulprit {
				c.Effects[td.ID] = sample(td.PassRates, rng)
			}
		}
	}
	// If not a culprit, Effects map remains empty (CL is safe).
	// --- CHANGED LOGIC END ---

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

// Step performs one iteration of the Sparse Bernoulli queue processing.
func (sq *PipelinedSubmitQueue) Step(currentTick int) int {
	pendingCount := len(sq.PendingChanges)
	if pendingCount == 0 {
		return 0
	}

	// 1. Determine Batch Sizing (N)
	// We take up to MaxMinibatchSize from the pending queue.
	n := pendingCount
	if n > sq.MaxMinibatchSize {
		n = sq.MaxMinibatchSize
	}
	batchChanges := sq.PendingChanges[:n]

	// 2. Sparse Bernoulli Parameters
	// T: Number of minibatches. Scale logarithmically with N.
	// Baseline: T ~ 10 * log10(N).
	// We clamp T to be at least a reasonable number to allow triangulation.
	tFloat := 10.0 * math.Log10(float64(n))
	t := int(math.Ceil(tFloat))
	if t < 4 {
		t = 4 // Minimum batches to have some discrimination
	}

	// k: Weight (minibatches per CL).
	// Optimal k approx T / (d + 1). Assuming d=2 (expected culprits).
	d := 2.0
	k := int(math.Round(float64(t) / (d + 1)))
	// Practical constraints:
	if k < 1 {
		k = 1
	}
	if k >= t {
		k = t - 1 // Cannot be in all batches (would offer no discrimination vs others)
	}
	if k > 6 {
		k = 6 // Cap weight to prevent saturation as per doc recommendation
	}

	// 3. Construct Encoding Matrix and Batches
	// We need T minibatches.
	minibatches := make([]Minibatch, t)
	for i := range minibatches {
		minibatches[i] = Minibatch{
			Changes: make([]*Change, 0, n/2), // heuristic cap
			// NewChanges not strictly used in this logic but kept for struct compatibility
			NewChanges: nil,
			LaneID:     i,
			DepthID:    0,
		}
	}

	// Matrix M[t][n] implicitly defined by which batches each CL is added to.
	// We track which batches each CL is in for scoring later.
	clBatchMap := make([][]int, n) // cl_index -> list of batch_indices

	for j, cl := range batchChanges {
		// Select k unique batches for this CL
		// We use Fisher-Yates shuffle on an index array to pick k unique
		indices := make([]int, t)
		for x := 0; x < t; x++ {
			indices[x] = x
		}
		// Partial shuffle to get first k
		for x := 0; x < k; x++ {
			randIdx := x + int(sq.rng.Float64()*float64(t-x))
			indices[x], indices[randIdx] = indices[randIdx], indices[x]
			
			// Assign CL to batch indices[x]
			bIdx := indices[x]
			minibatches[bIdx].Changes = append(minibatches[bIdx].Changes, cl)
			clBatchMap[j] = append(clBatchMap[j], bIdx)
		}
	}

	// 4. Evaluate Batches
	batchResults := make([]bool, t) // true = pass, false = fail
	failuresInTick := 0
	var hardFailCLs []*Change // Collect CLs that caused hard failures to fix them later

	for i := range minibatches {
		// Evaluate returns (passed, hardFailure)
		// We treat hardFailure same as regular failure for the boolean vector y
		passed, hard := minibatches[i].Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)
		
		// Fix hard breaks: Defer to after evaluation to simulate parallel execution
		// (Prevent side-effects from affecting other batches in the same tick)
		if hard {
			for _, cl := range minibatches[i].Changes {
				if cl.IsHardBreak() {
					hardFailCLs = append(hardFailCLs, cl)
				}
			}
		}

		batchResults[i] = passed
		sq.TotalMinibatches++
		if passed {
			sq.PassedMinibatches++
		} else {
			failuresInTick++
		}
	}

	// Fix Hard Failures now that all batches have "run"
	// Deduplicate in case multiple batches found the same hard break
	fixedCLs := make(map[int]bool)
	for _, cl := range hardFailCLs {
		if !fixedCLs[cl.ID] {
			cl.FixHardBreaks(sq.TestDefM, sq.rng)
			fixedCLs[cl.ID] = true
		}
	}

	// Update global failure rate estimate (EMA)
	currentRate := 0.0
	if t > 0 {
		currentRate = float64(failuresInTick) / float64(t)
	}
	sq.FailureRateEst = 0.8*sq.FailureRateEst + 0.2*currentRate

	// 5. Decode / Scoring
	// Score Sj = (Sum of failing batches CL j is in) / k
	rejectedIndices := make(map[int]bool)
	
	for j := 0; j < n; j++ {
		failCount := 0
		for _, bIdx := range clBatchMap[j] {
			if !batchResults[bIdx] {
				failCount++
			}
		}
		
		score := float64(failCount) / float64(k)
		
		// Decision: Reject if Score > Threshold
		// Doc suggests > 0.8 for primary culprit.
		// We use 0.75 as a safe cutoff given k=5 or 6.
		if score > 0.75 {
			rejectedIndices[j] = true
		}
	}

	// 6. Submit or Reject
	submittedCount := 0
	
	// We need to reconstruct PendingChanges removing BOTH submitted and rejected.
	// In this design, "Rejected" CLs are removed from the queue (failed).
	// "Submitted" CLs are applied and removed.
	
	// Apply effects for submitted
	for j, cl := range batchChanges {
		if !rejectedIndices[j] {
			sq.applyEffect(cl, currentTick)
			submittedCount++
		}
	}
	sq.TotalSubmitted += submittedCount
	
	// Apply flaky fixes based on submitted count
	if submittedCount > 0 {
		sq.ApplyFlakyFixes(submittedCount)
	}

	// Remove processed CLs (both submitted and rejected) from pending
	// Since we processed the first 'n', we just slice them off.
	// Wait! If we processed them, they are either Submitted (success) or Rejected (fail).
	// In either case, they leave the Pending Queue.
	sq.PendingChanges = sq.PendingChanges[n:]

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
			// PAffected here represents the marginal probability of this test being broken
			// by a random CL.
			// In the new logic, this is satisfied by P(Culprit) * P(Affected|Culprit).
			// With ProbCulprit = 3%, and PAffected = 0.5%,
			// a Culprit CL has a ~16% chance of hitting this specific test.
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