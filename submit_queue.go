package main

import (
	"fmt" // Keep fmt for printing results
	"math"
	"math/rand/v2" // Use math/rand/v2 for better performance and thread safety
	"sync"
	"time"
)

// --- Thread-Safe Fast RNG ---
const maxRand = 10_000_000

// Global precomputed randoms (read-only after init, so thread-safe)
var rands []float64

func init() {
	rands = make([]float64, maxRand)
	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), uint64(time.Now().UnixNano()/2))) // Use PCG for better quality
	for i := range rands {
		rands[i] = rng.Float64()
	}
}

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

// FixHardBreaks simulates quarantine & culprit finding of a failed run.
func (c *Change) FixHardBreaks(testDefMap map[int]*TestDefinition, testDefs []TestDefinition, rng *FastRNG) {
	for tid, effect := range c.Effects {
		if effect == 0.0 {
			tDef := testDefMap[tid]
			// Try to fix it by resampling a non-zero pass rate
			newEffect := 0.0
			for i := 0; i < 10; i++ {
				newEffect = sample(tDef.PassRates, rng)
				if newEffect > 0.0 {
					break
				}
			}
			if newEffect == 0.0 {
				newEffect = 1.0
			}
			c.Effects[tid] = newEffect
		}
	}
}

// Minibatch keeps a subset of the pending change which can be evaluated in a run.
type Minibatch struct {
	Changes    []*Change
	NewChanges []*Change
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

// PipelinedSubmitQueue keeps the state of the tests, pending changes, and minibatch statistics.
type PipelinedSubmitQueue struct {
	// --- Configuration ---
	TestDefs         []TestDefinition
	TestDefM         map[int]*TestDefinition
	AllTestIDs       []int
	PipelineDepth    int
	MaxMinibatchSize int

	// --- State ---
	RepoBasePPass   map[int]float64
	PendingChanges  []*Change
	ChangeIDCounter int
	UseParallelMode bool     // Toggle for dynamic fallback strategy
	rng             *FastRNG // Local RNG for this queue simulation

	// --- Statistics ---
	TotalMinibatches  int
	PassedMinibatches int
	TotalSubmitted    int
	TotalWaitTicks    int // Sum of (SubmitTick - CreationTick) for all submitted CLs
}

func NewPipelinedSubmitQueue(testDefs []TestDefinition, depth, maxMB int, rng *FastRNG) *PipelinedSubmitQueue {

	testDefMap := make(map[int]*TestDefinition, len(testDefs))
	for i := range testDefs {
		testDefMap[testDefs[i].ID] = &testDefs[i]
	}

	sq := &PipelinedSubmitQueue{
		TestDefs:         testDefs,
		TestDefM:         testDefMap,
		AllTestIDs:       make([]int, len(testDefs)),
		RepoBasePPass:    make(map[int]float64),
		PipelineDepth:    depth,
		MaxMinibatchSize: maxMB,
		PendingChanges:   make([]*Change, 0, 1024),
		UseParallelMode:  false,
		rng:              rng,
	}
	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
	}
	return sq
}

// ResetStats clears the cumulative statistics, used after the priming phase.
func (sq *PipelinedSubmitQueue) ResetStats() {
	sq.TotalMinibatches = 0
	sq.PassedMinibatches = 0
	sq.TotalSubmitted = 0
	sq.TotalWaitTicks = 0
}

func (sq *PipelinedSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		sq.PendingChanges = append(sq.PendingChanges, NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs, sq.rng))
		sq.ChangeIDCounter++
	}
}

func (sq *PipelinedSubmitQueue) Step(currentTick int) int {
	if len(sq.PendingChanges) == 0 {
		return 0
	}
	if sq.UseParallelMode {
		return sq.stepParallel(currentTick)
	}
	return sq.stepSpeculative(currentTick)
}

func (sq *PipelinedSubmitQueue) stepSpeculative(currentTick int) int {
	pendingCount := len(sq.PendingChanges)

	// 1. Dynamic Batch Sizing
	rawSize := int(math.Ceil(float64(pendingCount) / float64(sq.PipelineDepth)))
	baseBatchSize := rawSize
	if baseBatchSize < 1 {
		baseBatchSize = 1
	}
	if baseBatchSize > sq.MaxMinibatchSize {
		baseBatchSize = sq.MaxMinibatchSize
	}

	var minibatches []Minibatch
	cumulativeCls := make([]*Change, 0, baseBatchSize*sq.PipelineDepth)

	for i := 0; i < sq.PipelineDepth; i++ {
		start := i * baseBatchSize
		end := start + baseBatchSize
		if start >= pendingCount {
			break
		}
		if end > pendingCount {
			end = pendingCount
		}
		newClsForLevel := sq.PendingChanges[start:end]
		cumulativeCls = append(cumulativeCls, newClsForLevel...)
		mbChanges := make([]*Change, len(cumulativeCls))
		copy(mbChanges, cumulativeCls)
		minibatches = append(minibatches, Minibatch{Changes: mbChanges, NewChanges: newClsForLevel})
	}

	// 2. Evaluation
	allPassed := true
	results := make([]struct{ passed, hardFail bool }, len(minibatches))
	for i, mb := range minibatches {
		p, h := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)
		results[i] = struct{ passed, hardFail bool }{p, h}

		// Stats: Track minibatch pass rates
		sq.TotalMinibatches++
		if p {
			sq.PassedMinibatches++
		} else {
			allPassed = false
		}
	}

	// 3. Greedy Merging
	submittedCount := 0
	passStreakBroken := false
	for i, res := range results {
		if passStreakBroken {
			if res.hardFail {
				// Fix hard breaks in speculated batches that we won't submit
				for _, cl := range minibatches[i].NewChanges {
					if cl.IsHardBreak() {
						cl.FixHardBreaks(sq.TestDefM, sq.TestDefs, sq.rng)
					}
				}
			}
			continue
		}
		if res.passed {
			for _, cl := range minibatches[i].NewChanges {
				sq.applyEffect(cl, currentTick)
			}
			submittedCount += len(minibatches[i].NewChanges)
		} else {
			passStreakBroken = true
			if res.hardFail {
				for _, cl := range minibatches[i].NewChanges {
					if cl.IsHardBreak() {
						cl.FixHardBreaks(sq.TestDefM, sq.TestDefs, sq.rng)
					}
				}
			}
		}
	}

	if submittedCount > 0 {
		sq.PendingChanges = sq.PendingChanges[submittedCount:]
		sq.TotalSubmitted += submittedCount
	}

	// Decision for next step: stay speculative only if everything passed
	sq.UseParallelMode = !allPassed
	return submittedCount
}

type TestResult struct {
	passed, hardFail bool
}

func (sq *PipelinedSubmitQueue) stepParallel(currentTick int) int {
	pendingCount := len(sq.PendingChanges)

	// 1. Dynamic Batch Sizing (Same as speculative)
	rawSize := int(math.Ceil(float64(pendingCount) / float64(sq.PipelineDepth)))
	baseBatchSize := rawSize
	if baseBatchSize < 1 {
		baseBatchSize = 1
	}
	if baseBatchSize > sq.MaxMinibatchSize {
		baseBatchSize = sq.MaxMinibatchSize
	}

	var minibatches []Minibatch
	batchIndices := make([]struct{ start, end int }, 0, sq.PipelineDepth)

	for i := 0; i < sq.PipelineDepth; i++ {
		start := i * baseBatchSize
		end := start + baseBatchSize
		if start >= pendingCount {
			break
		}
		if end > pendingCount {
			end = pendingCount
		}

		// Parallel: Independent batches
		changes := sq.PendingChanges[start:end]
		minibatches = append(minibatches, Minibatch{Changes: changes, NewChanges: changes})
		batchIndices = append(batchIndices, struct{ start, end int }{start, end})
	}

	// 2. Evaluation
	allPassed := true
	results := make([]TestResult, len(minibatches))
	for i, mb := range minibatches {
		p, hardFail := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)
		results[i] = TestResult{p, hardFail}
		sq.TotalMinibatches++
		if p {
			sq.PassedMinibatches++
		} else {
			allPassed = false
		}
	}

	// 3. Submit FIRST passing batch only
	submittedCount := 0
	submittedBatchIdx := -1
	for i, res := range results {
		if res.passed {
			if submittedBatchIdx == -1 {
				submittedBatchIdx = i
				for _, cl := range minibatches[i].Changes {
					sq.applyEffect(cl, currentTick)
				}
				submittedCount = len(minibatches[i].Changes)
			}
			// We don't submit subsequent passing batches in this mode,
			// but they don't need hard-fixes either.
		} else if res.hardFail {
			for _, cl := range minibatches[i].Changes {
				if cl.IsHardBreak() {
					cl.FixHardBreaks(sq.TestDefM, sq.TestDefs, sq.rng)
				}
			}
		}
	}

	if submittedCount > 0 {
		sq.TotalSubmitted += submittedCount
		start := batchIndices[submittedBatchIdx].start
		end := batchIndices[submittedBatchIdx].end
		newPending := make([]*Change, 0, len(sq.PendingChanges)-submittedCount)
		newPending = append(newPending, sq.PendingChanges[:start]...)
		newPending = append(newPending, sq.PendingChanges[end:]...)
		sq.PendingChanges = newPending
		sq.ApplyFlakyFixes(submittedCount)
	}

	// Decision for next step: return to speculative only if ALL parallel batches passed.
	// (This is a strict criterion, but appropriate if we want to avoid "flapping" between modes)
	sq.UseParallelMode = !allPassed
	return submittedCount
}

// ApplyFlakyFixes assumes that highly-flaky tests get fixed or demoted with per
// CL submitted with likelihood proportional to their failure rate (severity).
func (sq *PipelinedSubmitQueue) ApplyFlakyFixes(n int) {
	for t, passRate := range sq.RepoBasePPass {
		// Each CL has an independent chance; we can use Pow to simulate this.
		// Assume a baseline of ~1w to fix a 1% flaky test, which is 8400 CLs
		// So divide by n by 84 to scale appropriately.
		if sq.rng.Float64() > math.Pow(passRate, float64(n)/84) {
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

var nChangesPer2Hour = []int{5, 5, 5, 5, 60, 60, 60, 60, 10, 10, 10, 10}

const idealThroughput = 25 // per 2hour, 12.5 per h
const nSamples = 100

// SimConfig defines the parameters for one simulation run.
type SimConfig struct {
	SeqID       int // For ordered output
	Parallelism int
	Traffic     int
	NTests      int
	MaxBatch    int
}

// SimResult holds the results of one simulation run.
type SimResult struct {
	Config        SimConfig
	Slowdown      float64
	AvgQueueSize  float64
	MBPassRate    float64
	AvgSubmitTime float64
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
			// Only 1% of CLs affect a test's pass rate post-smoketest (TreeHugger)
			// This means 0.5% of CLs are "bad" (increase a test's failure rate).
			PAffected: 0.01,
			// Distribution of new pass rates after a transition
			PassRates: []DistEntry{
				// Allow a 50% chance for flake to randomly get fixed,
				// in addition to the chance to purposely get fixed.
				// See ApplyFlakyFixes
				{0.5, 1.0},
				// ~40% chance of a "low flake rate" between 0.5% and 5%
				{0.75, 0.995},
				{0.85, 0.98},
				{0.90, 0.95},
				// 8% chance of a "high flake rate" between 20% and 80%
				{0.95, 0.80},
				{0.98, 0.20},
				// Some failures are breakages which can be
				// culprit-found and quarantined. These are probably
				// mid-air collisions between LKGB and TOT
				{1.0, 0.0},
			},
		}
	}

	sq := NewPipelinedSubmitQueue(testDefs, cfg.Parallelism, cfg.MaxBatch, rng)

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

		// Apply load (identical logic for both phases)
		qSize := len(sq.PendingChanges)
		changesToAdd := cfg.Traffic * nChangesPer2Hour[i%12]
		if qSize >= 200 && qSize < 400 {
			changesToAdd /= 2
		} else if qSize >= 400 && qSize < 800 {
			changesToAdd /= 4
		} else if qSize >= 800 {
			changesToAdd /= 8
		}
		if changesToAdd > 0 {
			sq.AddChanges(changesToAdd, i)
		}
	}

	mbPassRate := float64(sq.PassedMinibatches) / float64(sq.TotalMinibatches)
	avgSubmitTime := 1.0 + float64(sq.TotalWaitTicks)/float64(sq.TotalSubmitted)

	throughput := float64(submittedTotal) / float64(nIter)
	slowdown := float64(idealThroughput*cfg.Traffic) / throughput

	return SimResult{
		Config:        cfg,
		Slowdown:      slowdown,
		AvgQueueSize:  float64(totalQ) / float64(nIter),
		MBPassRate:    mbPassRate,
		AvgSubmitTime: avgSubmitTime,
	}
}

// runAveragedSimulation performs nSamples of the simulation and returns averaged results.
func runAveragedSimulation(cfg SimConfig) SimResult {
	var sumSlowdown, sumQSize, sumMBPassRate, sumSubmitTime float64

	for i := 0; i < nSamples; i++ {
		// Generate a unique seed for every sample of every config
		seed := int64((cfg.SeqID*nSamples + i) * 997)
		res := runSimulation(cfg, seed)

		sumSlowdown += res.Slowdown
		sumQSize += res.AvgQueueSize
		sumMBPassRate += res.MBPassRate
		sumSubmitTime += res.AvgSubmitTime
	}

	return SimResult{
		Config:        cfg,
		Slowdown:      sumSlowdown / float64(nSamples),
		AvgQueueSize:  sumQSize / float64(nSamples),
		MBPassRate:    sumMBPassRate / float64(nSamples),
		AvgSubmitTime: sumSubmitTime / float64(nSamples),
	}
}

// printIncremental handles the stateful header printing and the result row.
func printIncremental(res SimResult, lastCfg *SimConfig) {
	cfg := res.Config
	// Print headers if major parameters changed compared to the last printed result
	if lastCfg == nil || cfg.Parallelism != lastCfg.Parallelism {
		fmt.Printf("Pipeline depth/parallelism: %d\n", cfg.Parallelism)
	}
	if lastCfg == nil || cfg.Parallelism != lastCfg.Parallelism || cfg.Traffic != lastCfg.Traffic {
		fmt.Printf("\nIdeal throughput: %d CLs/2hour = Productivity 1/1x \n", idealThroughput*cfg.Traffic)
		fmt.Printf("%-10s | %-22s | %-14s | %-9s | %s\n",
			"Max Batch", "Productivity * 1/x", "Avg Queue Size", "Pass Rate", "Avg Time to Submit (h)")
		fmt.Println("-------------------------------------------------------------------------------------------------------------")
	}
	if lastCfg == nil || cfg.Parallelism != lastCfg.Parallelism || cfg.Traffic != lastCfg.Traffic || cfg.NTests != lastCfg.NTests {
		fmt.Printf("n_tests: %d\n", cfg.NTests)
	}

	fmt.Printf("%-10d | %-22.2f | %-14.0f | %-9.2f | %.2f\n",
		cfg.MaxBatch, res.Slowdown, res.AvgQueueSize, res.MBPassRate, res.AvgSubmitTime)
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
	for _, parallelism := range []int{2, 4, 8} {
		for _, traffic := range []int{1, 2} {
			for _, nTests := range []int{16, 32, 64} {
				for _, maxBatch := range []int{32, 64, 128} {
					wg.Add(1)
					cfg := SimConfig{
						SeqID:       seqCounter,
						Parallelism: parallelism,
						Traffic:     traffic,
						NTests:      nTests,
						MaxBatch:    maxBatch,
					}
					go func(c SimConfig) {
						defer wg.Done()
						resultsCh <- runAveragedSimulation(c)
					}(cfg)
					seqCounter++
				}
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
