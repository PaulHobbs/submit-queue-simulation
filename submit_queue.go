package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// --- RNG & Basic Structures (Unchanged) ---

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

type DistEntry struct {
	Limit float64
	Value float64
}

type Distribution []DistEntry

type TestDefinition struct {
	ID        int
	PAffected float64
	PassRates Distribution
}

func sample(d Distribution, rng *FastRNG) float64 {
	s := rng.Float64()
	for _, e := range d {
		if s < e.Limit {
			return e.Value
		}
	}
	return d[len(d)-1].Value
}

type ChangeState int

const (
	StateQueued ChangeState = iota
	StateInBatch
	StateSuspect
	StateVerifying
	StateFixing
	StateSubmitted
)

type Change struct {
	ID           int
	CreationTick int
	Effects      map[int]float64
	State        ChangeState
	VerifyDoneTick int
	FixDoneTick    int
}

func NewChange(id, tick int, testDefs []TestDefinition, rng *FastRNG) *Change {
	c := &Change{
		ID:           id,
		CreationTick: tick,
		Effects:      make(map[int]float64, len(testDefs)/10),
	}

	// 3% chance to be a "Culprit" (Bad CL)
	const ProbCulprit = 0.03

	if rng.Float64() < ProbCulprit {
		for _, td := range testDefs {
			pCatchGivenCulprit := td.PAffected / ProbCulprit
			if pCatchGivenCulprit > 1.0 {
				pCatchGivenCulprit = 1.0
			}
			if rng.Float64() < pCatchGivenCulprit {
				c.Effects[td.ID] = sample(td.PassRates, rng)
			}
		}
	}
	return c
}

// --- Minibatch Logic ---

type Minibatch struct {
	Changes []*Change
	// Lane/Depth removed: CulpritQueue batches are flat and independent
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
		// Determine if test passes based on effective probability
		if effP < 1.0 && rng.Float64() >= effP {
			passed = false
			if hardFailure {
				break
			}
		}
	}
	return passed, hardFailure
}

// --- CulpritQueue Submit Queue (Sparse Parallel Group Testing) ---

type CulpritQueueSubmitQueue struct {
	// Configuration
	TestDefs         []TestDefinition
	AllTestIDs       []int
	ResourceBudget   int // N: Number of parallel batches
	RedundancyK      int // K: Number of batches each CL is assigned to
	MaxMinibatchSize int

	// State
	RepoBasePPass   map[int]float64
	PendingChanges  []*Change
	VerificationQueue []*Change
	FixingQueue       []*Change
	ChangeIDCounter int
	rng             *FastRNG

	// Statistics
	TotalMinibatches  int
	PassedMinibatches int
	TotalSubmitted    int
	TotalWaitTicks    int
	TotalVerifications int
	TotalVictims       int
}

func NewCulpritQueueSubmitQueue(testDefs []TestDefinition, resources, maxMB int, rng *FastRNG) *CulpritQueueSubmitQueue {
	sq := &CulpritQueueSubmitQueue{
		TestDefs:         testDefs,
		AllTestIDs:       make([]int, len(testDefs)),
		RepoBasePPass:    make(map[int]float64),
		ResourceBudget:   resources, // This is N
		RedundancyK:      4,         // K (Definite Defectives redundancy)
		MaxMinibatchSize: maxMB,
		PendingChanges:   make([]*Change, 0, 1024),
		VerificationQueue: make([]*Change, 0, 1024),
		FixingQueue:       make([]*Change, 0, 1024),
		rng:              rng,
	}

	// Adjust K if N is small
	K := sq.RedundancyK
	N := sq.ResourceBudget
	if N > 0 && K > N {
		K = int(N / 2)
	}
	// Ensure we don't drop below K=2 unless N is very small, 
	// otherwise intersection logic is weak.
	if N >= 2 && K < 2 {
		K = 2
	}
	if K < 1 {
		K = 1
	}

	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
	}
	return sq
}

func (sq *CulpritQueueSubmitQueue) ResetStats() {
	sq.TotalMinibatches = 0
	sq.PassedMinibatches = 0
	sq.TotalSubmitted = 0
	sq.TotalWaitTicks = 0
	sq.TotalVerifications = 0
	sq.TotalVictims = 0
}

func (sq *CulpritQueueSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		sq.PendingChanges = append(sq.PendingChanges, NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs, sq.rng))
		sq.ChangeIDCounter++
	}
}

// pickDistinct selects k distinct integers from range [0, n)
func pickDistinct(n, k int, rng *FastRNG) []int {
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	// Fisher-Yates shuffle partial
	for i := 0; i < k; i++ {
		j := i + int(rng.Float64()*float64(n-i))
		indices[i], indices[j] = indices[j], indices[i]
	}
	return indices[:k]
}

// IsCulprit checks if the change actually breaks a test (Hard Failure).
// In a real system, this requires a Bisection/Culprit Finder step.
// In this simulation, we peek at the state to decide if we should 
// Reject (Remove) or Retry (Re-queue) the suspect.
func (c *Change) IsCulprit() bool {
	for _, e := range c.Effects {
		if e == 0.0 {
			return true
		}
	}
	return false
}

func (sq *CulpritQueueSubmitQueue) processVerificationQueue(currentTick int) ([]*Change, int) {
	submitted := make([]*Change, 0)
	activeVerifications := 0
	nextQueue := make([]*Change, 0, len(sq.VerificationQueue))

	// Count currently active verifications to respect resource budget
	for _, cl := range sq.VerificationQueue {
		if cl.State == StateVerifying {
			activeVerifications++
		}
	}

	for _, cl := range sq.VerificationQueue {
		// Start verification if resources allow
		if cl.State == StateSuspect {
			if activeVerifications < sq.ResourceBudget * 16 {
				cl.State = StateVerifying
				cl.VerifyDoneTick = currentTick + 2 // VerificationLatency
				activeVerifications++
				sq.TotalVerifications++
			}
		}

		if cl.State == StateVerifying {
			if currentTick >= cl.VerifyDoneTick {
				// Verification Complete: Run Isolated Test
				mb := Minibatch{Changes: []*Change{cl}}
				passed, _ := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)

				if passed {
					cl.State = StateSubmitted
					sq.TotalVictims++
					submitted = append(submitted, cl)
				} else {
					cl.State = StateFixing
					cl.FixDoneTick = currentTick + 60 // FixDelay
					sq.FixingQueue = append(sq.FixingQueue, cl)
				}
				// Do not add to nextQueue
			} else {
				nextQueue = append(nextQueue, cl)
			}
		} else {
			nextQueue = append(nextQueue, cl)
		}
	}
	sq.VerificationQueue = nextQueue
	return submitted, activeVerifications
}

func (sq *CulpritQueueSubmitQueue) processFixingQueue(currentTick int) {
	nextQueue := make([]*Change, 0, len(sq.FixingQueue))
	for _, cl := range sq.FixingQueue {
		if currentTick >= cl.FixDoneTick {
			// Fixed: Resample and Re-queue as new change
			newCL := NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs, sq.rng)
			sq.ChangeIDCounter++
			sq.PendingChanges = append(sq.PendingChanges, newCL)
		} else {
			nextQueue = append(nextQueue, cl)
		}
	}
	sq.FixingQueue = nextQueue
}

func (sq *CulpritQueueSubmitQueue) Step(currentTick int) int {
	sq.processFixingQueue(currentTick)
	verifiedSubmitted, _ := sq.processVerificationQueue(currentTick)

	N := sq.ResourceBudget  // - int(float32(activeVerifications) / 16)
	K := sq.RedundancyK

	if N > 0 && K > N {  // Clamp K to be at most N / 2.
		K = int(N / 2)
	}
	// Ensure we don't drop below K=2 unless N is very small, 
	// otherwise intersection logic is weak.
	if N >= 2 && K < 2 {
		K = 2
	}
	if K < 1 {
		K = 1
	}

	// 1. Select Candidate CLs
	limit := N * sq.MaxMinibatchSize
	if limit > len(sq.PendingChanges) {
		limit = len(sq.PendingChanges)
	}

	// If no resources or no changes, just handle verified
	if N <= 0 || limit == 0 {
		for _, cl := range verifiedSubmitted {
			sq.applyEffect(cl, currentTick)
		}
		sq.TotalSubmitted += len(verifiedSubmitted)
		sq.ApplyFlakyFixes(len(verifiedSubmitted))
		return len(verifiedSubmitted)
	}

	activeCLs := sq.PendingChanges[:limit]

	// 2. Sparse Assignment with Orthogonality Check
	batches := make([][]*Change, N)
	clAssignments := make(map[*Change][]int, len(activeCLs))
	
	// Keep track of what we assigned to ensure separation
	assignedHistory := make([][]int, 0, len(activeCLs))

	for _, cl := range activeCLs {
		cl.State = StateInBatch
		
		// NEW: Use pickSeparated instead of pickDistinct
		assignedIndices := pickSeparated(N, K, sq.rng, assignedHistory)
		
		// Store for future comparisons
		assignedHistory = append(assignedHistory, assignedIndices)
		
		clAssignments[cl] = assignedIndices
		for _, batchIdx := range assignedIndices {
			batches[batchIdx] = append(batches[batchIdx], cl)
		}
	}

	// 3. Execute Batches (Parallel)
	batchPassed := make([]bool, N)
	for i := 0; i < N; i++ {
		mb := Minibatch{Changes: batches[i]}
		passed, _ := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs, sq.rng)
		batchPassed[i] = passed
		
		sq.TotalMinibatches++
		if passed {
			sq.PassedMinibatches++
		}
	}

	// 4. Decode & Rebuild Queue
	submittedChanges := make([]*Change, 0)
	
	// Start the new pending queue with the CLs we didn't touch this tick (overflow)
	newPendingChanges := make([]*Change, 0, len(sq.PendingChanges))
	if limit < len(sq.PendingChanges) {
		newPendingChanges = append(newPendingChanges, sq.PendingChanges[limit:]...)
	}

	for _, cl := range activeCLs {
		indices := clAssignments[cl]
		isCleared := false
		
		// Check if any assigned batch passed
		for _, idx := range indices {
			if batchPassed[idx] {
				isCleared = true
				break
			}
		}

		if isCleared {
			cl.State = StateSubmitted
			submittedChanges = append(submittedChanges, cl)
		} else {
			// Suspect: Move to VerificationQueue
			cl.State = StateSuspect
			sq.VerificationQueue = append(sq.VerificationQueue, cl)
		}
	}

	// Merge verified submissions
	submittedChanges = append(submittedChanges, verifiedSubmitted...)

	sq.PendingChanges = newPendingChanges

	// 5. Apply Effects
	for _, cl := range submittedChanges {
		sq.applyEffect(cl, currentTick)
	}
	
	sq.TotalSubmitted += len(submittedChanges)

	sq.ApplyFlakyFixes(len(submittedChanges))

	return len(submittedChanges)
}


func (sq *CulpritQueueSubmitQueue) ApplyFlakyFixes(n int) {
	for t, passRate := range sq.RepoBasePPass {
		if sq.rng.Float64() > math.Pow(passRate, float64(n)/84.0) {
			sq.RepoBasePPass[t] = 1
		}
	}
}

func (sq *CulpritQueueSubmitQueue) applyEffect(cl *Change, currentTick int) {
	for tid, effect := range cl.Effects {
		current := sq.RepoBasePPass[tid]
		if _, ok := sq.RepoBasePPass[tid]; !ok {
			current = 1.0
		}
		// We simulate that if a bad change lands, the repo stays broken
		// until fixed.
		if effect < current {
			sq.RepoBasePPass[tid] = effect
		}
	}
	sq.TotalWaitTicks += (currentTick - cl.CreationTick)
}

// --- Simulation & Reporting ---

var nChangesPer2Hour = []int{5, 5, 5, 5, 60, 60, 60, 60, 10, 10, 10, 10}

const idealThroughput = 25 
const nSamples = 9

type SimConfig struct {
	SeqID     int
	Resources int // N (Batches)
	Traffic   int
	NTests    int
	MaxBatch  int
}

type SimResult struct {
	Config        SimConfig
	Slowdown      float64
	AvgQueueSize  float64
	MBPassRate    float64
	VictimRate    float64
	AvgSubmitTime float64
	AvgRunsPerSubmitted float64
}

func runSimulation(cfg SimConfig, seed int64) SimResult {
	const primingIter = 3 * 12 * 7 
	const nIter = 60 * 12 * 7      
	rng := NewFastRNG(seed)

	testDefs := make([]TestDefinition, cfg.NTests)
	for i := 0; i < cfg.NTests; i++ {
		testDefs[i] = TestDefinition{
			ID: i,
			PAffected: 0.005,
			PassRates: []DistEntry{
				{0.5, 1.0},
				{0.55, 0.995},
				{0.59, 0.98},
				{0.60, 0.95},
				{0.64, 0.80},
				{0.68, 0.20},
				{1.0, 0.0},
			},
		}
	}

	// Use the new CulpritQueue queue
	sq := NewCulpritQueueSubmitQueue(testDefs, cfg.Resources, cfg.MaxBatch, rng)
	
	sq.AddChanges(cfg.Traffic*nChangesPer2Hour[len(nChangesPer2Hour)-1], 0)

	totalQ := 0
	submittedTotal := 0

	for i := 0; i < primingIter+nIter; i++ {
		if i == primingIter {
			sq.ResetStats()
		}

		submitted := sq.Step(i)

		if i >= primingIter {
			submittedTotal += submitted
			totalQ += len(sq.PendingChanges)
		}

		qSize := len(sq.PendingChanges)
		changesToAdd := cfg.Traffic * nChangesPer2Hour[i%12]

		// Backpressure
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

	victimRate := 0.0
	if sq.TotalVerifications > 0 {
		victimRate = float64(sq.TotalVictims) / float64(sq.TotalVerifications)
	}

	avgRuns := 0.0
	if sq.TotalSubmitted > 0 {
		avgRuns = (float64(sq.TotalMinibatches) + float64(sq.TotalVerifications) / 16) / float64(sq.TotalSubmitted)
	}

	return SimResult{
		Config:        cfg,
		Slowdown:      slowdown,
		AvgQueueSize:  float64(totalQ) / float64(nIter),
		MBPassRate:    mbPassRate,
		VictimRate:    victimRate,
		AvgSubmitTime: avgSubmitTime,
		AvgRunsPerSubmitted: avgRuns,
	}
}

func runAveragedSimulation(cfg SimConfig) SimResult {
	var sumSlowdown, sumQSize, sumMBPassRate, sumSubmitTime, sumVictimRate, sumBuilds float64

	for i := 0; i < nSamples; i++ {
		seed := int64((cfg.SeqID*nSamples + i) * 997)
		res := runSimulation(cfg, seed)

		sumSlowdown += res.Slowdown
		sumQSize += res.AvgQueueSize
		sumMBPassRate += res.MBPassRate
		sumSubmitTime += res.AvgSubmitTime
		sumVictimRate += res.VictimRate
		sumBuilds += res.AvgRunsPerSubmitted
	}

	return SimResult{
		Config:        cfg,
		Slowdown:      sumSlowdown / float64(nSamples),
		AvgQueueSize:  sumQSize / float64(nSamples),
		MBPassRate:    sumMBPassRate / float64(nSamples),
		VictimRate:    sumVictimRate / float64(nSamples),
		AvgSubmitTime: sumSubmitTime / float64(nSamples),
		AvgRunsPerSubmitted: sumBuilds / float64(nSamples),
	}
}

func printIncremental(res SimResult, lastCfg *SimConfig) {
	cfg := res.Config
	if lastCfg == nil || cfg.Resources != lastCfg.Resources {
		fmt.Printf("CulpritQueue Resources (N Batches): %d\n", cfg.Resources)
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic {
		fmt.Printf("\nIdeal throughput: %d CLs/2hour\n", idealThroughput*cfg.Traffic)
		fmt.Printf("%-10s | %-12s | %-14s | %-9s | %-9s | %-10s | %s\n",
			"Max Batch", "Slowdown", "Avg Queue", "Pass Rate", "Victim%", "Runs/CL", "Avg Time (h)")
		fmt.Println("---------------------------------------------------------------------------------------------")
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic || cfg.NTests != lastCfg.NTests {
		fmt.Printf("n_tests: %d\n", cfg.NTests)
	}

	fmt.Printf("%-10d | %-12.2f | %-14.0f | %-9.2f | %-9.3f | %-10.2f | %.2f\n",
		cfg.MaxBatch, res.Slowdown, res.AvgQueueSize, res.MBPassRate, res.VictimRate, res.AvgRunsPerSubmitted, res.AvgSubmitTime)
}

func main() {
	resultsCh := make(chan SimResult, 100)
	var wg sync.WaitGroup
	start := time.Now()

	printDone := make(chan struct{})
	go func() {
		defer close(printDone)
		buffer := make(map[int]SimResult)
		nextExpectedID := 0
		var lastCfg *SimConfig

		for res := range resultsCh {
			buffer[res.Config.SeqID] = res
			for {
				if nextRes, ok := buffer[nextExpectedID]; ok {
					printIncremental(nextRes, lastCfg)
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

	seqCounter := 0
	// We simulate N=8 * traffic parallel batches.
	resources := 32

	for _, traffic := range []int{4, 8, 16} {
		for _, nTests := range []int{16, 32, 64} {
			for _, maxBatch := range []int{1024} {
				wg.Add(1)
				cfg := SimConfig{
					SeqID:     seqCounter,
					Resources: resources * traffic,
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

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	<-printDone
	fmt.Printf("\nAll CulpritQueue simulations complete in %v.\n", time.Since(start))
}

// computeOverlap calculates how many batches two assignments share.
// Assumes sorted inputs for O(K) efficiency, but for small K, O(K^2) is negligible.
func computeOverlap(a, b []int) int {
	count := 0
	// Since K is small (4-8), nested loop is faster than allocating maps
	for _, valA := range a {
		for _, valB := range b {
			if valA == valB {
				count++
				break // Optimization: move to next valA
			}
		}
	}
	return count
}

// pickSeparated tries to find a set of K indices that overlaps as little as possible
// with any existing assignment in currentBatchAssignments.
func pickSeparated(N, K int, rng *FastRNG, existingAssignments [][]int) []int {
	const MaxRetries = 20
	
	var bestAssignment []int
	minMaxOverlap := K + 1 // Start higher than possible

	for try := 0; try < MaxRetries; try++ {
		// Generate candidate
		candidate := pickDistinct(N, K, rng)
		
		// Check against neighbors
		currentMaxOverlap := 0
		for _, other := range existingAssignments {
			overlap := computeOverlap(candidate, other)
			if overlap > currentMaxOverlap {
				currentMaxOverlap = overlap
			}
			// Pruning: If we already found a bad overlap, this candidate is useless
			// unless it's better than our global best, but usually we just want "Good Enough"
			if currentMaxOverlap >= minMaxOverlap {
				break
			}
		}

		// Perfect case: Overlap is 0 or 1.
		// (Overlap 1 is usually acceptable and unavoidable in sparse designs).
		if currentMaxOverlap <= 1 {
			return candidate
		}

		// Keep track of the best we've seen so far (fallback)
		if currentMaxOverlap < minMaxOverlap {
			minMaxOverlap = currentMaxOverlap
			bestAssignment = candidate
		}
	}
	
	return bestAssignment
}