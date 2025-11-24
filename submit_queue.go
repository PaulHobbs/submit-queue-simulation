package main

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"sync"
	"time"
)

const ProbCulprit = 0.03

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
	ID             int
	CreationTick   int
	Effects        map[int]float64
	State          ChangeState
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

// --- MATRIX OPTIMIZER (High Performance Bitsets) ---

type MatrixKey struct {
	Rows   int
	Cols   int
	Weight int
	Optimize bool
}

var matrixCache sync.Map // Thread-safe memoization

// Matrix represents our testing grid using bitsets.
type Matrix struct {
	Cols      [][]uint64
	NumTests  int
	NumItems  int
	ColWeight int
	RowChunks int
}

func GetCachedMatrix(rows, cols, weight int, optimize bool) *Matrix {
	key := MatrixKey{rows, cols, weight, optimize}
	if val, ok := matrixCache.Load(key); ok {
		return val.(*Matrix)
	}

	// Not found, compute it (Optimize)
	// We use a fresh RNG for matrix generation to ensure deterministic structure 
	// based on the key if we wanted, but here we just need *a* good matrix.
	mat := NewMatrix(rows, cols, weight)
	if optimize {
		mat.Optimize(0.1) // Spend up to 0.1s optimizing this matrix configuration
	}
	
	matrixCache.Store(key, mat)
	return mat
}

func NewMatrix(rows, cols, weight int) *Matrix {
	chunks := (rows + 63) / 64
	m := &Matrix{
		Cols:      make([][]uint64, cols),
		NumTests:  rows,
		NumItems:  cols,
		ColWeight: weight,
		RowChunks: chunks,
	}
	for i := 0; i < cols; i++ {
		m.Cols[i] = make([]uint64, chunks)
		m.randomizeColumn(i)
	}
	return m
}

func (m *Matrix) randomizeColumn(colIdx int) {
	for k := 0; k < m.RowChunks; k++ {
		m.Cols[colIdx][k] = 0
	}
	setCount := 0
	// Simple random fill
	for setCount < m.ColWeight {
		r := rand.Intn(m.NumTests)
		chunk := r / 64
		bit := uint64(1) << (r % 64)
		if (m.Cols[colIdx][chunk] & bit) == 0 {
			m.Cols[colIdx][chunk] |= bit
			setCount++
		}
	}
}

// Optimize runs the Greedy Swap / Electron Repulsion algorithm
func (m *Matrix) Optimize(seconds float64) {
	startTime := time.Now()
	timeout := time.Duration(seconds * float64(time.Second))

	for time.Since(startTime) < timeout {
		maxOverlap, worstPair := m.MaxOverlap()
		if maxOverlap <= 1 {
			// Theoretical optimum reached for most sparse matrices
			break
		}

		colA := worstPair[0]
		colB := worstPair[1]

		// Find collisions
		collisionRows := m.findCollisions(colA, colB)
		if len(collisionRows) == 0 {
			continue
		}

		// Perturb colA
		rowToRemove := collisionRows[rand.Intn(len(collisionRows))]
		rowToAdd := m.findRandomEmptyRow(colA)

		// Execute Swap
		m.flipBit(colA, rowToRemove, 0)
		m.flipBit(colA, rowToAdd, 1)

		// Check if improved (Greedy)
		newOverlap, _ := m.MaxOverlap()
		if newOverlap > maxOverlap {
			// Revert
			m.flipBit(colA, rowToAdd, 0)
			m.flipBit(colA, rowToRemove, 1)
		}
	}
}

func (m *Matrix) MaxOverlap() (int, [2]int) {
	maxVal := 0
	pair := [2]int{0, 0}
	for i := 0; i < m.NumItems; i++ {
		for j := i + 1; j < m.NumItems; j++ {
			overlap := 0
			for k := 0; k < m.RowChunks; k++ {
				overlap += bits.OnesCount64(m.Cols[i][k] & m.Cols[j][k])
			}
			if overlap > maxVal {
				maxVal = overlap
				pair[0], pair[1] = i, j
			}
		}
	}
	return maxVal, pair
}

func (m *Matrix) findCollisions(idxA, idxB int) []int {
	var collisions []int
	for k := 0; k < m.RowChunks; k++ {
		common := m.Cols[idxA][k] & m.Cols[idxB][k]
		if common == 0 {
			continue
		}
		for bit := 0; bit < 64; bit++ {
			if (common & (uint64(1) << bit)) != 0 {
				rowAbs := k*64 + bit
				if rowAbs < m.NumTests {
					collisions = append(collisions, rowAbs)
				}
			}
		}
	}
	return collisions
}

func (m *Matrix) findRandomEmptyRow(colIdx int) int {
	for {
		r := rand.Intn(m.NumTests)
		chunk := r / 64
		bit := uint64(1) << (r % 64)
		if (m.Cols[colIdx][chunk] & bit) == 0 {
			return r
		}
	}
}

func (m *Matrix) flipBit(colIdx, rowIdx int, val int) {
	chunk := rowIdx / 64
	bit := uint64(1) << (rowIdx % 64)
	if val == 1 {
		m.Cols[colIdx][chunk] |= bit
	} else {
		m.Cols[colIdx][chunk] &^= bit
	}
}

func (m *Matrix) GetColumnIndices(colIdx int) []int {
	var indices []int
	for k := 0; k < m.RowChunks; k++ {
		for bit := 0; bit < 64; bit++ {
			mask := uint64(1) << bit
			if (m.Cols[colIdx][k] & mask) != 0 {
				rowAbs := k*64 + bit
				if rowAbs < m.NumTests {
					indices = append(indices, rowAbs)
				}
			}
		}
	}
	return indices
}

// --- Minibatch Logic ---

type Minibatch struct {
	Changes []*Change
}

func (mb *Minibatch) Evaluate(repoBasePPass []float64, allTestIDs []int, rng *FastRNG) (bool, bool, []int) {
	passed := true
	hardFailure := false
	var failedTests []int

	for _, tid := range allTestIDs {
		effP := repoBasePPass[tid]
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
			failedTests = append(failedTests, tid)
			if hardFailure {
				break
			}
		}
	}
	return passed, hardFailure, failedTests
}

// --- CulpritQueue Submit Queue (Sparse Parallel Group Testing) ---

type CulpritQueueSubmitQueue struct {
	// Configuration
	TestDefs         []TestDefinition
	AllTestIDs       []int
	ResourceBudget   int // N: Number of parallel batches
	MaxMinibatchSize int
	UseOptimizedMatrix bool
	MaxK             int
	KDivisor         int
	FlakeTolerance   float64

	// State
	RepoBasePPass      []float64
	PostsubmitFailureRate []float64
	activeTestIDs      []int
	PendingChanges     []*Change
	VerificationQueue  []*Change
	FixingQueue        []*Change
	ChangeIDCounter    int
	rng                *FastRNG

	// Statistics
	TotalMinibatches   int
	PassedMinibatches  int
	TotalSubmitted     int
	TotalWaitTicks     int
	TotalVerifications int
	TotalVictims       int
}

func NewCulpritQueueSubmitQueue(testDefs []TestDefinition, resources, maxMB int, rng *FastRNG, useOptimizedMatrix bool, maxK, kDivisor int, flakeTolerance float64) *CulpritQueueSubmitQueue {
	sq := &CulpritQueueSubmitQueue{
		TestDefs:          testDefs,
		AllTestIDs:        make([]int, len(testDefs)),
		RepoBasePPass:     make([]float64, len(testDefs)),
		PostsubmitFailureRate: make([]float64, len(testDefs)),
		activeTestIDs:     make([]int, 0, len(testDefs)),
		ResourceBudget:    resources, // This is N
		MaxMinibatchSize:  maxMB,
		PendingChanges:    make([]*Change, 0, 1024),
		VerificationQueue: make([]*Change, 0, 1024),
		FixingQueue:       make([]*Change, 0, 1024),
		rng:               rng,
		UseOptimizedMatrix: useOptimizedMatrix,
		MaxK:              maxK,
		KDivisor:          kDivisor,
		FlakeTolerance:    flakeTolerance,
	}

	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
		sq.PostsubmitFailureRate[t.ID] = 0.0
		sq.activeTestIDs = append(sq.activeTestIDs, t.ID)
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

func (sq *CulpritQueueSubmitQueue) getActiveTestIDs() []int {
	return sq.activeTestIDs
}

func (sq *CulpritQueueSubmitQueue) updateFailureRate(tid int, observed float64) {
	const alpha = 0.05
	sq.PostsubmitFailureRate[tid] = alpha*observed + (1-alpha)*sq.PostsubmitFailureRate[tid]
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

	activeTestIDs := sq.getActiveTestIDs()

	for _, cl := range sq.VerificationQueue {
		// Start verification if resources allow
		if cl.State == StateSuspect {
			if activeVerifications < sq.ResourceBudget*16 {
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
				passed, _, _ := mb.Evaluate(sq.RepoBasePPass, activeTestIDs, sq.rng)

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

	// 1. Select Candidate CLs
	limit := sq.MaxMinibatchSize
	if limit > len(sq.PendingChanges) {
		limit = len(sq.PendingChanges)
	}

	// Dynamic sizing of N (Resources) based on load
	N := int(float32(limit) / 2)
	if N == 0 {
		N = sq.ResourceBudget
	}
	
	// Ensure we don't exceed physical resource budget for parallelism
	// (Simulation simplification: we treat N as "tests in a compressed matrix")
	
	if N <= 0 || limit == 0 {
		for _, cl := range verifiedSubmitted {
			sq.applyEffect(cl, currentTick)
		}
		sq.TotalSubmitted += len(verifiedSubmitted)
		sq.ApplyFlakyFixes(len(verifiedSubmitted))
		return len(verifiedSubmitted)
	}

	activeCLs := sq.PendingChanges[:limit]
	
	// Determine Sparsity K
	K := sq.MaxK
	if K >= int(float32(N) / float32(sq.KDivisor)) {
		K = int(float32(N) / float32(sq.KDivisor))
	}
	if N >= 2 && K < 2 {
		K = 2
	}
	if K < 1 {
		K = 1
	}

	// 2. SPARSE ASSIGNMENT (Using Memoized Optimized Matrix)
	// We request a matrix optimized for:
	// Rows = N
	// Cols = MaxMinibatchSize (We generate a matrix wide enough for max load)
	// Weight = K
	
	optMatrix := GetCachedMatrix(N, sq.MaxMinibatchSize, K, sq.UseOptimizedMatrix)
	
	batches := make([][]*Change, N)
	clAssignments := make(map[*Change][]int, len(activeCLs))

	for i, cl := range activeCLs {
		cl.State = StateInBatch
		
		// Use the optimized column corresponding to the CL's index
		// Note: indices returned are 0..N-1
		assignedIndices := optMatrix.GetColumnIndices(i)
		
		clAssignments[cl] = assignedIndices
		for _, batchIdx := range assignedIndices {
			if batchIdx < N { // Safety check against N resizing
				batches[batchIdx] = append(batches[batchIdx], cl)
			}
		}
	}

	// 3. Execute Batches (Parallel)
	activeTestIDs := sq.getActiveTestIDs()

	batchPassed := make([]bool, N)
	batchFailedTests := make([][]int, N)

	for i := 0; i < N; i++ {
		mb := Minibatch{Changes: batches[i]}
		passed, _, failedTests := mb.Evaluate(sq.RepoBasePPass, activeTestIDs, sq.rng)
		batchPassed[i] = passed
		batchFailedTests[i] = failedTests
		
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
			if idx < len(batchPassed) && batchPassed[idx] {
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

	// Update stats from innocent batches (all CLs submitted)
	if len(submittedChanges) > 0 {
		submittedSet := make(map[int]bool, len(submittedChanges))
		for _, cl := range submittedChanges {
			submittedSet[cl.ID] = true
		}

		for i := 0; i < N; i++ {
			if len(batches[i]) == 0 {
				continue
			}
			allInnocent := true
			for _, cl := range batches[i] {
				if !submittedSet[cl.ID] {
					allInnocent = false
					break
				}
			}
			if allInnocent {
				failedMap := make(map[int]bool)
				for _, tid := range batchFailedTests[i] {
					failedMap[tid] = true
				}
				for _, tid := range activeTestIDs {
					observed := 0.0
					if failedMap[tid] {
						observed = 1.0
					}
					sq.updateFailureRate(tid, observed)
				}
			}
		}
	}

	if len(submittedChanges) > 0 {
		sq.runPostsubmit()
	}

	return len(submittedChanges)
}


func (sq *CulpritQueueSubmitQueue) ApplyFlakyFixes(n int) {
	for t, passRate := range sq.RepoBasePPass {
		if sq.rng.Float64() > math.Pow(passRate, float64(n)/84.0) {
			sq.RepoBasePPass[t] = 1
		}
	}
}

func (sq *CulpritQueueSubmitQueue) runPostsubmit() {
	
	newActive := make([]int, 0, len(sq.AllTestIDs))
	
	for _, tid := range sq.AllTestIDs {
		p := sq.RepoBasePPass[tid]

		passed := true
		if p < 1.0 && sq.rng.Float64() >= p {
			passed = false
		}

		observed := 0.0
		if !passed {
			observed = 1.0
		}

		sq.updateFailureRate(tid, observed)
		
		if sq.PostsubmitFailureRate[tid] <= sq.FlakeTolerance {
			newActive = append(newActive, tid)
		}
	}
	sq.activeTestIDs = newActive
}

func (sq *CulpritQueueSubmitQueue) applyEffect(cl *Change, currentTick int) {
	for tid, effect := range cl.Effects {
		current := sq.RepoBasePPass[tid]
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
const nSamples = 100

type SimConfig struct {
	SeqID     int
	Resources int // N (Batches)
	Traffic   int
	NTests    int
	MaxBatch  int
	UseOptimizedMatrix bool
	MaxK      int
	KDivisor  int
	FlakeTolerance float64
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

	sq := NewCulpritQueueSubmitQueue(testDefs, cfg.Resources, cfg.MaxBatch, rng, cfg.UseOptimizedMatrix, cfg.MaxK, cfg.KDivisor, cfg.FlakeTolerance)
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
		fmt.Printf("%-10s | %-5s | %-5s | %-5s | %-10s | %-12s | %-14s | %-9s | %-9s | %-10s | %s\n",
			"Optimized", "MaxK", "Div", "Flake", "Max Batch", "Slowdown", "Avg Queue", "Pass Rate", "Victim%", "Runs/CL", "Avg Time (h)")
		fmt.Println("-----------------------------------------------------------------------------------------------------------------------------------")
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic || cfg.NTests != lastCfg.NTests {
		fmt.Printf("n_tests: %d\n", cfg.NTests)
	}

	fmt.Printf("%-10v | %-5d | %-5d | %-5.2f | %-10d | %-12.2f | %-14.0f | %-9.2f | %-9.1f | %-10.2f | %.2f\n",
		cfg.UseOptimizedMatrix, cfg.MaxK, cfg.KDivisor, cfg.FlakeTolerance, cfg.MaxBatch, res.Slowdown, res.AvgQueueSize, res.MBPassRate, 100 * res.VictimRate, res.AvgRunsPerSubmitted, res.AvgSubmitTime)
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
	// We allow for up to N=resources * traffic parallel batches.
	resources := 4

	for _, traffic := range []int{8} {
		for _, optimized := range []bool{true} {
			for _, nTests := range []int{32} {
				for _, maxBatch := range []int{2048} {
					for _, maxK := range []int{12} {
						for _, kDiv := range []int{5} {
							for _, flakeTol := range []float64{0.05, 0.10, 0.15} {
								wg.Add(1)
								cfg := SimConfig{
									SeqID:              seqCounter,
									Resources:          resources * traffic,
									Traffic:            traffic,
									NTests:             nTests,
									MaxBatch:           maxBatch,
									UseOptimizedMatrix: optimized,
									MaxK:               maxK,
									KDivisor:           kDiv,
									FlakeTolerance:     flakeTol,
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
