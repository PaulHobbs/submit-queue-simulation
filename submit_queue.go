package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"os"
	"runtime/pprof"
	"sort"
	"strings"
	"sync"
	"time"
)

var ProbCulprit = 0.03 // Default culprit probability, can be overridden

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

// Cache statistics
var (
	matrixCacheHits   int64
	matrixCacheMisses int64
	cacheMutex        sync.Mutex
)

// quantizeValue rounds values to nearby quantized levels for better cache hit rates.
// For small values (<20), returns exact value.
// For larger values, rounds to ±5% tolerance on exponential scale.
func quantizeValue(val int) int {
	if val < 20 {
		return val // Keep exact for small values (cheap to optimize)
	}

	// Use exponential quantization: round to nearest value in geometric sequence
	// Scale factor chosen to give ~5% granularity
	scale := 1.05 // ~5% steps

	// Find the nearest quantized value
	// log(val) / log(scale) gives us the "index" in the sequence
	index := math.Log(float64(val)) / math.Log(scale)
	roundedIndex := math.Round(index)
	quantized := int(math.Pow(scale, roundedIndex))

	return quantized
}

// Matrix represents our testing grid using bitsets.
type Matrix struct {
	Cols      [][]uint64
	NumTests  int
	NumItems  int
	ColWeight int
	RowChunks int
}

func GetCachedMatrix(rows, cols, weight int, optimize bool) *Matrix {
	// N and K should already be quantized by caller
	key := MatrixKey{rows, cols, weight, optimize}
	if val, ok := matrixCache.Load(key); ok {
		cacheMutex.Lock()
		matrixCacheHits++
		cacheMutex.Unlock()
		return val.(*Matrix)
	}

	// Not found, compute it (Optimize)
	cacheMutex.Lock()
	matrixCacheMisses++
	cacheMutex.Unlock()

	// We use a fresh RNG for matrix generation to ensure deterministic structure
	// based on the key if we wanted, but here we just need *a* good matrix.
	mat := NewMatrix(rows, cols, weight)
	if optimize {
		mat.Optimize(1)
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
func (m *Matrix) Optimize(iterations int) {
	for i := 0; i < iterations; i++ {
		maxOverlap, worstPair := m.MaxOverlap()
		if maxOverlap <= 1 {
			// Theoretical optimum reached for most sparse matrices
			break
		}

		colA := worstPair[0]
		colB := worstPair[1]

		// Find collisions
		collisionRows := m.findCollisionRows(colA, colB)
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

// OptimizeHighDensity runs a more advanced optimization algorithm for potentially higher density matrices.
func (m *Matrix) OptimizeHighDensity(iterations int) {
	numCols := m.NumItems // 100

	// To break out of local minima
	stagnationCounter := 0

	// Track global best to know when to stop or re-roll
	currentMaxOverlap, _ := m.MaxOverlap()

	for i := 0; i < iterations; i++ {
		// 1. Pick a random pivot column
		colA := rand.Intn(numCols)

		// 2. Find the WORST enemy of colA (Linear scan O(C))
		colB, maxLocalOverlap := m.findWorstNeighbor(colA)

		// OPTIMIZATION: Adaptive Threshold
		// If colA's worst situation is already better than our global average/target,
		// don't waste time fixing it. Pick a different random column.
		if maxLocalOverlap < currentMaxOverlap-1 {
			continue
		}

		// 3. Attempt to fix colA relative to colB
		improved := m.attemptTargetedSwap(colA, colB, maxLocalOverlap)

		if improved {
			// Update our knowledge of the global state lazily
			stagnationCounter = 0
		} else {
			stagnationCounter++
		}

		// 4. Anti-Stagnation: If we can't improve, shake the matrix
		// With high density, we get stuck in local optima easily.
		if stagnationCounter > numCols*2 {
			m.randomizeColumn(rand.Intn(numCols)) // Nuke a random column
			stagnationCounter = 0
			// Re-calculate global baseline
			currentMaxOverlap, _ = m.MaxOverlap()
		}
	}
}

// findWorstNeighbor scans all columns to find who hurts colIdx the most.
// Complexity: O(C) - very fast for C=100
func (m *Matrix) findWorstNeighbor(colIdx int) (int, int) {
	worstCol := -1
	maxO := -1

	for j := 0; j < m.NumItems; j++ {
		if colIdx == j {
			continue
		}
		// Inline overlap calc for speed
		overlap := 0
		for k := 0; k < m.RowChunks; k++ {
			overlap += bits.OnesCount64(m.Cols[colIdx][k] & m.Cols[j][k])
		}

		if overlap > maxO {
			maxO = overlap
			worstCol = j
		}
	}
	return worstCol, maxO
}

// attemptTargetedSwap tries to move a bit from colA that collides with colB
// to a spot that doesn't create a WORSE collision with someone else.
func (m *Matrix) attemptTargetedSwap(colA, colB, oldOverlap int) bool {
	// 1. Identify colliding rows (Candidates to REMOVE)
	collisions := m.findCollisionRows(colA, colB)
	if len(collisions) == 0 {
		return false
	}

	// 2. Identify empty rows in A (Candidates to ADD)
	empties := m.findEmptyRows(colA)
	if len(empties) == 0 {
		return false
	}

	// 3. Try X random swaps to see if one improves the score
	currentScore := oldOverlap

	for attempt := 0; attempt < 5; attempt++ {
		rowOut := collisions[rand.Intn(len(collisions))]
		rowIn := empties[rand.Intn(len(empties))]

		// Provisional Flip
		m.flipBit(colA, rowOut, 0)
		m.flipBit(colA, rowIn, 1)

		// Check new Max Overlap for A
		_, newMaxOverlap := m.findWorstNeighbor(colA)

		if newMaxOverlap < currentScore {
			// We found an improvement! Keep it.
			return true
		}

		// Revert
		m.flipBit(colA, rowIn, 0)
		m.flipBit(colA, rowOut, 1)
	}

	return false
}

func (m *Matrix) findEmptyRows(colIdx int) []int {
	// Pre-allocating for performance assuming ~60% empty
	empties := make([]int, 0, m.NumTests/2)

	for k := 0; k < m.RowChunks; k++ {
		// If chunk is all 1s (unlikely but possible), skip
		if m.Cols[colIdx][k] == ^uint64(0) {
			continue
		}
		for bit := 0; bit < 64; bit++ {
			rowAbs := k*64 + bit
			if rowAbs >= m.NumTests {
				break
			}
			if (m.Cols[colIdx][k] & (uint64(1) << bit)) == 0 {
				empties = append(empties, rowAbs)
			}
		}
	}
	return empties
}

func (m *Matrix) MaxOverlap() (int, [2]int) {
	maxVal := 0
	pair := [2]int{0, 0}
	cols := m.Cols // Cache to reduce slice access overhead
	chunks := m.RowChunks

	for i := 0; i < m.NumItems; i++ {
		colI := cols[i]
		for j := i + 1; j < m.NumItems; j++ {
			overlap := 0
			colJ := cols[j]

			// Compute overlap with cached column refs
			for k := 0; k < chunks; k++ {
				overlap += bits.OnesCount64(colI[k] & colJ[k])
			}

			if overlap > maxVal {
				maxVal = overlap
				pair[0], pair[1] = i, j
			}
		}
	}
	return maxVal, pair
}

// findCollisionRows finds rows where both colA and colB have a bit set.
func (m *Matrix) findCollisionRows(idxA, idxB int) []int {
	collisions := make([]int, 0, 8) // Pre-allocate with reasonable capacity
	colA := m.Cols[idxA]
	colB := m.Cols[idxB]

	for k := 0; k < m.RowChunks; k++ {
		common := colA[k] & colB[k]
		if common == 0 {
			continue
		}
		// Extract set bits
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
	// Pre-allocate with exact capacity (ColWeight)
	indices := make([]int, 0, m.ColWeight)
	for k := 0; k < m.RowChunks; k++ {
		chunk := m.Cols[colIdx][k]
		if chunk == 0 {
			continue // Skip empty chunks
		}
		// Process bits in this chunk
		for bit := 0; bit < 64; bit++ {
			if (chunk & (uint64(1) << bit)) != 0 {
				rowAbs := k*64 + bit
				if rowAbs < m.NumTests {
					indices = append(indices, rowAbs)
					// Early exit optimization - we know the exact count
					if len(indices) == m.ColWeight {
						return indices
					}
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
	failedTests := make([]int, 0, 8) // Pre-allocate with reasonable capacity

	for _, tid := range allTestIDs {
		effP := repoBasePPass[tid]
		// Optimize: Check all changes for this test
		for _, cl := range mb.Changes {
			if eff, ok := cl.Effects[tid]; ok {
				if eff == 0.0 {
					effP = 0.0
					hardFailure = true
					break // Early exit if hard failure found
				}
				if eff < effP {
					effP = eff
				}
			}
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
	ImplicitParams   ImplicitParams

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

	// Enhanced statistics
	TotalCulpritsCreated int
	TotalCulpritsCaught  int
	TotalInnocentFlagged int
	WaitTimes            []int
	QueueDepths          []int
	MaxQueueDepth        int
	TotalBatchSlots      int
	UsedBatchSlots       int
	MaxVerificationQueue int
	TestDemotionEvents   int
}

func NewCulpritQueueSubmitQueue(testDefs []TestDefinition, resources, maxMB int, rng *FastRNG, useOptimizedMatrix bool, maxK, kDivisor int, flakeTolerance float64, implicitParams ImplicitParams) *CulpritQueueSubmitQueue {
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
		ImplicitParams:    implicitParams,
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
	sq.TotalCulpritsCreated = 0
	sq.TotalCulpritsCaught = 0
	sq.TotalInnocentFlagged = 0
	sq.WaitTimes = make([]int, 0, 10000)
	sq.QueueDepths = make([]int, 0, 10000)
	sq.MaxQueueDepth = 0
	sq.TotalBatchSlots = 0
	sq.UsedBatchSlots = 0
	sq.MaxVerificationQueue = 0
	sq.TestDemotionEvents = 0
}

func (sq *CulpritQueueSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		cl := NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs, sq.rng)
		sq.PendingChanges = append(sq.PendingChanges, cl)
		sq.ChangeIDCounter++

		// Track culprits created
		if cl.IsCulprit() {
			sq.TotalCulpritsCreated++
		}
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
			if activeVerifications < sq.ResourceBudget*sq.ImplicitParams.VerifyResourceMult {
				cl.State = StateVerifying
				cl.VerifyDoneTick = currentTick + sq.ImplicitParams.VerifyLatency
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
					sq.TotalInnocentFlagged++
					submitted = append(submitted, cl)
				} else {
					cl.State = StateFixing
					cl.FixDoneTick = currentTick + sq.ImplicitParams.FixDelay
					sq.FixingQueue = append(sq.FixingQueue, cl)
					sq.TotalCulpritsCaught++
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

	// Track queue depths
	queueDepth := len(sq.PendingChanges)
	sq.QueueDepths = append(sq.QueueDepths, queueDepth)
	if queueDepth > sq.MaxQueueDepth {
		sq.MaxQueueDepth = queueDepth
	}

	// Track verification queue
	verifyQueueDepth := len(sq.VerificationQueue)
	if verifyQueueDepth > sq.MaxVerificationQueue {
		sq.MaxVerificationQueue = verifyQueueDepth
	}

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

	// Quantize N early for cache efficiency (±5% tolerance for large N)
	N = quantizeValue(N)

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

	// Quantize K early for cache efficiency
	K = quantizeValue(K)

	// 2. SPARSE ASSIGNMENT (Using Memoized Optimized Matrix)
	// We request a matrix optimized for:
	// Rows = N
	// Cols = MaxMinibatchSize (We generate a matrix wide enough for max load)
	// Weight = K

	optMatrix := GetCachedMatrix(N, sq.MaxMinibatchSize, K, sq.UseOptimizedMatrix)

	// Pre-allocate batches with estimated capacity
	batches := make([][]*Change, N)
	for i := 0; i < N; i++ {
		batches[i] = make([]*Change, 0, limit/N+1)
	}
	clAssignments := make(map[*Change][]int, len(activeCLs))

	for i, cl := range activeCLs {
		cl.State = StateInBatch

		// Use the optimized column corresponding to the CL's index
		// Note: indices returned are 0..N-1
		assignedIndices := optMatrix.GetColumnIndices(i)

		clAssignments[cl] = assignedIndices
		for _, batchIdx := range assignedIndices {
			if batchIdx < N { // Safety check
				batches[batchIdx] = append(batches[batchIdx], cl)
			}
		}
	}

	// 3. Execute Batches (Parallel)
	activeTestIDs := sq.getActiveTestIDs()

	batchPassed := make([]bool, N)
	batchFailedTests := make([][]int, N)

	// Track batch utilization
	sq.TotalBatchSlots += N
	usedBatches := 0

	// Evaluate batches
	for i := 0; i < N; i++ {
		if len(batches[i]) == 0 {
			batchPassed[i] = true // Empty batches pass
			continue
		}
		usedBatches++
		mb := Minibatch{Changes: batches[i]}
		passed, _, failedTests := mb.Evaluate(sq.RepoBasePPass, activeTestIDs, sq.rng)
		batchPassed[i] = passed
		batchFailedTests[i] = failedTests

		sq.TotalMinibatches++
		if passed {
			sq.PassedMinibatches++
		}
	}
	sq.UsedBatchSlots += usedBatches

	// 4. Decode & Rebuild Queue
	submittedChanges := make([]*Change, 0, limit) // Pre-allocate with max possible size

	// Start the new pending queue with the CLs we didn't touch this tick (overflow)
	newPendingChanges := make([]*Change, 0, len(sq.PendingChanges))
	if limit < len(sq.PendingChanges) {
		newPendingChanges = append(newPendingChanges, sq.PendingChanges[limit:]...)
	}

	// Process changes in batch
	for _, cl := range activeCLs {
		indices := clAssignments[cl]

		// Check if any assigned batch passed
		isCleared := false
		for _, idx := range indices {
			if batchPassed[idx] { // idx bounds already checked during batch creation
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
	if len(submittedChanges) > 0 && len(submittedChanges) <= limit {
		// Build set of submitted IDs (only if reasonably sized)
		submittedSet := make(map[int]bool, len(submittedChanges))
		for _, cl := range submittedChanges {
			submittedSet[cl.ID] = true
		}

		for i := 0; i < N; i++ {
			batchLen := len(batches[i])
			if batchLen == 0 {
				continue
			}

			// Check if all CLs in this batch were submitted
			allInnocent := true
			for _, cl := range batches[i] {
				if !submittedSet[cl.ID] {
					allInnocent = false
					break
				}
			}

			if allInnocent && len(batchFailedTests[i]) > 0 {
				// Build failure map only if needed
				failedMap := make(map[int]bool, len(batchFailedTests[i]))
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
			} else if allInnocent {
				// No failures - update all with 0.0
				for _, tid := range activeTestIDs {
					sq.updateFailureRate(tid, 0.0)
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
	oldActiveCount := len(sq.activeTestIDs)

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

	// Track test demotions
	if len(newActive) < oldActiveCount {
		sq.TestDemotionEvents += oldActiveCount - len(newActive)
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
	waitTime := currentTick - cl.CreationTick
	sq.TotalWaitTicks += waitTime
	sq.WaitTimes = append(sq.WaitTimes, waitTime)
}

// --- ASCII Visualization Helpers ---

// percentile calculates the nth percentile of a sorted slice
func percentile(sorted []int, p float64) int {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	idx := p * float64(len(sorted)-1)
	lower := int(idx)
	upper := lower + 1
	if upper >= len(sorted) {
		return sorted[lower]
	}
	fraction := idx - float64(lower)
	return int(float64(sorted[lower])*(1-fraction) + float64(sorted[upper])*fraction)
}

// sparkline creates a mini ASCII graph using block characters
func sparkline(values []int, width int) string {
	if len(values) == 0 {
		return ""
	}

	// Sample the values if we have more than width
	sampled := make([]int, width)
	if len(values) <= width {
		copy(sampled, values)
		sampled = sampled[:len(values)]
	} else {
		step := float64(len(values)) / float64(width)
		for i := 0; i < width; i++ {
			idx := int(float64(i) * step)
			sampled[i] = values[idx]
		}
	}

	// Find min/max for scaling
	minVal, maxVal := sampled[0], sampled[0]
	for _, v := range sampled {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	if minVal == maxVal {
		return strings.Repeat("▄", len(sampled))
	}

	// Map to spark characters
	chars := []rune{'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'}
	var result strings.Builder
	for _, v := range sampled {
		normalized := float64(v-minVal) / float64(maxVal-minVal)
		idx := int(normalized * float64(len(chars)-1))
		result.WriteRune(chars[idx])
	}
	return result.String()
}

// histogram creates an ASCII histogram with given number of buckets
func histogram(values []int, buckets int, width int) string {
	if len(values) == 0 || buckets == 0 {
		return ""
	}

	// Find min/max
	minVal, maxVal := values[0], values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	if minVal == maxVal {
		return fmt.Sprintf("  All values: %d\n", minVal)
	}

	// Create buckets
	counts := make([]int, buckets)
	bucketSize := float64(maxVal-minVal) / float64(buckets)

	for _, v := range values {
		bucketIdx := int(float64(v-minVal) / bucketSize)
		if bucketIdx >= buckets {
			bucketIdx = buckets - 1
		}
		counts[bucketIdx]++
	}

	// Find max count for scaling
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}

	// Build histogram
	var result strings.Builder
	blockChars := []rune{'░', '▒', '▓', '█'}

	for i := 0; i < buckets; i++ {
		rangeStart := minVal + int(float64(i)*bucketSize)
		rangeEnd := minVal + int(float64(i+1)*bucketSize)

		// Calculate bar length
		barLength := 0
		if maxCount > 0 {
			barLength = int(float64(counts[i]) / float64(maxCount) * float64(width))
		}

		// Create bar with gradient
		var bar strings.Builder
		for j := 0; j < barLength; j++ {
			progress := float64(j) / float64(width)
			charIdx := int(progress * float64(len(blockChars)-1))
			if charIdx >= len(blockChars) {
				charIdx = len(blockChars) - 1
			}
			bar.WriteRune(blockChars[charIdx])
		}

		result.WriteString(fmt.Sprintf("  %5d-%5d │%-*s│ %d\n",
			rangeStart, rangeEnd, width, bar.String(), counts[i]))
	}

	return result.String()
}

// barChart creates a simple horizontal bar chart
func barChart(label string, value, max float64, width int) string {
	if max == 0 {
		max = 1
	}
	filled := int(value / max * float64(width))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}

	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	percentage := value / max * 100
	return fmt.Sprintf("  %-20s │%s│ %6.1f%%", label, bar, percentage)
}

// --- Simulation & Reporting ---

var nChangesPer2Hour = []int{5, 5, 5, 5, 60, 60, 60, 60, 10, 10, 10, 10}

const idealThroughput = 25
const nSamples = 32  // Reduced from 100 for faster testing during optimization

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

type ImplicitParams struct {
	VerifyLatency          int
	FixDelay               int
	VerifyResourceMult     int
	BackpressureThreshold1 int
	BackpressureThreshold2 int
	BackpressureThreshold3 int
}

type SimResult struct {
	Config              SimConfig
	Slowdown            float64
	AvgQueueSize        float64
	MBPassRate          float64
	VictimRate          float64
	AvgSubmitTime       float64
	AvgRunsPerSubmitted float64

	// Enhanced statistics
	CulpritsCreated     int
	CulpritsCaught      int
	InnocentFlagged     int
	FalseNegativeRate   float64
	TruePositiveRate    float64
	WaitTimeP50         int
	WaitTimeP95         int
	WaitTimeP99         int
	WaitTimeMin         int
	WaitTimeMax         int
	QueueDepthSparkline string
	MaxQueueDepth       int
	BatchUtilization    float64
	MaxVerifyQueue      int
	ActiveTests         int
	DemotedTests        int
	WaitTimeHistogram   string
}

func runSimulationWithStability(cfg SimConfig, seed int64, testStability float64, implicitParams ImplicitParams) SimResult {
	// Reduced iterations for faster testing during optimization
	// Original: primingIter = 3 * 12 * 7 = 252, nIter = 60 * 12 * 7 = 5040
	const primingIter = 3 * 12 * 7     // 36 iterations (was 252)
	const nIter = 60 * 12 * 7          // 720 iterations (was 5040)
	rng := NewFastRNG(seed)

	testDefs := make([]TestDefinition, cfg.NTests)
	for i := 0; i < cfg.NTests; i++ {
		// Scale pass rates by stability factor
		// testStability = 1.0: normal
		// testStability < 1.0: flakier tests (lower pass rates)
		basePassRates := []DistEntry{
			{0.5, 1.0},
			{0.55, 0.995},
			{0.59, 0.98},
			{0.60, 0.95},
			{0.64, 0.80},
			{0.68, 0.20},
			{1.0, 0.0},
		}

		// Scale pass rates: move them closer to 0 if stability < 1.0
		scaledPassRates := make([]DistEntry, len(basePassRates))
		for j, entry := range basePassRates {
			scaledValue := entry.Value * testStability
			if scaledValue < 0 {
				scaledValue = 0
			}
			scaledPassRates[j] = DistEntry{
				Limit: entry.Limit,
				Value: scaledValue,
			}
		}

		testDefs[i] = TestDefinition{
			ID: i,
			PAffected: 0.005,
			PassRates: scaledPassRates,
		}
	}

	sq := NewCulpritQueueSubmitQueue(testDefs, cfg.Resources, cfg.MaxBatch, rng, cfg.UseOptimizedMatrix, cfg.MaxK, cfg.KDivisor, cfg.FlakeTolerance, implicitParams)
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
		if qSize >= implicitParams.BackpressureThreshold1 && qSize < implicitParams.BackpressureThreshold2 {
			changesToAdd /= 2
		}
		if qSize >= implicitParams.BackpressureThreshold2 && qSize < implicitParams.BackpressureThreshold3 {
			changesToAdd /= 4
		}
		if qSize >= implicitParams.BackpressureThreshold3 {
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

	// Calculate culprit detection stats
	falseNegativeRate := 0.0
	if sq.TotalCulpritsCreated > 0 {
		culpritsEscaped := sq.TotalCulpritsCreated - sq.TotalCulpritsCaught
		falseNegativeRate = float64(culpritsEscaped) / float64(sq.TotalCulpritsCreated)
	}

	truePositiveRate := 0.0
	totalFlagged := sq.TotalCulpritsCaught + sq.TotalInnocentFlagged
	if totalFlagged > 0 {
		truePositiveRate = float64(sq.TotalCulpritsCaught) / float64(totalFlagged)
	}

	// Calculate wait time percentiles
	sortedWaitTimes := make([]int, len(sq.WaitTimes))
	copy(sortedWaitTimes, sq.WaitTimes)
	sort.Ints(sortedWaitTimes)

	waitP50 := percentile(sortedWaitTimes, 0.50)
	waitP95 := percentile(sortedWaitTimes, 0.95)
	waitP99 := percentile(sortedWaitTimes, 0.99)
	waitMin := 0
	waitMax := 0
	if len(sortedWaitTimes) > 0 {
		waitMin = sortedWaitTimes[0]
		waitMax = sortedWaitTimes[len(sortedWaitTimes)-1]
	}

	// Create sparkline for queue depth
	queueSparkline := sparkline(sq.QueueDepths, 50)

	// Calculate batch utilization
	batchUtil := 0.0
	if sq.TotalBatchSlots > 0 {
		batchUtil = float64(sq.UsedBatchSlots) / float64(sq.TotalBatchSlots)
	}

	// Create wait time histogram
	waitHistogram := histogram(sortedWaitTimes, 8, 30)

	return SimResult{
		Config:              cfg,
		Slowdown:            slowdown,
		AvgQueueSize:        float64(totalQ) / float64(nIter),
		MBPassRate:          mbPassRate,
		VictimRate:          victimRate,
		AvgSubmitTime:       avgSubmitTime,
		AvgRunsPerSubmitted: avgRuns,
		CulpritsCreated:     sq.TotalCulpritsCreated,
		CulpritsCaught:      sq.TotalCulpritsCaught,
		InnocentFlagged:     sq.TotalInnocentFlagged,
		FalseNegativeRate:   falseNegativeRate,
		TruePositiveRate:    truePositiveRate,
		WaitTimeP50:         waitP50,
		WaitTimeP95:         waitP95,
		WaitTimeP99:         waitP99,
		WaitTimeMin:         waitMin,
		WaitTimeMax:         waitMax,
		QueueDepthSparkline: queueSparkline,
		MaxQueueDepth:       sq.MaxQueueDepth,
		BatchUtilization:    batchUtil,
		MaxVerifyQueue:      sq.MaxVerificationQueue,
		ActiveTests:         len(sq.activeTestIDs),
		DemotedTests:        sq.TestDemotionEvents,
		WaitTimeHistogram:   waitHistogram,
	}
}

func runSimulation(cfg SimConfig, seed int64) SimResult {
	defaultImplicitParams := ImplicitParams{
		VerifyLatency:          2,
		FixDelay:               60,
		VerifyResourceMult:     16,
		BackpressureThreshold1: 200,
		BackpressureThreshold2: 400,
		BackpressureThreshold3: 800,
	}
	return runSimulationWithStability(cfg, seed, 1.0, defaultImplicitParams)
}

func runSimulationOriginal(cfg SimConfig, seed int64, implicitParams ImplicitParams) SimResult {
	// Reduced iterations for faster testing during optimization
	// Original: primingIter = 3 * 12 * 7 = 252, nIter = 60 * 12 * 7 = 5040
	const primingIter = 3 * 12 * 7     // 36 iterations (was 252)
	const nIter = 60 * 12 * 7          // 720 iterations (was 5040)
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

	sq := NewCulpritQueueSubmitQueue(testDefs, cfg.Resources, cfg.MaxBatch, rng, cfg.UseOptimizedMatrix, cfg.MaxK, cfg.KDivisor, cfg.FlakeTolerance, implicitParams)
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
		if qSize >= implicitParams.BackpressureThreshold1 && qSize < implicitParams.BackpressureThreshold2 {
			changesToAdd /= 2
		}
		if qSize >= implicitParams.BackpressureThreshold2 && qSize < implicitParams.BackpressureThreshold3 {
			changesToAdd /= 4
		}
		if qSize >= implicitParams.BackpressureThreshold3 {
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

	// Calculate culprit detection stats
	falseNegativeRate := 0.0
	if sq.TotalCulpritsCreated > 0 {
		culpritsEscaped := sq.TotalCulpritsCreated - sq.TotalCulpritsCaught
		falseNegativeRate = float64(culpritsEscaped) / float64(sq.TotalCulpritsCreated)
	}

	truePositiveRate := 0.0
	totalFlagged := sq.TotalCulpritsCaught + sq.TotalInnocentFlagged
	if totalFlagged > 0 {
		truePositiveRate = float64(sq.TotalCulpritsCaught) / float64(totalFlagged)
	}

	// Calculate wait time percentiles
	sortedWaitTimes := make([]int, len(sq.WaitTimes))
	copy(sortedWaitTimes, sq.WaitTimes)
	sort.Ints(sortedWaitTimes)

	waitP50 := percentile(sortedWaitTimes, 0.50)
	waitP95 := percentile(sortedWaitTimes, 0.95)
	waitP99 := percentile(sortedWaitTimes, 0.99)
	waitMin := 0
	waitMax := 0
	if len(sortedWaitTimes) > 0 {
		waitMin = sortedWaitTimes[0]
		waitMax = sortedWaitTimes[len(sortedWaitTimes)-1]
	}

	// Create sparkline for queue depth
	queueSparkline := sparkline(sq.QueueDepths, 50)

	// Calculate batch utilization
	batchUtil := 0.0
	if sq.TotalBatchSlots > 0 {
		batchUtil = float64(sq.UsedBatchSlots) / float64(sq.TotalBatchSlots)
	}

	// Create wait time histogram
	waitHistogram := histogram(sortedWaitTimes, 8, 30)

	return SimResult{
		Config:              cfg,
		Slowdown:            slowdown,
		AvgQueueSize:        float64(totalQ) / float64(nIter),
		MBPassRate:          mbPassRate,
		VictimRate:          victimRate,
		AvgSubmitTime:       avgSubmitTime,
		AvgRunsPerSubmitted: avgRuns,
		CulpritsCreated:     sq.TotalCulpritsCreated,
		CulpritsCaught:      sq.TotalCulpritsCaught,
		InnocentFlagged:     sq.TotalInnocentFlagged,
		FalseNegativeRate:   falseNegativeRate,
		TruePositiveRate:    truePositiveRate,
		WaitTimeP50:         waitP50,
		WaitTimeP95:         waitP95,
		WaitTimeP99:         waitP99,
		WaitTimeMin:         waitMin,
		WaitTimeMax:         waitMax,
		QueueDepthSparkline: queueSparkline,
		MaxQueueDepth:       sq.MaxQueueDepth,
		BatchUtilization:    batchUtil,
		MaxVerifyQueue:      sq.MaxVerificationQueue,
		ActiveTests:         len(sq.activeTestIDs),
		DemotedTests:        sq.TestDemotionEvents,
		WaitTimeHistogram:   waitHistogram,
	}
}

func runAveragedSimulation(cfg SimConfig) SimResult {
	var sumSlowdown, sumQSize, sumMBPassRate, sumSubmitTime, sumVictimRate, sumBuilds float64
	var sumCulpritsCreated, sumCulpritsCaught, sumInnocentFlagged int
	var sumFalseNegRate, sumTruePositiveRate float64
	var sumWaitP50, sumWaitP95, sumWaitP99, sumWaitMin, sumWaitMax int
	var sumMaxQueue, sumMaxVerify, sumActiveTests, sumDemotedTests int
	var sumBatchUtil float64
	var lastSparkline, lastHistogram string

	// Use goroutines to gather samples in parallel
	resultsChan := make(chan SimResult, nSamples)
	var wg sync.WaitGroup

	for i := 0; i < nSamples; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			seed := int64((cfg.SeqID*nSamples + idx) * 997)
			res := runSimulation(cfg, seed)
			resultsChan <- res
		}(i)
	}

	// Close channel once all goroutines complete
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect and aggregate results
	for res := range resultsChan {
		sumSlowdown += res.Slowdown
		sumQSize += res.AvgQueueSize
		sumMBPassRate += res.MBPassRate
		sumSubmitTime += res.AvgSubmitTime
		sumVictimRate += res.VictimRate
		sumBuilds += res.AvgRunsPerSubmitted

		sumCulpritsCreated += res.CulpritsCreated
		sumCulpritsCaught += res.CulpritsCaught
		sumInnocentFlagged += res.InnocentFlagged
		sumFalseNegRate += res.FalseNegativeRate
		sumTruePositiveRate += res.TruePositiveRate
		sumWaitP50 += res.WaitTimeP50
		sumWaitP95 += res.WaitTimeP95
		sumWaitP99 += res.WaitTimeP99
		sumWaitMin += res.WaitTimeMin
		sumWaitMax += res.WaitTimeMax
		sumMaxQueue += res.MaxQueueDepth
		sumMaxVerify += res.MaxVerifyQueue
		sumActiveTests += res.ActiveTests
		sumDemotedTests += res.DemotedTests
		sumBatchUtil += res.BatchUtilization

		lastSparkline = res.QueueDepthSparkline
		lastHistogram = res.WaitTimeHistogram
	}

	return SimResult{
		Config:              cfg,
		Slowdown:            sumSlowdown / float64(nSamples),
		AvgQueueSize:        sumQSize / float64(nSamples),
		MBPassRate:          sumMBPassRate / float64(nSamples),
		VictimRate:          sumVictimRate / float64(nSamples),
		AvgSubmitTime:       sumSubmitTime / float64(nSamples),
		AvgRunsPerSubmitted: sumBuilds / float64(nSamples),
		CulpritsCreated:     sumCulpritsCreated / nSamples,
		CulpritsCaught:      sumCulpritsCaught / nSamples,
		InnocentFlagged:     sumInnocentFlagged / nSamples,
		FalseNegativeRate:   sumFalseNegRate / float64(nSamples),
		TruePositiveRate:    sumTruePositiveRate / float64(nSamples),
		WaitTimeP50:         sumWaitP50 / nSamples,
		WaitTimeP95:         sumWaitP95 / nSamples,
		WaitTimeP99:         sumWaitP99 / nSamples,
		WaitTimeMin:         sumWaitMin / nSamples,
		WaitTimeMax:         sumWaitMax / nSamples,
		QueueDepthSparkline: lastSparkline,
		MaxQueueDepth:       sumMaxQueue / nSamples,
		BatchUtilization:    sumBatchUtil / float64(nSamples),
		MaxVerifyQueue:      sumMaxVerify / nSamples,
		ActiveTests:         sumActiveTests / nSamples,
		DemotedTests:        sumDemotedTests / nSamples,
		WaitTimeHistogram:   lastHistogram,
	}
}

func printDetailedStats(res SimResult) {
	fmt.Printf("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  DETAILED STATISTICS - FlakeTolerance: %.2f                                                                                      ║\n", res.Config.FlakeTolerance)
	fmt.Printf("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")

	// Culprit Detection
	fmt.Printf("║                                                                                                                                ║\n")
	fmt.Printf("║  🎯 CULPRIT DETECTION                                                                                                          ║\n")
	fmt.Printf("║    Culprits Created:    %8d                                                                                                ║\n", res.CulpritsCreated)
	fmt.Printf("║    Culprits Caught:     %8d                                                                                                ║\n", res.CulpritsCaught)
	fmt.Printf("║    Innocent Flagged:    %8d                                                                                                ║\n", res.InnocentFlagged)
	fmt.Printf("║    False Negative Rate: %7.1f%%                                                                                                ║\n", res.FalseNegativeRate*100)
	fmt.Printf("║    True Positive Rate:  %7.1f%%                                                                                                ║\n", res.TruePositiveRate*100)
	fmt.Printf("║                                                                                                                                ║\n")

	// Latency Distribution
	fmt.Printf("║  ⏱️  LATENCY DISTRIBUTION (ticks)                                                                                               ║\n")
	fmt.Printf("║    Min:  %6d  │  P50:  %6d  │  P95:  %6d  │  P99:  %6d  │  Max:  %6d                                        ║\n",
		res.WaitTimeMin, res.WaitTimeP50, res.WaitTimeP95, res.WaitTimeP99, res.WaitTimeMax)
	fmt.Printf("║                                                                                                                                ║\n")

	// Queue Health
	fmt.Printf("║  📊 QUEUE HEALTH                                                                                                               ║\n")
	fmt.Printf("║    Average Queue Depth: %8.0f                                                                                                 ║\n", res.AvgQueueSize)
	fmt.Printf("║    Max Queue Depth:     %8d                                                                                                 ║\n", res.MaxQueueDepth)
	fmt.Printf("║    Max Verify Queue:    %8d                                                                                                 ║\n", res.MaxVerifyQueue)
	fmt.Printf("║    Queue Depth Over Time (50 samples):                                                                                         ║\n")
	fmt.Printf("║    %s                                                                                 ║\n", res.QueueDepthSparkline)
	fmt.Printf("║                                                                                                                                ║\n")

	// Resource Utilization
	fmt.Printf("║  ⚙️  RESOURCE UTILIZATION                                                                                                       ║\n")
	fmt.Printf("║    Batch Utilization:   %7.1f%%                                                                                                ║\n", res.BatchUtilization*100)
	fmt.Printf("║%s║\n", barChart("Batch Usage", res.BatchUtilization, 1.0, 40))
	fmt.Printf("║                                                                                                                                ║\n")

	// Test Health
	fmt.Printf("║  🧪 TEST HEALTH                                                                                                                ║\n")
	fmt.Printf("║    Active Tests:        %8d / %d                                                                                            ║\n", res.ActiveTests, res.Config.NTests)
	fmt.Printf("║    Demoted Tests:       %8d                                                                                                 ║\n", res.DemotedTests)
	activePercentage := float64(res.ActiveTests) / float64(res.Config.NTests)
	fmt.Printf("║%s║\n", barChart("Active Tests", activePercentage, 1.0, 40))
	fmt.Printf("║                                                                                                                                ║\n")

	// Wait Time Histogram
	fmt.Printf("║  📈 WAIT TIME DISTRIBUTION                                                                                                     ║\n")
	histLines := strings.Split(strings.TrimRight(res.WaitTimeHistogram, "\n"), "\n")
	for _, line := range histLines {
		if line != "" {
			fmt.Printf("║%-127s║\n", line)
		}
	}

	fmt.Printf("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n")
}

func printIncremental(res SimResult, lastCfg *SimConfig) {
	cfg := res.Config
	if lastCfg == nil || cfg.Resources != lastCfg.Resources {
		fmt.Printf("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
		fmt.Printf("║  CulpritQueue Resources (N Batches): %-3d                                                                                      ║\n", cfg.Resources)
		fmt.Printf("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n")
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic {
		fmt.Printf("\n  Ideal Throughput: %d CLs/2hour\n\n", idealThroughput*cfg.Traffic)
		fmt.Printf("┌────────────┬───────┬───────┬───────┬────────────┬──────────────┬────────────────┬───────────┬───────────┬────────────┬────────────────┐\n")
		fmt.Printf("│ %-10s │ %-5s │ %-5s │ %-5s │ %-10s │ %-12s │ %-14s │ %-9s │ %-9s │ %-10s │ %-14s │\n",
			"Optimized", "MaxK", "Div", "Flake", "Max Batch", "Slowdown", "Avg Queue", "Pass Rate", "Victim%", "Runs/CL", "Avg Time (h)")
		fmt.Printf("├────────────┼───────┼───────┼───────┼────────────┼──────────────┼────────────────┼───────────┼───────────┼────────────┼────────────────┤\n")
	}
	if lastCfg == nil || cfg.Resources != lastCfg.Resources || cfg.Traffic != lastCfg.Traffic || cfg.NTests != lastCfg.NTests {
		fmt.Printf("│ Tests: %-3d                                                                                                                     │\n", cfg.NTests)
		fmt.Printf("├────────────┼───────┼───────┼───────┼────────────┼──────────────┼────────────────┼───────────┼───────────┼────────────┼────────────────┤\n")
	}

	fmt.Printf("│ %-10v │ %5d │ %5d │ %5.2f │ %10d │ %12.2f │ %14.0f │ %9.2f │ %8.1f%% │ %10.2f │ %14.2f │\n",
		cfg.UseOptimizedMatrix, cfg.MaxK, cfg.KDivisor, cfg.FlakeTolerance, cfg.MaxBatch, res.Slowdown, res.AvgQueueSize, res.MBPassRate, 100*res.VictimRate, res.AvgRunsPerSubmitted, res.AvgSubmitTime)

	// Print detailed stats for this configuration
	printDetailedStats(res)
}

// JSONResult represents the simulation result in JSON format
type JSONResult struct {
	Config              SimConfig `json:"config"`
	Slowdown            float64   `json:"slowdown"`
	AvgQueueSize        float64   `json:"avg_queue_size"`
	MBPassRate          float64   `json:"mb_pass_rate"`
	VictimRate          float64   `json:"victim_rate"`
	AvgSubmitTime       float64   `json:"avg_submit_time"`
	AvgRunsPerSubmitted float64   `json:"avg_runs_per_submitted"`
	CulpritsCreated     int       `json:"culprits_created"`
	CulpritsCaught      int       `json:"culprits_caught"`
	InnocentFlagged     int       `json:"innocent_flagged"`
	FalseNegativeRate   float64   `json:"false_negative_rate"`
	TruePositiveRate    float64   `json:"true_positive_rate"`
	WaitTimeP50         int       `json:"wait_time_p50"`
	WaitTimeP95         int       `json:"wait_time_p95"`
	WaitTimeP99         int       `json:"wait_time_p99"`
	MaxQueueDepth       int       `json:"max_queue_depth"`
	BatchUtilization    float64   `json:"batch_utilization"`
	MaxVerifyQueue      int       `json:"max_verify_queue"`
	ActiveTests         int       `json:"active_tests"`
	DemotedTests        int       `json:"demoted_tests"`
}

func main() {
	// Command-line flags for optimizer
	jsonOutput := flag.Bool("json", false, "Output results as JSON")
	resources := flag.Int("resources", 32, "Number of parallel batches")
	traffic := flag.Int("traffic", 8, "Traffic multiplier")
	nTests := flag.Int("ntests", 32, "Number of tests")
	maxBatch := flag.Int("maxbatch", 2048, "Maximum batch size")
	maxK := flag.Int("maxk", 12, "Maximum K sparsity")
	kDiv := flag.Int("kdiv", 5, "K divisor")
	flakeTol := flag.Float64("flaketol", 0.10, "Flake tolerance")
	useOpt := flag.Bool("optimized", false, "Use optimized matrix")
	seed := flag.Int64("seed", 0, "Random seed (0 for auto)")

	// Scenario variation parameters
	culpritProb := flag.Float64("culprit-prob", 0.03, "Probability that a CL is a culprit")
	testStability := flag.Float64("test-stability", 1.0, "Test stability multiplier (1.0=normal, <1.0=flakier)")

	// Level 2: Implicit hyperparameters (previously hardcoded)
	verifyLatency := flag.Int("verify-latency", 2, "Ticks to verify a suspect CL")
	fixDelay := flag.Int("fix-delay", 60, "Ticks to fix a culprit CL")
	verifyResourceMult := flag.Int("verify-resource-mult", 16, "Resource budget multiplier for verification")
	backpressureThreshold1 := flag.Int("bp-threshold-1", 200, "First backpressure threshold")
	backpressureThreshold2 := flag.Int("bp-threshold-2", 400, "Second backpressure threshold")
	backpressureThreshold3 := flag.Int("bp-threshold-3", 800, "Third backpressure threshold")

	flag.Parse()

	// Apply scenario parameters
	ProbCulprit = *culpritProb

	// Store implicit params for passing to simulation
	implicitParams := ImplicitParams{
		VerifyLatency:          *verifyLatency,
		FixDelay:               *fixDelay,
		VerifyResourceMult:     *verifyResourceMult,
		BackpressureThreshold1: *backpressureThreshold1,
		BackpressureThreshold2: *backpressureThreshold2,
		BackpressureThreshold3: *backpressureThreshold3,
	}

	// If JSON mode, run single simulation with given parameters
	if *jsonOutput {
		cfg := SimConfig{
			SeqID:              0,
			Resources:          *resources,
			Traffic:            *traffic,
			NTests:             *nTests,
			MaxBatch:           *maxBatch,
			UseOptimizedMatrix: *useOpt,
			MaxK:               *maxK,
			KDivisor:           *kDiv,
			FlakeTolerance:     *flakeTol,
		}

		if *seed == 0 {
			*seed = time.Now().UnixNano()
		}

		res := runSimulationWithStability(cfg, *seed, *testStability, implicitParams)

		jsonRes := JSONResult{
			Config:              res.Config,
			Slowdown:            res.Slowdown,
			AvgQueueSize:        res.AvgQueueSize,
			MBPassRate:          res.MBPassRate,
			VictimRate:          res.VictimRate,
			AvgSubmitTime:       res.AvgSubmitTime,
			AvgRunsPerSubmitted: res.AvgRunsPerSubmitted,
			CulpritsCreated:     res.CulpritsCreated,
			CulpritsCaught:      res.CulpritsCaught,
			InnocentFlagged:     res.InnocentFlagged,
			FalseNegativeRate:   res.FalseNegativeRate,
			TruePositiveRate:    res.TruePositiveRate,
			WaitTimeP50:         res.WaitTimeP50,
			WaitTimeP95:         res.WaitTimeP95,
			WaitTimeP99:         res.WaitTimeP99,
			MaxQueueDepth:       res.MaxQueueDepth,
			BatchUtilization:    res.BatchUtilization,
			MaxVerifyQueue:      res.MaxVerifyQueue,
			ActiveTests:         res.ActiveTests,
			DemotedTests:        res.DemotedTests,
		}

		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(jsonRes); err != nil {
			fmt.Fprintf(os.Stderr, "error encoding JSON: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Normal mode: run the original sweep
	// Enable CPU profiling if CPUPROFILE env var is set
	if cpuprofile := os.Getenv("CPUPROFILE"); cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "could not create CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "could not start CPU profile: %v\n", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

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
	// We allow for up to N=resourceMult * traffic parallel batches.
	resourceMult := 4

	for _, trafficLevel := range []int{8} {
		for _, optimized := range []bool{true} {
			for _, numTests := range []int{32} {
				for _, maxBatchSize := range []int{2048} {
					for _, maxKVal := range []int{12} {
						for _, kDivVal := range []int{5} {
							for _, flakeTolVal := range []float64{0.05, 0.10, 0.15} {
								wg.Add(1)
								cfg := SimConfig{
									SeqID:              seqCounter,
									Resources:          resourceMult * trafficLevel,
									Traffic:            trafficLevel,
									NTests:             numTests,
									MaxBatch:           maxBatchSize,
									UseOptimizedMatrix: optimized,
									MaxK:               maxKVal,
									KDivisor:           kDivVal,
									FlakeTolerance:     flakeTolVal,
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
	fmt.Printf("└────────────┴───────┴───────┴───────┴────────────┴──────────────┴────────────────┴───────────┴───────────┴────────────┴────────────────┘\n")

	elapsed := time.Since(start)
	fmt.Printf("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  SIMULATION COMPLETE                                                                                                           ║\n")
	fmt.Printf("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Elapsed Time: %-108v║\n", elapsed)

	// Print cache statistics
	cacheMutex.Lock()
	totalAccess := matrixCacheHits + matrixCacheMisses
	hitRate := 0.0
	if totalAccess > 0 {
		hitRate = float64(matrixCacheHits) / float64(totalAccess) * 100
	}
	fmt.Printf("║                                                                                                                                ║\n")
	fmt.Printf("║  Matrix Cache Statistics:                                                                                                      ║\n")
	fmt.Printf("║    • Cache Hits:    %8d                                                                                                    ║\n", matrixCacheHits)
	fmt.Printf("║    • Cache Misses:  %8d                                                                                                    ║\n", matrixCacheMisses)
	fmt.Printf("║    • Hit Rate:      %7.1f%%                                                                                                    ║\n", hitRate)
	cacheMutex.Unlock()
	fmt.Printf("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n")
}
