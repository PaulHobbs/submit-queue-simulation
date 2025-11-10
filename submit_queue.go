package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Precompute randoms to make rng very fast
const maxRand = 1_000_000

var (
	rands []float64
	randI int64
)

func init() {
	rands = make([]float64, maxRand)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range rands {
		rands[i] = rng.Float64()
	}
}

// fastRand uses precomputed pseudorandom numbers. This is *much* faster than
// calling rng.Float64() in a loop.
func fastRand() float64 {
	randI += 123456
	randI *= 777777
	randI %= int64(maxRand)
	return rands[randI]
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
func sample(d Distribution) float64 {
	s := fastRand()
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
	CreationTick int             // Feature 3: Track creation time for submit stats
	Effects      map[int]float64 // Map from Test ID to pass rate effect
}

func NewChange(id, tick int, testDefs []TestDefinition) *Change {
	c := &Change{
		ID:           id,
		CreationTick: tick,
		Effects:      make(map[int]float64, len(testDefs)/10), // pre-allocate a bit
	}
	for _, td := range testDefs {
		if fastRand() < td.PAffected {
			c.Effects[td.ID] = sample(td.PassRates)
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
func (c *Change) FixHardBreaks(testDefs []TestDefinition) {
	for tid, effect := range c.Effects {
		if effect == 0.0 {
			// Find the corresponding test definition
			var tDef *TestDefinition
			for i := range testDefs {
				if testDefs[i].ID == tid {
					tDef = &testDefs[i]
					break
				}
			}
			if tDef == nil {
				continue
			}

			// Try to fix it by resampling a non-zero pass rate
			newEffect := 0.0
			for i := 0; i < 10; i++ {
				newEffect = sample(tDef.PassRates)
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

func (mb *Minibatch) Evaluate(repoBasePPass map[int]float64, allTestIDs []int) (bool, bool) {
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
		if effP < 1.0 && fastRand() >= effP {
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
	AllTestIDs       []int
	PipelineDepth    int
	MaxMinibatchSize int

	// --- State ---
	RepoBasePPass   map[int]float64
	PendingChanges  []*Change
	ChangeIDCounter int
	UseParallelMode bool // Toggle for dynamic fallback strategy

	// --- Statistics ---
	TotalMinibatches  int
	PassedMinibatches int
	TotalSubmitted    int
	TotalWaitTicks    int // Sum of (SubmitTick - CreationTick) for all submitted CLs
}

func NewPipelinedSubmitQueue(testDefs []TestDefinition, depth, maxMB int) *PipelinedSubmitQueue {
	sq := &PipelinedSubmitQueue{
		TestDefs:         testDefs,
		AllTestIDs:       make([]int, len(testDefs)),
		RepoBasePPass:    make(map[int]float64),
		PipelineDepth:    depth,
		MaxMinibatchSize: maxMB,
		PendingChanges:   make([]*Change, 0, 1024),
		UseParallelMode:  false, // Start optimistically
	}
	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
	}
	return sq
}

func (sq *PipelinedSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		sq.PendingChanges = append(sq.PendingChanges, NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs))
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

		// Speculative: Batch N includes everything in Batches 0..N
		mbChanges := make([]*Change, len(cumulativeCls))
		copy(mbChanges, cumulativeCls)

		minibatches = append(minibatches, Minibatch{
			Changes:    mbChanges,
			NewChanges: newClsForLevel,
		})
	}

	// 2. Evaluation
	allPassed := true
	results := make([]struct{ passed, hardFail bool }, len(minibatches))
	for i, mb := range minibatches {
		p, h := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs)
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
						cl.FixHardBreaks(sq.TestDefs)
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
						cl.FixHardBreaks(sq.TestDefs)
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
		minibatches = append(minibatches, Minibatch{
			Changes:    changes,
			NewChanges: changes,
		})
		batchIndices = append(batchIndices, struct{ start, end int }{start, end})
	}

	// 2. Evaluation
	allPassed := true
	results := make([]TestResult, len(minibatches))
	for i, mb := range minibatches {
		p, hardFail := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs)
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
				// Found our winner
				submittedBatchIdx = i
				for _, cl := range minibatches[i].Changes {
					sq.applyEffect(cl, currentTick)
				}
				submittedCount = len(minibatches[i].Changes)
			}
			// We don't submit subsequent passing batches in this mode,
			// but they don't need hard-fixes either.
		} else if res.hardFail {
			// Must fix hard failures in all non-submitted batches
			for _, cl := range minibatches[i].Changes {
				if cl.IsHardBreak() {
					cl.FixHardBreaks(sq.TestDefs)
				}
			}
		}
	}

	if submittedCount > 0 {
		sq.TotalSubmitted += submittedCount
		// Reconstruct queue: remove the submitted batch from the middle
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
		if fastRand() > math.Pow(passRate, float64(n)/84) {
			sq.RepoBasePPass[t] = 1
		}
	}
}

func (sq *PipelinedSubmitQueue) applyEffect(cl *Change, currentTick int) {
	for tid, effect := range cl.Effects {
		current := sq.RepoBasePPass[tid]
		// Ensure default is 1.0 if not present
		if _, ok := sq.RepoBasePPass[tid]; !ok {
			current = 1.0
		}
		if effect < current {
			sq.RepoBasePPass[tid] = effect
		}
	}
	sq.TotalWaitTicks += (currentTick - cl.CreationTick)
}

// Averages to 25/hour
var nChangesPerHour = []int{5, 5, 5, 5, 5, 5, 5, 5, 60, 60, 60, 60, 60, 60, 60, 60, 10, 10, 10, 10, 10, 10, 10, 10}

// Avg of 25 CLs/hour in base traffic.
const idealThroughput = 25

func main() {
	const nIter = 30000

	for _, parallelism := range []int{1, 2, 4, 8} {
		fmt.Printf("Pipeline depth/parellism: %d\n", parallelism)
		for _, traffic := range []int{1, 2, 3, 4} {
			fmt.Printf("")
			fmt.Printf("Ideal throughput: %d CLs/hour = Productivity 1/1x \n", 25*traffic)
			fmt.Printf("%-10s | %-22s | %-14s | %-9s | %s\n",
				"Max Batch", "Productivity * 1/x", "Avg Queue Size", "Pass Rate", "Avg Time to Submit (h)")
			fmt.Println("-------------------------------------------------------------------------------------------------------------")
			for _, nTests := range []int{16, 32, 64, 128} {
				fmt.Printf("n_tests: %d\n", nTests)
				for _, maxBatch := range []int{32, 64, 128} {
					totalQ := 0

					testDefs := make([]TestDefinition, nTests)
					for i := 0; i < nTests; i++ {
						testDefs[i] = TestDefinition{
							ID: i,
							// Only 0.1% of CLs affect a test's pass rate post-smoketest (TreeHugger)
							// This means 0.05% of CLs are "bad" (increase a test's failure rate).
							PAffected: 0.001,
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

					sq := NewPipelinedSubmitQueue(testDefs, parallelism, maxBatch)

					// Initial load
					sq.AddChanges(traffic*nChangesPerHour[len(nChangesPerHour)-1], 0)

					submittedTotal := 0
					for i := 0; i < nIter; i++ {
						submittedTotal += sq.Step(i)
						totalQ += len(sq.PendingChanges)

						// Developers adding work based on current queue size
						qSize := len(sq.PendingChanges)
						changesToAdd := traffic * nChangesPerHour[i%24]
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

					avgQ := float64(totalQ) / float64(nIter)
					mbPassRate := 0.0
					if sq.TotalMinibatches > 0 {
						mbPassRate = float64(sq.PassedMinibatches) / float64(sq.TotalMinibatches)
					}

					avgSubmitTime := 1.0 // minimum 1h per submit.
					if sq.TotalSubmitted > 0 {
						avgSubmitTime += float64(sq.TotalWaitTicks) / float64(sq.TotalSubmitted)
					}

					throughput := float64(submittedTotal) / float64(nIter)
					slowdown := float64(idealThroughput*traffic) / throughput
					fmt.Printf("%-10d | %-22.2f | %-14.0f | %-9.2f | %.2f\n",
						maxBatch, slowdown, avgQ, mbPassRate, avgSubmitTime)
				}
			}
		}
	}
}
