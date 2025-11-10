package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Precompute randoms to make rng very fast
const maxRand = 10_000_000

var (
	rands []float64
	randI int64
)

func init() {
	rands = make([]float64, maxRand)
	// Use a fixed seed for reproducibility similar to the original if it were seeded.
	rng := rand.New(rand.NewSource(42))
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
		// If effP is 1.0, rand() >= 1.0 is always false (assuming rand() < 1.0), so it passes.
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
	RepoBasePPass    map[int]float64
	PendingChanges   []*Change
	ChangeIDCounter  int

	// --- Statistics ---
	TotalMinibatches    int
	PassedMinibatches   int
	TotalSubmitted      int
	TotalWaitTicks      int // Sum of (SubmitTick - CreationTick) for all submitted CLs
}

func NewPipelinedSubmitQueue(testDefs []TestDefinition, depth, maxMB int) *PipelinedSubmitQueue {
	sq := &PipelinedSubmitQueue{
		TestDefs:         testDefs,
		AllTestIDs:       make([]int, len(testDefs)),
		RepoBasePPass:    make(map[int]float64),
		PipelineDepth:    depth,
		MaxMinibatchSize: maxMB,
		PendingChanges:   make([]*Change, 0, 1024),
	}
	for i, t := range testDefs {
		sq.AllTestIDs[i] = t.ID
		sq.RepoBasePPass[t.ID] = 1.0
	}
	return sq
}

// AddChanges queues up more changes to be processed.
func (sq *PipelinedSubmitQueue) AddChanges(n, currentTick int) {
	for i := 0; i < n; i++ {
		sq.PendingChanges = append(sq.PendingChanges, NewChange(sq.ChangeIDCounter, currentTick, sq.TestDefs))
		sq.ChangeIDCounter++
	}
}

// Step runs a single step of the submit queue.
func (sq *PipelinedSubmitQueue) Step(currentTick int) int {
	pendingCount := len(sq.PendingChanges)
	if pendingCount == 0 {
		return 0
	}

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

		// Create a snapshot of cumulative changes for this minibatch
		mbChanges := make([]*Change, len(cumulativeCls))
		copy(mbChanges, cumulativeCls)

		minibatches = append(minibatches, Minibatch{
			Changes:    mbChanges,
			NewChanges: newClsForLevel,
		})
	}

	// 2. Speculative Execution
	results := make([]struct{ passed, hardFail bool }, len(minibatches))
	for i, mb := range minibatches {
		p, h := mb.Evaluate(sq.RepoBasePPass, sq.AllTestIDs)
		results[i] = struct{ passed, hardFail bool }{p, h}

		// Feature 2 Stats: Track minibatch pass rates
		sq.TotalMinibatches++
		if p {
			sq.PassedMinibatches++
		}
	}

	// 3. Greedy Merging
	submittedCount := 0
	passStreakBroken := false

	for i, res := range results {
		if passStreakBroken {
			break
		}

		if res.passed {
			for _, cl := range minibatches[i].NewChanges {
				for tid, effect := range cl.Effects {
					current := sq.RepoBasePPass[tid]
					// If effect is worse than current base, degrade the base
					// (Assuming untracked TIDs are 1.0, handled by map zero value if we are careful, but explicit is safer)
					if _, ok := sq.RepoBasePPass[tid]; !ok {
						current = 1.0
					}

					if effect < current {
						sq.RepoBasePPass[tid] = effect
					}
				}
				// Feature 3 Stats: Track time to submit
				sq.TotalWaitTicks += (currentTick - cl.CreationTick)
			}
			submittedCount += len(minibatches[i].NewChanges)
		} else {
			passStreakBroken = true
			if res.hardFail {
				// Fix hard breaks in the failing batch so they might pass next tick
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

	return submittedCount
}

// Averages to 50/hour
var nChangesPerHour = []int{10, 10, 10, 10, 10, 10, 10, 10, 120, 120, 120, 120, 120, 120, 120, 120, 20, 20, 20, 20, 20, 20, 20, 20}

func main() {
	const nIter = 10000
	const fixedDepth = 4

	fmt.Printf("Pipeline Depth fixed at: %d\n", fixedDepth)
	fmt.Printf("Ideal throughput: 50 CLs/hour\n")
	// updated header with new columns
	fmt.Printf("%-10s | %-22s | %-14s | %-9s | %s\n",
		"Max Batch", "Productivity * 1/x", "Avg Queue Size", "Pass Rate", "Avg Time to Submit (h)")
	fmt.Println("-------------------------------------------------------------------------------------------------------------")

	for _, nTests := range []int{8, 16, 32, 64} {
		fmt.Printf("n_tests: %d\n", nTests)
		for _, maxMB := range []int{32, 64, 128, 256} {
			totalQ := 0

			// Initialize Test Definitions
			testDefs := make([]TestDefinition, nTests)
			for i := 0; i < nTests; i++ {
				testDefs[i] = TestDefinition{
					ID:        i,
					PAffected: 0.01,
					PassRates: []DistEntry{
						{0.1, 1.0},
						{0.15, 0.98},
						{0.20, 0.995},
						{1.0, 0.0},
					},
				}
			}

			sq := NewPipelinedSubmitQueue(testDefs, fixedDepth, maxMB)

			// Initial load
			sq.AddChanges(nChangesPerHour[len(nChangesPerHour)-1], 0)

			submittedTotal := 0
			for i := 0; i < nIter; i++ {
				submittedTotal += sq.Step(i)
				totalQ += len(sq.PendingChanges)

				// Developers adding work based on current queue size
				qSize := len(sq.PendingChanges)
				changesToAdd := nChangesPerHour[i%24]
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

			avgSubmitTime := 0.0
			if sq.TotalSubmitted > 0 {
				avgSubmitTime = float64(sq.TotalWaitTicks) / float64(sq.TotalSubmitted)
			}

			throughput := float64(submittedTotal) / float64(nIter)
			slowdown := 50/throughput
			fmt.Printf("%-10d | %-22.2f | %-14.0f | %-9.2f | %.2f\n",
				maxMB, slowdown, avgQ, mbPassRate, avgSubmitTime)
		}
	}
}
