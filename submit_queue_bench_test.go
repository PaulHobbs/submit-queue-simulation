package main

import (
	"testing"
)

// Benchmark the hot path: Minibatch.Evaluate
func BenchmarkMinibatchEvaluate(b *testing.B) {
	rng := NewFastRNG(12345)

	// Setup similar to simulation
	nTests := 32
	testDefs := make([]TestDefinition, nTests)
	for i := 0; i < nTests; i++ {
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

	allTestIDs := make([]int, nTests)
	repoBasePPass := make([]float64, nTests)
	for i := 0; i < nTests; i++ {
		allTestIDs[i] = i
		repoBasePPass[i] = 1.0
	}

	// Create a batch with some changes
	batchSize := 20
	changes := make([]*Change, batchSize)
	for i := 0; i < batchSize; i++ {
		changes[i] = NewChange(i, 0, testDefs, rng)
	}

	mb := Minibatch{Changes: changes}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mb.Evaluate(repoBasePPass, allTestIDs, rng)
	}
}

// Benchmark NewChange creation
func BenchmarkNewChange(b *testing.B) {
	rng := NewFastRNG(12345)
	nTests := 32
	testDefs := make([]TestDefinition, nTests)
	for i := 0; i < nTests; i++ {
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewChange(i, 0, testDefs, rng)
	}
}

// Benchmark the main Step function (small queue)
func BenchmarkSubmitQueueStepSmall(b *testing.B) {
	rng := NewFastRNG(12345)
	nTests := 16 // Smaller test set
	testDefs := make([]TestDefinition, nTests)
	for i := 0; i < nTests; i++ {
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

	sq := NewCulpritQueueSubmitQueue(testDefs, 8, 100, rng, true, 6, 5, 0.10)
	sq.AddChanges(50, 0) // Smaller batch

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sq.Step(i)
		// Add some changes to keep the queue populated
		if i%5 == 0 {
			sq.AddChanges(20, i)
		}
	}
}

// Benchmark a single iteration of the simulation loop
func BenchmarkSimulationIteration(b *testing.B) {
	nTests := 16
	testDefs := make([]TestDefinition, nTests)
	for i := 0; i < nTests; i++ {
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng := NewFastRNG(int64(i * 997))
		sq := NewCulpritQueueSubmitQueue(testDefs, 8, 100, rng, true, 6, 5, 0.10)
		sq.AddChanges(30, 0)

		// Run 10 steps
		for j := 0; j < 10; j++ {
			sq.Step(j)
			if j%2 == 0 {
				sq.AddChanges(5, j)
			}
		}
	}
}

// Benchmark matrix operations
func BenchmarkMatrixGetColumnIndices(b *testing.B) {
	mat := GetCachedMatrix(32, 2048, 12, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.GetColumnIndices(i % 100)
	}
}

// Benchmark the full simulation (smaller version)
func BenchmarkSmallSimulation(b *testing.B) {
	cfg := SimConfig{
		SeqID:              0,
		Resources:          32,
		Traffic:            8,
		NTests:             32,
		MaxBatch:           2048,
		UseOptimizedMatrix: true,
		MaxK:               12,
		KDivisor:           5,
		FlakeTolerance:     0.10,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runSimulation(cfg, int64(i*997))
	}
}
