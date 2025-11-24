package main

import (
	"math/rand"
	"testing"
)

// Helper to create a basic queue for testing
func createTestQueue(nTests, resources, maxBatch int) *PipelinedSubmitQueue {
	rng := NewFastRNG(12345)
	testDefs := make([]TestDefinition, nTests)
	for i := 0; i < nTests; i++ {
		testDefs[i] = TestDefinition{
			ID:        i,
			PAffected: 0.0, // Start clean
			PassRates: []DistEntry{{1.0, 1.0}}, // Always pass by default
		}
	}
	return NewPipelinedSubmitQueue(testDefs, resources, maxBatch, rng)
}

// FuzzTestStep_Properties tests core properties of the Step function with varied inputs.

func FuzzStep_Properties(f *testing.F) {
	f.Add(10, 1, 0, 32) // Initial seed values: numChanges, numCulprits, culpritTestID, maxBatchSize

	f.Fuzz(func(t *testing.T, numChanges, numCulprits, culpritTestID, maxBatchSize int) {
		// Normalize inputs to reasonable bounds
		numChanges = numChanges % 100 // Max 99 changes
		if numChanges < 1 {
			numChanges = 1
		}
		numCulprits = numCulprits % numChanges // Max culprits = numChanges
		if numCulprits < 0 {
			numCulprits = 0
		}
		culpritTestID = culpritTestID % 5 // Test IDs 0-4
		if culpritTestID < 0 {
			culpritTestID = 0
		}
		maxBatchSize = maxBatchSize % 64 // Max 63
		if maxBatchSize < 1 {
			maxBatchSize = 1
		}

		const nTests = 5
		sq := createTestQueue(nTests, 8, maxBatchSize)
		sq.AddChanges(numChanges, 0)

		// Make all clean first to ensure precise culprit injection
		for _, cl := range sq.PendingChanges {
			cl.Effects = make(map[int]float64)
		}

		// Identify the actual batch of changes that will be processed in this step
		// This is the number of changes from PendingChanges that 'Step' will look at.
		nToProcess := len(sq.PendingChanges)
		if nToProcess > sq.MaxMinibatchSize {
			nToProcess = sq.MaxMinibatchSize
		}
		if nToProcess == 0 {
			// If no changes to process, step should return 0.
			if sq.Step(1) != 0 {
				t.Errorf("Expected 0 submitted for empty queue, got %d", sq.Step(1))
			}
			return
		}

		// Capture the exact changes that will be processed in this step.
		// We need to work with the *actual* change objects from sq.PendingChanges.
		// Inject culprits into these direct references.
		changesToProcess := sq.PendingChanges[:nToProcess]

		culpritIDsInProcessedBatch := make(map[int]bool)
		culpritCountInProcessedBatch := 0
		// Use a deterministic random source for selecting culprits from changesToProcess
		randSrc := rand.NewSource(int64(numCulprits) + int64(culpritTestID))
		r := rand.New(randSrc)

		// Ensure numCulprits does not exceed the number of changes to be processed
		actualNumCulprits := numCulprits
		if actualNumCulprits > nToProcess {
			actualNumCulprits = nToProcess // Cannot have more culprits than changes to process
		}

		if nToProcess > 0 {
			// Shuffle the changesToProcess and pick the first actualNumCulprits to be culprits
			shuffledIndices := r.Perm(nToProcess)

			for i := 0; i < actualNumCulprits; i++ {
				idx := shuffledIndices[i]
				cl := changesToProcess[idx]
				cl.Effects[culpritTestID] = 0.0 // Hard fail
				culpritIDsInProcessedBatch[cl.ID] = true
				culpritCountInProcessedBatch++
			}
		}

		oldPendingCount := len(sq.PendingChanges) // This is the total, before Step
		submitted := sq.Step(1)                   // Call Step

		// --- Property 1: No Culprits Submitted ---
		// The number of submitted changes should be less than or equal to
		// the total number of processed changes minus the *actual* number of culprits in that processed batch.
		if submitted > (nToProcess - culpritCountInProcessedBatch) {
			t.Errorf("Property violation: Submitted %d changes, expected at most %d (processed %d - culprits %d). Culprit potentially submitted.\nFuzz args: numChanges=%d, numCulprits=%d, culpritTestID=%d, maxBatchSize=%d",
				submitted, nToProcess-culpritCountInProcessedBatch, nToProcess, culpritCountInProcessedBatch,
				numChanges, numCulprits, culpritTestID, maxBatchSize)
		}

		// --- Property 2: Queue Size Reduction ---
		// The total number of changes removed from pending should be exactly nToProcess
		expectedPending := oldPendingCount - nToProcess
		if len(sq.PendingChanges) != expectedPending {
			t.Errorf("Property violation: Expected %d pending changes, got %d (old: %d, processed: %d)\nFuzz args: numChanges=%d, numCulprits=%d, culpritTestID=%d, maxBatchSize=%d",
				expectedPending, len(sq.PendingChanges), oldPendingCount, nToProcess,
				numChanges, numCulprits, culpritTestID, maxBatchSize)
		}

		// --- Property 3: Submitted Count Consistency ---
		// Submitted changes should not exceed the number of changes actually processed.
		if submitted < 0 || submitted > nToProcess {
			t.Errorf("Property violation: Submitted %d changes, but only %d changes were processed.\nFuzz args: numChanges=%d, numCulprits=%d, culpritTestID=%d, maxBatchSize=%d",
				submitted, nToProcess,
				numChanges, numCulprits, culpritTestID, maxBatchSize)
		}

		t.Logf("numChanges=%d, numCulprits=%d, culpritTestID=%d, maxBatchSize=%d -> Processed=%d, Submitted=%d, RemainingPending=%d",
			numChanges, numCulprits, culpritTestID, maxBatchSize, nToProcess, submitted, len(sq.PendingChanges))
	})
}