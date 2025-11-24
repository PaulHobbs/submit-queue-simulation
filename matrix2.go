package main

import (
	"fmt"
	"math/bits"
	"math/rand"
	"time"
)

// Configuration
const (
	NumItems   = 200 // (N) Columns
	NumTests   = 50 // (T) Rows
	ColWeight  = 8   // (K) Ones per column
	MaxSeconds = 2   // Time budget for optimization
)

// Matrix represents our testing grid.
// To optimize for speed, we store the matrix in "Column Major" format.
// Each column is a list of uint64 chunks.
// For 100 tests, we need slice of length 2 (2 * 64 bits = 128 capacity).
type Matrix struct {
	Cols      [][]uint64
	NumTests  int
	NumItems  int
	ColWeight int
	RowChunks int // How many uint64s needed to store one column
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Printf("Initializing %dx%d Matrix (Weight=%d)...\n", NumTests, NumItems, ColWeight)
	
	// 1. Create and Fill Randomly
	mat := NewMatrix(NumTests, NumItems, ColWeight)
	
	initialOverlap, _ := mat.MaxOverlap()
	fmt.Printf("Initial Random Overlap: %d\n", initialOverlap)

	// 2. Optimize
	start := time.Now()
	mat.Optimize(MaxSeconds)
	duration := time.Since(start)

	// 3. Report
	finalOverlap, _ := mat.MaxOverlap()
	fmt.Printf("Optimization Complete in %s.\n", duration)
	fmt.Printf("Final Max Overlap: %d\n", finalOverlap)

	// Example: Check a specific column
	// fmt.Println("Column 0 layout:", mat.GetColumnIndices(0))
}

// NewMatrix creates a random matrix with constant column weight
func NewMatrix(rows, cols, weight int) *Matrix {
	// Calculate how many uint64s we need per column
	chunks := (rows + 63) / 64
	
	m := &Matrix{
		Cols:      make([][]uint64, cols),
		NumTests:  rows,
		NumItems:  cols,
		ColWeight: weight,
		RowChunks: chunks,
	}

	// Initialize each column
	for i := 0; i < cols; i++ {
		m.Cols[i] = make([]uint64, chunks)
		m.randomizeColumn(i)
	}
	return m
}

// randomizeColumn clears a column and sets 'weight' random bits to 1
func (m *Matrix) randomizeColumn(colIdx int) {
	// Clear column
	for k := 0; k < m.RowChunks; k++ {
		m.Cols[colIdx][k] = 0
	}

	// Set random bits
	setCount := 0
	for setCount < m.ColWeight {
		r := rand.Intn(m.NumTests)
		chunk := r / 64
		bit := uint64(1) << (r % 64)

		// Check if already set
		if (m.Cols[colIdx][chunk] & bit) == 0 {
			m.Cols[colIdx][chunk] |= bit
			setCount++
		}
	}
}

// Optimize runs the "Electron Repulsion" / Greedy Swap algorithm
func (m *Matrix) Optimize(seconds float64) {
	startTime := time.Now()
	timeout := time.Duration(seconds * float64(time.Second))

	// Loop until timeout or perfect score (0 or 1 overlap is ideal)
	steps := 0
	for time.Since(startTime) < timeout {
		steps++
		
		// 1. Find the worst pair (highest overlap)
		maxOverlap, worstPair := m.MaxOverlap()
		
		// If overlap is 1, we physically cannot do better (unless orthogonal, which is impossible here)
		if maxOverlap <= 1 {
			fmt.Println("Theoretical Limit Reached!")
			break
		}

		colA := worstPair[0]
		colB := worstPair[1]

		// 2. Attempt to Perturb colA to move it away from colB
		// We need to identify the specific rows causing the collision
		collisionRows := m.findCollisions(colA, colB)
		
		if len(collisionRows) == 0 {
			continue // Should not happen if overlap > 0
		}

		// Pick a collision to fix (remove this 1)
		rowToRemove := collisionRows[rand.Intn(len(collisionRows))]

		// Pick an empty spot to move it to (add a 1 here)
		rowToAdd := m.findRandomEmptyRow(colA)

		// 3. Execute Swap
		m.flipBit(colA, rowToRemove, 0) // Turn off
		m.flipBit(colA, rowToAdd, 1)    // Turn on

		// 4. Check: Did we make the GLOBAL state worse?
		// Note: Calculating global overlap is O(N^2). 
		// For N=200, this is fast enough in Go. 
		newOverlap, _ := m.MaxOverlap()

		if newOverlap > maxOverlap {
			// Revert! We made it worse.
			m.flipBit(colA, rowToAdd, 0)
			m.flipBit(colA, rowToRemove, 1)
		} else {
			// Keep it. (Implicitly accepts equal states to allow traversing plateaus)
		}
	}
	fmt.Printf("Ran %d iterations.\n", steps)
}

// MaxOverlap calculates the maximum overlap between ANY two columns
func (m *Matrix) MaxOverlap() (int, [2]int) {
	maxVal := 0
	pair := [2]int{0, 0}

	for i := 0; i < m.NumItems; i++ {
		for j := i + 1; j < m.NumItems; j++ {
			// Fast Bitwise Intersection
			overlap := 0
			for k := 0; k < m.RowChunks; k++ {
				// PopCount( A & B )
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

// Helper: Find all row indices where both colA and colB are 1
func (m *Matrix) findCollisions(idxA, idxB int) []int {
	var collisions []int
	for k := 0; k < m.RowChunks; k++ {
		// Intersection bits
		common := m.Cols[idxA][k] & m.Cols[idxB][k]
		if common == 0 {
			continue
		}
		// Extract bit indices
		for bit := 0; bit < 64; bit++ {
			mask := uint64(1) << bit
			if (common & mask) != 0 {
				rowAbs := k*64 + bit
				if rowAbs < m.NumTests {
					collisions = append(collisions, rowAbs)
				}
			}
		}
	}
	return collisions
}

// Helper: Find a random row index where col has a 0
func (m *Matrix) findRandomEmptyRow(colIdx int) int {
	// Simple rejection sampling is fastest here since matrix is sparse
	for {
		r := rand.Intn(m.NumTests)
		chunk := r / 64
		bit := uint64(1) << (r % 64)
		
		if (m.Cols[colIdx][chunk] & bit) == 0 {
			return r
		}
	}
}

// Helper: Set a specific bit in a column
func (m *Matrix) flipBit(colIdx, rowIdx int, val int) {
	chunk := rowIdx / 64
	bit := uint64(1) << (rowIdx % 64)
	
	if val == 1 {
		m.Cols[colIdx][chunk] |= bit
	} else {
		m.Cols[colIdx][chunk] &^= bit // Bit clear
	}
}

// Debug helper to see the rows as a list of ints
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
