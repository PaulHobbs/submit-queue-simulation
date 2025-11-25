package main

import (
	"fmt"
	"testing"
)

func TestQuantizeValue(t *testing.T) {
	tests := []struct {
		input    int
		expected int
	}{
		// Small values (<20) should be exact
		{1, 1},
		{10, 10},
		{19, 19},

		// Large values should quantize to ~5% steps
		{20, 20},
		{21, 21},
		{22, 21},
		{30, 30},
		{32, 32},
		{33, 32},
		{50, 51},
		{100, 100},
		{105, 105},
	}

	fmt.Println("Quantization Examples:")
	fmt.Println("Input → Quantized (% diff)")
	fmt.Println("─────────────────────────")

	for _, tt := range tests {
		result := quantizeValue(tt.input)
		diff := float64(result-tt.input) / float64(tt.input) * 100
		fmt.Printf("%3d → %3d (%+.1f%%)\n", tt.input, result, diff)
	}
}

func TestQuantizeCacheEfficiency(t *testing.T) {
	// Test that nearby values quantize to the same value
	values := []int{30, 31, 32, 33, 34, 50, 51, 52, 100, 102, 105}

	fmt.Println("\nCache Groupings:")
	quantized := make(map[int][]int)

	for _, v := range values {
		q := quantizeValue(v)
		quantized[q] = append(quantized[q], v)
	}

	for q, vals := range quantized {
		if len(vals) > 1 {
			fmt.Printf("Quantized=%d: %v (cache hit rate: %d/%d)\n",
				q, vals, len(vals)-1, len(vals))
		}
	}
}

func BenchmarkQuantizeValue(b *testing.B) {
	for i := 0; i < b.N; i++ {
		quantizeValue(32)
		quantizeValue(100)
		quantizeValue(10)
	}
}
