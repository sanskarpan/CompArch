/*
SIMD Operations
===============

Single Instruction Multiple Data operations for parallel processing.

Features:
- Vector operations (add, sub, mul, div)
- Vector comparisons
- Horizontal operations (sum, min, max)
- Data-level parallelism simulation
- Performance comparison with scalar operations

Applications:
- Understanding SIMD performance benefits
- Vectorization optimization
- Multimedia and scientific computing
*/

package cpu

import (
	"fmt"
	"math"
	"time"
)

// =============================================================================
// SIMD Vector Types
// =============================================================================

// Vec4f represents a 4-element float32 vector
type Vec4f [4]float32

// Vec8f represents an 8-element float32 vector
type Vec8f [8]float32

// Vec4i represents a 4-element int32 vector
type Vec4i [4]int32

// Vec8i represents an 8-element int32 vector
type Vec8i [8]int32

// =============================================================================
// SIMD Float Operations
// =============================================================================

// AddVec4f adds two Vec4f vectors
func AddVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
		a[3] + b[3],
	}
}

// SubVec4f subtracts two Vec4f vectors
func SubVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
		a[3] - b[3],
	}
}

// MulVec4f multiplies two Vec4f vectors element-wise
func MulVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		a[0] * b[0],
		a[1] * b[1],
		a[2] * b[2],
		a[3] * b[3],
	}
}

// DivVec4f divides two Vec4f vectors element-wise
func DivVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		a[0] / b[0],
		a[1] / b[1],
		a[2] / b[2],
		a[3] / b[3],
	}
}

// DotVec4f computes the dot product of two Vec4f vectors
func DotVec4f(a, b Vec4f) float32 {
	mul := MulVec4f(a, b)
	return mul[0] + mul[1] + mul[2] + mul[3]
}

// SqrtVec4f computes the square root of each element
func SqrtVec4f(a Vec4f) Vec4f {
	return Vec4f{
		float32(math.Sqrt(float64(a[0]))),
		float32(math.Sqrt(float64(a[1]))),
		float32(math.Sqrt(float64(a[2]))),
		float32(math.Sqrt(float64(a[3]))),
	}
}

// MaxVec4f returns the maximum of two Vec4f vectors element-wise
func MaxVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		max(a[0], b[0]),
		max(a[1], b[1]),
		max(a[2], b[2]),
		max(a[3], b[3]),
	}
}

// MinVec4f returns the minimum of two Vec4f vectors element-wise
func MinVec4f(a, b Vec4f) Vec4f {
	return Vec4f{
		min(a[0], b[0]),
		min(a[1], b[1]),
		min(a[2], b[2]),
		min(a[3], b[3]),
	}
}

// HorizontalAddVec4f sums all elements
func HorizontalAddVec4f(a Vec4f) float32 {
	return a[0] + a[1] + a[2] + a[3]
}

// HorizontalMaxVec4f returns the maximum element
func HorizontalMaxVec4f(a Vec4f) float32 {
	return max(max(a[0], a[1]), max(a[2], a[3]))
}

// HorizontalMinVec4f returns the minimum element
func HorizontalMinVec4f(a Vec4f) float32 {
	return min(min(a[0], a[1]), min(a[2], a[3]))
}

// =============================================================================
// SIMD Integer Operations
// =============================================================================

// AddVec4i adds two Vec4i vectors
func AddVec4i(a, b Vec4i) Vec4i {
	return Vec4i{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
		a[3] + b[3],
	}
}

// SubVec4i subtracts two Vec4i vectors
func SubVec4i(a, b Vec4i) Vec4i {
	return Vec4i{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
		a[3] - b[3],
	}
}

// MulVec4i multiplies two Vec4i vectors element-wise
func MulVec4i(a, b Vec4i) Vec4i {
	return Vec4i{
		a[0] * b[0],
		a[1] * b[1],
		a[2] * b[2],
		a[3] * b[3],
	}
}

// MaxVec4i returns the maximum of two Vec4i vectors element-wise
func MaxVec4i(a, b Vec4i) Vec4i {
	return Vec4i{
		max(a[0], b[0]),
		max(a[1], b[1]),
		max(a[2], b[2]),
		max(a[3], b[3]),
	}
}

// MinVec4i returns the minimum of two Vec4i vectors element-wise
func MinVec4i(a, b Vec4i) Vec4i {
	return Vec4i{
		min(a[0], b[0]),
		min(a[1], b[1]),
		min(a[2], b[2]),
		min(a[3], b[3]),
	}
}

// =============================================================================
// Wide SIMD Operations (Vec8f)
// =============================================================================

// AddVec8f adds two Vec8f vectors
func AddVec8f(a, b Vec8f) Vec8f {
	return Vec8f{
		a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3],
		a[4] + b[4], a[5] + b[5], a[6] + b[6], a[7] + b[7],
	}
}

// MulVec8f multiplies two Vec8f vectors
func MulVec8f(a, b Vec8f) Vec8f {
	return Vec8f{
		a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3],
		a[4] * b[4], a[5] * b[5], a[6] * b[6], a[7] * b[7],
	}
}

// HorizontalAddVec8f sums all elements
func HorizontalAddVec8f(a Vec8f) float32 {
	return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}

// =============================================================================
// SIMD Processor Simulator
// =============================================================================

// SIMDProcessor simulates a SIMD execution unit
type SIMDProcessor struct {
	VectorWidth     int     // Number of elements processed in parallel
	ClockSpeed      float64 // GHz
	VectorRegisters int     // Number of vector registers

	// Statistics
	VectorOps  uint64
	ScalarOps  uint64
	TotalCycles uint64
}

// NewSIMDProcessor creates a new SIMD processor
func NewSIMDProcessor(vectorWidth int, clockSpeed float64) *SIMDProcessor {
	return &SIMDProcessor{
		VectorWidth:     vectorWidth,
		ClockSpeed:      clockSpeed,
		VectorRegisters: 32,
	}
}

// ExecuteVectorAdd simulates vector addition
func (s *SIMDProcessor) ExecuteVectorAdd(numElements int) {
	numVectorOps := (numElements + s.VectorWidth - 1) / s.VectorWidth
	s.VectorOps += uint64(numVectorOps)
	s.TotalCycles += uint64(numVectorOps) // 1 cycle per vector op
}

// ExecuteScalarAdd simulates scalar addition
func (s *SIMDProcessor) ExecuteScalarAdd(numElements int) {
	s.ScalarOps += uint64(numElements)
	s.TotalCycles += uint64(numElements) // 1 cycle per scalar op
}

// GetSpeedup returns the speedup of SIMD vs scalar
func (s *SIMDProcessor) GetSpeedup(numElements int) float64 {
	scalarCycles := float64(numElements)
	vectorCycles := float64((numElements + s.VectorWidth - 1) / s.VectorWidth)
	return scalarCycles / vectorCycles
}

// GetStats returns SIMD statistics
func (s *SIMDProcessor) GetStats() SIMDStats {
	return SIMDStats{
		VectorOps:   s.VectorOps,
		ScalarOps:   s.ScalarOps,
		TotalCycles: s.TotalCycles,
		VectorWidth: s.VectorWidth,
	}
}

// SIMDStats contains SIMD execution statistics
type SIMDStats struct {
	VectorOps   uint64
	ScalarOps   uint64
	TotalCycles uint64
	VectorWidth int
}

// String returns a string representation of the stats
func (s SIMDStats) String() string {
	return fmt.Sprintf(`SIMD Statistics:
  Vector Operations: %d
  Scalar Operations: %d
  Total Cycles: %d
  Vector Width: %d
  Theoretical Speedup: %.2fx`,
		s.VectorOps,
		s.ScalarOps,
		s.TotalCycles,
		s.VectorWidth,
		float64(s.VectorWidth))
}

// =============================================================================
// SIMD Benchmarks
// =============================================================================

// BenchmarkSIMD compares SIMD vs scalar performance
func BenchmarkSIMD(size int) SIMDBenchmarkResult {
	// Create test data
	a := make([]float32, size)
	b := make([]float32, size)
	result := make([]float32, size)

	for i := 0; i < size; i++ {
		a[i] = float32(i)
		b[i] = float32(i * 2)
	}

	// Scalar benchmark
	scalarStart := time.Now()
	for i := 0; i < size; i++ {
		result[i] = a[i] + b[i]
	}
	scalarTime := time.Since(scalarStart)

	// SIMD benchmark (Vec4f)
	simdStart := time.Now()
	for i := 0; i < size; i += 4 {
		if i+4 <= size {
			va := Vec4f{a[i], a[i+1], a[i+2], a[i+3]}
			vb := Vec4f{b[i], b[i+1], b[i+2], b[i+3]}
			vr := AddVec4f(va, vb)
			result[i] = vr[0]
			result[i+1] = vr[1]
			result[i+2] = vr[2]
			result[i+3] = vr[3]
		}
	}
	simdTime := time.Since(simdStart)

	return SIMDBenchmarkResult{
		Size:        size,
		ScalarTime:  scalarTime,
		SIMDTime:    simdTime,
		Speedup:     float64(scalarTime) / float64(simdTime),
	}
}

// SIMDBenchmarkResult contains benchmark results
type SIMDBenchmarkResult struct {
	Size       int
	ScalarTime time.Duration
	SIMDTime   time.Duration
	Speedup    float64
}

// String returns a string representation of the result
func (r SIMDBenchmarkResult) String() string {
	return fmt.Sprintf(`SIMD Benchmark (size=%d):
  Scalar Time: %v
  SIMD Time: %v
  Speedup: %.2fx`,
		r.Size,
		r.ScalarTime,
		r.SIMDTime,
		r.Speedup)
}

// =============================================================================
// Helper Functions
// =============================================================================

func max[T ~int32 | ~float32](a, b T) T {
	if a > b {
		return a
	}
	return b
}

func min[T ~int32 | ~float32](a, b T) T {
	if a < b {
		return a
	}
	return b
}
