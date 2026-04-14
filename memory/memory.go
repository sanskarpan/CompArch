/*
Memory Systems
==============

Comprehensive memory systems implementation including RAM types,
memory access patterns, bandwidth/latency measurement, and optimization.

Features:
- Different RAM types (DDR, GDDR, HBM)
- Memory bandwidth and latency characteristics
- Memory access patterns (sequential, random, strided)
- Cache-friendly vs cache-unfriendly access
- Memory-bound vs compute-bound analysis
- Data locality optimization

Applications:
- Understanding memory hierarchy performance
- Optimizing memory access patterns
- Analyzing memory bottlenecks
*/

package memory

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// =============================================================================
// RAM Types
// =============================================================================

// RAMType represents different types of RAM
type RAMType int

const (
	DDR4   RAMType = 0 // Standard DDR4
	DDR5   RAMType = 1 // Modern DDR5
	GDDR6  RAMType = 2 // Graphics DDR6
	GDDR6X RAMType = 3 // GDDR6X (NVIDIA)
	HBM2   RAMType = 4 // High Bandwidth Memory 2
	HBM2E  RAMType = 5 // HBM2E (enhanced)
	HBM3   RAMType = 6 // Latest HBM3
)

// RAMCharacteristics defines RAM properties
type RAMCharacteristics struct {
	Type          RAMType
	Bandwidth     float64 // GB/s
	Latency       int     // nanoseconds
	BusWidth      int     // bits
	ClockSpeed    float64 // MHz
	VoltageRange  string
	PowerConsumption float64 // Watts
	Capacity      uint64  // GB
}

// GetRAMCharacteristics returns characteristics for different RAM types
func GetRAMCharacteristics(ramType RAMType) RAMCharacteristics {
	specs := map[RAMType]RAMCharacteristics{
		DDR4: {
			Type:             DDR4,
			Bandwidth:        25.6,  // GB/s per channel
			Latency:          13,    // ns
			BusWidth:         64,
			ClockSpeed:       3200,
			VoltageRange:     "1.2V",
			PowerConsumption: 3.0,
			Capacity:         32,
		},
		DDR5: {
			Type:             DDR5,
			Bandwidth:        51.2,  // GB/s per channel
			Latency:          14,    // ns
			BusWidth:         64,
			ClockSpeed:       6400,
			VoltageRange:     "1.1V",
			PowerConsumption: 2.5,
			Capacity:         64,
		},
		GDDR6: {
			Type:             GDDR6,
			Bandwidth:        448.0, // GB/s (256-bit bus)
			Latency:          10,
			BusWidth:         256,
			ClockSpeed:       14000,
			VoltageRange:     "1.35V",
			PowerConsumption: 15.0,
			Capacity:         8,
		},
		GDDR6X: {
			Type:             GDDR6X,
			Bandwidth:        760.0, // GB/s (PAM4 signaling)
			Latency:          9,
			BusWidth:         384,
			ClockSpeed:       19000,
			VoltageRange:     "1.35V",
			PowerConsumption: 28.0,
			Capacity:         24,
		},
		HBM2: {
			Type:             HBM2,
			Bandwidth:        307.0, // GB/s per stack
			Latency:          6,
			BusWidth:         1024,
			ClockSpeed:       2400,
			VoltageRange:     "1.2V",
			PowerConsumption: 8.0,
			Capacity:         16,
		},
		HBM2E: {
			Type:             HBM2E,
			Bandwidth:        460.0,
			Latency:          5,
			BusWidth:         1024,
			ClockSpeed:       3600,
			VoltageRange:     "1.2V",
			PowerConsumption: 10.0,
			Capacity:         24,
		},
		HBM3: {
			Type:             HBM3,
			Bandwidth:        819.0,
			Latency:          4,
			BusWidth:         1024,
			ClockSpeed:       6400,
			VoltageRange:     "1.1V",
			PowerConsumption: 12.0,
			Capacity:         32,
		},
	}

	return specs[ramType]
}

// String returns a string representation
func (r RAMCharacteristics) String() string {
	typeNames := map[RAMType]string{
		DDR4: "DDR4", DDR5: "DDR5",
		GDDR6: "GDDR6", GDDR6X: "GDDR6X",
		HBM2: "HBM2", HBM2E: "HBM2E", HBM3: "HBM3",
	}

	return fmt.Sprintf(`%s Specifications:
  Bandwidth: %.1f GB/s
  Latency: %d ns
  Bus Width: %d bits
  Clock Speed: %.0f MHz
  Voltage: %s
  Power: %.1f W
  Capacity: %d GB`,
		typeNames[r.Type],
		r.Bandwidth,
		r.Latency,
		r.BusWidth,
		r.ClockSpeed,
		r.VoltageRange,
		r.PowerConsumption,
		r.Capacity)
}

// =============================================================================
// Memory System
// =============================================================================

// MemorySystem represents a memory subsystem
type MemorySystem struct {
	Type          RAMType
	Characteristics RAMCharacteristics
	Data          []byte
	Size          uint64

	// Statistics
	Reads         uint64
	Writes        uint64
	BytesRead     uint64
	BytesWritten  uint64
	TotalLatency  uint64
	mu            sync.RWMutex
}

// NewMemorySystem creates a new memory system
func NewMemorySystem(ramType RAMType, sizeGB uint64) *MemorySystem {
	chars := GetRAMCharacteristics(ramType)
	chars.Capacity = sizeGB

	return &MemorySystem{
		Type:            ramType,
		Characteristics: chars,
		Data:            make([]byte, sizeGB*1024*1024*1024),
		Size:            sizeGB * 1024 * 1024 * 1024,
	}
}

// Read reads data from memory
func (m *MemorySystem) Read(addr uint64, size int) []byte {
	m.mu.RLock()
	defer m.mu.RUnlock()

	m.Reads++
	m.BytesRead += uint64(size)
	m.TotalLatency += uint64(m.Characteristics.Latency)

	if addr+uint64(size) > m.Size {
		return nil
	}

	data := make([]byte, size)
	copy(data, m.Data[addr:addr+uint64(size)])
	return data
}

// Write writes data to memory
func (m *MemorySystem) Write(addr uint64, data []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.Writes++
	m.BytesWritten += uint64(len(data))
	m.TotalLatency += uint64(m.Characteristics.Latency)

	if addr+uint64(len(data)) <= m.Size {
		copy(m.Data[addr:], data)
	}
}

// GetBandwidthUtilization calculates actual bandwidth utilization
func (m *MemorySystem) GetBandwidthUtilization(duration time.Duration) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	totalBytes := float64(m.BytesRead + m.BytesWritten)
	seconds := duration.Seconds()
	actualBandwidth := totalBytes / seconds / (1024 * 1024 * 1024) // GB/s

	return actualBandwidth / m.Characteristics.Bandwidth
}

// GetStats returns memory statistics
func (m *MemorySystem) GetStats() MemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return MemoryStats{
		Reads:        m.Reads,
		Writes:       m.Writes,
		BytesRead:    m.BytesRead,
		BytesWritten: m.BytesWritten,
		TotalLatency: m.TotalLatency,
		AvgLatency:   float64(m.TotalLatency) / float64(m.Reads+m.Writes),
	}
}

// MemoryStats contains memory access statistics
type MemoryStats struct {
	Reads        uint64
	Writes       uint64
	BytesRead    uint64
	BytesWritten uint64
	TotalLatency uint64
	AvgLatency   float64
}

// =============================================================================
// Memory Access Patterns
// =============================================================================

// AccessPattern represents different memory access patterns
type AccessPattern int

const (
	SequentialAccess AccessPattern = 0
	RandomAccess     AccessPattern = 1
	StridedAccess    AccessPattern = 2
	BlockAccess      AccessPattern = 3
)

// MemoryBenchmark benchmarks different access patterns
type MemoryBenchmark struct {
	Memory      *MemorySystem
	Pattern     AccessPattern
	AccessSize  int
	NumAccesses int
}

// RunBenchmark runs the memory benchmark
func (mb *MemoryBenchmark) RunBenchmark() BenchmarkResult {
	start := time.Now()

	switch mb.Pattern {
	case SequentialAccess:
		mb.runSequential()
	case RandomAccess:
		mb.runRandom()
	case StridedAccess:
		mb.runStrided(64) // 64-byte stride
	case BlockAccess:
		mb.runBlock(4096) // 4KB blocks
	}

	duration := time.Since(start)
	stats := mb.Memory.GetStats()

	totalBytes := stats.BytesRead + stats.BytesWritten
	bandwidth := float64(totalBytes) / duration.Seconds() / (1024 * 1024 * 1024)

	return BenchmarkResult{
		Pattern:         mb.Pattern,
		Duration:        duration,
		BytesAccessed:   totalBytes,
		Bandwidth:       bandwidth,
		AvgLatency:      stats.AvgLatency,
		BandwidthUtil:   bandwidth / mb.Memory.Characteristics.Bandwidth,
	}
}

func (mb *MemoryBenchmark) runSequential() {
	addr := uint64(0)
	for i := 0; i < mb.NumAccesses; i++ {
		mb.Memory.Read(addr, mb.AccessSize)
		addr += uint64(mb.AccessSize)
		if addr >= mb.Memory.Size {
			addr = 0
		}
	}
}

func (mb *MemoryBenchmark) runRandom() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < mb.NumAccesses; i++ {
		addr := uint64(rng.Int63n(int64(mb.Memory.Size - uint64(mb.AccessSize))))
		mb.Memory.Read(addr, mb.AccessSize)
	}
}

func (mb *MemoryBenchmark) runStrided(stride int) {
	addr := uint64(0)
	for i := 0; i < mb.NumAccesses; i++ {
		mb.Memory.Read(addr, mb.AccessSize)
		addr += uint64(stride)
		if addr >= mb.Memory.Size {
			addr = 0
		}
	}
}

func (mb *MemoryBenchmark) runBlock(blockSize int) {
	addr := uint64(0)
	for i := 0; i < mb.NumAccesses; i++ {
		mb.Memory.Read(addr, blockSize)
		addr += uint64(blockSize)
		if addr >= mb.Memory.Size {
			addr = 0
		}
	}
}

// BenchmarkResult contains benchmark results
type BenchmarkResult struct {
	Pattern       AccessPattern
	Duration      time.Duration
	BytesAccessed uint64
	Bandwidth     float64
	AvgLatency    float64
	BandwidthUtil float64
}

// String returns a string representation
func (r BenchmarkResult) String() string {
	patternNames := map[AccessPattern]string{
		SequentialAccess: "Sequential",
		RandomAccess:     "Random",
		StridedAccess:    "Strided",
		BlockAccess:      "Block",
	}

	return fmt.Sprintf(`%s Access Pattern:
  Duration: %v
  Bytes Accessed: %d MB
  Bandwidth: %.2f GB/s
  Avg Latency: %.2f ns
  Bandwidth Utilization: %.2f%%`,
		patternNames[r.Pattern],
		r.Duration,
		r.BytesAccessed/(1024*1024),
		r.Bandwidth,
		r.AvgLatency,
		100.0*r.BandwidthUtil)
}

// =============================================================================
// Data Locality Analysis
// =============================================================================

// LocalityAnalyzer analyzes memory access locality
type LocalityAnalyzer struct {
	Accesses      []uint64
	CacheLineSize uint64
}

// NewLocalityAnalyzer creates a new locality analyzer
func NewLocalityAnalyzer(cacheLineSize uint64) *LocalityAnalyzer {
	return &LocalityAnalyzer{
		Accesses:      make([]uint64, 0),
		CacheLineSize: cacheLineSize,
	}
}

// RecordAccess records a memory access
func (la *LocalityAnalyzer) RecordAccess(addr uint64) {
	la.Accesses = append(la.Accesses, addr)
}

// AnalyzeLocality analyzes temporal and spatial locality
func (la *LocalityAnalyzer) AnalyzeLocality() LocalityMetrics {
	if len(la.Accesses) == 0 {
		return LocalityMetrics{}
	}

	// Temporal locality: how often same addresses are accessed
	accessMap := make(map[uint64]int)
	for _, addr := range la.Accesses {
		accessMap[addr]++
	}

	repeatedAccesses := 0
	for _, count := range accessMap {
		if count > 1 {
			repeatedAccesses += count - 1
		}
	}

	temporalLocality := float64(repeatedAccesses) / float64(len(la.Accesses))

	// Spatial locality: how often consecutive addresses are accessed
	consecutiveAccesses := 0
	for i := 1; i < len(la.Accesses); i++ {
		diff := int64(la.Accesses[i]) - int64(la.Accesses[i-1])
		if diff >= 0 && diff <= int64(la.CacheLineSize) {
			consecutiveAccesses++
		}
	}

	spatialLocality := float64(consecutiveAccesses) / float64(len(la.Accesses)-1)

	return LocalityMetrics{
		TemporalLocality: temporalLocality,
		SpatialLocality:  spatialLocality,
		UniqueAddresses:  len(accessMap),
		TotalAccesses:    len(la.Accesses),
	}
}

// LocalityMetrics contains locality analysis results
type LocalityMetrics struct {
	TemporalLocality float64
	SpatialLocality  float64
	UniqueAddresses  int
	TotalAccesses    int
}

// String returns a string representation
func (lm LocalityMetrics) String() string {
	return fmt.Sprintf(`Locality Analysis:
  Temporal Locality: %.2f%%
  Spatial Locality: %.2f%%
  Unique Addresses: %d
  Total Accesses: %d
  Reuse Ratio: %.2fx`,
		100.0*lm.TemporalLocality,
		100.0*lm.SpatialLocality,
		lm.UniqueAddresses,
		lm.TotalAccesses,
		float64(lm.TotalAccesses)/float64(lm.UniqueAddresses))
}

// =============================================================================
// Memory-Bound vs Compute-Bound Analysis
// =============================================================================

// WorkloadType represents the type of workload
type WorkloadType int

const (
	ComputeBound WorkloadType = 0
	MemoryBound  WorkloadType = 1
	Balanced     WorkloadType = 2
)

// RooflineModel implements the roofline performance model
type RooflineModel struct {
	PeakComputePerformance float64 // FLOPS
	PeakMemoryBandwidth    float64 // GB/s
}

// NewRooflineModel creates a new roofline model
func NewRooflineModel(computeFLOPS, memoryBandwidth float64) *RooflineModel {
	return &RooflineModel{
		PeakComputePerformance: computeFLOPS,
		PeakMemoryBandwidth:    memoryBandwidth,
	}
}

// AnalyzeWorkload analyzes workload characteristics
func (rm *RooflineModel) AnalyzeWorkload(flops, bytesAccessed float64) WorkloadAnalysis {
	// Arithmetic intensity: FLOPS per byte
	intensity := flops / bytesAccessed

	// Ridge point: where compute and memory bound regions meet
	ridgePoint := rm.PeakComputePerformance / rm.PeakMemoryBandwidth

	var workloadType WorkloadType
	if intensity < ridgePoint*0.5 {
		workloadType = MemoryBound
	} else if intensity > ridgePoint*2.0 {
		workloadType = ComputeBound
	} else {
		workloadType = Balanced
	}

	// Actual performance limited by bottleneck
	actualPerformance := 0.0
	if intensity < ridgePoint {
		// Memory bound
		actualPerformance = intensity * rm.PeakMemoryBandwidth
	} else {
		// Compute bound
		actualPerformance = rm.PeakComputePerformance
	}

	return WorkloadAnalysis{
		ArithmeticIntensity: intensity,
		RidgePoint:          ridgePoint,
		WorkloadType:        workloadType,
		ActualPerformance:   actualPerformance,
		PeakPerformance:     rm.PeakComputePerformance,
		Efficiency:          actualPerformance / rm.PeakComputePerformance,
	}
}

// WorkloadAnalysis contains workload analysis results
type WorkloadAnalysis struct {
	ArithmeticIntensity float64
	RidgePoint          float64
	WorkloadType        WorkloadType
	ActualPerformance   float64
	PeakPerformance     float64
	Efficiency          float64
}

// String returns a string representation
func (wa WorkloadAnalysis) String() string {
	typeNames := map[WorkloadType]string{
		ComputeBound: "Compute-Bound",
		MemoryBound:  "Memory-Bound",
		Balanced:     "Balanced",
	}

	return fmt.Sprintf(`Workload Analysis (Roofline Model):
  Arithmetic Intensity: %.2f FLOPS/byte
  Ridge Point: %.2f FLOPS/byte
  Workload Type: %s
  Actual Performance: %.2f GFLOPS
  Peak Performance: %.2f GFLOPS
  Efficiency: %.2f%%`,
		wa.ArithmeticIntensity,
		wa.RidgePoint,
		typeNames[wa.WorkloadType],
		wa.ActualPerformance/1e9,
		wa.PeakPerformance/1e9,
		100.0*wa.Efficiency)
}
