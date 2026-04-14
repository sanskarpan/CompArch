package memory

import (
	"testing"
)

func TestRAMCharacteristics(t *testing.T) {
	types := []RAMType{DDR4, DDR5, GDDR6, GDDR6X, HBM2, HBM2E, HBM3}

	for _, ramType := range types {
		chars := GetRAMCharacteristics(ramType)

		if chars.Bandwidth <= 0 {
			t.Errorf("%v: Bandwidth should be positive", ramType)
		}
		if chars.Latency <= 0 {
			t.Errorf("%v: Latency should be positive", ramType)
		}
		if chars.BusWidth <= 0 {
			t.Errorf("%v: BusWidth should be positive", ramType)
		}
	}
}

func TestRAMBandwidthProgression(t *testing.T) {
	ddr4 := GetRAMCharacteristics(DDR4)
	ddr5 := GetRAMCharacteristics(DDR5)
	hbm3 := GetRAMCharacteristics(HBM3)

	// DDR5 should have higher bandwidth than DDR4
	if ddr5.Bandwidth <= ddr4.Bandwidth {
		t.Error("DDR5 should have higher bandwidth than DDR4")
	}

	// HBM3 should have much higher bandwidth than DDR
	if hbm3.Bandwidth <= ddr5.Bandwidth {
		t.Error("HBM3 should have higher bandwidth than DDR5")
	}
}

func TestMemorySystemCreation(t *testing.T) {
	mem := NewMemorySystem(DDR4, 8)

	if mem.Size != 8*1024*1024*1024 {
		t.Errorf("Expected 8GB, got %d bytes", mem.Size)
	}
	if mem.Type != DDR4 {
		t.Error("Memory type not set correctly")
	}
}

func TestMemoryReadWrite(t *testing.T) {
	mem := NewMemorySystem(DDR4, 1)

	data := []byte{1, 2, 3, 4, 5}
	mem.Write(0, data)

	readData := mem.Read(0, 5)

	for i := 0; i < 5; i++ {
		if readData[i] != data[i] {
			t.Errorf("Data mismatch at index %d", i)
		}
	}

	stats := mem.GetStats()
	if stats.Writes != 1 {
		t.Errorf("Expected 1 write, got %d", stats.Writes)
	}
	if stats.Reads != 1 {
		t.Errorf("Expected 1 read, got %d", stats.Reads)
	}
}

func TestSequentialAccess(t *testing.T) {
	mem := NewMemorySystem(DDR4, 1)

	benchmark := &MemoryBenchmark{
		Memory:      mem,
		Pattern:     SequentialAccess,
		AccessSize:  64,
		NumAccesses: 1000,
	}

	result := benchmark.RunBenchmark()

	if result.BytesAccessed == 0 {
		t.Error("No bytes accessed")
	}
	if result.Bandwidth <= 0 {
		t.Error("Bandwidth should be positive")
	}
}

func TestRandomAccess(t *testing.T) {
	mem := NewMemorySystem(DDR4, 1)

	benchmark := &MemoryBenchmark{
		Memory:      mem,
		Pattern:     RandomAccess,
		AccessSize:  64,
		NumAccesses: 1000,
	}

	result := benchmark.RunBenchmark()

	if result.BytesAccessed == 0 {
		t.Error("No bytes accessed")
	}
}

func TestLocalityAnalysis(t *testing.T) {
	analyzer := NewLocalityAnalyzer(64)

	// Sequential access - good locality
	for i := uint64(0); i < 100; i++ {
		analyzer.RecordAccess(i * 8)
	}

	metrics := analyzer.AnalyzeLocality()

	if metrics.TotalAccesses != 100 {
		t.Errorf("Expected 100 accesses, got %d", metrics.TotalAccesses)
	}
	if metrics.SpatialLocality <= 0 {
		t.Error("Spatial locality should be positive for sequential access")
	}
}

func TestRooflineModel(t *testing.T) {
	model := NewRooflineModel(100e12, 1000.0) // 100 TFLOPS, 1000 GB/s

	// Compute-bound workload
	analysis := model.AnalyzeWorkload(1e12, 1e9) // 1 TFLOP, 1 GB data

	if analysis.ArithmeticIntensity <= 0 {
		t.Error("Arithmetic intensity should be positive")
	}
	if analysis.ActualPerformance <= 0 {
		t.Error("Actual performance should be positive")
	}
}

func TestRooflineComputeBound(t *testing.T) {
	model := NewRooflineModel(100e12, 1000.0)

	// High arithmetic intensity - compute bound
	// intensity = 1e14 / 1e8 = 1e6 FLOPS/byte
	// ridge = 100e12 / 1000 = 100e9 FLOPS/byte
	// 1e6 > 100e9 * 2.0 ? No, that's still too low
	// Let's use: 1e14 / 1e6 = 1e8 FLOPS/byte, still less than ridge
	// Actually: 1e14 / 100 = 1e12 FLOPS/byte which is > 200e9
	analysis := model.AnalyzeWorkload(1e14, 100)

	if analysis.WorkloadType != ComputeBound {
		t.Errorf("High arithmetic intensity should be compute-bound, got %v (intensity=%f, ridge=%f)",
			analysis.WorkloadType, analysis.ArithmeticIntensity, analysis.RidgePoint)
	}
}

func TestRooflineMemoryBound(t *testing.T) {
	model := NewRooflineModel(100e12, 1000.0)

	// Low arithmetic intensity - memory bound
	analysis := model.AnalyzeWorkload(1e10, 1e10)

	if analysis.WorkloadType != MemoryBound {
		t.Error("Low arithmetic intensity should be memory-bound")
	}
}

func TestLocalityTemporalReuse(t *testing.T) {
	analyzer := NewLocalityAnalyzer(64)

	// Access same addresses multiple times
	for i := 0; i < 3; i++ {
		for j := uint64(0); j < 10; j++ {
			analyzer.RecordAccess(j * 64)
		}
	}

	metrics := analyzer.AnalyzeLocality()

	// With 3 iterations, we should have 20 repeated accesses (3-1)*10
	if metrics.TemporalLocality <= 0 {
		t.Error("Temporal locality should be positive with repeated accesses")
	}
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkSequentialAccess(b *testing.B) {
	mem := NewMemorySystem(DDR4, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem.Read(uint64(i%1000)*64, 64)
	}
}

func BenchmarkRandomAccess(b *testing.B) {
	mem := NewMemorySystem(DDR4, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem.Read(uint64(i*4096)%1000000, 64)
	}
}
