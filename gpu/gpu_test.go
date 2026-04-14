package gpu

import (
	"testing"
)

func TestGPUCreation(t *testing.T) {
	gpu := NewNVIDIA_A100()

	if gpu.Name != "NVIDIA A100" {
		t.Errorf("Expected NVIDIA A100, got %s", gpu.Name)
	}
	if gpu.NumSMs != 108 {
		t.Errorf("Expected 108 SMs, got %d", gpu.NumSMs)
	}
	if gpu.TotalCores != 6912 {
		t.Errorf("Expected 6912 cores, got %d", gpu.TotalCores)
	}
}

func TestGPUModels(t *testing.T) {
	models := []struct {
		name     string
		gpu      *GPU
		numSMs   int
		numCores int
	}{
		{"V100", NewNVIDIA_V100(), 80, 5120},
		{"A100", NewNVIDIA_A100(), 108, 6912},
		{"H100", NewNVIDIA_H100(), 132, 16896},
	}

	for _, m := range models {
		if m.gpu.NumSMs != m.numSMs {
			t.Errorf("%s: Expected %d SMs, got %d", m.name, m.numSMs, m.gpu.NumSMs)
		}
		if m.gpu.TotalCores != m.numCores {
			t.Errorf("%s: Expected %d cores, got %d", m.name, m.numCores, m.gpu.TotalCores)
		}
	}
}

func TestKernelLaunch(t *testing.T) {
	gpu := NewNVIDIA_A100()

	gridSize := 256
	blockSize := 1024

	kernel := gpu.LaunchKernel(gridSize, blockSize)

	expectedThreads := gridSize * blockSize
	if kernel.NumThreads != expectedThreads {
		t.Errorf("Expected %d threads, got %d", expectedThreads, kernel.NumThreads)
	}

	expectedWarps := (expectedThreads + 31) / 32
	if kernel.NumWarps != expectedWarps {
		t.Errorf("Expected %d warps, got %d", expectedWarps, kernel.NumWarps)
	}
}

func TestWarpCreation(t *testing.T) {
	warp := NewWarp(0)

	if len(warp.Threads) != 32 {
		t.Errorf("Expected 32 threads, got %d", len(warp.Threads))
	}

	activeThreads := warp.GetActiveThreads()
	if activeThreads != 32 {
		t.Errorf("Expected 32 active threads, got %d", activeThreads)
	}
}

func TestGlobalMemory(t *testing.T) {
	mem := NewGlobalMem(1.0) // 1 GB

	// Write data
	data := []byte{1, 2, 3, 4, 5}
	mem.Write(0, data)

	// Read data back
	readData := mem.Read(0, 5)

	for i := 0; i < 5; i++ {
		if readData[i] != data[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, data[i], readData[i])
		}
	}

	if mem.Accesses != 2 {
		t.Errorf("Expected 2 accesses (1 write + 1 read), got %d", mem.Accesses)
	}
}

func TestSharedMemory(t *testing.T) {
	sharedMem := NewSharedMem(64) // 64 KB

	data := []byte{10, 20, 30, 40}
	sharedMem.Write(0, data)

	readData := sharedMem.Read(0, 4)

	for i := 0; i < 4; i++ {
		if readData[i] != data[i] {
			t.Errorf("Data mismatch at index %d", i)
		}
	}
}

func TestSMCreation(t *testing.T) {
	sm := NewSM(0, 64, 1.0)

	if sm.NumCores != 64 {
		t.Errorf("Expected 64 cores, got %d", sm.NumCores)
	}
	if len(sm.CUDACores) != 64 {
		t.Errorf("Expected 64 CUDA cores, got %d", len(sm.CUDACores))
	}
	if sm.SharedMemory == nil {
		t.Error("Shared memory not initialized")
	}
}

func TestWarpScheduler(t *testing.T) {
	scheduler := NewWarpScheduler(64)

	// Add warps
	for i := 0; i < 10; i++ {
		warp := NewWarp(i)
		if !scheduler.AddWarp(warp) {
			t.Errorf("Failed to add warp %d", i)
		}
	}

	// Select warps for execution
	selected := scheduler.SelectWarps(4)

	if len(selected) > 4 {
		t.Errorf("Expected at most 4 warps, got %d", len(selected))
	}
}

func TestThreadGlobalID(t *testing.T) {
	thread := &Thread{
		ThreadIDX: 10,
		BlockIDX:  2,
	}

	globalID := thread.GetGlobalID()
	expected := 2*256 + 10 // blockID * blockSize + threadID

	if globalID != expected {
		t.Errorf("Expected global ID %d, got %d", expected, globalID)
	}
}

func TestL1Cache(t *testing.T) {
	l1 := NewL1Cache(128)

	// First read - miss
	_, hit := l1.Read(0)
	if hit {
		t.Error("Expected cache miss on first read")
	}

	// Write to cache
	data := []byte{1, 2, 3, 4}
	l1.Write(0, data)

	// Second read - hit
	readData, hit := l1.Read(0)
	if !hit {
		t.Error("Expected cache hit after write")
	}
	if readData[0] != 1 {
		t.Error("Data not correctly cached")
	}
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkKernelLaunch(b *testing.B) {
	gpu := NewNVIDIA_A100()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gpu.LaunchKernel(256, 1024)
	}
}

func BenchmarkGlobalMemoryAccess(b *testing.B) {
	mem := NewGlobalMem(1.0)
	data := make([]byte, 64)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mem.Write(uint64(i%1000)*64, data)
		mem.Read(uint64(i%1000)*64, 64)
	}
}
