/*
GPU Architecture
================

Comprehensive GPU architecture simulator including CUDA cores,
streaming multiprocessors, memory hierarchy, and warp scheduling.

Features:
- CUDA core simulation
- Streaming Multiprocessor (SM) architecture
- Warp scheduling and execution
- GPU memory hierarchy (global, shared, constant, local, registers)
- Thread block and grid management
- Performance metrics (occupancy, throughput, bandwidth)

Applications:
- Understanding GPU parallel execution
- Performance optimization for ML workloads
- Memory access pattern analysis
*/

package gpu

import (
	"fmt"
	"sync"
	"time"
)

// =============================================================================
// GPU Thread Hierarchy
// =============================================================================

// Thread represents a single GPU thread
type Thread struct {
	ThreadIDX int // Thread ID within block (x dimension)
	ThreadIDY int // Thread ID within block (y dimension)
	ThreadIDZ int // Thread ID within block (z dimension)
	BlockIDX  int // Block ID (x dimension)
	BlockIDY  int // Block ID (y dimension)
	BlockIDZ  int // Block ID (z dimension)
	WarpID    int // Warp ID
	LaneID    int // Lane ID within warp (0-31)
}

// GetGlobalID returns the global thread ID for a 1-D kernel.
// blockSize must match the kernel launch parameter (threads per block).
func (t *Thread) GetGlobalID(blockSize int) int {
	return t.BlockIDX*blockSize + t.ThreadIDX
}

// GetGlobalID3D returns the global thread ID for a 3-D kernel launch.
func (t *Thread) GetGlobalID3D(blockDimX, blockDimY, blockDimZ int) (int, int, int) {
	gx := t.BlockIDX*blockDimX + t.ThreadIDX
	gy := t.BlockIDY*blockDimY + t.ThreadIDY
	gz := t.BlockIDZ*blockDimZ + t.ThreadIDZ
	return gx, gy, gz
}

// Warp represents a group of 32 threads executing in lockstep
type Warp struct {
	ID             int
	Threads        []*Thread
	PC             uint64     // Program counter
	ActiveMask     uint32     // Which threads are active (bitmask)
	State          WarpState
	InstructionsExecuted uint64
}

// WarpState represents the execution state of a warp
type WarpState int

const (
	WarpReady      WarpState = 0
	WarpRunning    WarpState = 1
	WarpStalled    WarpState = 2
	WarpCompleted  WarpState = 3
)

// NewWarp creates a new warp
func NewWarp(id int) *Warp {
	warp := &Warp{
		ID:         id,
		Threads:    make([]*Thread, 32),
		ActiveMask: 0xFFFFFFFF, // All threads active
		State:      WarpReady,
	}

	for i := 0; i < 32; i++ {
		warp.Threads[i] = &Thread{
			ThreadIDX: i,
			WarpID:    id,
			LaneID:    i,
		}
	}

	return warp
}

// GetActiveThreads returns the number of active threads
func (w *Warp) GetActiveThreads() int {
	count := 0
	for i := 0; i < 32; i++ {
		if (w.ActiveMask & (1 << i)) != 0 {
			count++
		}
	}
	return count
}

// =============================================================================
// GPU Memory Hierarchy
// =============================================================================

// MemoryType represents different GPU memory types
type MemoryType int

const (
	GlobalMemory   MemoryType = 0 // Large, high latency
	SharedMemory   MemoryType = 1 // Small, low latency, per-SM
	ConstantMemory MemoryType = 2 // Read-only, cached
	LocalMemory    MemoryType = 3 // Private per-thread
	RegisterMemory MemoryType = 4 // Fastest, limited
	TextureMemory  MemoryType = 5 // Specialized, cached
)

// MemoryCharacteristics defines memory properties
type MemoryCharacteristics struct {
	Type      MemoryType
	Size      uint64  // Bytes
	Bandwidth float64 // GB/s
	Latency   int     // Cycles
}

// GlobalMem represents GPU global memory (VRAM)
type GlobalMem struct {
	Data      []byte
	Size      uint64  // Total size in bytes
	Bandwidth float64 // GB/s
	Latency   int     // Access latency in cycles
	Accesses  uint64
	mu        sync.RWMutex
}

// NewGlobalMem creates global memory
func NewGlobalMem(sizeGB float64) *GlobalMem {
	return &GlobalMem{
		Data:      make([]byte, uint64(sizeGB*1024*1024*1024)),
		Size:      uint64(sizeGB * 1024 * 1024 * 1024),
		Bandwidth: 900.0, // GB/s (e.g., A100)
		Latency:   400,   // cycles
	}
}

// Read reads from global memory.
// Uses a full write lock because Accesses counter is modified.
func (g *GlobalMem) Read(addr uint64, size int) []byte {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.Accesses++

	if addr+uint64(size) > g.Size {
		return nil
	}

	data := make([]byte, size)
	copy(data, g.Data[addr:addr+uint64(size)])
	return data
}

// Write writes to global memory
func (g *GlobalMem) Write(addr uint64, data []byte) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.Accesses++

	if addr+uint64(len(data)) <= g.Size {
		copy(g.Data[addr:], data)
	}
}

// SharedMem represents shared memory per SM
type SharedMem struct {
	Data       []byte
	Size       uint64
	Bandwidth  float64
	Latency    int
	BankConflicts uint64
	mu         sync.RWMutex
}

// NewSharedMem creates shared memory
func NewSharedMem(sizeKB float64) *SharedMem {
	return &SharedMem{
		Data:      make([]byte, uint64(sizeKB*1024)),
		Size:      uint64(sizeKB * 1024),
		Bandwidth: 15000.0, // GB/s (much higher than global)
		Latency:   28,      // cycles (much lower than global)
	}
}

// Read reads from shared memory
func (s *SharedMem) Read(addr uint64, size int) []byte {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if addr+uint64(size) > s.Size {
		return nil
	}

	data := make([]byte, size)
	copy(data, s.Data[addr:addr+uint64(size)])
	return data
}

// Write writes to shared memory
func (s *SharedMem) Write(addr uint64, data []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if addr+uint64(len(data)) <= s.Size {
		copy(s.Data[addr:], data)
	}
}

// RegisterFile represents the register file per SM
type RegisterFile struct {
	Registers     []uint32
	NumRegisters  int
	RegistersPerThread int
}

// NewRegisterFile creates a register file
func NewRegisterFile(numRegisters int) *RegisterFile {
	return &RegisterFile{
		Registers:          make([]uint32, numRegisters),
		NumRegisters:       numRegisters,
		RegistersPerThread: 255, // Max per thread
	}
}

// =============================================================================
// CUDA Core
// =============================================================================

// CUDACore represents a single CUDA core (ALU)
type CUDACore struct {
	ID             int
	Busy           bool
	CurrentOp      string
	Utilization    float64
	OpsExecuted    uint64
}

// NewCUDACore creates a new CUDA core
func NewCUDACore(id int) *CUDACore {
	return &CUDACore{
		ID: id,
	}
}

// Execute executes an operation
func (c *CUDACore) Execute(op string) {
	c.Busy = true
	c.CurrentOp = op
	c.OpsExecuted++
}

// =============================================================================
// Streaming Multiprocessor (SM)
// =============================================================================

// SM represents a Streaming Multiprocessor
type SM struct {
	ID             int
	CUDACores      []*CUDACore
	NumCores       int
	WarpScheduler  *WarpScheduler
	SharedMemory   *SharedMem
	RegisterFile   *RegisterFile
	L1Cache        *L1Cache
	ActiveWarps    []*Warp
	MaxWarps       int
	ClockSpeed     float64 // GHz

	// Statistics
	CyclesExecuted uint64
	Utilization    float64
	mu             sync.Mutex
}

// NewSM creates a new Streaming Multiprocessor
func NewSM(id int, numCores int, clockSpeed float64) *SM {
	cores := make([]*CUDACore, numCores)
	for i := 0; i < numCores; i++ {
		cores[i] = NewCUDACore(i)
	}

	return &SM{
		ID:            id,
		CUDACores:     cores,
		NumCores:      numCores,
		WarpScheduler: NewWarpScheduler(64), // Max 64 warps per SM
		SharedMemory:  NewSharedMem(64),      // 64KB shared memory
		RegisterFile:  NewRegisterFile(65536), // 64K registers
		L1Cache:       NewL1Cache(128),       // 128KB L1
		MaxWarps:      64,
		ClockSpeed:    clockSpeed,
	}
}

// Schedule schedules warps for execution
func (sm *SM) Schedule() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Warp scheduler selects ready warps
	selectedWarps := sm.WarpScheduler.SelectWarps(4) // Issue up to 4 warps

	// Execute warps on CUDA cores
	for _, warp := range selectedWarps {
		// Simulate warp execution
		warp.InstructionsExecuted++
		sm.CyclesExecuted++
	}
}

// GetOccupancy calculates SM occupancy
func (sm *SM) GetOccupancy() float64 {
	return float64(len(sm.ActiveWarps)) / float64(sm.MaxWarps)
}

// =============================================================================
// Warp Scheduler
// =============================================================================

// WarpScheduler schedules warps for execution
type WarpScheduler struct {
	ReadyWarps    []*Warp
	StalledWarps  []*Warp
	MaxWarps      int
	ScheduledWarps uint64
	mu            sync.Mutex
}

// NewWarpScheduler creates a new warp scheduler
func NewWarpScheduler(maxWarps int) *WarpScheduler {
	return &WarpScheduler{
		ReadyWarps:   make([]*Warp, 0),
		StalledWarps: make([]*Warp, 0),
		MaxWarps:     maxWarps,
	}
}

// AddWarp adds a warp to the scheduler
func (ws *WarpScheduler) AddWarp(warp *Warp) bool {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if len(ws.ReadyWarps)+len(ws.StalledWarps) >= ws.MaxWarps {
		return false // No space
	}

	ws.ReadyWarps = append(ws.ReadyWarps, warp)
	return true
}

// SelectWarps selects warps for execution (greedy round-robin)
func (ws *WarpScheduler) SelectWarps(count int) []*Warp {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	selected := make([]*Warp, 0, count)

	for i := 0; i < len(ws.ReadyWarps) && len(selected) < count; i++ {
		warp := ws.ReadyWarps[i]
		if warp.State == WarpReady {
			selected = append(selected, warp)
			warp.State = WarpRunning
			ws.ScheduledWarps++
		}
	}

	return selected
}

// =============================================================================
// L1 Cache (per SM)
// =============================================================================

// L1Cache represents L1 cache per SM
type L1Cache struct {
	Size      uint64
	LineSize  uint64
	Lines     map[uint64][]byte
	Hits      uint64
	Misses    uint64
	mu        sync.RWMutex
}

// NewL1Cache creates a new L1 cache
func NewL1Cache(sizeKB uint64) *L1Cache {
	return &L1Cache{
		Size:     sizeKB * 1024,
		LineSize: 128, // 128B cache lines
		Lines:    make(map[uint64][]byte),
	}
}

// Read reads from L1 cache.
// Uses a full write lock because Hits/Misses counters are modified.
func (l1 *L1Cache) Read(addr uint64) ([]byte, bool) {
	l1.mu.Lock()
	defer l1.mu.Unlock()

	lineAddr := addr / l1.LineSize

	if data, exists := l1.Lines[lineAddr]; exists {
		l1.Hits++
		return data, true
	}

	l1.Misses++
	return nil, false
}

// Write writes to L1 cache
func (l1 *L1Cache) Write(addr uint64, data []byte) {
	l1.mu.Lock()
	defer l1.mu.Unlock()

	lineAddr := addr / l1.LineSize
	l1.Lines[lineAddr] = data
}

// =============================================================================
// GPU Device
// =============================================================================

// GPU represents a complete GPU device
type GPU struct {
	Name           string
	SMs            []*SM
	NumSMs         int
	CoresPerSM     int
	TotalCores     int
	GlobalMemory   *GlobalMem
	ConstantMemory []byte
	ClockSpeed     float64 // GHz
	MemoryBandwidth float64 // GB/s

	// Compute capability
	ComputeCapability string
	MaxThreadsPerBlock int
	MaxBlocksPerSM     int
	MaxWarpsPerSM      int

	// Statistics
	KernelLaunches  uint64
	TotalCycles     uint64
	ActiveTime      time.Duration
	mu              sync.Mutex
}

// NewGPU creates a new GPU
func NewGPU(name string, numSMs int, coresPerSM int, memoryGB float64, clockSpeed float64) *GPU {
	sms := make([]*SM, numSMs)
	for i := 0; i < numSMs; i++ {
		sms[i] = NewSM(i, coresPerSM, clockSpeed)
	}

	return &GPU{
		Name:               name,
		SMs:                sms,
		NumSMs:             numSMs,
		CoresPerSM:         coresPerSM,
		TotalCores:         numSMs * coresPerSM,
		GlobalMemory:       NewGlobalMem(memoryGB),
		ConstantMemory:     make([]byte, 64*1024), // 64KB constant memory
		ClockSpeed:         clockSpeed,
		MemoryBandwidth:    900.0,
		ComputeCapability:  "8.0",
		MaxThreadsPerBlock: 1024,
		MaxBlocksPerSM:     16,
		MaxWarpsPerSM:      64,
	}
}

// LaunchKernel launches a kernel on the GPU
func (g *GPU) LaunchKernel(gridSize, blockSize int) *KernelExecution {
	g.mu.Lock()
	g.KernelLaunches++
	g.mu.Unlock()

	start := time.Now()

	kernel := &KernelExecution{
		GridSize:   gridSize,
		BlockSize:  blockSize,
		NumThreads: gridSize * blockSize,
		NumWarps:   (gridSize * blockSize + 31) / 32,
		StartTime:  start,
	}

	// Distribute warps across SMs
	warpsPerSM := kernel.NumWarps / g.NumSMs
	for i, sm := range g.SMs {
		for j := 0; j < warpsPerSM; j++ {
			warp := NewWarp(i*warpsPerSM + j)
			sm.WarpScheduler.AddWarp(warp)
		}
	}

	// Simulate execution
	g.ExecuteKernel(kernel)

	kernel.EndTime = time.Now()
	kernel.Duration = kernel.EndTime.Sub(kernel.StartTime)

	return kernel
}

// ExecuteKernel executes a kernel
func (g *GPU) ExecuteKernel(kernel *KernelExecution) {
	var wg sync.WaitGroup

	// Execute on all SMs in parallel
	for _, sm := range g.SMs {
		wg.Add(1)
		go func(s *SM) {
			defer wg.Done()
			// Simulate warp scheduling and execution
			for i := 0; i < 100; i++ {
				s.Schedule()
			}
		}(sm)
	}

	wg.Wait()

	// Update statistics
	g.mu.Lock()
	g.TotalCycles += 100
	g.mu.Unlock()
}

// GetStats returns GPU statistics
func (g *GPU) GetStats() GPUStats {
	g.mu.Lock()
	defer g.mu.Unlock()

	stats := GPUStats{
		Name:            g.Name,
		NumSMs:          g.NumSMs,
		TotalCores:      g.TotalCores,
		KernelLaunches:  g.KernelLaunches,
		TotalCycles:     g.TotalCycles,
		SMStats:         make([]SMStats, g.NumSMs),
	}

	var totalOccupancy float64
	for i, sm := range g.SMs {
		stats.SMStats[i] = SMStats{
			SMID:       sm.ID,
			Occupancy:  sm.GetOccupancy(),
			ActiveWarps: len(sm.ActiveWarps),
		}
		totalOccupancy += sm.GetOccupancy()
	}

	stats.AvgOccupancy = totalOccupancy / float64(g.NumSMs)

	return stats
}

// KernelExecution represents a kernel execution
type KernelExecution struct {
	GridSize   int
	BlockSize  int
	NumThreads int
	NumWarps   int
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
}

// GPUStats contains GPU performance statistics
type GPUStats struct {
	Name           string
	NumSMs         int
	TotalCores     int
	KernelLaunches uint64
	TotalCycles    uint64
	AvgOccupancy   float64
	SMStats        []SMStats
}

// SMStats contains SM statistics
type SMStats struct {
	SMID        int
	Occupancy   float64
	ActiveWarps int
}

// String returns a string representation of the stats
func (s GPUStats) String() string {
	return fmt.Sprintf(`GPU Statistics:
  Name: %s
  Streaming Multiprocessors: %d
  Total CUDA Cores: %d
  Kernel Launches: %d
  Total Cycles: %d
  Average Occupancy: %.2f%%`,
		s.Name,
		s.NumSMs,
		s.TotalCores,
		s.KernelLaunches,
		s.TotalCycles,
		100.0*s.AvgOccupancy)
}

// =============================================================================
// Common GPU Configurations
// =============================================================================

// NewNVIDIA_A100 creates an NVIDIA A100 GPU simulator
func NewNVIDIA_A100() *GPU {
	return NewGPU("NVIDIA A100", 108, 64, 80.0, 1.41)
}

// NewNVIDIA_H100 creates an NVIDIA H100 GPU simulator
func NewNVIDIA_H100() *GPU {
	return NewGPU("NVIDIA H100", 132, 128, 80.0, 1.83)
}

// NewNVIDIA_V100 creates an NVIDIA V100 GPU simulator
func NewNVIDIA_V100() *GPU {
	return NewGPU("NVIDIA V100", 80, 64, 32.0, 1.53)
}
