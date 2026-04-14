/*
Multi-Core Processor
====================

Complete multi-core processor simulator integrating all CPU components.

Features:
- Multiple cores with independent pipelines
- Private L1/L2 caches per core
- Shared L3 cache
- Cache coherence (MESI protocol)
- Thread scheduling and assignment
- Performance monitoring and statistics

Applications:
- Understanding multi-core architecture
- Parallel processing simulation
- Performance optimization analysis
*/

package cpu

import (
	"fmt"
	"sync"
	"time"
)

// =============================================================================
// Core
// =============================================================================

// Core represents a single CPU core
type Core struct {
	ID             int
	Pipeline       *Pipeline
	L1Cache        *Cache
	L2Cache        *Cache
	CoherentCache  *CoherentCache
	SIMDUnit       *SIMDProcessor
	Running        bool
	CurrentThread  *Thread
	mu             sync.Mutex

	// Statistics
	InstructionsExecuted uint64
	CyclesActive         uint64
	CyclesIdle           uint64
}

// NewCore creates a new CPU core
func NewCore(id int, bus *CoherenceBus) *Core {
	// L1 Cache: 32KB, 8-way
	l1 := NewCache(CacheConfig{
		Size:          32 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	// L2 Cache: 256KB, 8-way
	l2 := NewCache(CacheConfig{
		Size:          256 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	l1.NextLevel = l2

	return &Core{
		ID:            id,
		Pipeline:      NewPipeline(),
		L1Cache:       l1,
		L2Cache:       l2,
		CoherentCache: NewCoherentCache(id, 64, bus),
		SIMDUnit:      NewSIMDProcessor(8, 3.0),
	}
}

// Execute runs the core for a specified number of cycles
func (c *Core) Execute(cycles uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.Running || c.CurrentThread == nil {
		c.CyclesIdle += cycles
		return
	}

	for i := uint64(0); i < cycles && c.Running; i++ {
		c.Pipeline.Cycle()
		c.CyclesActive++
		c.InstructionsExecuted++
	}
}

// AssignThread assigns a thread to this core
func (c *Core) AssignThread(thread *Thread) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.CurrentThread = thread
	c.Running = true
}

// GetStats returns core statistics
func (c *Core) GetStats() CoreStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	return CoreStats{
		CoreID:               c.ID,
		InstructionsExecuted: c.InstructionsExecuted,
		CyclesActive:         c.CyclesActive,
		CyclesIdle:           c.CyclesIdle,
		Utilization:          float64(c.CyclesActive) / float64(c.CyclesActive + c.CyclesIdle),
		L1Stats:              c.L1Cache.GetStats(),
		L2Stats:              c.L2Cache.GetStats(),
	}
}

// CoreStats contains core statistics
type CoreStats struct {
	CoreID               int
	InstructionsExecuted uint64
	CyclesActive         uint64
	CyclesIdle           uint64
	Utilization          float64
	L1Stats              CacheStats
	L2Stats              CacheStats
}

// =============================================================================
// Thread
// =============================================================================

// Thread represents a software thread
type Thread struct {
	ID          int
	Program     []uint32
	Priority    int
	State       ThreadState
	AssignedCore int
}

// ThreadState represents the state of a thread
type ThreadState int

const (
	ThreadReady   ThreadState = 0
	ThreadRunning ThreadState = 1
	ThreadBlocked ThreadState = 2
	ThreadDone    ThreadState = 3
)

// =============================================================================
// Multi-Core Processor
// =============================================================================

// MultiCoreProcessor represents a complete multi-core system
type MultiCoreProcessor struct {
	Cores          []*Core
	L3Cache        *Cache
	CoherenceBus   *CoherenceBus
	Scheduler      *Scheduler
	NumCores       int
	ClockSpeed     float64 // GHz
	TotalCycles    uint64
	mu             sync.Mutex

	// Performance counters
	TotalInstructions uint64
	TotalCacheHits    uint64
	TotalCacheMisses  uint64
}

// NewMultiCoreProcessor creates a new multi-core processor
func NewMultiCoreProcessor(numCores int, clockSpeed float64) *MultiCoreProcessor {
	// L3 Cache: 8MB, 16-way, shared across all cores
	l3 := NewCache(CacheConfig{
		Size:          8 * 1024 * 1024,
		LineSize:      64,
		Associativity: 16,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	bus := NewCoherenceBus()
	cores := make([]*Core, numCores)

	for i := 0; i < numCores; i++ {
		cores[i] = NewCore(i, bus)
		cores[i].L2Cache.NextLevel = l3
	}

	return &MultiCoreProcessor{
		Cores:        cores,
		L3Cache:      l3,
		CoherenceBus: bus,
		Scheduler:    NewScheduler(numCores),
		NumCores:     numCores,
		ClockSpeed:   clockSpeed,
	}
}

// RunCycles runs the processor for a specified number of cycles
func (p *MultiCoreProcessor) RunCycles(cycles uint64) {
	var wg sync.WaitGroup

	for i := uint64(0); i < cycles; i++ {
		p.mu.Lock()
		p.TotalCycles++
		p.mu.Unlock()

		// Execute all cores in parallel
		for _, core := range p.Cores {
			wg.Add(1)
			go func(c *Core) {
				defer wg.Done()
				c.Execute(1)
			}(core)
		}

		wg.Wait()
	}
}

// ScheduleThread schedules a thread on an available core
func (p *MultiCoreProcessor) ScheduleThread(thread *Thread) bool {
	coreID := p.Scheduler.ScheduleThread(thread)
	if coreID >= 0 && coreID < p.NumCores {
		p.Cores[coreID].AssignThread(thread)
		return true
	}
	return false
}

// GetSystemStats returns comprehensive system statistics
func (p *MultiCoreProcessor) GetSystemStats() SystemStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	stats := SystemStats{
		NumCores:      p.NumCores,
		ClockSpeed:    p.ClockSpeed,
		TotalCycles:   p.TotalCycles,
		CoreStats:     make([]CoreStats, p.NumCores),
		L3Stats:       p.L3Cache.GetStats(),
		CoherenceStats: make(map[int]CoherenceStats),
	}

	var totalInstructions uint64
	var totalActive uint64
	var totalIdle uint64

	for i, core := range p.Cores {
		coreStats := core.GetStats()
		stats.CoreStats[i] = coreStats
		totalInstructions += coreStats.InstructionsExecuted
		totalActive += coreStats.CyclesActive
		totalIdle += coreStats.CyclesIdle

		stats.CoherenceStats[i] = core.CoherentCache.GetStats()
	}

	stats.TotalInstructions = totalInstructions
	stats.OverallUtilization = float64(totalActive) / float64(totalActive + totalIdle)
	stats.IPC = float64(totalInstructions) / float64(p.TotalCycles)
	stats.BusStats = p.CoherenceBus.GetStats()

	return stats
}

// SystemStats contains complete system statistics
type SystemStats struct {
	NumCores            int
	ClockSpeed          float64
	TotalCycles         uint64
	TotalInstructions   uint64
	OverallUtilization  float64
	IPC                 float64
	CoreStats           []CoreStats
	L3Stats             CacheStats
	CoherenceStats      map[int]CoherenceStats
	BusStats            BusStats
}

// String returns a string representation of the stats
func (s SystemStats) String() string {
	result := fmt.Sprintf(`Multi-Core Processor Statistics:
  Cores: %d
  Clock Speed: %.2f GHz
  Total Cycles: %d
  Total Instructions: %d
  Overall IPC: %.2f
  Overall Utilization: %.2f%%

L3 Cache:
%s

`, s.NumCores, s.ClockSpeed, s.TotalCycles, s.TotalInstructions,
		s.IPC, 100.0*s.OverallUtilization, s.L3Stats.String())

	for i, coreStats := range s.CoreStats {
		result += fmt.Sprintf("\nCore %d:\n", i)
		result += fmt.Sprintf("  Instructions: %d\n", coreStats.InstructionsExecuted)
		result += fmt.Sprintf("  Active Cycles: %d\n", coreStats.CyclesActive)
		result += fmt.Sprintf("  Idle Cycles: %d\n", coreStats.CyclesIdle)
		result += fmt.Sprintf("  Utilization: %.2f%%\n", 100.0*coreStats.Utilization)
	}

	return result
}

// =============================================================================
// Scheduler
// =============================================================================

// Scheduler schedules threads to cores
type Scheduler struct {
	NumCores      int
	ReadyQueue    []*Thread
	RunningThreads map[int]*Thread // CoreID -> Thread
	mu            sync.Mutex
}

// NewScheduler creates a new scheduler
func NewScheduler(numCores int) *Scheduler {
	return &Scheduler{
		NumCores:       numCores,
		ReadyQueue:     make([]*Thread, 0),
		RunningThreads: make(map[int]*Thread),
	}
}

// ScheduleThread schedules a thread using round-robin
func (s *Scheduler) ScheduleThread(thread *Thread) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find available core
	for i := 0; i < s.NumCores; i++ {
		if _, exists := s.RunningThreads[i]; !exists {
			s.RunningThreads[i] = thread
			thread.AssignedCore = i
			thread.State = ThreadRunning
			return i
		}
	}

	// No available core, add to ready queue
	s.ReadyQueue = append(s.ReadyQueue, thread)
	thread.State = ThreadReady
	return -1
}

// CompleteThread marks a thread as complete and schedules next
func (s *Scheduler) CompleteThread(coreID int) *Thread {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.RunningThreads, coreID)

	// Schedule next thread from ready queue
	if len(s.ReadyQueue) > 0 {
		thread := s.ReadyQueue[0]
		s.ReadyQueue = s.ReadyQueue[1:]
		s.RunningThreads[coreID] = thread
		thread.AssignedCore = coreID
		thread.State = ThreadRunning
		return thread
	}

	return nil
}

// =============================================================================
// Performance Comparison
// =============================================================================

// BenchmarkMultiCore compares single-core vs multi-core performance
func BenchmarkMultiCore(numCores int, workload [][]uint32) BenchmarkResult {
	// Single-core benchmark
	singleCore := NewMultiCoreProcessor(1, 3.0)
	start := time.Now()
	for _, program := range workload {
		thread := &Thread{Program: program}
		singleCore.ScheduleThread(thread)
	}
	singleCore.RunCycles(10000)
	singleTime := time.Since(start)
	singleStats := singleCore.GetSystemStats()

	// Multi-core benchmark
	multiCore := NewMultiCoreProcessor(numCores, 3.0)
	start = time.Now()
	for _, program := range workload {
		thread := &Thread{Program: program}
		multiCore.ScheduleThread(thread)
	}
	multiCore.RunCycles(10000)
	multiTime := time.Since(start)
	multiStats := multiCore.GetSystemStats()

	return BenchmarkResult{
		SingleCoreTime:        singleTime,
		MultiCoreTime:         multiTime,
		Speedup:               float64(singleTime) / float64(multiTime),
		SingleCoreIPC:         singleStats.IPC,
		MultiCoreIPC:          multiStats.IPC,
		ParallelEfficiency:    (float64(singleTime) / float64(multiTime)) / float64(numCores),
	}
}

// BenchmarkResult contains benchmark comparison results
type BenchmarkResult struct {
	SingleCoreTime     time.Duration
	MultiCoreTime      time.Duration
	Speedup            float64
	SingleCoreIPC      float64
	MultiCoreIPC       float64
	ParallelEfficiency float64
}

// String returns a string representation of the result
func (r BenchmarkResult) String() string {
	return fmt.Sprintf(`Multi-Core Benchmark:
  Single-Core Time: %v
  Multi-Core Time: %v
  Speedup: %.2fx
  Single-Core IPC: %.2f
  Multi-Core IPC: %.2f
  Parallel Efficiency: %.2f%%`,
		r.SingleCoreTime,
		r.MultiCoreTime,
		r.Speedup,
		r.SingleCoreIPC,
		r.MultiCoreIPC,
		100.0*r.ParallelEfficiency)
}
