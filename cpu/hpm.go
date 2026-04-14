/*
Hardware Performance Monitor (HPM / PMU)
=========================================

Simulates the performance monitoring unit found in modern CPUs.

Features:
- Fixed and programmable performance counters
- Hardware events: cycles, instructions, cache events, branch events,
  TLB events, memory events
- Counter overflow and interrupt simulation
- Per-core and system-wide aggregation
- PAPI-style preset event names

Applications:
- Profiling simulated workloads
- Identifying bottlenecks
- Validating architecture models
*/

package cpu

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// =============================================================================
// Event Types
// =============================================================================

// PMUEvent identifies a hardware event to count
type PMUEvent int

const (
	// Cycles and instructions
	EventCycles         PMUEvent = 0
	EventInstructions   PMUEvent = 1
	EventStalledCycles  PMUEvent = 2

	// Cache events
	EventL1DLoads       PMUEvent = 10
	EventL1DLoadMisses  PMUEvent = 11
	EventL1DStores      PMUEvent = 12
	EventL1DStoreMisses PMUEvent = 13
	EventL1ILoads       PMUEvent = 14
	EventL1ILoadMisses  PMUEvent = 15
	EventL2Loads        PMUEvent = 16
	EventL2LoadMisses   PMUEvent = 17
	EventL3Loads        PMUEvent = 18
	EventL3LoadMisses   PMUEvent = 19
	EventCacheRefs      PMUEvent = 20
	EventCacheMisses    PMUEvent = 21

	// Branch events
	EventBranches        PMUEvent = 30
	EventBranchMisses    PMUEvent = 31
	EventBranchTaken     PMUEvent = 32

	// TLB events
	EventDTLBLoads       PMUEvent = 40
	EventDTLBLoadMisses  PMUEvent = 41
	EventITLBLoads       PMUEvent = 42
	EventITLBLoadMisses  PMUEvent = 43

	// Memory events
	EventMemLoads        PMUEvent = 50
	EventMemStores       PMUEvent = 51
	EventMemBandwidth    PMUEvent = 52

	// Misc
	EventContextSwitches PMUEvent = 60
	EventPageFaults      PMUEvent = 61
	EventNumEvents       PMUEvent = 70
)

var eventNames = map[PMUEvent]string{
	EventCycles:          "cycles",
	EventInstructions:    "instructions",
	EventStalledCycles:   "stalled-cycles-frontend",
	EventL1DLoads:        "L1-dcache-loads",
	EventL1DLoadMisses:   "L1-dcache-load-misses",
	EventL1DStores:       "L1-dcache-stores",
	EventL1DStoreMisses:  "L1-dcache-store-misses",
	EventL1ILoads:        "L1-icache-loads",
	EventL1ILoadMisses:   "L1-icache-load-misses",
	EventL2Loads:         "L2-cache-loads",
	EventL2LoadMisses:    "L2-cache-load-misses",
	EventL3Loads:         "L3-cache-loads",
	EventL3LoadMisses:    "L3-cache-load-misses",
	EventCacheRefs:       "cache-references",
	EventCacheMisses:     "cache-misses",
	EventBranches:        "branches",
	EventBranchMisses:    "branch-misses",
	EventBranchTaken:     "branches-taken",
	EventDTLBLoads:       "dTLB-loads",
	EventDTLBLoadMisses:  "dTLB-load-misses",
	EventITLBLoads:       "iTLB-loads",
	EventITLBLoadMisses:  "iTLB-load-misses",
	EventMemLoads:        "mem-loads",
	EventMemStores:       "mem-stores",
	EventMemBandwidth:    "mem-bandwidth",
	EventContextSwitches: "context-switches",
	EventPageFaults:      "page-faults",
}

// EventName returns the human-readable name of a PMU event
func EventName(e PMUEvent) string {
	if name, ok := eventNames[e]; ok {
		return name
	}
	return fmt.Sprintf("event-%d", int(e))
}

// =============================================================================
// Performance Counter
// =============================================================================

// PerfCounter is an individual performance counter
type PerfCounter struct {
	event    PMUEvent
	count    int64         // atomic
	overflow int64         // number of times it wrapped
	enabled  bool
	overflow_threshold int64 // 0 = no overflow interrupts
}

// Increment atomically increments the counter by delta
func (pc *PerfCounter) Increment(delta int64) {
	if !pc.enabled {
		return
	}
	newVal := atomic.AddInt64(&pc.count, delta)
	if pc.overflow_threshold > 0 && newVal >= pc.overflow_threshold {
		atomic.AddInt64(&pc.overflow, 1)
		atomic.StoreInt64(&pc.count, newVal-pc.overflow_threshold)
	}
}

// Value returns the current counter value
func (pc *PerfCounter) Value() int64 {
	return atomic.LoadInt64(&pc.count)
}

// Reset resets the counter to zero
func (pc *PerfCounter) Reset() {
	atomic.StoreInt64(&pc.count, 0)
	atomic.StoreInt64(&pc.overflow, 0)
}

// =============================================================================
// PMU (Per-Core Performance Monitoring Unit)
// =============================================================================

// PMU represents the performance monitoring unit for a single CPU core
type PMU struct {
	CoreID   int
	counters [EventNumEvents]PerfCounter
	enabled  bool
	mu       sync.RWMutex
}

// NewPMU creates a new PMU for the given core
func NewPMU(coreID int) *PMU {
	pmu := &PMU{
		CoreID:  coreID,
		enabled: true,
	}
	for i := range pmu.counters {
		pmu.counters[i] = PerfCounter{
			event:   PMUEvent(i),
			enabled: true,
		}
	}
	return pmu
}

// Record atomically increments the counter for a hardware event
func (p *PMU) Record(event PMUEvent, count int64) {
	p.mu.RLock()
	if !p.enabled {
		p.mu.RUnlock()
		return
	}
	p.mu.RUnlock()

	if int(event) < len(p.counters) {
		p.counters[event].Increment(count)
	}
}

// Get returns the current value of a counter
func (p *PMU) Get(event PMUEvent) int64 {
	if int(event) >= len(p.counters) {
		return 0
	}
	return p.counters[event].Value()
}

// Reset resets all counters to zero
func (p *PMU) Reset() {
	for i := range p.counters {
		p.counters[i].Reset()
	}
}

// Enable enables the PMU
func (p *PMU) Enable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabled = true
}

// Disable disables the PMU (counters stop incrementing)
func (p *PMU) Disable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabled = false
}

// Snapshot returns a map of all event counts
func (p *PMU) Snapshot() PMUSnapshot {
	snap := PMUSnapshot{
		CoreID: p.CoreID,
		Events: make(map[PMUEvent]int64, int(EventNumEvents)),
	}
	for e := PMUEvent(0); e < EventNumEvents; e++ {
		v := p.counters[e].Value()
		if v != 0 {
			snap.Events[e] = v
		}
	}
	return snap
}

// PMUSnapshot is a point-in-time capture of all counter values
type PMUSnapshot struct {
	CoreID int
	Events map[PMUEvent]int64
}

// CPI returns Cycles Per Instruction from the snapshot
func (s PMUSnapshot) CPI() float64 {
	cycles := s.Events[EventCycles]
	insns := s.Events[EventInstructions]
	if insns == 0 {
		return 0
	}
	return float64(cycles) / float64(insns)
}

// IPC returns Instructions Per Cycle
func (s PMUSnapshot) IPC() float64 {
	cycles := s.Events[EventCycles]
	if cycles == 0 {
		return 0
	}
	return float64(s.Events[EventInstructions]) / float64(cycles)
}

// L1HitRate returns the L1 data cache hit rate
func (s PMUSnapshot) L1HitRate() float64 {
	hits := s.Events[EventL1DLoads]
	misses := s.Events[EventL1DLoadMisses]
	total := hits + misses
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

// BranchMissPct returns the branch misprediction percentage
func (s PMUSnapshot) BranchMissPct() float64 {
	branches := s.Events[EventBranches]
	misses := s.Events[EventBranchMisses]
	if branches == 0 {
		return 0
	}
	return 100.0 * float64(misses) / float64(branches)
}

// String formats the snapshot as a perf-stat-style report
func (s PMUSnapshot) String() string {
	cycles := s.Events[EventCycles]
	insns := s.Events[EventInstructions]
	result := fmt.Sprintf("Performance counter stats for Core %d:\n\n", s.CoreID)
	result += fmt.Sprintf("  %15d      %-35s\n", cycles, EventName(EventCycles))
	result += fmt.Sprintf("  %15d      %-35s\n", insns, EventName(EventInstructions))
	if cycles > 0 && insns > 0 {
		result += fmt.Sprintf("  %15s      %-35s  #  %.4f insns/cycle\n",
			"", "IPC", float64(insns)/float64(cycles))
	}

	for e := PMUEvent(2); e < EventNumEvents; e++ {
		v := s.Events[e]
		if v == 0 {
			continue
		}
		result += fmt.Sprintf("  %15d      %-35s\n", v, EventName(e))
	}
	return result
}

// =============================================================================
// System PMU – aggregates all core PMUs
// =============================================================================

// SystemPMU aggregates PMUs across all cores
type SystemPMU struct {
	CorePMUs []*PMU
	mu       sync.RWMutex
}

// NewSystemPMU creates a system-wide PMU for n cores
func NewSystemPMU(numCores int) *SystemPMU {
	pmUs := make([]*PMU, numCores)
	for i := range pmUs {
		pmUs[i] = NewPMU(i)
	}
	return &SystemPMU{CorePMUs: pmUs}
}

// Record increments a counter on a specific core
func (sp *SystemPMU) Record(coreID int, event PMUEvent, count int64) {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	if coreID >= 0 && coreID < len(sp.CorePMUs) {
		sp.CorePMUs[coreID].Record(event, count)
	}
}

// Aggregate returns the sum of all per-core counters for a given event
func (sp *SystemPMU) Aggregate(event PMUEvent) int64 {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	var total int64
	for _, pmu := range sp.CorePMUs {
		total += pmu.Get(event)
	}
	return total
}

// Snapshot returns a per-core snapshot of all counters
func (sp *SystemPMU) Snapshot() []PMUSnapshot {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	snaps := make([]PMUSnapshot, len(sp.CorePMUs))
	for i, pmu := range sp.CorePMUs {
		snaps[i] = pmu.Snapshot()
	}
	return snaps
}

// AggregateSnapshot returns one snapshot summing all cores
func (sp *SystemPMU) AggregateSnapshot() PMUSnapshot {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	agg := PMUSnapshot{
		CoreID: -1, // -1 = system aggregate
		Events: make(map[PMUEvent]int64),
	}
	for _, pmu := range sp.CorePMUs {
		snap := pmu.Snapshot()
		for e, v := range snap.Events {
			agg.Events[e] += v
		}
	}
	return agg
}

// Reset resets all per-core counters
func (sp *SystemPMU) Reset() {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	for _, pmu := range sp.CorePMUs {
		pmu.Reset()
	}
}
