package cpu

import (
	"sync"
	"testing"
)

// =============================================================================
// PerfCounter Tests
// =============================================================================

func TestPerfCounterIncrement(t *testing.T) {
	c := &PerfCounter{enabled: true, event: EventCycles}

	c.Increment(1)
	c.Increment(4)

	if c.Value() != 5 {
		t.Errorf("Value: got %d, want 5", c.Value())
	}
}

func TestPerfCounterDisabled(t *testing.T) {
	c := &PerfCounter{enabled: false, event: EventCycles}
	c.Increment(10)
	if c.Value() != 0 {
		t.Error("disabled counter should not increment")
	}
}

func TestPerfCounterReset(t *testing.T) {
	c := &PerfCounter{enabled: true, event: EventInstructions}
	c.Increment(100)
	c.Reset()
	if c.Value() != 0 {
		t.Error("counter should be 0 after Reset")
	}
}

func TestPerfCounterOverflow(t *testing.T) {
	c := &PerfCounter{
		enabled:            true,
		event:              EventCycles,
		overflow_threshold: 10,
	}
	c.Increment(15)

	// After overflow, count = 15 - 10 = 5
	if c.Value() != 5 {
		t.Errorf("overflow wraparound: got %d, want 5", c.Value())
	}
	if c.overflow != 1 {
		t.Errorf("overflow count: got %d, want 1", c.overflow)
	}
}

// =============================================================================
// PMU Tests
// =============================================================================

func TestPMURecord(t *testing.T) {
	pmu := NewPMU(0)

	pmu.Record(EventCycles, 100)
	pmu.Record(EventInstructions, 80)
	pmu.Record(EventCacheMisses, 5)

	if pmu.Get(EventCycles) != 100 {
		t.Errorf("cycles: got %d, want 100", pmu.Get(EventCycles))
	}
	if pmu.Get(EventInstructions) != 80 {
		t.Errorf("instructions: got %d, want 80", pmu.Get(EventInstructions))
	}
	if pmu.Get(EventCacheMisses) != 5 {
		t.Errorf("cache misses: got %d, want 5", pmu.Get(EventCacheMisses))
	}
}

func TestPMUOutOfBoundsEvent(t *testing.T) {
	pmu := NewPMU(0)
	// Should not panic
	pmu.Record(PMUEvent(1000), 1)
	val := pmu.Get(PMUEvent(1000))
	if val != 0 {
		t.Errorf("out-of-range event should return 0, got %d", val)
	}
}

func TestPMUReset(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventCycles, 500)
	pmu.Record(EventInstructions, 400)
	pmu.Reset()

	if pmu.Get(EventCycles) != 0 {
		t.Error("cycles should be 0 after Reset")
	}
	if pmu.Get(EventInstructions) != 0 {
		t.Error("instructions should be 0 after Reset")
	}
}

func TestPMUEnableDisable(t *testing.T) {
	pmu := NewPMU(0)

	pmu.Disable()
	pmu.Record(EventCycles, 100)
	if pmu.Get(EventCycles) != 0 {
		t.Error("disabled PMU should not record events")
	}

	pmu.Enable()
	pmu.Record(EventCycles, 50)
	if pmu.Get(EventCycles) != 50 {
		t.Errorf("re-enabled PMU: got %d, want 50", pmu.Get(EventCycles))
	}
}

func TestPMUSnapshot(t *testing.T) {
	pmu := NewPMU(2)
	pmu.Record(EventCycles, 200)
	pmu.Record(EventInstructions, 150)
	pmu.Record(EventL1DLoadMisses, 10)

	snap := pmu.Snapshot()

	if snap.CoreID != 2 {
		t.Errorf("CoreID: got %d, want 2", snap.CoreID)
	}
	if snap.Events[EventCycles] != 200 {
		t.Errorf("cycles: got %d, want 200", snap.Events[EventCycles])
	}
	if snap.Events[EventInstructions] != 150 {
		t.Errorf("instructions: got %d, want 150", snap.Events[EventInstructions])
	}
}

func TestPMUSnapshotOmitsZero(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventCycles, 50)

	snap := pmu.Snapshot()
	if _, ok := snap.Events[EventInstructions]; ok {
		t.Error("zero-value events should not appear in snapshot")
	}
}

// =============================================================================
// PMUSnapshot Derived Metrics
// =============================================================================

func TestPMUSnapshotCPI(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventCycles, 200)
	pmu.Record(EventInstructions, 100)

	snap := pmu.Snapshot()
	cpi := snap.CPI()
	if cpi != 2.0 {
		t.Errorf("CPI: got %.2f, want 2.00", cpi)
	}
}

func TestPMUSnapshotIPC(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventCycles, 100)
	pmu.Record(EventInstructions, 300)

	snap := pmu.Snapshot()
	ipc := snap.IPC()
	if ipc != 3.0 {
		t.Errorf("IPC: got %.2f, want 3.00", ipc)
	}
}

func TestPMUSnapshotCPIZero(t *testing.T) {
	snap := PMUSnapshot{Events: make(map[PMUEvent]int64)}
	if snap.CPI() != 0 {
		t.Error("CPI should be 0 when no instructions")
	}
	if snap.IPC() != 0 {
		t.Error("IPC should be 0 when no cycles")
	}
}

func TestPMUSnapshotL1HitRate(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventL1DLoads, 90)
	pmu.Record(EventL1DLoadMisses, 10)

	snap := pmu.Snapshot()
	hr := snap.L1HitRate()
	expected := 0.9
	if hr != expected {
		t.Errorf("L1 hit rate: got %.2f, want %.2f", hr, expected)
	}
}

func TestPMUSnapshotL1HitRateZero(t *testing.T) {
	snap := PMUSnapshot{Events: make(map[PMUEvent]int64)}
	if snap.L1HitRate() != 0 {
		t.Error("L1HitRate should be 0 when no loads")
	}
}

func TestPMUSnapshotBranchMissPct(t *testing.T) {
	pmu := NewPMU(0)
	pmu.Record(EventBranches, 200)
	pmu.Record(EventBranchMisses, 20)

	snap := pmu.Snapshot()
	pct := snap.BranchMissPct()
	expected := 10.0
	if pct != expected {
		t.Errorf("branch miss pct: got %.2f, want %.2f", pct, expected)
	}
}

func TestPMUSnapshotBranchMissPctZero(t *testing.T) {
	snap := PMUSnapshot{Events: make(map[PMUEvent]int64)}
	if snap.BranchMissPct() != 0 {
		t.Error("BranchMissPct should be 0 when no branches")
	}
}

func TestPMUSnapshotString(t *testing.T) {
	pmu := NewPMU(1)
	pmu.Record(EventCycles, 1000)
	pmu.Record(EventInstructions, 900)
	pmu.Record(EventCacheMisses, 50)

	snap := pmu.Snapshot()
	str := snap.String()
	if len(str) == 0 {
		t.Error("Snapshot.String() returned empty string")
	}
}

// =============================================================================
// SystemPMU Tests
// =============================================================================

func TestSystemPMURecord(t *testing.T) {
	spmu := NewSystemPMU(4)

	spmu.Record(0, EventCycles, 100)
	spmu.Record(1, EventCycles, 200)
	spmu.Record(2, EventCycles, 300)
	spmu.Record(3, EventCycles, 400)

	total := spmu.Aggregate(EventCycles)
	if total != 1000 {
		t.Errorf("aggregate cycles: got %d, want 1000", total)
	}
}

func TestSystemPMUOutOfBoundsCore(t *testing.T) {
	spmu := NewSystemPMU(2)
	// Should not panic for out-of-range core ID
	spmu.Record(99, EventCycles, 1)
	total := spmu.Aggregate(EventCycles)
	if total != 0 {
		t.Errorf("out-of-range core should not record, got %d", total)
	}
}

func TestSystemPMUSnapshot(t *testing.T) {
	spmu := NewSystemPMU(2)
	spmu.Record(0, EventCycles, 500)
	spmu.Record(1, EventInstructions, 400)

	snaps := spmu.Snapshot()
	if len(snaps) != 2 {
		t.Errorf("snapshot count: got %d, want 2", len(snaps))
	}
	if snaps[0].Events[EventCycles] != 500 {
		t.Errorf("core 0 cycles: got %d, want 500", snaps[0].Events[EventCycles])
	}
}

func TestSystemPMUAggregateSnapshot(t *testing.T) {
	spmu := NewSystemPMU(3)
	for i := 0; i < 3; i++ {
		spmu.Record(i, EventInstructions, int64(100*(i+1)))
	}

	agg := spmu.AggregateSnapshot()
	if agg.CoreID != -1 {
		t.Errorf("aggregate CoreID should be -1, got %d", agg.CoreID)
	}
	total := agg.Events[EventInstructions]
	if total != 600 { // 100+200+300
		t.Errorf("aggregate instructions: got %d, want 600", total)
	}
}

func TestSystemPMUReset(t *testing.T) {
	spmu := NewSystemPMU(2)
	spmu.Record(0, EventCycles, 999)
	spmu.Record(1, EventCycles, 888)
	spmu.Reset()

	total := spmu.Aggregate(EventCycles)
	if total != 0 {
		t.Errorf("after reset, aggregate should be 0, got %d", total)
	}
}

// =============================================================================
// Concurrency Tests
// =============================================================================

func TestPMUConcurrentRecord(t *testing.T) {
	pmu := NewPMU(0)
	const goroutines = 8
	const increments = 1000

	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < increments; j++ {
				pmu.Record(EventCycles, 1)
			}
		}()
	}
	wg.Wait()

	expected := int64(goroutines * increments)
	if pmu.Get(EventCycles) != expected {
		t.Errorf("concurrent record: got %d, want %d", pmu.Get(EventCycles), expected)
	}
}

// =============================================================================
// EventName Tests
// =============================================================================

func TestEventName(t *testing.T) {
	name := EventName(EventCycles)
	if name != "cycles" {
		t.Errorf("EventName(EventCycles): got %q, want %q", name, "cycles")
	}

	// Unknown event
	name = EventName(PMUEvent(999))
	if len(name) == 0 {
		t.Error("unknown event name should not be empty")
	}
}
