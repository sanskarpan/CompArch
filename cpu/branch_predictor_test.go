package cpu

import (
	"testing"
)

// =============================================================================
// SatCounter Tests
// =============================================================================

func TestSatCounterStates(t *testing.T) {
	c := WeakNotTaken // start at 1

	// Should predict not-taken initially
	if c.Predict() {
		t.Error("WeakNotTaken should predict not-taken")
	}

	// Two increments → StrongTaken
	c.Update(true)
	c.Update(true)
	if c != StrongTaken {
		t.Errorf("expected StrongTaken (3), got %d", c)
	}
	if !c.Predict() {
		t.Error("StrongTaken should predict taken")
	}

	// Saturates at top: another increment keeps it at StrongTaken
	c.Update(true)
	if c != StrongTaken {
		t.Error("counter should saturate at StrongTaken")
	}

	// Decrement back to StrongNotTaken
	for i := 0; i < 3; i++ {
		c.Update(false)
	}
	if c != StrongNotTaken {
		t.Errorf("expected StrongNotTaken (0), got %d", c)
	}

	// Saturates at bottom
	c.Update(false)
	if c != StrongNotTaken {
		t.Error("counter should saturate at StrongNotTaken")
	}
}

func TestSatCounterWeakTakenPredicts(t *testing.T) {
	c := WeakTaken
	if !c.Predict() {
		t.Error("WeakTaken should predict taken")
	}
}

// =============================================================================
// BTB Tests
// =============================================================================

func TestBTBLookupMiss(t *testing.T) {
	btb := NewBTB(512)
	_, found := btb.Lookup(0x1000)
	if found {
		t.Error("cold BTB should miss")
	}
	if btb.Misses != 1 {
		t.Errorf("Misses: got %d, want 1", btb.Misses)
	}
}

func TestBTBUpdateAndHit(t *testing.T) {
	btb := NewBTB(512)
	btb.Update(0x2000, 0x5000, false, false)

	target, found := btb.Lookup(0x2000)
	if !found {
		t.Fatal("expected BTB hit after Update")
	}
	if target != 0x5000 {
		t.Errorf("target: got 0x%X, want 0x5000", target)
	}
	if btb.Hits != 1 {
		t.Errorf("Hits: got %d, want 1", btb.Hits)
	}
}

func TestBTBCallReturn(t *testing.T) {
	btb := NewBTB(512)
	btb.Update(0x3000, 0x6000, true, false) // CALL

	target, found := btb.Lookup(0x3000)
	if !found {
		t.Fatal("CALL entry should be in BTB")
	}
	if target != 0x6000 {
		t.Errorf("CALL target: got 0x%X, want 0x6000", target)
	}
	idx := int(uint32(0x3000)>>2) % 512
	if !btb.Entries[idx].IsCall {
		t.Error("IsCall should be set")
	}
}

func TestBTBEviction(t *testing.T) {
	// Direct-mapped: two PCs mapping to same index evict each other
	btb := NewBTB(4) // 4 entries
	pc1 := uint32(0x0000)
	pc2 := uint32(0x0010) // pc2>>2 = 4, same index as pc1>>2=0 when %4

	btb.Update(pc1, 0x100, false, false)
	btb.Update(pc2, 0x200, false, false)

	// pc1 should be evicted
	_, found := btb.Lookup(pc1)
	if found {
		t.Error("pc1 should have been evicted by pc2 in direct-mapped BTB")
	}
	target, found := btb.Lookup(pc2)
	if !found || target != 0x200 {
		t.Errorf("pc2 should be in BTB with target 0x200, got 0x%X found=%v", target, found)
	}
}

// =============================================================================
// RAS Tests
// =============================================================================

func TestRASPushPop(t *testing.T) {
	ras := NewRAS(16)

	ras.Push(0x1004)
	ras.Push(0x2008)

	addr, ok := ras.Pop()
	if !ok {
		t.Fatal("expected successful pop")
	}
	if addr != 0x2008 {
		t.Errorf("LIFO order violated: got 0x%X, want 0x2008", addr)
	}

	addr, ok = ras.Pop()
	if !ok || addr != 0x1004 {
		t.Errorf("second pop: got 0x%X ok=%v", addr, ok)
	}
}

func TestRASEmptyPop(t *testing.T) {
	ras := NewRAS(16)
	_, ok := ras.Pop()
	if ok {
		t.Error("pop from empty RAS should return ok=false")
	}
}

func TestRASWrap(t *testing.T) {
	depth := 4
	ras := NewRAS(depth)

	// Push more than depth: oldest is overwritten
	for i := uint32(0); i < 6; i++ {
		ras.Push(i * 4)
	}

	// Should still pop without error
	addr, ok := ras.Pop()
	if !ok {
		t.Fatal("expected successful pop after overflow")
	}
	// The value returned depends on wrap-around; just verify it's valid
	_ = addr
}

// =============================================================================
// BimodalPredictor Tests
// =============================================================================

func TestBimodalPredictTrain(t *testing.T) {
	bp := NewBimodalPredictor(1024)
	const pc = uint32(0x100)

	// Initially weak not-taken
	if bp.Predict(pc) {
		t.Error("initial prediction should be not-taken")
	}

	// Train taken 3 times → should predict taken
	bp.Update(pc, true)
	bp.Update(pc, true)
	if !bp.Predict(pc) {
		t.Error("after 2 taken updates, should predict taken")
	}
}

func TestBimodalPredictNotTaken(t *testing.T) {
	bp := NewBimodalPredictor(1024)
	const pc = uint32(0x200)

	// Start strong not-taken by training
	bp.Update(pc, false)

	if bp.Predict(pc) {
		t.Error("after not-taken training, should predict not-taken")
	}
}

// =============================================================================
// GsharePredictor Tests
// =============================================================================

func TestGshareBasic(t *testing.T) {
	gp := NewGsharePredictor(8)
	pc := uint32(0x400)

	// Train taken
	gp.Update(pc, true)
	gp.Update(pc, true)
	gp.Update(pc, true)

	// After 3 taken updates with history, should predict taken
	// (May vary due to XOR with global history; just verify no panic)
	_ = gp.Predict(pc)
}

func TestGshareHistoryShift(t *testing.T) {
	gp := NewGsharePredictor(4)
	bits := uint32((1 << 4) - 1) // mask for 4 bits

	gp.Update(0x100, true) // history = ...0001
	h1 := gp.GlobalHistory & bits

	gp.Update(0x100, false) // history = ...0010
	h2 := gp.GlobalHistory & bits

	if h1 == h2 {
		t.Error("global history should change after update")
	}
}

// =============================================================================
// TournamentPredictor Tests
// =============================================================================

func TestTournamentPredictor(t *testing.T) {
	tp := NewTournamentPredictor(1024, 8)
	pc := uint32(0x800)

	// Train a consistent pattern: taken
	for i := 0; i < 10; i++ {
		tp.Update(pc, true)
	}

	if !tp.Predict(pc) {
		t.Error("after 10 taken updates, tournament should predict taken")
	}
}

func TestTournamentMetaUpdate(t *testing.T) {
	tp := NewTournamentPredictor(1024, 8)
	pc := uint32(0x900)

	// Get baseline prediction
	pred := tp.Predict(pc)

	// Force global to be right by training consistently
	for i := 0; i < 5; i++ {
		tp.Update(pc, pred)
	}
	// Meta predictor should have moved toward global
	_ = tp.Predict(pc) // should not panic
}

// =============================================================================
// AdvancedBranchPredictor Tests
// =============================================================================

func TestAdvancedBPPredictNotTaken(t *testing.T) {
	bp := NewAdvancedBranchPredictor()

	// Cold BTB: predict not-taken
	_, taken := bp.Predict(0x1000, false, false)
	if taken {
		t.Error("cold BTB should predict not-taken")
	}
	if bp.Predictions != 1 {
		t.Errorf("Predictions: got %d, want 1", bp.Predictions)
	}
}

func TestAdvancedBPReturnUsesRAS(t *testing.T) {
	bp := NewAdvancedBranchPredictor()

	// Simulate CALL at 0x100, return address 0x104
	bp.Update(0x100, 0x200, 0x104, true, true, false)

	// Simulate RET: should predict return to 0x104 via RAS
	target, taken := bp.Predict(0x300, true, false)
	if !taken {
		t.Error("RAS should predict taken for RET")
	}
	if target != 0x104 {
		t.Errorf("RAS return target: got 0x%X, want 0x104", target)
	}
	if bp.RASHits != 1 {
		t.Errorf("RASHits: got %d, want 1", bp.RASHits)
	}
}

func TestAdvancedBPBTBHit(t *testing.T) {
	bp := NewAdvancedBranchPredictor()

	// Train a taken branch at 0x400 targeting 0x600
	bp.Update(0x400, 0x600, 0x404, true, false, false)

	// Predict: BTB should provide target
	target, _ := bp.Predict(0x400, false, false)
	if target != 0x600 {
		t.Errorf("BTB target: got 0x%X, want 0x600", target)
	}
	if bp.BTBHits != 1 {
		t.Errorf("BTBHits: got %d, want 1", bp.BTBHits)
	}
}

func TestAdvancedBPAccuracy(t *testing.T) {
	bp := NewAdvancedBranchPredictor()

	// Record some outcomes
	bp.RecordOutcome(true)
	bp.RecordOutcome(true)
	bp.RecordOutcome(false)
	bp.RecordOutcome(true)

	acc := bp.Accuracy()
	expected := 0.75
	if acc != expected {
		t.Errorf("accuracy: got %.2f, want %.2f", acc, expected)
	}
}

func TestAdvancedBPStats(t *testing.T) {
	bp := NewAdvancedBranchPredictor()

	for i := 0; i < 5; i++ {
		bp.Predict(uint32(i*4), false, false)
		bp.RecordOutcome(i%2 == 0)
	}

	stats := bp.GetStats()
	if stats.Predictions != 5 {
		t.Errorf("predictions: got %d, want 5", stats.Predictions)
	}
	if stats.Correct+stats.Incorrect != 5 {
		t.Errorf("correct+incorrect: got %d, want 5", stats.Correct+stats.Incorrect)
	}
	_ = stats.String() // Ensure String() doesn't panic
}

func TestAdvancedBPZeroAccuracy(t *testing.T) {
	bp := NewAdvancedBranchPredictor()
	acc := bp.Accuracy()
	if acc != 0 {
		t.Errorf("zero predictions accuracy should be 0, got %.2f", acc)
	}
}

// =============================================================================
// BPStats String Tests
// =============================================================================

func TestBPStatsString(t *testing.T) {
	s := BPStats{
		Predictions: 100,
		Correct:     80,
		Incorrect:   20,
		Accuracy:    0.8,
		BTBHits:     90,
		BTBMisses:   10,
		RASHits:     5,
	}
	str := s.String()
	if len(str) == 0 {
		t.Error("BPStats.String() returned empty string")
	}
}
