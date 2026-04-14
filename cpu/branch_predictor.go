/*
Branch Prediction
=================

Advanced branch prediction hardware simulation.

Predictors implemented:
1. Static predictor         – always-not-taken baseline
2. 1-bit predictor          – last outcome
3. 2-bit saturating counter – standard textbook predictor (bimodal)
4. Gshare                   – global history XOR'd with PC
5. Tournament (hybrid)      – chooses between local and global predictors
6. Branch Target Buffer     – caches predicted target addresses

Applications:
- Understanding branch misprediction penalties
- Evaluating branch predictor accuracy
- Pipeline performance analysis
*/

package cpu

import (
	"fmt"
	"sync"
)

// =============================================================================
// 2-bit Saturating Counter
// =============================================================================

// SatCounter is a 2-bit saturating counter: 0=StrongNT, 1=WeakNT, 2=WeakT, 3=StrongT
type SatCounter uint8

const (
	StrongNotTaken SatCounter = 0
	WeakNotTaken   SatCounter = 1
	WeakTaken      SatCounter = 2
	StrongTaken    SatCounter = 3
)

// Predict returns true if this counter predicts "taken"
func (c SatCounter) Predict() bool {
	return c >= WeakTaken
}

// Update increments (taken=true) or decrements (taken=false) the counter
func (c *SatCounter) Update(taken bool) {
	if taken {
		if *c < StrongTaken {
			*c++
		}
	} else {
		if *c > StrongNotTaken {
			*c--
		}
	}
}

// =============================================================================
// Branch Target Buffer
// =============================================================================

// BTBEntry is one entry in the Branch Target Buffer
type BTBEntry struct {
	Valid     bool
	PC        uint32 // branch PC
	Target    uint32 // predicted target
	IsCall    bool   // is this a CALL instruction?
	IsReturn  bool   // is this a RET instruction?
}

// BTB is a direct-mapped Branch Target Buffer
type BTB struct {
	Entries []BTBEntry
	Size    int
	Hits    uint64
	Misses  uint64
	mu      sync.Mutex
}

// NewBTB creates a new Branch Target Buffer with 2^n entries
func NewBTB(size int) *BTB {
	return &BTB{
		Entries: make([]BTBEntry, size),
		Size:    size,
	}
}

// Lookup looks up a PC in the BTB. Returns (target, found).
func (b *BTB) Lookup(pc uint32) (uint32, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	idx := int(pc>>2) % b.Size
	e := &b.Entries[idx]
	if e.Valid && e.PC == pc {
		b.Hits++
		return e.Target, true
	}
	b.Misses++
	return 0, false
}

// Update installs or updates a BTB entry
func (b *BTB) Update(pc, target uint32, isCall, isReturn bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	idx := int(pc>>2) % b.Size
	b.Entries[idx] = BTBEntry{
		Valid:    true,
		PC:       pc,
		Target:   target,
		IsCall:   isCall,
		IsReturn: isReturn,
	}
}

// =============================================================================
// Return Address Stack
// =============================================================================

// RAS is a Return Address Stack for predicting RET targets
type RAS struct {
	Stack []uint32
	Top   int
	Size  int
	mu    sync.Mutex
}

// NewRAS creates a return address stack of the given depth
func NewRAS(depth int) *RAS {
	return &RAS{
		Stack: make([]uint32, depth),
		Size:  depth,
	}
}

// Push pushes a return address (called on CALL)
func (r *RAS) Push(returnAddr uint32) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.Stack[r.Top%r.Size] = returnAddr
	r.Top++
}

// Pop pops and returns the predicted return address (called on RET)
func (r *RAS) Pop() (uint32, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.Top == 0 {
		return 0, false
	}
	r.Top--
	return r.Stack[r.Top%r.Size], true
}

// =============================================================================
// Bimodal Predictor (2-bit saturating counter table indexed by PC)
// =============================================================================

// BimodalPredictor is a simple 2-bit counter table
type BimodalPredictor struct {
	Table []SatCounter
	Size  int
	mu    sync.Mutex
}

// NewBimodalPredictor creates a bimodal predictor with 2^n entries
func NewBimodalPredictor(size int) *BimodalPredictor {
	table := make([]SatCounter, size)
	for i := range table {
		table[i] = WeakNotTaken // initialise to "weakly not-taken"
	}
	return &BimodalPredictor{Table: table, Size: size}
}

func (p *BimodalPredictor) index(pc uint32) int {
	return int(pc>>2) % p.Size
}

// Predict predicts taken/not-taken for the given branch PC
func (p *BimodalPredictor) Predict(pc uint32) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.Table[p.index(pc)].Predict()
}

// Update updates the counter after the branch resolves
func (p *BimodalPredictor) Update(pc uint32, taken bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Table[p.index(pc)].Update(taken)
}

// =============================================================================
// Gshare Predictor
// =============================================================================

// GsharePredictor XORs global history with the PC to index a counter table
type GsharePredictor struct {
	Table         []SatCounter
	Size          int
	HistoryLength int
	GlobalHistory uint32 // shift register of recent outcomes
	mu            sync.Mutex
}

// NewGsharePredictor creates a gshare predictor
// historyBits: length of global history register (e.g. 12)
func NewGsharePredictor(historyBits int) *GsharePredictor {
	size := 1 << historyBits
	table := make([]SatCounter, size)
	for i := range table {
		table[i] = WeakNotTaken
	}
	return &GsharePredictor{
		Table:         table,
		Size:          size,
		HistoryLength: historyBits,
	}
}

func (p *GsharePredictor) index(pc uint32) int {
	mask := uint32((1 << p.HistoryLength) - 1)
	return int(((pc >> 2) ^ p.GlobalHistory) & mask)
}

// Predict predicts taken/not-taken
func (p *GsharePredictor) Predict(pc uint32) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.Table[p.index(pc)].Predict()
}

// Update updates the predictor after branch resolution
func (p *GsharePredictor) Update(pc uint32, taken bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.Table[p.index(pc)].Update(taken)
	// Shift global history register
	p.GlobalHistory = (p.GlobalHistory << 1) & uint32((1<<p.HistoryLength)-1)
	if taken {
		p.GlobalHistory |= 1
	}
}

// =============================================================================
// Tournament (Hybrid) Predictor
// =============================================================================

// TournamentPredictor combines a local (bimodal) and global (gshare) predictor
// using a "meta-predictor" that tracks which one was more accurate recently.
type TournamentPredictor struct {
	Local    *BimodalPredictor
	Global   *GsharePredictor
	Meta     []SatCounter // chooses between Local (0/1) and Global (2/3)
	MetaSize int
	mu       sync.Mutex
}

// NewTournamentPredictor creates a tournament predictor.
// tableSize: number of entries in each sub-predictor table
// historyBits: gshare global history length
func NewTournamentPredictor(tableSize, historyBits int) *TournamentPredictor {
	meta := make([]SatCounter, tableSize)
	for i := range meta {
		meta[i] = WeakTaken // start by trusting global slightly
	}
	return &TournamentPredictor{
		Local:    NewBimodalPredictor(tableSize),
		Global:   NewGsharePredictor(historyBits),
		Meta:     meta,
		MetaSize: tableSize,
	}
}

// Predict returns the combined prediction
func (p *TournamentPredictor) Predict(pc uint32) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	metaIdx := int(pc>>2) % p.MetaSize
	useGlobal := p.Meta[metaIdx].Predict()
	if useGlobal {
		return p.Global.Predict(pc)
	}
	return p.Local.Predict(pc)
}

// Update updates both sub-predictors and the meta-predictor
func (p *TournamentPredictor) Update(pc uint32, taken bool) {
	localPred := p.Local.Predict(pc)
	globalPred := p.Global.Predict(pc)

	p.Local.Update(pc, taken)
	p.Global.Update(pc, taken)

	// Update meta: prefer global if global was right and local was wrong, vice versa
	p.mu.Lock()
	defer p.mu.Unlock()

	metaIdx := int(pc>>2) % p.MetaSize
	localCorrect := localPred == taken
	globalCorrect := globalPred == taken
	if globalCorrect && !localCorrect {
		p.Meta[metaIdx].Update(true) // lean toward global
	} else if localCorrect && !globalCorrect {
		p.Meta[metaIdx].Update(false) // lean toward local
	}
}

// =============================================================================
// BranchPredictor – unified frontend combining BTB + Tournament + RAS
// =============================================================================

// AdvancedBranchPredictor is the full branch prediction unit
type AdvancedBranchPredictor struct {
	BTB        *BTB
	Predictor  *TournamentPredictor
	RAS        *RAS
	mu         sync.Mutex

	// Statistics
	Predictions  uint64
	Correct      uint64
	Incorrect    uint64
	BTBHits      uint64
	BTBMisses    uint64
	RASHits      uint64
}

// NewAdvancedBranchPredictor creates a production-grade branch predictor
func NewAdvancedBranchPredictor() *AdvancedBranchPredictor {
	return &AdvancedBranchPredictor{
		BTB:       NewBTB(512),
		Predictor: NewTournamentPredictor(4096, 12),
		RAS:       NewRAS(16),
	}
}

// Predict returns (predictedTarget, predictedTaken) for a branch at pc.
// isReturn indicates whether the instruction is a RET.
// isCall indicates whether the instruction is a CALL.
func (bp *AdvancedBranchPredictor) Predict(pc uint32, isReturn, isCall bool) (uint32, bool) {
	bp.mu.Lock()
	bp.Predictions++
	bp.mu.Unlock()

	if isReturn {
		if target, ok := bp.RAS.Pop(); ok {
			bp.mu.Lock()
			bp.RASHits++
			bp.mu.Unlock()
			return target, true
		}
		// RAS empty – fall back to BTB
	}

	if target, ok := bp.BTB.Lookup(pc); ok {
		bp.mu.Lock()
		bp.BTBHits++
		bp.mu.Unlock()
		taken := bp.Predictor.Predict(pc)
		return target, taken
	}

	bp.mu.Lock()
	bp.BTBMisses++
	bp.mu.Unlock()
	// No BTB entry – predict not taken, target unknown
	return 0, false
}

// Update trains all components with the actual branch outcome.
// actualTarget: the real branch target
// taken:        whether the branch was actually taken
// isCall:       whether it was a CALL (push return address onto RAS)
// returnAddr:   the instruction after CALL (pushed to RAS)
func (bp *AdvancedBranchPredictor) Update(pc, actualTarget, returnAddr uint32, taken, isCall, isReturn bool) {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	if taken {
		bp.BTB.Update(pc, actualTarget, isCall, isReturn)
	}
	bp.Predictor.Update(pc, taken)
	if isCall {
		bp.RAS.Push(returnAddr)
	}
}

// RecordOutcome records whether a prediction was correct
func (bp *AdvancedBranchPredictor) RecordOutcome(correct bool) {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	if correct {
		bp.Correct++
	} else {
		bp.Incorrect++
	}
}

// Accuracy returns prediction accuracy as a fraction [0,1]
func (bp *AdvancedBranchPredictor) Accuracy() float64 {
	bp.mu.Lock()
	defer bp.mu.Unlock()
	total := bp.Correct + bp.Incorrect
	if total == 0 {
		return 0
	}
	return float64(bp.Correct) / float64(total)
}

// GetStats returns branch predictor statistics
func (bp *AdvancedBranchPredictor) GetStats() BPStats {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	total := bp.Correct + bp.Incorrect
	accuracy := float64(0)
	if total > 0 {
		accuracy = float64(bp.Correct) / float64(total)
	}
	return BPStats{
		Predictions: bp.Predictions,
		Correct:     bp.Correct,
		Incorrect:   bp.Incorrect,
		Accuracy:    accuracy,
		BTBHits:     bp.BTBHits,
		BTBMisses:   bp.BTBMisses,
		RASHits:     bp.RASHits,
	}
}

// BPStats contains branch predictor statistics
type BPStats struct {
	Predictions uint64
	Correct     uint64
	Incorrect   uint64
	Accuracy    float64
	BTBHits     uint64
	BTBMisses   uint64
	RASHits     uint64
}

// String returns a human-readable representation
func (s BPStats) String() string {
	mispredictions := uint64(0)
	if s.Predictions > 0 {
		mispredictions = s.Incorrect
	}
	return fmt.Sprintf(`Branch Predictor Statistics:
  Predictions:    %d
  Correct:        %d (%.2f%%)
  Mispredictions: %d
  BTB Hits:       %d
  BTB Misses:     %d
  RAS Hits:       %d`,
		s.Predictions,
		s.Correct, 100.0*s.Accuracy,
		mispredictions,
		s.BTBHits,
		s.BTBMisses,
		s.RASHits)
}
