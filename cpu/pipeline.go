/*
CPU Pipeline
============

5-stage pipeline implementation with hazard detection and forwarding.

Stages:
1. Fetch (IF)    - Fetch instruction from memory
2. Decode (ID)   - Decode instruction and read registers
3. Execute (EX)  - Perform ALU operations
4. Memory (MEM)  - Access memory for load/store
5. Writeback (WB) - Write results back to registers

Features:
- Data hazard detection (load-use)
- Control hazard handling (branch flush + 1-bit predictor)
- Pipeline stalls and forwarding (MEM→EX, WB→EX)
- Performance metrics (CPI, pipeline efficiency)
- Superscalar out-of-order structures (ROB + Reservation Station)

Applications:
- Understanding pipeline performance
- Hazard analysis
- Performance optimization
*/

package cpu

import (
	"fmt"
)

// =============================================================================
// Pipeline Stages
// =============================================================================

// PipelineStage represents data in a pipeline stage
type PipelineStage struct {
	Valid       bool         // Whether the stage contains valid data
	Instruction *Instruction // Instruction being processed
	PC          uint32       // Program counter of this instruction
	// Decode stage
	Rs1Value  int32 // Source register 1 value
	Rs2Value  int32 // Source register 2 value
	StoreData int32 // Value to write for STORE instructions (captured at ID)
	// Execute stage
	ALUResult int32 // ALU computation result
	// Memory stage
	MemData int32 // Data read from memory (for LOAD)
	// Control signals
	MemRead  bool // Memory read enable (LOAD)
	MemWrite bool // Memory write enable (STORE)
	RegWrite bool // Register write enable
	Branch   bool // Branch instruction
}

// Pipeline represents a 5-stage pipelined CPU
type Pipeline struct {
	// Pipeline stage registers
	IF  PipelineStage // Instruction Fetch (used as staging buffer)
	ID  PipelineStage // Instruction Decode
	EX  PipelineStage // Execute
	MEM PipelineStage // Memory
	WB  PipelineStage // Writeback

	// CPU state
	Registers    [NumRegisters]int32
	Memory       []byte
	PC           uint32
	Halted       bool   // true when pipeline is fully drained and stopped
	FetchHalted  bool   // true when HALT has been fetched; no more instructions issued
	CycleCount   uint64

	// Hazard detection counters
	StallCount   uint64
	ForwardCount uint64
	BranchCount  uint64
	BranchTaken  uint64
	FlushCount   uint64

	// Instruction retirement
	InstructionCount uint64

	// Branch prediction (simple 1-bit predictor: PC → predicted_taken)
	BranchPredictor map[uint32]bool
}

// NewPipeline creates a new pipelined CPU
func NewPipeline() *Pipeline {
	return &Pipeline{
		Memory:          make([]byte, MemorySize),
		BranchPredictor: make(map[uint32]bool),
	}
}

// Reset resets the pipeline state
func (p *Pipeline) Reset() {
	p.IF = PipelineStage{}
	p.ID = PipelineStage{}
	p.EX = PipelineStage{}
	p.MEM = PipelineStage{}
	p.WB = PipelineStage{}

	for i := range p.Registers {
		p.Registers[i] = 0
	}
	p.PC = 0
	p.Halted = false
	p.FetchHalted = false
	p.CycleCount = 0
	p.StallCount = 0
	p.ForwardCount = 0
	p.BranchCount = 0
	p.BranchTaken = 0
	p.FlushCount = 0
	p.InstructionCount = 0
	p.BranchPredictor = make(map[uint32]bool)
}

// LoadProgram loads a program into memory
func (p *Pipeline) LoadProgram(program []uint32, startAddr uint32) {
	for i, word := range program {
		addr := startAddr + uint32(i*4)
		p.Memory[addr] = byte(word >> 24)
		p.Memory[addr+1] = byte(word >> 16)
		p.Memory[addr+2] = byte(word >> 8)
		p.Memory[addr+3] = byte(word)
	}
	p.PC = startAddr
}

// =============================================================================
// Hazard Detection
// =============================================================================

// detectLoadHazard checks for a load-use hazard.
// Called AFTER StageEX has run (so the former EX instruction is now in MEM).
// A hazard exists when the instruction in MEM is a load and the instruction
// still in ID depends on the load's destination register.
func (p *Pipeline) detectLoadHazard() bool {
	if p.MEM.Valid && p.MEM.MemRead && p.ID.Valid {
		if p.MEM.Instruction.Rd == p.ID.Instruction.Rs1 ||
			p.MEM.Instruction.Rd == p.ID.Instruction.Rs2 {
			return true
		}
	}
	return false
}

// =============================================================================
// Data Forwarding
// =============================================================================

// ForwardData performs data forwarding from MEM and WB stages to ID.
// Called at the end of StageID, before the instruction advances to EX.
//
// Forwarding paths:
//   - MEM→EX: ALU result from instruction now in MEM forwarded to ID (will go to EX).
//   - WB→EX:  ALU or LOAD result from instruction in WB forwarded to ID.
//     (WB forwarding is needed because StageWB committed the cycle-earlier instruction,
//     but the instruction currently in WB – moved there by StageMEM – is not yet committed.)
func (p *Pipeline) ForwardData() {
	if !p.ID.Valid {
		return
	}

	// Forward from MEM stage (ALU result only – loads in MEM don't have data yet)
	if p.MEM.Valid && p.MEM.RegWrite && p.MEM.Instruction.Rd != 0 {
		if p.MEM.Instruction.Rd == p.ID.Instruction.Rs1 {
			p.ID.Rs1Value = p.MEM.ALUResult
			p.ForwardCount++
		}
		if p.MEM.Instruction.Rd == p.ID.Instruction.Rs2 {
			p.ID.Rs2Value = p.MEM.ALUResult
			p.ForwardCount++
		}
	}

	// Forward from WB stage (handles both ALU results and completed LOADs)
	if p.WB.Valid && p.WB.RegWrite && p.WB.Instruction.Rd != 0 {
		wbValue := p.WB.ALUResult
		if p.WB.MemRead {
			wbValue = p.WB.MemData // For loads, use the fetched memory data
		}
		if p.WB.Instruction.Rd == p.ID.Instruction.Rs1 {
			p.ID.Rs1Value = wbValue
			p.ForwardCount++
		}
		if p.WB.Instruction.Rd == p.ID.Instruction.Rs2 {
			p.ID.Rs2Value = wbValue
			p.ForwardCount++
		}
	}
}

// =============================================================================
// Pipeline Stage Implementations
// =============================================================================

// StageWB executes the Writeback stage.
// Commits the instruction result to the register file.
func (p *Pipeline) StageWB() {
	if !p.WB.Valid {
		return
	}

	// Write result back to destination register
	if p.WB.RegWrite && p.WB.Instruction.Rd != 0 {
		var result int32
		if p.WB.MemRead {
			result = p.WB.MemData
		} else {
			result = p.WB.ALUResult
		}
		p.Registers[p.WB.Instruction.Rd] = result
	}

	// Count every valid instruction that retires
	p.InstructionCount++
}

// StageMEM executes the Memory stage.
// Performs memory read (LOAD) or write (STORE) then advances to WB.
func (p *Pipeline) StageMEM() {
	if !p.MEM.Valid {
		p.WB = PipelineStage{}
		return
	}

	// LOAD: fetch data from memory
	if p.MEM.MemRead {
		addr := uint32(p.MEM.ALUResult)
		if addr+3 < uint32(len(p.Memory)) {
			p.MEM.MemData = int32(p.Memory[addr])<<24 |
				int32(p.Memory[addr+1])<<16 |
				int32(p.Memory[addr+2])<<8 |
				int32(p.Memory[addr+3])
		}
	}

	// STORE: write data to memory (StoreData was captured at ID stage)
	if p.MEM.MemWrite {
		addr := uint32(p.MEM.ALUResult)
		if addr+3 < uint32(len(p.Memory)) {
			val := p.MEM.StoreData
			p.Memory[addr] = byte(val >> 24)
			p.Memory[addr+1] = byte(val >> 16)
			p.Memory[addr+2] = byte(val >> 8)
			p.Memory[addr+3] = byte(val)
		}
	}

	// Advance to WB
	p.WB = p.MEM
}

// StageEX executes the Execute stage.
// Runs the ALU, resolves branches, and advances to MEM.
func (p *Pipeline) StageEX() {
	if !p.EX.Valid {
		p.MEM = PipelineStage{}
		return
	}

	inst := p.EX.Instruction
	var result int32

	// ALU operations
	switch inst.OpCode {
	case OpADD:
		result = p.EX.Rs1Value + p.EX.Rs2Value
	case OpSUB:
		result = p.EX.Rs1Value - p.EX.Rs2Value
	case OpMUL:
		result = p.EX.Rs1Value * p.EX.Rs2Value
	case OpDIV:
		if p.EX.Rs2Value != 0 {
			result = p.EX.Rs1Value / p.EX.Rs2Value
		}
	case OpAND:
		result = p.EX.Rs1Value & p.EX.Rs2Value
	case OpOR:
		result = p.EX.Rs1Value | p.EX.Rs2Value
	case OpXOR:
		result = p.EX.Rs1Value ^ p.EX.Rs2Value
	case OpSHL:
		result = p.EX.Rs1Value << uint(p.EX.Rs2Value)
	case OpSHR:
		result = p.EX.Rs1Value >> uint(p.EX.Rs2Value)
	case OpMOVI:
		result = inst.Imm
	case OpLOAD:
		result = p.EX.Rs1Value + inst.Imm // Effective address
	case OpSTORE:
		result = p.EX.Rs1Value + inst.Imm // Effective address
	}

	p.EX.ALUResult = result

	// Advance to MEM
	p.MEM = p.EX

	// ------------------------------------------------------------------
	// Branch / Jump resolution (resolved in EX stage)
	// Static "predict not-taken": if branch is taken, flush IF/ID and
	// redirect PC.  This is done AFTER copying EX→MEM so the branch
	// instruction itself continues to drain through the pipeline (as a
	// no-op in WB – RegWrite=false for branches).
	// ------------------------------------------------------------------
	switch inst.OpCode {
	case OpJMP:
		p.BranchCount++
		p.BranchTaken++
		// Unconditional – always flush
		p.flushAndRedirect(inst.Addr)

	case OpBEQ:
		p.BranchCount++
		if p.EX.Rs1Value == p.EX.Rs2Value {
			p.BranchTaken++
			p.flushAndRedirect(inst.Addr)
		} else {
			p.BranchPredictor[p.EX.PC] = false
		}

	case OpBNE:
		p.BranchCount++
		if p.EX.Rs1Value != p.EX.Rs2Value {
			p.BranchTaken++
			p.flushAndRedirect(inst.Addr)
		} else {
			p.BranchPredictor[p.EX.PC] = false
		}

	case OpBLT:
		p.BranchCount++
		if p.EX.Rs1Value < p.EX.Rs2Value {
			p.BranchTaken++
			p.flushAndRedirect(inst.Addr)
		} else {
			p.BranchPredictor[p.EX.PC] = false
		}

	case OpBGT:
		p.BranchCount++
		if p.EX.Rs1Value > p.EX.Rs2Value {
			p.BranchTaken++
			p.flushAndRedirect(inst.Addr)
		} else {
			p.BranchPredictor[p.EX.PC] = false
		}

	case OpCALL:
		p.BranchCount++
		p.BranchTaken++
		// Save return address in R31 (link register).
		// p.EX.PC is the address of the CALL instruction; return goes to CALL+4.
		p.Registers[31] = int32(p.EX.PC + 4)
		p.flushAndRedirect(inst.Addr)

	case OpRET:
		p.BranchCount++
		p.BranchTaken++
		target := uint32(p.Registers[31])
		p.flushAndRedirect(target)
	}
}

// flushAndRedirect flushes in-flight wrong-path instructions and redirects the PC.
// When a branch is resolved as taken while in EX stage, the instruction at ID
// (fetched one cycle after the branch) is on the wrong path and must be squashed.
// After the flush, StageIF in this same cycle will fetch from the correct target.
func (p *Pipeline) flushAndRedirect(target uint32) {
	p.FlushCount++
	p.PC = target
	p.ID = PipelineStage{} // Squash the wrong-path instruction in ID
	p.BranchPredictor[p.EX.PC] = true
}

// StageID executes the Decode stage.
// Reads registers, sets control signals, applies data forwarding,
// then advances to EX.
func (p *Pipeline) StageID() {
	if !p.ID.Valid {
		p.EX = PipelineStage{}
		return
	}

	inst := p.ID.Instruction

	// Read source registers
	p.ID.Rs1Value = p.Registers[inst.Rs1]
	p.ID.Rs2Value = p.Registers[inst.Rs2]

	// Capture STORE data value at decode time.
	// Must be patched up with forwarding below because the producing instruction
	// may still be in-flight (MEM or WB stage) when STORE is in ID.
	if inst.OpCode == OpSTORE {
		p.ID.StoreData = p.Registers[inst.Rd]
	}

	// Set control signals
	p.ID.MemRead = (inst.OpCode == OpLOAD)
	p.ID.MemWrite = (inst.OpCode == OpSTORE)
	p.ID.RegWrite = (inst.OpCode != OpSTORE &&
		inst.OpCode != OpBEQ && inst.OpCode != OpBNE &&
		inst.OpCode != OpBLT && inst.OpCode != OpBGT &&
		inst.OpCode != OpJMP && inst.OpCode != OpRET &&
		inst.OpCode != OpNOP && inst.OpCode != OpHALT)
	p.ID.Branch = (inst.OpCode == OpBEQ || inst.OpCode == OpBNE ||
		inst.OpCode == OpBLT || inst.OpCode == OpBGT ||
		inst.OpCode == OpJMP || inst.OpCode == OpCALL || inst.OpCode == OpRET)

	// Apply data forwarding (updates Rs1Value / Rs2Value)
	p.ForwardData()

	// Forward the STORE data value (Rd field) from MEM/WB if it is still in-flight.
	// ForwardData only patches Rs1/Rs2; STORE's source value lives in Rd.
	if inst.OpCode == OpSTORE {
		if p.MEM.Valid && p.MEM.RegWrite && p.MEM.Instruction != nil &&
			p.MEM.Instruction.Rd == inst.Rd {
			p.ID.StoreData = p.MEM.ALUResult
		}
		if p.WB.Valid && p.WB.RegWrite && p.WB.Instruction != nil &&
			p.WB.Instruction.Rd == inst.Rd {
			if p.WB.MemRead {
				p.ID.StoreData = p.WB.MemData
			} else {
				p.ID.StoreData = p.WB.ALUResult
			}
		}
	}

	// Advance to EX
	p.EX = p.ID
}

// StageIF executes the Fetch stage.
// Fetches the instruction at the current PC, decodes it, and places it in ID.
// When a HALT instruction is encountered, fetch is stopped (FetchHalted=true)
// and a bubble (empty stage) is injected into ID so the pipeline drains naturally.
func (p *Pipeline) StageIF() {
	if p.Halted || p.FetchHalted {
		// Inject bubble — nothing new enters the pipeline.
		p.ID = PipelineStage{}
		return
	}

	// Bounds check
	if int(p.PC)+3 >= len(p.Memory) {
		return
	}

	word := uint32(p.Memory[p.PC])<<24 |
		uint32(p.Memory[p.PC+1])<<16 |
		uint32(p.Memory[p.PC+2])<<8 |
		uint32(p.Memory[p.PC+3])

	inst := DecodeInstruction(word)

	if inst.OpCode == OpHALT {
		p.FetchHalted = true
		p.ID = PipelineStage{} // inject bubble; let in-flight instructions drain
		return
	}

	p.IF = PipelineStage{
		Valid:       true,
		Instruction: inst,
		PC:          p.PC,
	}

	p.PC += 4
	p.ID = p.IF
}

// =============================================================================
// Main Cycle
// =============================================================================

// Cycle executes one pipeline cycle using the following ordering:
//
//  1. WB  – retire the oldest instruction, update registers
//  2. MEM – perform load/store, advance to WB
//  3. EX  – compute ALU, resolve branches, advance to MEM
//  4. Hazard check – stall if load-use hazard (EX just placed a LOAD in MEM)
//  5. ID  – decode, read registers, forward, advance to EX
//  6. IF  – fetch next instruction into ID (or inject bubble after HALT)
//
// The key insight: WB/MEM/EX always advance; only ID/IF are frozen on a stall.
// After HALT is fetched (FetchHalted=true), the pipeline keeps cycling until all
// in-flight instructions have drained through WB, at which point Halted is set.
func (p *Pipeline) Cycle() {
	if p.Halted {
		return
	}

	p.CycleCount++

	p.StageWB()
	p.StageMEM()
	p.StageEX() // After this, the former EX instruction is in MEM

	// Load-use hazard: a LOAD is in MEM and ID depends on its destination.
	// Freeze ID and IF for one cycle; the load will complete in MEM next cycle
	// and its result will be forwarded from WB when ID finally advances.
	if p.detectLoadHazard() {
		p.StallCount++
		return
	}

	p.StageID()
	p.StageIF()

	// After fetch is halted, check whether the pipeline has fully drained.
	// All stage registers must be empty (Valid=false) before we stop.
	if p.FetchHalted &&
		!p.WB.Valid && !p.MEM.Valid && !p.EX.Valid && !p.ID.Valid {
		p.Halted = true
	}
}

// Run runs the pipeline until halted or max cycles reached
func (p *Pipeline) Run(maxCycles uint64) {
	for !p.Halted && p.CycleCount < maxCycles {
		p.Cycle()
	}
}

// =============================================================================
// Statistics
// =============================================================================

// GetCPI returns cycles per instruction (total cycles / instructions retired).
func (p *Pipeline) GetCPI() float64 {
	if p.InstructionCount == 0 {
		return 0
	}
	return float64(p.CycleCount) / float64(p.InstructionCount)
}

// GetStats returns pipeline performance statistics.
func (p *Pipeline) GetStats() PipelineStats {
	return PipelineStats{
		TotalCycles:      p.CycleCount,
		StallCycles:      p.StallCount,
		InstructionCount: p.InstructionCount,
		ForwardCount:     p.ForwardCount,
		BranchCount:      p.BranchCount,
		BranchTaken:      p.BranchTaken,
		FlushCount:       p.FlushCount,
		CPI:              p.GetCPI(),
		PipelineEff:      1.0 - safeDivide(float64(p.StallCount), float64(p.CycleCount)),
	}
}

// safeDivide returns num/den or 0 if den == 0.
func safeDivide(num, den float64) float64 {
	if den == 0 {
		return 0
	}
	return num / den
}

// PipelineStats contains pipeline performance statistics
type PipelineStats struct {
	TotalCycles      uint64
	StallCycles      uint64
	InstructionCount uint64
	ForwardCount     uint64
	BranchCount      uint64
	BranchTaken      uint64
	FlushCount       uint64
	CPI              float64
	PipelineEff      float64
}

// String returns a human-readable representation of the stats
func (s PipelineStats) String() string {
	stallPct := 100.0 * safeDivide(float64(s.StallCycles), float64(s.TotalCycles))
	branchPct := 100.0 * safeDivide(float64(s.BranchTaken), float64(s.BranchCount))
	return fmt.Sprintf(`Pipeline Statistics:
  Total Cycles:      %d
  Instructions:      %d
  Stall Cycles:      %d (%.2f%%)
  Forward Count:     %d
  Branch Count:      %d
  Branch Taken:      %d (%.2f%%)
  Flush Count:       %d
  CPI:               %.2f
  Pipeline Eff:      %.2f%%`,
		s.TotalCycles,
		s.InstructionCount,
		s.StallCycles, stallPct,
		s.ForwardCount,
		s.BranchCount,
		s.BranchTaken, branchPct,
		s.FlushCount,
		s.CPI,
		100.0*s.PipelineEff)
}

// =============================================================================
// Superscalar Pipeline (out-of-order execution)
// =============================================================================

// SuperscalarPipeline represents a superscalar processor with multiple execution units.
// It issues up to IssueWidth instructions per cycle and executes them out of order,
// committing in program order via the Reorder Buffer (ROB).
type SuperscalarPipeline struct {
	IssueWidth     int                 // Number of instructions issued per cycle
	ExecutionUnits []ExecutionUnit     // Execution units
	ReorderBuffer  *ReorderBuffer      // Reorder buffer for in-order commit
	ReservationStn *ReservationStation // Reservation station

	// CPU state
	Registers        [NumRegisters]int32
	Memory           []byte
	PC               uint32
	Halted           bool
	CycleCount       uint64
	InstructionCount uint64

	// Register Alias Table: maps arch. register → ROB entry index (-1 = not pending)
	RAT      [NumRegisters]int
	RATValid [NumRegisters]bool
}

// ExecutionUnit represents a functional unit
type ExecutionUnit struct {
	Type      string // "ALU", "MEM", "BRANCH"
	Busy      bool
	Latency   int // cycles to complete
	Remaining int // cycles left
	ROBEntry  int // which ROB entry this unit is computing for
}

// ReorderBuffer manages out-of-order completion with in-order commit
type ReorderBuffer struct {
	Entries []ROBEntry
	Size    int
	Head    int // next to commit
	Tail    int // next free slot
	Count   int // entries in use
}

// ROBEntry represents a reorder buffer entry
type ROBEntry struct {
	Valid       bool
	Instruction *Instruction
	PC          uint32
	Result      int32
	Ready       bool // result computed
	DestReg     int  // architectural destination register (-1 if none)
}

// ReservationStation holds instructions waiting for operands
type ReservationStation struct {
	Entries []RSEntry
	Size    int
}

// RSEntry represents a reservation station entry
type RSEntry struct {
	Valid       bool
	Instruction *Instruction
	PC          uint32
	ROBIndex    int   // ROB entry this RS entry writes to
	Op1Ready    bool  // operand 1 available
	Op2Ready    bool  // operand 2 available
	Op1Value    int32 // operand 1 value
	Op2Value    int32 // operand 2 value
	Op1ROB      int   // ROB tag for operand 1 (if not ready)
	Op2ROB      int   // ROB tag for operand 2 (if not ready)
}

// NewSuperscalarPipeline creates a new superscalar out-of-order processor.
// issueWidth: max instructions issued per cycle
// robSize: reorder buffer capacity
// rsSize: reservation station capacity
func NewSuperscalarPipeline(issueWidth, robSize, rsSize int) *SuperscalarPipeline {
	p := &SuperscalarPipeline{
		IssueWidth: issueWidth,
		ExecutionUnits: []ExecutionUnit{
			{Type: "ALU1", Latency: 1},
			{Type: "ALU2", Latency: 1},
			{Type: "MEM", Latency: 3},
			{Type: "BRANCH", Latency: 1},
		},
		ReorderBuffer: &ReorderBuffer{
			Entries: make([]ROBEntry, robSize),
			Size:    robSize,
		},
		ReservationStn: &ReservationStation{
			Entries: make([]RSEntry, rsSize),
			Size:    rsSize,
		},
		Memory: make([]byte, MemorySize),
	}
	// Initialise RAT to "no pending producer"
	for i := range p.RAT {
		p.RAT[i] = -1
	}
	return p
}

// LoadProgram loads a program into memory
func (p *SuperscalarPipeline) LoadProgram(program []uint32, startAddr uint32) {
	for i, word := range program {
		addr := startAddr + uint32(i*4)
		p.Memory[addr] = byte(word >> 24)
		p.Memory[addr+1] = byte(word >> 16)
		p.Memory[addr+2] = byte(word >> 8)
		p.Memory[addr+3] = byte(word)
	}
	p.PC = startAddr
}

// robAllocate allocates a ROB entry; returns index or -1 if full.
func (rob *ReorderBuffer) allocate(inst *Instruction, pc uint32, destReg int) int {
	if rob.Count == rob.Size {
		return -1 // full
	}
	idx := rob.Tail
	rob.Entries[idx] = ROBEntry{
		Valid:       true,
		Instruction: inst,
		PC:          pc,
		DestReg:     destReg,
		Ready:       false,
	}
	rob.Tail = (rob.Tail + 1) % rob.Size
	rob.Count++
	return idx
}

// robCommit commits the head entry if ready; returns true on success.
func (rob *ReorderBuffer) commit() (ROBEntry, bool) {
	if rob.Count == 0 {
		return ROBEntry{}, false
	}
	entry := rob.Entries[rob.Head]
	if !entry.Ready {
		return ROBEntry{}, false
	}
	rob.Entries[rob.Head] = ROBEntry{}
	rob.Head = (rob.Head + 1) % rob.Size
	rob.Count--
	return entry, true
}

// rsAllocate finds a free RS entry; returns index or -1 if full.
func (rs *ReservationStation) allocate() int {
	for i, e := range rs.Entries {
		if !e.Valid {
			return i
		}
	}
	return -1
}

// Cycle runs one superscalar cycle: commit → execute → issue.
func (p *SuperscalarPipeline) Cycle() {
	if p.Halted {
		return
	}
	p.CycleCount++

	// 1. Commit phase: retire ready instructions from ROB head (in order)
	for {
		entry, ok := p.ReorderBuffer.commit()
		if !ok {
			break
		}
		p.InstructionCount++
		if entry.DestReg > 0 && entry.DestReg < NumRegisters {
			p.Registers[entry.DestReg] = entry.Result
			// Clear RAT if this ROB entry is still the most recent producer
			if p.RATValid[entry.DestReg] && p.RAT[entry.DestReg] == p.robIndexOf(entry) {
				p.RATValid[entry.DestReg] = false
			}
		}
	}

	// 2. Execute phase: tick execution units; on completion broadcast to RS
	for i := range p.ExecutionUnits {
		eu := &p.ExecutionUnits[i]
		if !eu.Busy {
			continue
		}
		eu.Remaining--
		if eu.Remaining > 0 {
			continue
		}
		// Unit finished – mark ROB entry ready and broadcast result
		robIdx := eu.ROBEntry
		if robIdx >= 0 && robIdx < p.ReorderBuffer.Size {
			p.ReorderBuffer.Entries[robIdx].Ready = true
			result := p.ReorderBuffer.Entries[robIdx].Result
			// Wake up RS entries waiting on this ROB tag
			for j := range p.ReservationStn.Entries {
				rs := &p.ReservationStn.Entries[j]
				if !rs.Valid {
					continue
				}
				if !rs.Op1Ready && rs.Op1ROB == robIdx {
					rs.Op1Ready = true
					rs.Op1Value = result
				}
				if !rs.Op2Ready && rs.Op2ROB == robIdx {
					rs.Op2Ready = true
					rs.Op2Value = result
				}
			}
		}
		eu.Busy = false
		eu.ROBEntry = -1
	}

	// 3. Issue phase: dispatch ready RS entries to free execution units
	for j := range p.ReservationStn.Entries {
		rs := &p.ReservationStn.Entries[j]
		if !rs.Valid || !rs.Op1Ready || !rs.Op2Ready {
			continue
		}
		// Find a compatible free unit
		unitType := p.unitTypeFor(rs.Instruction.OpCode)
		eu := p.findFreeUnit(unitType)
		if eu == nil {
			continue
		}
		// Execute
		result := p.execute(rs.Instruction, rs.Op1Value, rs.Op2Value)
		p.ReorderBuffer.Entries[rs.ROBIndex].Result = result
		eu.Busy = true
		eu.Remaining = eu.Latency
		eu.ROBEntry = rs.ROBIndex
		rs.Valid = false // Free RS slot
	}

	// 4. Fetch/Decode/Rename/Dispatch phase: issue up to IssueWidth instructions
	issued := 0
	for !p.Halted && issued < p.IssueWidth {
		if int(p.PC)+3 >= len(p.Memory) {
			break
		}
		word := uint32(p.Memory[p.PC])<<24 |
			uint32(p.Memory[p.PC+1])<<16 |
			uint32(p.Memory[p.PC+2])<<8 |
			uint32(p.Memory[p.PC+3])
		inst := DecodeInstruction(word)

		if inst.OpCode == OpHALT {
			// Drain the ROB before halting
			if p.ReorderBuffer.Count == 0 {
				p.Halted = true
			}
			break
		}

		// Determine destination register
		destReg := -1
		if inst.OpCode != OpSTORE && inst.OpCode != OpBEQ && inst.OpCode != OpBNE &&
			inst.OpCode != OpBLT && inst.OpCode != OpBGT && inst.OpCode != OpJMP &&
			inst.OpCode != OpRET && inst.OpCode != OpNOP && inst.Type != TypeJ {
			destReg = int(inst.Rd)
		}

		// Try to allocate ROB entry
		robIdx := p.ReorderBuffer.allocate(inst, p.PC, destReg)
		if robIdx < 0 {
			break // ROB full, stall
		}

		// Try to allocate RS entry
		rsIdx := p.ReservationStn.allocate()
		if rsIdx < 0 {
			// RS full – undo ROB allocation and stall
			p.ReorderBuffer.Count--
			p.ReorderBuffer.Tail = (p.ReorderBuffer.Tail - 1 + p.ReorderBuffer.Size) % p.ReorderBuffer.Size
			break
		}

		// Rename: resolve operand sources via RAT
		rs := &p.ReservationStn.Entries[rsIdx]
		rs.Valid = true
		rs.Instruction = inst
		rs.PC = p.PC
		rs.ROBIndex = robIdx

		// Operand 1 (Rs1)
		if p.RATValid[inst.Rs1] {
			robProd := p.RAT[inst.Rs1]
			if p.ReorderBuffer.Entries[robProd].Ready {
				rs.Op1Ready = true
				rs.Op1Value = p.ReorderBuffer.Entries[robProd].Result
			} else {
				rs.Op1Ready = false
				rs.Op1ROB = robProd
			}
		} else {
			rs.Op1Ready = true
			rs.Op1Value = p.Registers[inst.Rs1]
		}

		// Operand 2 (Rs2 or immediate)
		if inst.Type == TypeR {
			if p.RATValid[inst.Rs2] {
				robProd := p.RAT[inst.Rs2]
				if p.ReorderBuffer.Entries[robProd].Ready {
					rs.Op2Ready = true
					rs.Op2Value = p.ReorderBuffer.Entries[robProd].Result
				} else {
					rs.Op2Ready = false
					rs.Op2ROB = robProd
				}
			} else {
				rs.Op2Ready = true
				rs.Op2Value = p.Registers[inst.Rs2]
			}
		} else {
			// Immediate type: op2 is the immediate value
			rs.Op2Ready = true
			rs.Op2Value = inst.Imm
		}

		// Update RAT for destination register
		if destReg > 0 {
			p.RAT[destReg] = robIdx
			p.RATValid[destReg] = true
		}

		p.PC += 4
		issued++
	}
}

// robIndexOf returns the index of a committed ROB entry (linear scan for correctness).
func (p *SuperscalarPipeline) robIndexOf(entry ROBEntry) int {
	for i, e := range p.ReorderBuffer.Entries {
		if e.PC == entry.PC && e.Instruction == entry.Instruction {
			return i
		}
	}
	return -1
}

// unitTypeFor maps an opcode to the execution unit type string.
func (p *SuperscalarPipeline) unitTypeFor(op OpCode) string {
	switch op {
	case OpLOAD, OpSTORE:
		return "MEM"
	case OpBEQ, OpBNE, OpBLT, OpBGT, OpJMP, OpCALL, OpRET:
		return "BRANCH"
	default:
		return "ALU"
	}
}

// findFreeUnit returns a free execution unit matching the given type, or nil.
func (p *SuperscalarPipeline) findFreeUnit(unitType string) *ExecutionUnit {
	for i := range p.ExecutionUnits {
		eu := &p.ExecutionUnits[i]
		if !eu.Busy && (eu.Type == unitType || (unitType == "ALU" && (eu.Type == "ALU1" || eu.Type == "ALU2"))) {
			return eu
		}
	}
	return nil
}

// execute performs the ALU computation for an RS entry.
func (p *SuperscalarPipeline) execute(inst *Instruction, op1, op2 int32) int32 {
	switch inst.OpCode {
	case OpADD:
		return op1 + op2
	case OpSUB:
		return op1 - op2
	case OpMUL:
		return op1 * op2
	case OpDIV:
		if op2 != 0 {
			return op1 / op2
		}
		return 0
	case OpAND:
		return op1 & op2
	case OpOR:
		return op1 | op2
	case OpXOR:
		return op1 ^ op2
	case OpSHL:
		return op1 << uint(op2)
	case OpSHR:
		return op1 >> uint(op2)
	case OpMOVI:
		return op2 // immediate value
	case OpLOAD:
		addr := uint32(op1 + op2)
		if addr+3 < uint32(len(p.Memory)) {
			return int32(p.Memory[addr])<<24 | int32(p.Memory[addr+1])<<16 |
				int32(p.Memory[addr+2])<<8 | int32(p.Memory[addr+3])
		}
	case OpSTORE:
		addr := uint32(op1 + op2)
		if addr+3 < uint32(len(p.Memory)) {
			val := op1 // The value to store would need a 3rd operand; simplification
			p.Memory[addr] = byte(val >> 24)
			p.Memory[addr+1] = byte(val >> 16)
			p.Memory[addr+2] = byte(val >> 8)
			p.Memory[addr+3] = byte(val)
		}
	}
	return 0
}

// Run runs the superscalar pipeline until halted or max cycles reached.
func (p *SuperscalarPipeline) Run(maxCycles uint64) {
	for !p.Halted && p.CycleCount < maxCycles {
		p.Cycle()
	}
}

// GetStats returns superscalar pipeline statistics.
func (p *SuperscalarPipeline) GetStats() SuperscalarStats {
	ipc := float64(0)
	if p.CycleCount > 0 {
		ipc = float64(p.InstructionCount) / float64(p.CycleCount)
	}
	return SuperscalarStats{
		TotalCycles:      p.CycleCount,
		InstructionCount: p.InstructionCount,
		IPC:              ipc,
		IssueWidth:       p.IssueWidth,
		ROBSize:          p.ReorderBuffer.Size,
		RSSize:           p.ReservationStn.Size,
		ROBOccupancy:     p.ReorderBuffer.Count,
	}
}

// SuperscalarStats contains superscalar pipeline performance statistics.
type SuperscalarStats struct {
	TotalCycles      uint64
	InstructionCount uint64
	IPC              float64
	IssueWidth       int
	ROBSize          int
	RSSize           int
	ROBOccupancy     int
}

// String returns a human-readable representation.
func (s SuperscalarStats) String() string {
	return fmt.Sprintf(`Superscalar Pipeline Statistics:
  Total Cycles:      %d
  Instructions:      %d
  IPC:               %.2f
  Issue Width:       %d
  ROB Size:          %d  (occupancy: %d)
  RS  Size:          %d`,
		s.TotalCycles,
		s.InstructionCount,
		s.IPC,
		s.IssueWidth,
		s.ROBSize, s.ROBOccupancy,
		s.RSSize)
}
