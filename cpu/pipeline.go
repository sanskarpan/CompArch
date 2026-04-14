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
- Data hazard detection
- Control hazard handling (branch prediction)
- Pipeline stalls and forwarding
- Performance metrics (CPI, pipeline efficiency)

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
	PC          uint32       // Program counter
	// Decode stage
	Rs1Value int32 // Source register 1 value
	Rs2Value int32 // Source register 2 value
	// Execute stage
	ALUResult int32 // ALU computation result
	// Memory stage
	MemData int32 // Data read from memory
	// Control signals
	MemRead  bool // Memory read enable
	MemWrite bool // Memory write enable
	RegWrite bool // Register write enable
	Branch   bool // Branch instruction
}

// Pipeline represents a 5-stage pipelined CPU
type Pipeline struct {
	// Pipeline stages
	IF  PipelineStage // Instruction Fetch
	ID  PipelineStage // Instruction Decode
	EX  PipelineStage // Execute
	MEM PipelineStage // Memory
	WB  PipelineStage // Writeback

	// CPU state
	Registers  [NumRegisters]int32
	Memory     []byte
	PC         uint32
	Halted     bool
	CycleCount uint64

	// Hazard detection
	StallCount     uint64
	ForwardCount   uint64
	BranchCount    uint64
	BranchTaken    uint64
	FlushCount     uint64

	// Branch prediction (simple 1-bit predictor)
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
	p.CycleCount = 0
	p.StallCount = 0
	p.ForwardCount = 0
	p.BranchCount = 0
	p.BranchTaken = 0
	p.FlushCount = 0
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

// DetectHazards detects data and control hazards
func (p *Pipeline) DetectHazards() bool {
	// Data hazard: Load-use hazard
	// If EX stage has a load instruction and ID stage needs that register
	if p.EX.Valid && p.EX.MemRead && p.ID.Valid {
		if p.EX.Instruction.Rd == p.ID.Instruction.Rs1 ||
			p.EX.Instruction.Rd == p.ID.Instruction.Rs2 {
			// Stall required
			return true
		}
	}

	return false
}

// ForwardData handles data forwarding
func (p *Pipeline) ForwardData() {
	if !p.ID.Valid {
		return
	}

	// Forward from MEM stage
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

	// Forward from WB stage
	if p.WB.Valid && p.WB.RegWrite && p.WB.Instruction.Rd != 0 {
		if p.WB.Instruction.Rd == p.ID.Instruction.Rs1 {
			p.ID.Rs1Value = p.WB.ALUResult
			p.ForwardCount++
		}
		if p.WB.Instruction.Rd == p.ID.Instruction.Rs2 {
			p.ID.Rs2Value = p.WB.ALUResult
			p.ForwardCount++
		}
	}
}

// Writeback stage
func (p *Pipeline) StageWB() {
	if !p.WB.Valid {
		return
	}

	// Write result back to register
	if p.WB.RegWrite && p.WB.Instruction.Rd != 0 {
		var result int32
		if p.WB.MemRead {
			result = p.WB.MemData
		} else {
			result = p.WB.ALUResult
		}
		p.Registers[p.WB.Instruction.Rd] = result
	}
}

// Memory stage
func (p *Pipeline) StageMEM() {
	if !p.MEM.Valid {
		return
	}

	inst := p.MEM.Instruction

	// Memory operations
	if p.MEM.MemRead {
		addr := uint32(p.MEM.ALUResult)
		if addr+3 < uint32(len(p.Memory)) {
			p.MEM.MemData = int32(p.Memory[addr])<<24 |
				int32(p.Memory[addr+1])<<16 |
				int32(p.Memory[addr+2])<<8 |
				int32(p.Memory[addr+3])
		}
	}

	if p.MEM.MemWrite {
		addr := uint32(p.MEM.ALUResult)
		if addr+3 < uint32(len(p.Memory)) {
			val := p.Registers[inst.Rd]
			p.Memory[addr] = byte(val >> 24)
			p.Memory[addr+1] = byte(val >> 16)
			p.Memory[addr+2] = byte(val >> 8)
			p.Memory[addr+3] = byte(val)
		}
	}

	// Move to WB stage
	p.WB = p.MEM
}

// Execute stage
func (p *Pipeline) StageEX() {
	if !p.EX.Valid {
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
		result = p.EX.Rs1Value + inst.Imm
	case OpSTORE:
		result = p.EX.Rs1Value + inst.Imm
	}

	p.EX.ALUResult = result

	// Move to MEM stage
	p.MEM = p.EX
}

// Decode stage
func (p *Pipeline) StageID() {
	if !p.ID.Valid {
		return
	}

	inst := p.ID.Instruction

	// Read register values
	p.ID.Rs1Value = p.Registers[inst.Rs1]
	p.ID.Rs2Value = p.Registers[inst.Rs2]

	// Set control signals
	p.ID.MemRead = (inst.OpCode == OpLOAD)
	p.ID.MemWrite = (inst.OpCode == OpSTORE)
	p.ID.RegWrite = (inst.OpCode != OpSTORE && inst.OpCode != OpBEQ &&
		inst.OpCode != OpBNE && inst.OpCode != OpBLT && inst.OpCode != OpBGT &&
		inst.OpCode != OpJMP && inst.OpCode != OpNOP && inst.OpCode != OpHALT)
	p.ID.Branch = (inst.OpCode == OpBEQ || inst.OpCode == OpBNE ||
		inst.OpCode == OpBLT || inst.OpCode == OpBGT || inst.OpCode == OpJMP)

	// Handle data forwarding
	p.ForwardData()

	// Move to EX stage
	p.EX = p.ID
}

// Fetch stage
func (p *Pipeline) StageIF() {
	if p.Halted {
		return
	}

	// Fetch instruction from memory
	word := uint32(p.Memory[p.PC])<<24 |
		uint32(p.Memory[p.PC+1])<<16 |
		uint32(p.Memory[p.PC+2])<<8 |
		uint32(p.Memory[p.PC+3])

	inst := DecodeInstruction(word)

	// Check for HALT
	if inst.OpCode == OpHALT {
		p.Halted = true
		return
	}

	p.IF = PipelineStage{
		Valid:       true,
		Instruction: inst,
		PC:          p.PC,
	}

	// Increment PC
	p.PC += 4

	// Move to ID stage
	p.ID = p.IF
}

// Cycle executes one pipeline cycle
func (p *Pipeline) Cycle() {
	if p.Halted {
		return
	}

	p.CycleCount++

	// Check for hazards
	if p.DetectHazards() {
		// Stall the pipeline
		p.StallCount++
		// Insert bubble in EX stage
		p.EX = PipelineStage{}
		return
	}

	// Execute stages in reverse order (WB -> IF)
	p.StageWB()
	p.StageMEM()
	p.StageEX()
	p.StageID()
	p.StageIF()
}

// Run runs the pipeline until it halts or reaches max cycles
func (p *Pipeline) Run(maxCycles uint64) {
	for !p.Halted && p.CycleCount < maxCycles {
		p.Cycle()
	}
}

// GetCPI returns the cycles per instruction
func (p *Pipeline) GetCPI() float64 {
	if p.CycleCount == 0 {
		return 0
	}
	return float64(p.CycleCount) / float64(p.CycleCount-p.StallCount)
}

// GetStats returns pipeline statistics
func (p *Pipeline) GetStats() PipelineStats {
	return PipelineStats{
		TotalCycles:   p.CycleCount,
		StallCycles:   p.StallCount,
		ForwardCount:  p.ForwardCount,
		BranchCount:   p.BranchCount,
		BranchTaken:   p.BranchTaken,
		FlushCount:    p.FlushCount,
		CPI:           p.GetCPI(),
		PipelineEff:   1.0 - (float64(p.StallCount) / float64(p.CycleCount)),
	}
}

// PipelineStats contains pipeline performance statistics
type PipelineStats struct {
	TotalCycles  uint64  // Total cycles executed
	StallCycles  uint64  // Cycles stalled due to hazards
	ForwardCount uint64  // Number of forwarding operations
	BranchCount  uint64  // Number of branch instructions
	BranchTaken  uint64  // Number of branches taken
	FlushCount   uint64  // Number of pipeline flushes
	CPI          float64 // Cycles per instruction
	PipelineEff  float64 // Pipeline efficiency (1 - stall_rate)
}

// String returns a string representation of the stats
func (s PipelineStats) String() string {
	return fmt.Sprintf(`Pipeline Statistics:
  Total Cycles: %d
  Stall Cycles: %d (%.2f%%)
  Forward Count: %d
  Branch Count: %d
  Branch Taken: %d (%.2f%%)
  Flush Count: %d
  CPI: %.2f
  Pipeline Efficiency: %.2f%%`,
		s.TotalCycles,
		s.StallCycles, 100.0*float64(s.StallCycles)/float64(s.TotalCycles),
		s.ForwardCount,
		s.BranchCount,
		s.BranchTaken, 100.0*float64(s.BranchTaken)/float64(s.BranchCount),
		s.FlushCount,
		s.CPI,
		100.0*s.PipelineEff)
}

// =============================================================================
// Superscalar Pipeline (out-of-order execution)
// =============================================================================

// SuperscalarPipeline represents a superscalar processor with multiple execution units
type SuperscalarPipeline struct {
	IssueWidth     int               // Number of instructions issued per cycle
	ExecutionUnits []ExecutionUnit   // Execution units
	ReorderBuffer  *ReorderBuffer    // Reorder buffer for in-order commit
	ReservationStn *ReservationStation // Reservation station
	CycleCount     uint64
}

// ExecutionUnit represents a functional unit
type ExecutionUnit struct {
	Type     string // ALU, MEM, BRANCH
	Busy     bool
	Latency  int
	Remaining int
}

// ReorderBuffer manages out-of-order completion
type ReorderBuffer struct {
	Entries []ROBEntry
	Size    int
	Head    int
	Tail    int
}

// ROBEntry represents a reorder buffer entry
type ROBEntry struct {
	Valid       bool
	Instruction *Instruction
	Result      int32
	Ready       bool
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
	Op1Ready    bool
	Op2Ready    bool
	Op1Value    int32
	Op2Value    int32
}
