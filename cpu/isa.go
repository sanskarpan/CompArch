/*
Instruction Set Architecture (ISA)
==================================

A simple RISC-style instruction set architecture simulator.

Features:
- Register-based architecture (32 registers)
- Memory operations (load/store)
- Arithmetic and logical operations
- Control flow (branches, jumps)
- Instruction encoding and decoding

Applications:
- Understanding CPU instruction execution
- Pipeline simulation foundation
- Performance analysis
*/

package cpu

import (
	"fmt"
)

// =============================================================================
// Instruction Types
// =============================================================================

// OpCode represents an instruction operation code
type OpCode byte

const (
	// Arithmetic operations
	OpADD OpCode = 0x01 // Add
	OpSUB OpCode = 0x02 // Subtract
	OpMUL OpCode = 0x03 // Multiply
	OpDIV OpCode = 0x04 // Divide
	OpAND OpCode = 0x05 // Bitwise AND
	OpOR  OpCode = 0x06 // Bitwise OR
	OpXOR OpCode = 0x07 // Bitwise XOR
	OpSHL OpCode = 0x08 // Shift left
	OpSHR OpCode = 0x09 // Shift right

	// Memory operations
	OpLOAD  OpCode = 0x10 // Load from memory
	OpSTORE OpCode = 0x11 // Store to memory
	OpMOVI  OpCode = 0x12 // Move immediate

	// Control flow
	OpJMP  OpCode = 0x20 // Unconditional jump
	OpBEQ  OpCode = 0x21 // Branch if equal
	OpBNE  OpCode = 0x22 // Branch if not equal
	OpBLT  OpCode = 0x23 // Branch if less than
	OpBGT  OpCode = 0x24 // Branch if greater than
	OpCALL OpCode = 0x25 // Function call
	OpRET  OpCode = 0x26 // Return

	// Special operations
	OpNOP  OpCode = 0x00 // No operation
	OpHALT OpCode = 0xFF // Halt execution
)

// InstructionType represents the format of an instruction
type InstructionType byte

const (
	TypeR InstructionType = 0 // Register type (3 registers)
	TypeI InstructionType = 1 // Immediate type (2 registers + immediate)
	TypeJ InstructionType = 2 // Jump type (address)
)

// Instruction represents a single CPU instruction
type Instruction struct {
	OpCode OpCode          // Operation code
	Type   InstructionType // Instruction format
	Rd     byte            // Destination register
	Rs1    byte            // Source register 1
	Rs2    byte            // Source register 2
	Imm    int32           // Immediate value
	Addr   uint32          // Address (for jumps)
}

// Encode encodes an instruction into a 32-bit word
func (inst *Instruction) Encode() uint32 {
	var word uint32
	word |= uint32(inst.OpCode) << 24
	word |= uint32(inst.Type) << 22
	word |= uint32(inst.Rd) << 17
	word |= uint32(inst.Rs1) << 12
	word |= uint32(inst.Rs2) << 7

	// Immediate or address in lower bits
	if inst.Type == TypeI {
		word |= uint32(inst.Imm) & 0x7F
	} else if inst.Type == TypeJ {
		word |= inst.Addr & 0x3FFFFF
	}

	return word
}

// DecodeInstruction decodes a 32-bit word into an instruction
func DecodeInstruction(word uint32) *Instruction {
	inst := &Instruction{
		OpCode: OpCode((word >> 24) & 0xFF),
		Type:   InstructionType((word >> 22) & 0x03),
		Rd:     byte((word >> 17) & 0x1F),
		Rs1:    byte((word >> 12) & 0x1F),
		Rs2:    byte((word >> 7) & 0x1F),
	}

	if inst.Type == TypeI {
		inst.Imm = int32(word & 0x7F)
		// Sign extend
		if inst.Imm&0x40 != 0 {
			inst.Imm |= ^0x7F
		}
	} else if inst.Type == TypeJ {
		inst.Addr = word & 0x3FFFFF
	}

	return inst
}

// String returns a human-readable representation of the instruction
func (inst *Instruction) String() string {
	opName := map[OpCode]string{
		OpADD: "ADD", OpSUB: "SUB", OpMUL: "MUL", OpDIV: "DIV",
		OpAND: "AND", OpOR: "OR", OpXOR: "XOR", OpSHL: "SHL", OpSHR: "SHR",
		OpLOAD: "LOAD", OpSTORE: "STORE", OpMOVI: "MOVI",
		OpJMP: "JMP", OpBEQ: "BEQ", OpBNE: "BNE", OpBLT: "BLT", OpBGT: "BGT",
		OpCALL: "CALL", OpRET: "RET", OpNOP: "NOP", OpHALT: "HALT",
	}[inst.OpCode]

	switch inst.Type {
	case TypeR:
		return fmt.Sprintf("%s R%d, R%d, R%d", opName, inst.Rd, inst.Rs1, inst.Rs2)
	case TypeI:
		return fmt.Sprintf("%s R%d, R%d, %d", opName, inst.Rd, inst.Rs1, inst.Imm)
	case TypeJ:
		return fmt.Sprintf("%s 0x%X", opName, inst.Addr)
	default:
		return "UNKNOWN"
	}
}

// =============================================================================
// CPU with Registers and Memory
// =============================================================================

const (
	NumRegisters = 32
	MemorySize   = 65536 // 64KB
)

// CPU represents a simple CPU
type CPU struct {
	Registers [NumRegisters]int32 // General-purpose registers
	PC        uint32              // Program counter
	Memory    []byte              // Main memory
	Halted    bool                // Halt flag
	CycleCount uint64             // Number of cycles executed
}

// NewCPU creates a new CPU
func NewCPU() *CPU {
	return &CPU{
		Memory: make([]byte, MemorySize),
	}
}

// Reset resets the CPU state
func (c *CPU) Reset() {
	for i := range c.Registers {
		c.Registers[i] = 0
	}
	c.PC = 0
	c.Halted = false
	c.CycleCount = 0
}

// LoadProgram loads a program into memory
func (c *CPU) LoadProgram(program []uint32, startAddr uint32) {
	for i, word := range program {
		addr := startAddr + uint32(i*4)
		c.Memory[addr] = byte(word >> 24)
		c.Memory[addr+1] = byte(word >> 16)
		c.Memory[addr+2] = byte(word >> 8)
		c.Memory[addr+3] = byte(word)
	}
	c.PC = startAddr
}

// FetchInstruction fetches the instruction at the current PC
func (c *CPU) FetchInstruction() uint32 {
	word := uint32(c.Memory[c.PC])<<24 |
		uint32(c.Memory[c.PC+1])<<16 |
		uint32(c.Memory[c.PC+2])<<8 |
		uint32(c.Memory[c.PC+3])

	return word
}

// Execute executes a single instruction
func (c *CPU) Execute(inst *Instruction) {
	c.CycleCount++

	switch inst.OpCode {
	// Arithmetic operations
	case OpADD:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] + c.Registers[inst.Rs2]
	case OpSUB:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] - c.Registers[inst.Rs2]
	case OpMUL:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] * c.Registers[inst.Rs2]
	case OpDIV:
		if c.Registers[inst.Rs2] != 0 {
			c.Registers[inst.Rd] = c.Registers[inst.Rs1] / c.Registers[inst.Rs2]
		}
	case OpAND:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] & c.Registers[inst.Rs2]
	case OpOR:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] | c.Registers[inst.Rs2]
	case OpXOR:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] ^ c.Registers[inst.Rs2]
	case OpSHL:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] << uint(c.Registers[inst.Rs2])
	case OpSHR:
		c.Registers[inst.Rd] = c.Registers[inst.Rs1] >> uint(c.Registers[inst.Rs2])

	// Immediate operations
	case OpMOVI:
		c.Registers[inst.Rd] = inst.Imm

	// Memory operations
	case OpLOAD:
		addr := uint32(c.Registers[inst.Rs1] + inst.Imm)
		if addr+3 < uint32(len(c.Memory)) {
			c.Registers[inst.Rd] = int32(c.Memory[addr])<<24 |
				int32(c.Memory[addr+1])<<16 |
				int32(c.Memory[addr+2])<<8 |
				int32(c.Memory[addr+3])
		}
	case OpSTORE:
		addr := uint32(c.Registers[inst.Rs1] + inst.Imm)
		if addr+3 < uint32(len(c.Memory)) {
			val := c.Registers[inst.Rd]
			c.Memory[addr] = byte(val >> 24)
			c.Memory[addr+1] = byte(val >> 16)
			c.Memory[addr+2] = byte(val >> 8)
			c.Memory[addr+3] = byte(val)
		}

	// Control flow
	case OpJMP:
		c.PC = inst.Addr
		return // Don't increment PC
	case OpBEQ:
		if c.Registers[inst.Rs1] == c.Registers[inst.Rs2] {
			c.PC = inst.Addr
			return
		}
	case OpBNE:
		if c.Registers[inst.Rs1] != c.Registers[inst.Rs2] {
			c.PC = inst.Addr
			return
		}
	case OpBLT:
		if c.Registers[inst.Rs1] < c.Registers[inst.Rs2] {
			c.PC = inst.Addr
			return
		}
	case OpBGT:
		if c.Registers[inst.Rs1] > c.Registers[inst.Rs2] {
			c.PC = inst.Addr
			return
		}
	case OpCALL:
		// Save return address in R31 (link register)
		c.Registers[31] = int32(c.PC + 4)
		c.PC = inst.Addr
		return
	case OpRET:
		// Return to address in R31
		c.PC = uint32(c.Registers[31])
		return

	// Special
	case OpHALT:
		c.Halted = true
	case OpNOP:
		// No operation
	}

	// Increment PC
	c.PC += 4
}

// Step executes a single instruction cycle
func (c *CPU) Step() {
	if c.Halted {
		return
	}

	word := c.FetchInstruction()
	inst := DecodeInstruction(word)
	c.Execute(inst)
}

// Run runs the CPU until it halts or reaches max cycles
func (c *CPU) Run(maxCycles uint64) {
	for !c.Halted && c.CycleCount < maxCycles {
		c.Step()
	}
}

// =============================================================================
// Instruction Builder Helpers
// =============================================================================

// NewRInstruction creates a register-type instruction
func NewRInstruction(op OpCode, rd, rs1, rs2 byte) *Instruction {
	return &Instruction{
		OpCode: op,
		Type:   TypeR,
		Rd:     rd,
		Rs1:    rs1,
		Rs2:    rs2,
	}
}

// NewIInstruction creates an immediate-type instruction
func NewIInstruction(op OpCode, rd, rs1 byte, imm int32) *Instruction {
	return &Instruction{
		OpCode: op,
		Type:   TypeI,
		Rd:     rd,
		Rs1:    rs1,
		Imm:    imm,
	}
}

// NewJInstruction creates a jump-type instruction
func NewJInstruction(op OpCode, addr uint32) *Instruction {
	return &Instruction{
		OpCode: op,
		Type:   TypeJ,
		Addr:   addr,
	}
}

// =============================================================================
// Assembler - Convert assembly to machine code
// =============================================================================

// Assembler converts assembly instructions to machine code
type Assembler struct {
	instructions []*Instruction
}

// NewAssembler creates a new assembler
func NewAssembler() *Assembler {
	return &Assembler{}
}

// Add adds an instruction to the program
func (a *Assembler) Add(inst *Instruction) {
	a.instructions = append(a.instructions, inst)
}

// Assemble assembles the program into machine code
func (a *Assembler) Assemble() []uint32 {
	program := make([]uint32, len(a.instructions))
	for i, inst := range a.instructions {
		program[i] = inst.Encode()
	}
	return program
}

// GetInstructions returns the list of instructions
func (a *Assembler) GetInstructions() []*Instruction {
	return a.instructions
}
