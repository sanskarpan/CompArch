package cpu

import (
	"testing"
)

// =============================================================================
// ISA Tests
// =============================================================================

func TestCPUBasicArithmetic(t *testing.T) {
	cpu := NewCPU()

	// Create a program: R3 = R1 + R2
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 5))      // R1 = 5
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 3))      // R2 = 3
	assembler.Add(NewRInstruction(OpADD, 3, 1, 2))       // R3 = R1 + R2
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	cpu.LoadProgram(program, 0)
	cpu.Run(100)

	if cpu.Registers[3] != 8 {
		t.Errorf("Expected R3 = 8, got %d", cpu.Registers[3])
	}
}

func TestCPUSubtraction(t *testing.T) {
	cpu := NewCPU()

	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 10))
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 3))
	assembler.Add(NewRInstruction(OpSUB, 3, 1, 2))       // R3 = 10 - 3
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	cpu.LoadProgram(program, 0)
	cpu.Run(100)

	if cpu.Registers[3] != 7 {
		t.Errorf("Expected R3 = 7, got %d", cpu.Registers[3])
	}
}

func TestCPUMultiplication(t *testing.T) {
	cpu := NewCPU()

	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 6))
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 7))
	assembler.Add(NewRInstruction(OpMUL, 3, 1, 2))       // R3 = 6 * 7
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	cpu.LoadProgram(program, 0)
	cpu.Run(100)

	if cpu.Registers[3] != 42 {
		t.Errorf("Expected R3 = 42, got %d", cpu.Registers[3])
	}
}

func TestInstructionEncoding(t *testing.T) {
	inst := NewRInstruction(OpADD, 3, 1, 2)
	encoded := inst.Encode()
	decoded := DecodeInstruction(encoded)

	if decoded.OpCode != OpADD {
		t.Errorf("Expected OpADD, got %v", decoded.OpCode)
	}
	if decoded.Rd != 3 || decoded.Rs1 != 1 || decoded.Rs2 != 2 {
		t.Errorf("Register fields not preserved")
	}
}

// =============================================================================
// Pipeline Tests
// =============================================================================

func TestPipelineExecution(t *testing.T) {
	pipeline := NewPipeline()

	assembler := NewAssembler()
	for i := 0; i < 10; i++ {
		assembler.Add(NewIInstruction(OpMOVI, byte(i), 0, int32(i*10)))
	}
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(100)

	stats := pipeline.GetStats()
	if stats.TotalCycles == 0 {
		t.Error("Pipeline did not execute any cycles")
	}
	if stats.CPI < 0 || stats.CPI > 10 {
		t.Errorf("Unexpected CPI: %.2f", stats.CPI)
	}
}

func TestPipelineDataForwarding(t *testing.T) {
	pipeline := NewPipeline()

	// Create data dependency to test forwarding
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 5))
	assembler.Add(NewRInstruction(OpADD, 2, 1, 1))       // Uses R1 immediately
	assembler.Add(NewRInstruction(OpADD, 3, 2, 2))       // Uses R2 immediately
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(100)

	stats := pipeline.GetStats()
	if stats.ForwardCount == 0 {
		t.Error("Expected data forwarding to occur")
	}
}

// =============================================================================
// Cache Tests
// =============================================================================

func TestCacheHit(t *testing.T) {
	cache := NewCache(CacheConfig{
		Size:          32 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	// First access - miss
	_, hit := cache.Read(0, 64)
	if hit {
		t.Error("Expected cache miss on first access")
	}

	// Second access to same line - hit
	_, hit = cache.Read(0, 64)
	if !hit {
		t.Error("Expected cache hit on second access")
	}
}

func TestCacheWriteBack(t *testing.T) {
	cache := NewCache(CacheConfig{
		Size:          32 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	data := []byte{1, 2, 3, 4}
	cache.Write(0, data)

	stats := cache.GetStats()
	if stats.Writes == 0 {
		t.Error("Expected write to be recorded")
	}
}

func TestMemoryHierarchy(t *testing.T) {
	hierarchy := NewMemoryHierarchy()

	// Sequential access should have good hit rate
	for i := uint64(0); i < 100; i++ {
		hierarchy.Read(i*64, 64)
	}

	stats := hierarchy.GetStats()
	l1Stats := stats["L1"]

	if l1Stats.Accesses == 0 {
		t.Error("L1 should have been accessed")
	}
}

// =============================================================================
// Cache Coherence Tests
// =============================================================================

func TestCoherenceWrite(t *testing.T) {
	system := NewCoherentSystem(2, 64)

	data := []byte{1, 2, 3, 4}
	system.Write(0, 0, data)

	readData := system.Read(0, 0)
	if readData[0] != 1 || readData[1] != 2 {
		t.Error("Data not correctly written/read")
	}
}

func TestCoherenceInvalidation(t *testing.T) {
	system := NewCoherentSystem(2, 64)

	// Core 0 writes
	data := []byte{1, 2, 3, 4}
	system.Write(0, 0, data)

	// Core 1 reads (should see data via coherence)
	readData := system.Read(1, 0)
	if readData[0] != 1 {
		t.Error("Coherence not working correctly")
	}

	// Core 1 writes (should invalidate Core 0's copy)
	newData := []byte{5, 6, 7, 8}
	system.Write(1, 0, newData)

	stats := system.GetAllStats()
	core0Stats := stats["Core0"].(CoherenceStats)
	if core0Stats.Invalidations == 0 {
		t.Error("Expected invalidation to occur")
	}
}

// =============================================================================
// SIMD Tests
// =============================================================================

func TestSIMDAddition(t *testing.T) {
	a := Vec4f{1, 2, 3, 4}
	b := Vec4f{5, 6, 7, 8}
	c := AddVec4f(a, b)

	expected := Vec4f{6, 8, 10, 12}
	for i := 0; i < 4; i++ {
		if c[i] != expected[i] {
			t.Errorf("Expected %v, got %v", expected, c)
			break
		}
	}
}

func TestSIMDMultiplication(t *testing.T) {
	a := Vec4f{2, 3, 4, 5}
	b := Vec4f{10, 10, 10, 10}
	c := MulVec4f(a, b)

	expected := Vec4f{20, 30, 40, 50}
	for i := 0; i < 4; i++ {
		if c[i] != expected[i] {
			t.Errorf("Expected %v, got %v", expected, c)
			break
		}
	}
}

func TestSIMDDotProduct(t *testing.T) {
	a := Vec4f{1, 2, 3, 4}
	b := Vec4f{5, 6, 7, 8}
	dot := DotVec4f(a, b)

	// 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
	if dot != 70.0 {
		t.Errorf("Expected 70, got %f", dot)
	}
}

func TestSIMDHorizontalAdd(t *testing.T) {
	a := Vec4f{1, 2, 3, 4}
	sum := HorizontalAddVec4f(a)

	if sum != 10.0 {
		t.Errorf("Expected 10, got %f", sum)
	}
}

// =============================================================================
// Multi-Core Tests
// =============================================================================

func TestMultiCoreCreation(t *testing.T) {
	processor := NewMultiCoreProcessor(4, 3.0)

	if processor.NumCores != 4 {
		t.Errorf("Expected 4 cores, got %d", processor.NumCores)
	}
	if len(processor.Cores) != 4 {
		t.Errorf("Expected 4 core instances, got %d", len(processor.Cores))
	}
}

func TestMultiCoreThreadScheduling(t *testing.T) {
	processor := NewMultiCoreProcessor(2, 3.0)

	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 10))
	assembler.Add(NewJInstruction(OpHALT, 0))

	thread := &Thread{
		ID:      0,
		Program: assembler.Assemble(),
	}

	scheduled := processor.ScheduleThread(thread)
	if !scheduled {
		t.Error("Thread should have been scheduled")
	}
}

func TestMultiCoreExecution(t *testing.T) {
	processor := NewMultiCoreProcessor(2, 3.0)

	for i := 0; i < 4; i++ {
		assembler := NewAssembler()
		for j := 0; j < 10; j++ {
			assembler.Add(NewIInstruction(OpMOVI, 1, 0, int32(j)))
		}
		assembler.Add(NewJInstruction(OpHALT, 0))

		thread := &Thread{
			ID:      i,
			Program: assembler.Assemble(),
		}
		processor.ScheduleThread(thread)
	}

	processor.RunCycles(100)

	stats := processor.GetSystemStats()
	if stats.TotalCycles == 0 {
		t.Error("Processor did not execute any cycles")
	}
}

// =============================================================================
// Pipeline CPI and Instruction Count Tests
// =============================================================================

func TestPipelineInstructionCount(t *testing.T) {
	pipeline := NewPipeline()

	assembler := NewAssembler()
	// 5 MOVI instructions + HALT
	for i := 0; i < 5; i++ {
		assembler.Add(NewIInstruction(OpMOVI, byte(i+1), 0, int32(i*10)))
	}
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(200)

	stats := pipeline.GetStats()
	if stats.InstructionCount == 0 {
		t.Error("InstructionCount should be > 0 after execution")
	}
	// CPI should be positive and reasonable
	if stats.CPI <= 0 {
		t.Errorf("CPI should be > 0, got %.2f", stats.CPI)
	}
	if stats.CPI > 10 {
		t.Errorf("CPI unreasonably high: %.2f", stats.CPI)
	}
}

func TestPipelineCPIFormula(t *testing.T) {
	pipeline := NewPipeline()

	// Ideal case: no deps, no branches — CPI should be close to 1
	assembler := NewAssembler()
	for i := 0; i < 20; i++ {
		assembler.Add(NewIInstruction(OpMOVI, byte(i%32), 0, int32(i)))
	}
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(500)

	stats := pipeline.GetStats()
	// CPI must equal TotalCycles / InstructionCount
	if stats.InstructionCount > 0 {
		expectedCPI := float64(stats.TotalCycles) / float64(stats.InstructionCount)
		if stats.CPI != expectedCPI {
			t.Errorf("CPI formula: got %.4f, want %.4f", stats.CPI, expectedCPI)
		}
	}
}

func TestPipelineLoadUseStall(t *testing.T) {
	pipeline := NewPipeline()

	// Store a value first so LOAD has something to read
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 42))  // R1 = 42
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 0))   // R2 = 0 (base addr for STORE)
	assembler.Add(NewRInstruction(OpSTORE, 1, 2, 0))  // mem[R2+0] = R1
	assembler.Add(NewIInstruction(OpLOAD, 3, 2, 0))   // R3 = mem[R2+0]  ← load
	assembler.Add(NewRInstruction(OpADD, 4, 3, 1))    // R4 = R3 + R1  ← uses R3 immediately
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(200)

	stats := pipeline.GetStats()
	if stats.StallCycles == 0 {
		t.Error("load-use hazard should cause at least one stall cycle")
	}
}

func TestPipelineBranchFlush(t *testing.T) {
	pipeline := NewPipeline()

	assembler := NewAssembler()
	// Program layout (each instruction is 4 bytes):
	//   PC=0x00  MOVI R1=5
	//   PC=0x04  MOVI R2=5
	//   PC=0x08  BEQ R0,R0 → 0x10   (R0==R0 always-taken; uses Rs1=0,Rs2=0 to
	//                                  avoid corrupting the target address in the
	//                                  J-type encoding where Addr overlaps Rs fields)
	//   PC=0x0C  MOVI R10=99        ← wrong-path, must be flushed
	//   PC=0x10  MOVI R3=7          ← branch target
	//   PC=0x14  HALT
	beq := &Instruction{OpCode: OpBEQ, Type: TypeJ, Rs1: 0, Rs2: 0, Addr: 0x10}
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 5))   // PC=0x00
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 5))   // PC=0x04
	assembler.Add(beq)                                 // PC=0x08
	assembler.Add(NewIInstruction(OpMOVI, 10, 0, 99)) // PC=0x0C (skipped)
	assembler.Add(NewIInstruction(OpMOVI, 3, 0, 7))   // PC=0x10 (target)
	assembler.Add(NewJInstruction(OpHALT, 0))          // PC=0x14

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(200)

	stats := pipeline.GetStats()
	if stats.BranchCount == 0 {
		t.Error("pipeline should have counted the BEQ")
	}
	if stats.FlushCount == 0 {
		t.Error("taken branch should flush wrong-path instruction")
	}
	// R10 must be 0 (initial value): the wrong-path MOVI was flushed.
	// Note: Imm=99 would encode as -29 in the 7-bit ISA, so we check for 0
	// (not 99) to confirm the instruction never executed.
	if pipeline.Registers[10] != 0 {
		t.Errorf("wrong-path MOVI R10 should have been flushed, R10=%d want 0",
			pipeline.Registers[10])
	}
	// R3 must be 7: the branch target instruction must execute
	if pipeline.Registers[3] != 7 {
		t.Errorf("branch target should execute: R3 = %d, want 7", pipeline.Registers[3])
	}
}

func TestPipelineJMPBranch(t *testing.T) {
	pipeline := NewPipeline()

	// PC=0x00  MOVI R1=1
	// PC=0x04  JMP → 0x0C           (skip the instruction at 0x08)
	// PC=0x08  MOVI R2=55           ← must be skipped (R2 stays 0)
	// PC=0x0C  MOVI R3=42           ← branch target (R3 should be 42)
	// PC=0x10  HALT
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 1))   // PC=0x00
	jmp := &Instruction{OpCode: OpJMP, Type: TypeJ, Addr: 0x0C}
	assembler.Add(jmp)                                 // PC=0x04
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 55))  // PC=0x08 (skipped; 55 ≤ 63 ✓)
	assembler.Add(NewIInstruction(OpMOVI, 3, 0, 42))  // PC=0x0C (target)
	assembler.Add(NewJInstruction(OpHALT, 0))          // PC=0x10

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(100)

	// R2 must still be 0: the JMP flushed MOVI R2=55 before it could write back
	if pipeline.Registers[2] != 0 {
		t.Errorf("JMP should flush MOVI R2: R2 = %d, want 0", pipeline.Registers[2])
	}
	if pipeline.Registers[3] != 42 {
		t.Errorf("JMP target should execute: R3 = %d, want 42", pipeline.Registers[3])
	}
}

func TestPipelineDataForwardingMEMtoEX(t *testing.T) {
	pipeline := NewPipeline()

	// Chain of dependent ALU ops — requires forwarding
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 10))  // R1 = 10
	assembler.Add(NewRInstruction(OpADD, 2, 1, 1))    // R2 = R1 + R1 = 20 (uses R1 via forwarding)
	assembler.Add(NewRInstruction(OpMUL, 3, 2, 2))    // R3 = R2 * R2 = 400 (uses R2 via forwarding)
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(100)

	if pipeline.Registers[1] != 10 {
		t.Errorf("R1: got %d, want 10", pipeline.Registers[1])
	}
	if pipeline.Registers[2] != 20 {
		t.Errorf("R2: got %d, want 20", pipeline.Registers[2])
	}
	if pipeline.Registers[3] != 400 {
		t.Errorf("R3: got %d, want 400", pipeline.Registers[3])
	}

	stats := pipeline.GetStats()
	if stats.ForwardCount == 0 {
		t.Error("dependent ALU chain should require data forwarding")
	}
}

func TestPipelineStoreLoad(t *testing.T) {
	pipeline := NewPipeline()

	// Use Imm=42 (fits in 7-bit positive range ≤ 63; 77 would be sign-extended to -51)
	const storeVal = int32(42)

	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, storeVal)) // R1 = 42
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 0))        // R2 = 0 (base address)
	// STORE: mem[R2+0] = R1
	storeInst := &Instruction{OpCode: OpSTORE, Type: TypeI, Rd: 1, Rs1: 2, Imm: 0}
	assembler.Add(storeInst)
	// Gap instruction prevents immediate load-after-store forwarding issues
	assembler.Add(NewIInstruction(OpMOVI, 5, 0, 0))
	// LOAD: R3 = mem[R2+0]
	loadInst := &Instruction{OpCode: OpLOAD, Type: TypeI, Rd: 3, Rs1: 2, Imm: 0}
	assembler.Add(loadInst)
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(200)

	// R3 must equal what was stored
	if pipeline.Registers[3] != storeVal {
		t.Errorf("store/load round-trip: R3 = %d, want %d", pipeline.Registers[3], storeVal)
	}
}

func TestPipelineReset(t *testing.T) {
	pipeline := NewPipeline()
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 99))
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(50)

	pipeline.Reset()

	if pipeline.CycleCount != 0 {
		t.Error("CycleCount should be 0 after Reset")
	}
	if pipeline.InstructionCount != 0 {
		t.Error("InstructionCount should be 0 after Reset")
	}
	if pipeline.Registers[1] != 0 {
		t.Error("registers should be 0 after Reset")
	}
}

func TestPipelineStatsPipelineEff(t *testing.T) {
	pipeline := NewPipeline()
	// Zero-stall run
	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 1))
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(50)

	stats := pipeline.GetStats()
	if stats.PipelineEff < 0 || stats.PipelineEff > 1 {
		t.Errorf("PipelineEff out of range [0,1]: %.2f", stats.PipelineEff)
	}
	_ = stats.String() // Ensure String() doesn't panic
}

// =============================================================================
// Superscalar Pipeline Tests
// =============================================================================

func TestSuperscalarCreation(t *testing.T) {
	sp := NewSuperscalarPipeline(4, 32, 16)

	if sp.IssueWidth != 4 {
		t.Errorf("IssueWidth: got %d, want 4", sp.IssueWidth)
	}
	if sp.ReorderBuffer.Size != 32 {
		t.Errorf("ROB size: got %d, want 32", sp.ReorderBuffer.Size)
	}
	if sp.ReservationStn.Size != 16 {
		t.Errorf("RS size: got %d, want 16", sp.ReservationStn.Size)
	}
	// RAT should be initialised to -1 (no pending producer)
	for i, v := range sp.RAT {
		if v != -1 {
			t.Errorf("RAT[%d] should be -1, got %d", i, v)
		}
	}
}

func TestSuperscalarSimpleExecution(t *testing.T) {
	sp := NewSuperscalarPipeline(2, 16, 8)

	assembler := NewAssembler()
	assembler.Add(NewIInstruction(OpMOVI, 1, 0, 10))
	assembler.Add(NewIInstruction(OpMOVI, 2, 0, 20))
	assembler.Add(NewRInstruction(OpADD, 3, 1, 2))  // R3 = 30
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	sp.LoadProgram(program, 0)

	for i := 0; i < 50; i++ {
		sp.Cycle()
	}

	if sp.CycleCount == 0 {
		t.Error("superscalar should have advanced cycle count")
	}
}

func TestSuperscalarROBFull(t *testing.T) {
	// Tiny ROB to exercise full-ROB stall
	sp := NewSuperscalarPipeline(4, 4, 8)

	assembler := NewAssembler()
	for i := 0; i < 16; i++ {
		assembler.Add(NewIInstruction(OpMOVI, byte(i%32), 0, int32(i)))
	}
	assembler.Add(NewJInstruction(OpHALT, 0))

	program := assembler.Assemble()
	sp.LoadProgram(program, 0)

	// Should not panic or deadlock
	for i := 0; i < 100; i++ {
		sp.Cycle()
	}
}

func TestSuperscalarHalted(t *testing.T) {
	sp := NewSuperscalarPipeline(2, 16, 8)
	sp.Halted = true

	before := sp.CycleCount
	sp.Cycle()
	if sp.CycleCount != before {
		t.Error("halted superscalar should not advance cycle count")
	}
}

// =============================================================================
// Cache Config Validation
// =============================================================================

func TestCacheConfigValidation(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewCache with zero LineSize should panic")
		}
	}()
	NewCache(CacheConfig{
		Size:          4096,
		LineSize:      0, // invalid
		Associativity: 4,
		WritePolicy:   "write-back",
	})
}

func TestCacheDefaultPolicy(t *testing.T) {
	// Should not panic even without explicit Policy
	cache := NewCache(CacheConfig{
		Size:          4096,
		LineSize:      64,
		Associativity: 4,
		WritePolicy:   "write-back",
		Policy:        nil, // should default to LRU
	})
	_, _ = cache.Read(0, 64) // should not panic
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkCPUExecution(b *testing.B) {
	cpu := NewCPU()

	assembler := NewAssembler()
	for i := 0; i < 100; i++ {
		assembler.Add(NewIInstruction(OpMOVI, byte(i%32), 0, int32(i)))
	}
	assembler.Add(NewJInstruction(OpHALT, 0))
	program := assembler.Assemble()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpu.Reset()
		cpu.LoadProgram(program, 0)
		cpu.Run(1000)
	}
}

func BenchmarkCacheAccess(b *testing.B) {
	cache := NewCache(CacheConfig{
		Size:          32 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Read(uint64(i%1000)*64, 64)
	}
}

func BenchmarkSIMDAdd(b *testing.B) {
	a := Vec4f{1, 2, 3, 4}
	c := Vec4f{5, 6, 7, 8}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = AddVec4f(a, c)
	}
}
