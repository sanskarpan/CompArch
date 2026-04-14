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
