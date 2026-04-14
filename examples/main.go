package main

import (
	"fmt"
	"strings"

	"github.com/sanskarpan/CompArch/cpu"
	"github.com/sanskarpan/CompArch/gpu"
	"github.com/sanskarpan/CompArch/hardware"
	"github.com/sanskarpan/CompArch/memory"
)

func main() {
	fmt.Println("=============================================================================")
	fmt.Println("Computer Architecture & Hardware Examples")
	fmt.Println("=============================================================================")
	fmt.Println()

	// Run all examples
	runCPUExamples()
	runGPUExamples()
	runMemoryExamples()
	runHardwareComparison()
}

func runCPUExamples() {
	fmt.Println()
	fmt.Println("--- CPU Architecture Examples ---")
	fmt.Println()

	// 1. ISA and Basic CPU
	fmt.Println("1. Instruction Set Architecture (ISA)")
	fmt.Println("--------------------------------------")
	demonstrateISA()

	// 2. Pipeline
	fmt.Println("\n2. CPU Pipeline (5-stage)")
	fmt.Println("--------------------------------------")
	demonstratePipeline()

	// 3. Cache Hierarchy
	fmt.Println("\n3. Cache Hierarchy (L1/L2/L3)")
	fmt.Println("--------------------------------------")
	demonstrateCache()

	// 4. Cache Coherence
	fmt.Println("\n4. Cache Coherence (MESI Protocol)")
	fmt.Println("--------------------------------------")
	demonstrateCoherence()

	// 5. SIMD Operations
	fmt.Println("\n5. SIMD Operations")
	fmt.Println("--------------------------------------")
	demonstrateSIMD()

	// 6. Multi-Core Processor
	fmt.Println("\n6. Multi-Core Processor")
	fmt.Println("--------------------------------------")
	demonstrateMultiCore()
}

func demonstrateISA() {
	// Create CPU
	processor := cpu.NewCPU()

	// Create a simple program: calculate 5 + 3
	assembler := cpu.NewAssembler()

	// MOVI R1, 5
	assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, 1, 0, 5))
	// MOVI R2, 3
	assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, 2, 0, 3))
	// ADD R3, R1, R2
	assembler.Add(cpu.NewRInstruction(cpu.OpADD, 3, 1, 2))
	// HALT
	assembler.Add(cpu.NewJInstruction(cpu.OpHALT, 0))

	program := assembler.Assemble()

	// Load and run
	processor.LoadProgram(program, 0)
	processor.Run(100)

	fmt.Printf("Result: R3 = %d (expected: 8)\n", processor.Registers[3])
	fmt.Printf("Cycles executed: %d\n", processor.CycleCount)
}

func demonstratePipeline() {
	pipeline := cpu.NewPipeline()

	// Create a simple program
	assembler := cpu.NewAssembler()
	for i := 0; i < 10; i++ {
		assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, byte(i), 0, int32(i*10)))
		assembler.Add(cpu.NewRInstruction(cpu.OpADD, byte(i+1), byte(i), byte(i)))
	}
	assembler.Add(cpu.NewJInstruction(cpu.OpHALT, 0))

	program := assembler.Assemble()
	pipeline.LoadProgram(program, 0)
	pipeline.Run(1000)

	stats := pipeline.GetStats()
	fmt.Println(stats.String())
}

func demonstrateCache() {
	// Create a 3-level cache hierarchy
	hierarchy := cpu.NewMemoryHierarchy()

	// Sequential access pattern (cache-friendly)
	fmt.Println("Sequential access pattern:")
	for i := uint64(0); i < 1000; i++ {
		hierarchy.Read(i*64, 64) // Read cache lines sequentially
	}

	stats := hierarchy.GetStats()
	fmt.Printf("L1 Hit Rate: %.2f%%\n", 100.0*stats["L1"].HitRate)
	fmt.Printf("L2 Hit Rate: %.2f%%\n", 100.0*stats["L2"].HitRate)
	fmt.Printf("L3 Hit Rate: %.2f%%\n", 100.0*stats["L3"].HitRate)

	// Reset and try random access
	hierarchy.L1.Reset()
	hierarchy.L2.Reset()
	hierarchy.L3.Reset()

	fmt.Println("\nRandom access pattern:")
	for i := uint64(0); i < 1000; i++ {
		addr := (i * 4096) % (1024 * 1024) // Random-ish access
		hierarchy.Read(addr, 64)
	}

	stats = hierarchy.GetStats()
	fmt.Printf("L1 Hit Rate: %.2f%%\n", 100.0*stats["L1"].HitRate)
	fmt.Printf("L2 Hit Rate: %.2f%%\n", 100.0*stats["L2"].HitRate)
	fmt.Printf("L3 Hit Rate: %.2f%%\n", 100.0*stats["L3"].HitRate)
}

func demonstrateCoherence() {
	// Create a 4-core system with cache coherence
	system := cpu.NewCoherentSystem(4, 64)

	// Core 0 writes to address 0
	data := []byte{1, 2, 3, 4}
	system.Write(0, 0, data)

	// Core 1 reads from address 0 (should get data from Core 0)
	readData := system.Read(1, 0)
	fmt.Printf("Core 1 read: %v\n", readData[:4])

	// Core 2 writes to address 0 (should invalidate others)
	newData := []byte{5, 6, 7, 8}
	system.Write(2, 0, newData)

	// Core 3 reads (should get latest data from Core 2)
	readData = system.Read(3, 0)
	fmt.Printf("Core 3 read: %v\n", readData[:4])

	// Get statistics
	allStats := system.GetAllStats()
	for name, stat := range allStats {
		if coherenceStat, ok := stat.(cpu.CoherenceStats); ok {
			fmt.Printf("%s: %d invalidations, %d downgrades\n",
				name, coherenceStat.Invalidations, coherenceStat.Downgrades)
		}
	}
}

func demonstrateSIMD() {
	// Vector operations
	a := cpu.Vec4f{1.0, 2.0, 3.0, 4.0}
	b := cpu.Vec4f{5.0, 6.0, 7.0, 8.0}

	// Vector addition
	c := cpu.AddVec4f(a, b)
	fmt.Printf("Vec4f Add: %v + %v = %v\n", a, b, c)

	// Vector multiplication
	d := cpu.MulVec4f(a, b)
	fmt.Printf("Vec4f Mul: %v * %v = %v\n", a, b, d)

	// Dot product
	dot := cpu.DotVec4f(a, b)
	fmt.Printf("Dot Product: %.2f\n", dot)

	// Benchmark SIMD vs scalar
	result := cpu.BenchmarkSIMD(1000000)
	fmt.Println("\n" + result.String())
}

func demonstrateMultiCore() {
	// Create a 4-core processor
	processor := cpu.NewMultiCoreProcessor(4, 3.0)

	// Create some threads
	for i := 0; i < 8; i++ {
		// Simple program for each thread
		assembler := cpu.NewAssembler()
		for j := 0; j < 100; j++ {
			assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, 1, 0, int32(i*j)))
		}
		assembler.Add(cpu.NewJInstruction(cpu.OpHALT, 0))

		thread := &cpu.Thread{
			ID:      i,
			Program: assembler.Assemble(),
		}
		processor.ScheduleThread(thread)
	}

	// Run for 10000 cycles
	processor.RunCycles(10000)

	// Get statistics
	stats := processor.GetSystemStats()
	fmt.Println(stats.String())
}

func runGPUExamples() {
	fmt.Println()
	fmt.Println()
	fmt.Println("--- GPU Architecture Examples ---")
	fmt.Println()

	// 1. GPU Basics
	fmt.Println("1. GPU Kernel Execution")
	fmt.Println("--------------------------------------")
	demonstrateGPU()

	// 2. GPU Comparison
	fmt.Println("\n2. GPU Models Comparison")
	fmt.Println("--------------------------------------")
	compareGPUs()
}

func demonstrateGPU() {
	// Create an NVIDIA A100 GPU
	gpuDevice := gpu.NewNVIDIA_A100()

	fmt.Printf("GPU: %s\n", gpuDevice.Name)
	fmt.Printf("Streaming Multiprocessors: %d\n", gpuDevice.NumSMs)
	fmt.Printf("Total CUDA Cores: %d\n", gpuDevice.TotalCores)
	fmt.Printf("Memory: %.0f GB\n", float64(gpuDevice.GlobalMemory.Size)/(1024*1024*1024))

	// Launch a kernel
	gridSize := 256   // Number of blocks
	blockSize := 1024 // Threads per block

	fmt.Printf("\nLaunching kernel: Grid(%d) x Block(%d) = %d threads\n",
		gridSize, blockSize, gridSize*blockSize)

	kernel := gpuDevice.LaunchKernel(gridSize, blockSize)
	fmt.Printf("Kernel execution time: %v\n", kernel.Duration)
	fmt.Printf("Total warps: %d\n", kernel.NumWarps)

	// Get statistics
	stats := gpuDevice.GetStats()
	fmt.Println("\n" + stats.String())
}

func compareGPUs() {
	models := []struct {
		name string
		gpu  *gpu.GPU
	}{
		{"V100", gpu.NewNVIDIA_V100()},
		{"A100", gpu.NewNVIDIA_A100()},
		{"H100", gpu.NewNVIDIA_H100()},
	}

	fmt.Printf("%-10s %8s %12s %15s\n", "Model", "SMs", "Cores", "Memory (GB)")
	fmt.Println("--------------------------------------------------")

	for _, m := range models {
		fmt.Printf("%-10s %8d %12d %15.0f\n",
			m.name,
			m.gpu.NumSMs,
			m.gpu.TotalCores,
			float64(m.gpu.GlobalMemory.Size)/(1024*1024*1024))
	}
}

func runMemoryExamples() {
	fmt.Println()
	fmt.Println()
	fmt.Println("--- Memory Systems Examples ---")
	fmt.Println()

	// 1. RAM Types
	fmt.Println("1. RAM Types Comparison")
	fmt.Println("--------------------------------------")
	compareRAMTypes()

	// 2. Memory Access Patterns
	fmt.Println("\n2. Memory Access Patterns")
	fmt.Println("--------------------------------------")
	demonstrateAccessPatterns()

	// 3. Data Locality
	fmt.Println("\n3. Data Locality Analysis")
	fmt.Println("--------------------------------------")
	demonstrateLocality()

	// 4. Roofline Model
	fmt.Println("\n4. Roofline Model (Compute vs Memory Bound)")
	fmt.Println("--------------------------------------")
	demonstrateRoofline()
}

func compareRAMTypes() {
	types := []memory.RAMType{
		memory.DDR4,
		memory.DDR5,
		memory.GDDR6,
		memory.GDDR6X,
		memory.HBM2,
		memory.HBM2E,
		memory.HBM3,
	}

	fmt.Printf("%-10s %12s %12s %12s\n", "Type", "BW (GB/s)", "Latency (ns)", "Power (W)")
	fmt.Println("-------------------------------------------------------")

	for _, ramType := range types {
		chars := memory.GetRAMCharacteristics(ramType)
		typeNames := map[memory.RAMType]string{
			memory.DDR4: "DDR4", memory.DDR5: "DDR5",
			memory.GDDR6: "GDDR6", memory.GDDR6X: "GDDR6X",
			memory.HBM2: "HBM2", memory.HBM2E: "HBM2E", memory.HBM3: "HBM3",
		}
		fmt.Printf("%-10s %12.1f %12d %12.1f\n",
			typeNames[ramType],
			chars.Bandwidth,
			chars.Latency,
			chars.PowerConsumption)
	}
}

func demonstrateAccessPatterns() {
	mem := memory.NewMemorySystem(memory.DDR4, 8)

	patterns := []memory.AccessPattern{
		memory.SequentialAccess,
		memory.RandomAccess,
		memory.StridedAccess,
		memory.BlockAccess,
	}

	for _, pattern := range patterns {
		mem.Reads = 0
		mem.Writes = 0
		mem.BytesRead = 0
		mem.BytesWritten = 0

		benchmark := &memory.MemoryBenchmark{
			Memory:      mem,
			Pattern:     pattern,
			AccessSize:  64,
			NumAccesses: 10000,
		}

		result := benchmark.RunBenchmark()
		fmt.Println(result.String())
		fmt.Println()
	}
}

func demonstrateLocality() {
	analyzer := memory.NewLocalityAnalyzer(64)

	// Sequential access (good locality)
	fmt.Println("Sequential access:")
	for i := uint64(0); i < 1000; i++ {
		analyzer.RecordAccess(i * 8)
	}
	metrics := analyzer.AnalyzeLocality()
	fmt.Println(metrics.String())

	// Random access (poor locality)
	analyzer = memory.NewLocalityAnalyzer(64)
	fmt.Println("\nRandom access:")
	for i := uint64(0); i < 1000; i++ {
		analyzer.RecordAccess((i * 4096) % 100000)
	}
	metrics = analyzer.AnalyzeLocality()
	fmt.Println(metrics.String())
}

func demonstrateRoofline() {
	// Example: NVIDIA A100
	model := memory.NewRooflineModel(
		312e12,  // 312 TFLOPS (FP64 tensor cores)
		1935.0,  // 1935 GB/s memory bandwidth
	)

	workloads := []struct {
		name          string
		flops         float64
		bytesAccessed float64
	}{
		{"Matrix Multiplication (Large)", 1e12, 1e9},       // Compute-bound
		{"Vector Addition", 1e9, 1e9},                      // Memory-bound
		{"Convolution (Optimized)", 5e11, 5e8},           // Balanced
		{"Sparse Matrix-Vector", 1e10, 1e10},              // Memory-bound
	}

	for _, w := range workloads {
		analysis := model.AnalyzeWorkload(w.flops, w.bytesAccessed)
		fmt.Printf("\nWorkload: %s\n", w.name)
		fmt.Println(analysis.String())
	}
}

func runHardwareComparison() {
	fmt.Println()
	fmt.Println()
	fmt.Println("--- Specialized Hardware Examples ---")
	fmt.Println()

	// 1. TPU
	fmt.Println("1. TPU (Tensor Processing Unit)")
	fmt.Println("--------------------------------------")
	demonstrateTPU()

	// 2. NPU
	fmt.Println("\n2. NPU (Neural Processing Unit)")
	fmt.Println("--------------------------------------")
	demonstrateNPU()

	// 3. FPGA
	fmt.Println("\n3. FPGA for ML")
	fmt.Println("--------------------------------------")
	demonstrateFPGA()

	// 4. Edge Devices
	fmt.Println("\n4. Edge Devices")
	fmt.Println("--------------------------------------")
	demonstrateEdge()

	// 5. Hardware Comparison
	fmt.Println("\n5. Hardware Accelerator Comparison")
	fmt.Println("--------------------------------------")
	fmt.Print(hardware.PrintComparison())
}

func demonstrateTPU() {
	tpu := hardware.NewTPUv5()

	fmt.Printf("TPU: %s (Generation %s)\n", tpu.Name, tpu.Generation)
	fmt.Printf("Matrix Size: %dx%d\n", tpu.MatrixSize, tpu.MatrixSize)
	fmt.Printf("Peak Performance: %.1f TFLOPS\n", tpu.PeakTFLOPS)
	fmt.Printf("Memory Bandwidth: %.1f GB/s\n", tpu.MemoryBandwidth)

	// Simulate matrix multiplication
	fmt.Println("\nExecuting 1024x1024 matrix multiplication:")
	duration := tpu.MatrixMultiply(1024, 1024, 1024)
	fmt.Printf("Execution time: %v\n", duration)

	stats := tpu.GetStats()
	fmt.Println("\n" + stats.String())
}

func demonstrateNPU() {
	npu := hardware.NewAppleNeuralEngine()

	fmt.Printf("NPU: %s\n", npu.Name)
	fmt.Printf("Manufacturer: %s\n", npu.Manufacturer)
	fmt.Printf("Performance: %.1f TOPS\n", npu.TOPs)
	fmt.Printf("Power: %.1f W\n", npu.PowerConsumption)
	fmt.Printf("Efficiency: %.2f TOPS/W\n", npu.TOPs/npu.PowerConsumption)

	// Run inference
	ops := uint64(1e10) // 10 billion operations
	duration := npu.RunInference(ops)
	fmt.Printf("\nInference (10B ops) time: %v\n", duration)

	stats := npu.GetStats()
	fmt.Println("\n" + stats.String())
}

func demonstrateFPGA() {
	fpga := hardware.NewXilinxVersal()

	fmt.Println(fpga.GetCharacteristics())

	// Configure for matrix multiplication
	configTime := fpga.Configure("MatrixMultiply")
	fmt.Printf("\nConfiguration time: %v\n", configTime)

	// Execute
	execTime := fpga.Execute()
	fmt.Printf("Execution time: %v\n", execTime)
	fmt.Printf("Total time (config + exec): %v\n", configTime+execTime)
}

func demonstrateEdge() {
	mobile := hardware.NewMobileDevice()
	iot := hardware.NewIoTDevice()

	fmt.Println("Mobile Device:")
	fmt.Println(mobile.GetCharacteristics())

	modelSize := uint64(50 * 1024 * 1024) // 50 MB model
	ops := uint64(1e9)                     // 1 billion ops

	duration, power := mobile.RunInference(modelSize, ops)
	fmt.Printf("\nInference time: %v\n", duration)
	fmt.Printf("Power used: %.2f W\n", power)

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("\nIoT Device:")
	fmt.Println(iot.GetCharacteristics())
}
