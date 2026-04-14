## Computer Architecture & Hardware in Go

Comprehensive computer architecture implementation covering CPU, GPU, memory systems, and specialized hardware accelerators for ML/AI workloads.

## Table of Contents

1. [CPU Architecture](#cpu-architecture)
2. [GPU Architecture](#gpu-architecture)
3. [Memory Systems](#memory-systems)
4. [Specialized Hardware](#specialized-hardware)
5. [Examples](#examples)
6. [Performance Analysis](#performance-analysis)

## CPU Architecture

### Instruction Set Architecture (ISA)

A simple RISC-style ISA with register-based operations.

**Features:**
- 32 general-purpose registers
- Arithmetic operations (ADD, SUB, MUL, DIV)
- Logical operations (AND, OR, XOR, SHL, SHR)
- Memory operations (LOAD, STORE, MOVI)
- Control flow (JMP, BEQ, BNE, BLT, BGT, CALL, RET)
- Instruction encoding/decoding

**Example:**
```go
// Create CPU
processor := cpu.NewCPU()

// Create a simple program
assembler := cpu.NewAssembler()
assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, 1, 0, 5))  // R1 = 5
assembler.Add(cpu.NewIInstruction(cpu.OpMOVI, 2, 0, 3))  // R2 = 3
assembler.Add(cpu.NewRInstruction(cpu.OpADD, 3, 1, 2))   // R3 = R1 + R2
assembler.Add(cpu.NewJInstruction(cpu.OpHALT, 0))

program := assembler.Assemble()
processor.LoadProgram(program, 0)
processor.Run(100)

fmt.Printf("Result: R3 = %d\n", processor.Registers[3])  // Output: 8
```

### CPU Pipeline

5-stage pipeline with hazard detection and data forwarding.

**Stages:**
1. **IF (Instruction Fetch)** - Fetch instruction from memory
2. **ID (Instruction Decode)** - Decode and read registers
3. **EX (Execute)** - Perform ALU operations
4. **MEM (Memory)** - Access memory for load/store
5. **WB (Writeback)** - Write results to registers

**Features:**
- Data hazard detection
- Pipeline stalls
- Data forwarding
- CPI (Cycles Per Instruction) calculation
- Pipeline efficiency metrics

**Example:**
```go
pipeline := cpu.NewPipeline()
pipeline.LoadProgram(program, 0)
pipeline.Run(1000)

stats := pipeline.GetStats()
fmt.Printf("CPI: %.2f\n", stats.CPI)
fmt.Printf("Pipeline Efficiency: %.2f%%\n", 100.0*stats.PipelineEff)
```

### Cache Hierarchy

Multi-level cache (L1/L2/L3) with various replacement policies.

**Cache Levels:**
- **L1 Cache:** 32KB, 8-way, write-back, LRU
- **L2 Cache:** 256KB, 8-way, write-back, LRU
- **L3 Cache:** 8MB, 16-way, write-back, LRU (shared)

**Replacement Policies:**
- LRU (Least Recently Used)
- FIFO (First-In-First-Out)
- Random
- LFU (Least Frequently Used)

**Example:**
```go
hierarchy := cpu.NewMemoryHierarchy()

// Sequential access (cache-friendly)
for i := uint64(0); i < 1000; i++ {
    hierarchy.Read(i*64, 64)
}

stats := hierarchy.GetStats()
fmt.Printf("L1 Hit Rate: %.2f%%\n", 100.0*stats["L1"].HitRate)
fmt.Printf("L2 Hit Rate: %.2f%%\n", 100.0*stats["L2"].HitRate)
```

### Cache Coherence (MESI Protocol)

Multi-core cache coherence using MESI protocol.

**MESI States:**
- **Modified (M):** Cache line is dirty and exclusive
- **Exclusive (E):** Cache line is clean and exclusive
- **Shared (S):** Cache line may be in other caches
- **Invalid (I):** Cache line is not valid

**Example:**
```go
system := cpu.NewCoherentSystem(4, 64)  // 4 cores

// Core 0 writes
system.Write(0, addr, data)

// Core 1 reads (gets data via coherence)
readData := system.Read(1, addr)

// Core 2 writes (invalidates other copies)
system.Write(2, addr, newData)

stats := system.GetAllStats()
```

### SIMD Operations

Single Instruction Multiple Data for parallel processing.

**Vector Types:**
- Vec4f (4x float32)
- Vec8f (8x float32)
- Vec4i (4x int32)
- Vec8i (8x int32)

**Operations:**
- Vector arithmetic (add, sub, mul, div)
- Dot product
- Horizontal operations (sum, max, min)
- Comparison operations

**Example:**
```go
a := cpu.Vec4f{1.0, 2.0, 3.0, 4.0}
b := cpu.Vec4f{5.0, 6.0, 7.0, 8.0}

c := cpu.AddVec4f(a, b)          // Vector addition
d := cpu.MulVec4f(a, b)          // Element-wise multiply
dot := cpu.DotVec4f(a, b)        // Dot product
sum := cpu.HorizontalAddVec4f(c) // Sum all elements

// Benchmark SIMD vs scalar
result := cpu.BenchmarkSIMD(1000000)
fmt.Printf("Speedup: %.2fx\n", result.Speedup)
```

### Multi-Core Processor

Complete multi-core system with scheduling and statistics.

**Features:**
- Multiple cores with independent pipelines
- Private L1/L2 caches per core
- Shared L3 cache
- Cache coherence (MESI)
- Thread scheduling
- Performance monitoring

**Example:**
```go
processor := cpu.NewMultiCoreProcessor(4, 3.0)  // 4 cores @ 3GHz

// Create and schedule threads
for i := 0; i < 8; i++ {
    thread := &cpu.Thread{
        ID:      i,
        Program: program,
    }
    processor.ScheduleThread(thread)
}

processor.RunCycles(10000)
stats := processor.GetSystemStats()

fmt.Printf("Overall IPC: %.2f\n", stats.IPC)
fmt.Printf("Utilization: %.2f%%\n", 100.0*stats.OverallUtilization)
```

## GPU Architecture

### GPU Components

**NVIDIA GPU Simulator:**
- Streaming Multiprocessors (SMs)
- CUDA cores
- Warp scheduler
- Memory hierarchy

**Example Configurations:**
```go
v100 := gpu.NewNVIDIA_V100()  // 80 SMs, 5120 cores
a100 := gpu.NewNVIDIA_A100()  // 108 SMs, 6912 cores
h100 := gpu.NewNVIDIA_H100()  // 132 SMs, 16896 cores
```

### GPU Memory Hierarchy

**Memory Types:**
- **Global Memory:** Large (80GB), high latency (400 cycles)
- **Shared Memory:** Small (64KB/SM), low latency (28 cycles)
- **Constant Memory:** Read-only, cached (64KB)
- **Register File:** Fastest, limited (65K registers/SM)
- **L1 Cache:** 128KB per SM
- **L2 Cache:** Shared across GPU

### Warp Scheduling

**Warp Characteristics:**
- 32 threads per warp
- SIMT (Single Instruction Multiple Threads) execution
- Warp divergence handling
- Occupancy calculation

**Example:**
```go
gpuDevice := gpu.NewNVIDIA_A100()

// Launch kernel
gridSize := 256      // Number of blocks
blockSize := 1024    // Threads per block
kernel := gpuDevice.LaunchKernel(gridSize, blockSize)

fmt.Printf("Total threads: %d\n", kernel.NumThreads)
fmt.Printf("Total warps: %d\n", kernel.NumWarps)
fmt.Printf("Execution time: %v\n", kernel.Duration)
```

## Memory Systems

### RAM Types

Comprehensive RAM characteristics and comparison.

**Types Supported:**
- **DDR4:** 25.6 GB/s, 13ns latency
- **DDR5:** 51.2 GB/s, 14ns latency
- **GDDR6:** 448 GB/s, 10ns latency
- **GDDR6X:** 760 GB/s, 9ns latency (PAM4 signaling)
- **HBM2:** 307 GB/s, 6ns latency
- **HBM2E:** 460 GB/s, 5ns latency
- **HBM3:** 819 GB/s, 4ns latency

**Example:**
```go
// Create memory system
mem := memory.NewMemorySystem(memory.HBM3, 80)  // 80GB HBM3

// Get characteristics
chars := memory.GetRAMCharacteristics(memory.HBM3)
fmt.Printf("Bandwidth: %.1f GB/s\n", chars.Bandwidth)
fmt.Printf("Latency: %d ns\n", chars.Latency)
```

### Memory Access Patterns

Benchmark different access patterns.

**Patterns:**
- **Sequential Access:** Cache-friendly, high bandwidth
- **Random Access:** Cache-unfriendly, low bandwidth
- **Strided Access:** Depends on stride size
- **Block Access:** Good for burst transfers

**Example:**
```go
benchmark := &memory.MemoryBenchmark{
    Memory:      mem,
    Pattern:     memory.SequentialAccess,
    AccessSize:  64,
    NumAccesses: 10000,
}

result := benchmark.RunBenchmark()
fmt.Printf("Bandwidth: %.2f GB/s\n", result.Bandwidth)
fmt.Printf("Bandwidth Utilization: %.2f%%\n", 100.0*result.BandwidthUtil)
```

### Data Locality Analysis

Analyze temporal and spatial locality of memory accesses.

**Example:**
```go
analyzer := memory.NewLocalityAnalyzer(64)

// Record accesses
for i := uint64(0); i < 1000; i++ {
    analyzer.RecordAccess(i * 8)  // Sequential
}

metrics := analyzer.AnalyzeLocality()
fmt.Printf("Temporal Locality: %.2f%%\n", 100.0*metrics.TemporalLocality)
fmt.Printf("Spatial Locality: %.2f%%\n", 100.0*metrics.SpatialLocality)
```

### Roofline Model

Analyze compute-bound vs memory-bound workloads.

**Example:**
```go
// NVIDIA A100 roofline
model := memory.NewRooflineModel(
    312e12,  // 312 TFLOPS (FP64 tensor)
    1935.0,  // 1935 GB/s bandwidth
)

// Analyze workload
analysis := model.AnalyzeWorkload(
    1e12,   // 1 TFLOP operations
    1e9,    // 1 GB data accessed
)

fmt.Printf("Arithmetic Intensity: %.2f FLOPS/byte\n", analysis.ArithmeticIntensity)
fmt.Printf("Workload Type: %s\n", analysis.WorkloadType)  // Compute/Memory bound
fmt.Printf("Efficiency: %.2f%%\n", 100.0*analysis.Efficiency)
```

## Specialized Hardware

### TPU (Tensor Processing Unit)

Google's TPU architecture with systolic arrays.

**Features:**
- Systolic array for matrix operations
- High throughput for matrix multiplication
- Specialized for TensorFlow workloads
- Energy efficient

**Example:**
```go
tpu := hardware.NewTPUv5()  // Latest TPU

fmt.Printf("Peak Performance: %.1f TFLOPS\n", tpu.PeakTFLOPS)
fmt.Printf("Memory Bandwidth: %.1f GB/s\n", tpu.MemoryBandwidth)

// Execute matrix multiplication
duration := tpu.MatrixMultiply(1024, 1024, 1024)
fmt.Printf("Execution time: %v\n", duration)

stats := tpu.GetStats()
```

### NPU (Neural Processing Unit)

On-device AI accelerators for edge inference.

**Supported NPUs:**
- Apple Neural Engine (M-series)
- Qualcomm Hexagon
- MediaTek APU

**Example:**
```go
npu := hardware.NewAppleNeuralEngine()

fmt.Printf("Performance: %.1f TOPS\n", npu.TOPs)
fmt.Printf("Power: %.1f W\n", npu.PowerConsumption)
fmt.Printf("Efficiency: %.2f TOPS/W\n", npu.TOPs/npu.PowerConsumption)

// Run inference
ops := uint64(1e10)  // 10 billion operations
duration := npu.RunInference(ops)
```

### FPGA for ML

Reconfigurable hardware for custom ML operations.

**Features:**
- Reconfigurable logic
- Low latency
- Custom operations
- Energy efficient

**Example:**
```go
fpga := hardware.NewXilinxVersal()

// Configure for specific operation
configTime := fpga.Configure("MatrixMultiply")
execTime := fpga.Execute()

fmt.Printf("Total time: %v\n", configTime+execTime)
```

### Edge Devices

Mobile and IoT devices for on-device AI.

**Example:**
```go
mobile := hardware.NewMobileDevice()

modelSize := uint64(50 * 1024 * 1024)  // 50 MB
ops := uint64(1e9)                      // 1B operations

duration, power := mobile.RunInference(modelSize, ops)
fmt.Printf("Inference time: %v\n", duration)
fmt.Printf("Power used: %.2f W\n", power)
```

### Hardware Comparison

Compare different accelerators.

```go
// Print comparison table
fmt.Print(hardware.PrintComparison())
```

**Output:**
```
Type                 Name                      TFLOPS      BW (GB/s)     Power (W)  Eff (TF/W)    Cost ($)
------------------------------------------------------------------------------------------------------------------------
CPU                  Intel Xeon Platinum 8380     4.0         204.8         270.0        0.02        8800
GPU                  NVIDIA A100                312.0        1935.0         400.0        0.78       15000
GPU                  NVIDIA H100                989.0        3350.0         700.0        1.41       30000
TPU                  Google TPU v4              275.0        1200.0         175.0        1.57        5000
TPU                  Google TPU v5e             459.0        1600.0         200.0        2.30        7000
NPU                  Apple M2 Neural Engine      15.8         200.0           8.0        1.98           0
FPGA                 Xilinx Versal AI Core      100.0         400.0          45.0        2.22       12000
```

## Examples

Run all examples:

```bash
cd examples
go run main.go
```

## Performance Analysis

### CPU Performance Metrics

- **CPI (Cycles Per Instruction)**
- **Pipeline Efficiency**
- **Cache Hit Rates**
- **Branch Prediction Accuracy**
- **IPC (Instructions Per Cycle)**

### GPU Performance Metrics

- **SM Occupancy**
- **Warp Efficiency**
- **Memory Bandwidth Utilization**
- **Kernel Execution Time**
- **FLOPS (Floating Point Operations Per Second)**

### Memory Metrics

- **Bandwidth Utilization**
- **Average Latency**
- **Temporal/Spatial Locality**
- **Cache Hit/Miss Rates**
- **Arithmetic Intensity**

## Building and Testing

Build all modules:
```bash
go build ./...
```

Run tests (when available):
```bash
go test ./... -v
```

Run examples:
```bash
cd examples
go run main.go
```

## Architecture Overview

```
CompArch/
├── cpu/                    # CPU Architecture
│   ├── isa.go             # Instruction Set Architecture
│   ├── pipeline.go        # 5-stage pipeline
│   ├── cache.go           # Cache hierarchy (L1/L2/L3)
│   ├── coherence.go       # MESI cache coherence
│   ├── simd.go            # SIMD operations
│   └── multicore.go       # Multi-core processor
│
├── gpu/                    # GPU Architecture
│   └── gpu.go             # GPU, SM, warps, memory
│
├── memory/                 # Memory Systems
│   └── memory.go          # RAM types, patterns, roofline
│
├── hardware/               # Specialized Hardware
│   └── accelerators.go    # TPU, NPU, FPGA, Edge
│
└── examples/               # Examples
    └── main.go            # Comprehensive examples
```

## Use Cases

### For ML/AI Optimization

- Understand memory bottlenecks in training
- Optimize data access patterns
- Choose appropriate hardware for workloads
- Analyze compute vs memory bound operations

### For Computer Architecture Education

- Learn CPU pipeline behavior
- Understand cache hierarchies
- Study cache coherence protocols
- Explore GPU architecture

### For Performance Analysis

- Benchmark different memory types
- Compare hardware accelerators
- Analyze workload characteristics
- Optimize for specific hardware

## Best Practices

1. **Memory Access Optimization:**
   - Use sequential access when possible
   - Maximize cache line utilization
   - Consider data layout for SIMD

2. **Cache Optimization:**
   - Minimize cache misses
   - Use blocking for large arrays
   - Consider cache line size (64 bytes)

3. **GPU Optimization:**
   - Maximize warp occupancy
   - Minimize warp divergence
   - Use shared memory for frequently accessed data
   - Coalesce global memory accesses

4. **Hardware Selection:**
   - GPUs for massively parallel workloads
   - TPUs for TensorFlow/matrix operations
   - NPUs for on-device inference
   - FPGAs for low-latency custom operations

## References

- Computer Architecture: A Quantitative Approach (Hennessy & Patterson)
- CUDA Programming Guide (NVIDIA)
- TPU Architecture (Google)
- Intel Architecture Optimization Manual
- ARM Architecture Reference Manual

## License

Part of Phase0_Core research project.
