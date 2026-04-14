/*
Specialized Hardware Accelerators
==================================

Implementation of specialized ML/AI hardware accelerators including
TPUs, NPUs, FPGAs, and edge devices.

Features:
- TPU (Tensor Processing Unit) architecture and systolic arrays
- NPU (Neural Processing Unit) characteristics
- FPGA concepts for ML acceleration
- Edge device considerations (mobile, IoT)
- Hardware accelerator comparison

Applications:
- Understanding specialized AI hardware
- Performance comparison across accelerators
- Choosing appropriate hardware for ML workloads
*/

package hardware

import (
	"fmt"
	"time"
)

// =============================================================================
// Accelerator Types
// =============================================================================

// AcceleratorType represents different types of accelerators
type AcceleratorType int

const (
	AcceleratorCPU  AcceleratorType = 0
	AcceleratorGPU  AcceleratorType = 1
	AcceleratorTPU  AcceleratorType = 2
	AcceleratorNPU  AcceleratorType = 3
	AcceleratorFPGA AcceleratorType = 4
	AcceleratorEdge AcceleratorType = 5
)

// =============================================================================
// TPU (Tensor Processing Unit)
// =============================================================================

// TPU represents a Google Tensor Processing Unit
type TPU struct {
	Generation       string
	Name             string
	SystolicArray    *SystolicArray
	MatrixSize       int
	ClockSpeed       float64 // GHz
	PeakTFLOPS       float64 // Tera FLOPS
	MemoryBandwidth  float64 // GB/s
	MemorySize       uint64  // GB
	PowerConsumption float64 // Watts

	// Statistics
	MatrixOps      uint64
	TotalCycles    uint64
	Utilization    float64
}

// NewTPUv4 creates a TPU v4 (current generation)
func NewTPUv4() *TPU {
	return &TPU{
		Generation:       "v4",
		Name:             "Google TPU v4",
		SystolicArray:    NewSystolicArray(128, 128),
		MatrixSize:       128,
		ClockSpeed:       0.9,
		PeakTFLOPS:       275.0,
		MemoryBandwidth:  1200.0,
		MemorySize:       32,
		PowerConsumption: 175.0,
	}
}

// NewTPUv5 creates a TPU v5 (latest)
func NewTPUv5() *TPU {
	return &TPU{
		Generation:       "v5e",
		Name:             "Google TPU v5e",
		SystolicArray:    NewSystolicArray(256, 256),
		MatrixSize:       256,
		ClockSpeed:       1.1,
		PeakTFLOPS:       459.0,
		MemoryBandwidth:  1600.0,
		MemorySize:       16,
		PowerConsumption: 200.0,
	}
}

// MatrixMultiply performs matrix multiplication on TPU
func (t *TPU) MatrixMultiply(m, n, k int) time.Duration {
	// m x k * k x n = m x n
	ops := uint64(2 * m * n * k) // Each multiply-add is 2 ops

	// Calculate cycles needed
	opsPerCycle := float64(t.MatrixSize * t.MatrixSize)
	cycles := uint64(float64(ops) / opsPerCycle)

	t.MatrixOps++
	t.TotalCycles += cycles

	// Calculate time
	cycleTime := 1.0 / (t.ClockSpeed * 1e9)
	duration := time.Duration(float64(cycles) * cycleTime * 1e9)

	return duration
}

// GetStats returns TPU statistics
func (t *TPU) GetStats() TPUStats {
	return TPUStats{
		Name:             t.Name,
		Generation:       t.Generation,
		MatrixOps:        t.MatrixOps,
		TotalCycles:      t.TotalCycles,
		PeakTFLOPS:       t.PeakTFLOPS,
		Utilization:      t.Utilization,
		PowerConsumption: t.PowerConsumption,
	}
}

// TPUStats contains TPU statistics
type TPUStats struct {
	Name             string
	Generation       string
	MatrixOps        uint64
	TotalCycles      uint64
	PeakTFLOPS       float64
	Utilization      float64
	PowerConsumption float64
}

// String returns a string representation
func (s TPUStats) String() string {
	return fmt.Sprintf(`TPU Statistics:
  Name: %s (Generation: %s)
  Matrix Operations: %d
  Total Cycles: %d
  Peak Performance: %.1f TFLOPS
  Utilization: %.2f%%
  Power Consumption: %.1f W`,
		s.Name,
		s.Generation,
		s.MatrixOps,
		s.TotalCycles,
		s.PeakTFLOPS,
		100.0*s.Utilization,
		s.PowerConsumption)
}

// =============================================================================
// Systolic Array (core TPU architecture)
// =============================================================================

// SystolicArray represents a systolic array for matrix operations
type SystolicArray struct {
	Rows    int
	Cols    int
	Cells   [][]*ProcessingElement
	Cycles  uint64
}

// ProcessingElement represents a single PE in systolic array
type ProcessingElement struct {
	Row       int
	Col       int
	Accumulator float32
	Weight    float32
	Input     float32
}

// NewSystolicArray creates a new systolic array
func NewSystolicArray(rows, cols int) *SystolicArray {
	cells := make([][]*ProcessingElement, rows)
	for i := range cells {
		cells[i] = make([]*ProcessingElement, cols)
		for j := range cells[i] {
			cells[i][j] = &ProcessingElement{
				Row: i,
				Col: j,
			}
		}
	}

	return &SystolicArray{
		Rows:  rows,
		Cols:  cols,
		Cells: cells,
	}
}

// Execute executes a matrix multiplication
func (sa *SystolicArray) Execute(matrixA, matrixB [][]float32) [][]float32 {
	m := len(matrixA)
	n := len(matrixB[0])
	k := len(matrixB)

	result := make([][]float32, m)
	for i := range result {
		result[i] = make([]float32, n)
	}

	// Simplified systolic array execution
	for cycle := 0; cycle < m+k+n; cycle++ {
		sa.Cycles++
		// In real systolic array, data flows through PEs
		// This is a simplified simulation
	}

	return result
}

// =============================================================================
// NPU (Neural Processing Unit)
// =============================================================================

// NPU represents a Neural Processing Unit
type NPU struct {
	Name             string
	Manufacturer     string
	TOPs             float64 // Tera Operations per Second
	PowerConsumption float64 // Watts
	ProcessNode      string  // e.g., "5nm"
	MemoryBandwidth  float64 // GB/s
	SupportedOps     []string

	// Statistics
	InferenceCount uint64
	TotalOps       uint64
}

// NewAppleNeuralEngine creates Apple Neural Engine (M-series)
func NewAppleNeuralEngine() *NPU {
	return &NPU{
		Name:             "Apple Neural Engine",
		Manufacturer:     "Apple",
		TOPs:             15.8,
		PowerConsumption: 8.0,
		ProcessNode:      "5nm",
		MemoryBandwidth:  200.0,
		SupportedOps:     []string{"Conv2D", "MatMul", "Pooling", "Activation"},
	}
}

// NewQualcommHexagon creates Qualcomm Hexagon NPU
func NewQualcommHexagon() *NPU {
	return &NPU{
		Name:             "Qualcomm Hexagon",
		Manufacturer:     "Qualcomm",
		TOPs:             26.0,
		PowerConsumption: 6.5,
		ProcessNode:      "4nm",
		MemoryBandwidth:  150.0,
		SupportedOps:     []string{"Conv2D", "MatMul", "RNN", "Transformer"},
	}
}

// RunInference runs inference on the NPU
func (n *NPU) RunInference(ops uint64) time.Duration {
	n.InferenceCount++
	n.TotalOps += ops

	// Calculate time based on TOPs
	opsPerSecond := n.TOPs * 1e12
	seconds := float64(ops) / opsPerSecond

	return time.Duration(seconds * 1e9) // nanoseconds
}

// GetStats returns NPU statistics
func (n *NPU) GetStats() NPUStats {
	return NPUStats{
		Name:             n.Name,
		Manufacturer:     n.Manufacturer,
		InferenceCount:   n.InferenceCount,
		TotalOps:         n.TotalOps,
		TOPs:             n.TOPs,
		PowerConsumption: n.PowerConsumption,
	}
}

// NPUStats contains NPU statistics
type NPUStats struct {
	Name             string
	Manufacturer     string
	InferenceCount   uint64
	TotalOps         uint64
	TOPs             float64
	PowerConsumption float64
}

// String returns a string representation
func (s NPUStats) String() string {
	return fmt.Sprintf(`NPU Statistics:
  Name: %s (%s)
  Inference Count: %d
  Total Operations: %d
  Peak Performance: %.1f TOPS
  Power Consumption: %.1f W
  Efficiency: %.2f TOPS/W`,
		s.Name,
		s.Manufacturer,
		s.InferenceCount,
		s.TotalOps,
		s.TOPs,
		s.PowerConsumption,
		s.TOPs/s.PowerConsumption)
}

// =============================================================================
// FPGA for ML
// =============================================================================

// FPGA represents an FPGA configured for ML
type FPGA struct {
	Name             string
	Manufacturer     string
	LogicCells       int
	DSPBlocks        int
	MemoryBlocks     int
	ClockSpeed       float64
	Reconfigurable   bool
	Latency          time.Duration
	PowerConsumption float64

	// Configuration
	CurrentConfig string
	ConfigTime    time.Duration
}

// NewXilinxVersal creates Xilinx Versal AI Core
func NewXilinxVersal() *FPGA {
	return &FPGA{
		Name:             "Xilinx Versal AI Core",
		Manufacturer:     "AMD Xilinx",
		LogicCells:       1897000,
		DSPBlocks:        1968,
		MemoryBlocks:     967,
		ClockSpeed:       0.74,
		Reconfigurable:   true,
		Latency:          10 * time.Microsecond,
		PowerConsumption: 45.0,
		ConfigTime:       100 * time.Millisecond,
	}
}

// NewIntelStratix creates Intel Stratix 10
func NewIntelStratix() *FPGA {
	return &FPGA{
		Name:             "Intel Stratix 10",
		Manufacturer:     "Intel",
		LogicCells:       5500000,
		DSPBlocks:        5760,
		MemoryBlocks:     1100,
		ClockSpeed:       1.0,
		Reconfigurable:   true,
		Latency:          8 * time.Microsecond,
		PowerConsumption: 60.0,
		ConfigTime:       150 * time.Millisecond,
	}
}

// Configure configures the FPGA for a specific operation
func (f *FPGA) Configure(operation string) time.Duration {
	f.CurrentConfig = operation
	return f.ConfigTime
}

// Execute executes the configured operation
func (f *FPGA) Execute() time.Duration {
	return f.Latency
}

// GetCharacteristics returns FPGA characteristics
func (f *FPGA) GetCharacteristics() string {
	return fmt.Sprintf(`FPGA: %s (%s)
  Logic Cells: %d
  DSP Blocks: %d
  Memory Blocks: %d
  Clock Speed: %.2f GHz
  Reconfigurable: %v
  Latency: %v
  Power: %.1f W
  Config Time: %v`,
		f.Name,
		f.Manufacturer,
		f.LogicCells,
		f.DSPBlocks,
		f.MemoryBlocks,
		f.ClockSpeed,
		f.Reconfigurable,
		f.Latency,
		f.PowerConsumption,
		f.ConfigTime)
}

// =============================================================================
// Edge Devices
// =============================================================================

// EdgeDevice represents an edge computing device
type EdgeDevice struct {
	Name             string
	Type             string // Phone, IoT, Embedded
	Processor        string
	NPU              *NPU
	MemorySize       uint64  // MB
	PowerBudget      float64 // Watts
	BatteryLife      time.Duration
	SupportedModels  []string
}

// NewMobileDevice creates a mobile edge device
func NewMobileDevice() *EdgeDevice {
	return &EdgeDevice{
		Name:            "Modern Smartphone",
		Type:            "Mobile",
		Processor:       "ARM Cortex-A78",
		NPU:             NewAppleNeuralEngine(),
		MemorySize:      8192,
		PowerBudget:     5.0,
		BatteryLife:     10 * time.Hour,
		SupportedModels: []string{"MobileNet", "EfficientNet", "TinyYOLO"},
	}
}

// NewIoTDevice creates an IoT edge device
func NewIoTDevice() *EdgeDevice {
	return &EdgeDevice{
		Name:            "IoT Edge Device",
		Type:            "IoT",
		Processor:       "ARM Cortex-M7",
		MemorySize:      512,
		PowerBudget:     2.0,
		BatteryLife:     24 * time.Hour,
		SupportedModels: []string{"TinyML", "MicroNet"},
	}
}

// RunInference runs inference considering power constraints
func (e *EdgeDevice) RunInference(modelSize uint64, ops uint64) (time.Duration, float64) {
	// Check if model fits in memory
	if modelSize > e.MemorySize*1024*1024 {
		return 0, 0 // Model too large
	}

	var duration time.Duration
	var powerUsed float64

	if e.NPU != nil {
		duration = e.NPU.RunInference(ops)
		powerUsed = e.NPU.PowerConsumption
	} else {
		// CPU fallback (slower, more power)
		duration = time.Duration(float64(ops) / 1e9) // 1 GOPS
		powerUsed = e.PowerBudget * 0.8
	}

	return duration, powerUsed
}

// GetCharacteristics returns edge device characteristics
func (e *EdgeDevice) GetCharacteristics() string {
	return fmt.Sprintf(`Edge Device: %s (%s)
  Processor: %s
  Memory: %d MB
  Power Budget: %.1f W
  Battery Life: %v
  Supported Models: %v`,
		e.Name,
		e.Type,
		e.Processor,
		e.MemorySize,
		e.PowerBudget,
		e.BatteryLife,
		e.SupportedModels)
}

// =============================================================================
// Hardware Comparison
// =============================================================================

// AcceleratorSpec represents hardware accelerator specifications
type AcceleratorSpec struct {
	Type             AcceleratorType
	Name             string
	PeakPerformance  float64 // TFLOPS
	MemoryBandwidth  float64 // GB/s
	PowerConsumption float64 // Watts
	Efficiency       float64 // TFLOPS/W
	Cost             float64 // USD
	BestUseCase      string
}

// CompareAccelerators compares different accelerators
func CompareAccelerators() []AcceleratorSpec {
	return []AcceleratorSpec{
		{
			Type:             AcceleratorCPU,
			Name:             "Intel Xeon Platinum 8380",
			PeakPerformance:  4.0,
			MemoryBandwidth:  204.8,
			PowerConsumption: 270.0,
			Efficiency:       0.015,
			Cost:             8800.0,
			BestUseCase:      "General purpose, diverse workloads",
		},
		{
			Type:             AcceleratorGPU,
			Name:             "NVIDIA A100",
			PeakPerformance:  312.0,
			MemoryBandwidth:  1935.0,
			PowerConsumption: 400.0,
			Efficiency:       0.78,
			Cost:             15000.0,
			BestUseCase:      "Training large models, flexible compute",
		},
		{
			Type:             AcceleratorGPU,
			Name:             "NVIDIA H100",
			PeakPerformance:  989.0,
			MemoryBandwidth:  3350.0,
			PowerConsumption: 700.0,
			Efficiency:       1.41,
			Cost:             30000.0,
			BestUseCase:      "Large-scale training, transformers",
		},
		{
			Type:             AcceleratorTPU,
			Name:             "Google TPU v4",
			PeakPerformance:  275.0,
			MemoryBandwidth:  1200.0,
			PowerConsumption: 175.0,
			Efficiency:       1.57,
			Cost:             5000.0,
			BestUseCase:      "TensorFlow workloads, inference",
		},
		{
			Type:             AcceleratorTPU,
			Name:             "Google TPU v5e",
			PeakPerformance:  459.0,
			MemoryBandwidth:  1600.0,
			PowerConsumption: 200.0,
			Efficiency:       2.30,
			Cost:             7000.0,
			BestUseCase:      "Efficient training and inference",
		},
		{
			Type:             AcceleratorNPU,
			Name:             "Apple M2 Neural Engine",
			PeakPerformance:  15.8,
			MemoryBandwidth:  200.0,
			PowerConsumption: 8.0,
			Efficiency:       1.98,
			Cost:             0.0, // Integrated
			BestUseCase:      "On-device inference, privacy",
		},
		{
			Type:             AcceleratorFPGA,
			Name:             "Xilinx Versal AI Core",
			PeakPerformance:  100.0,
			MemoryBandwidth:  400.0,
			PowerConsumption: 45.0,
			Efficiency:       2.22,
			Cost:             12000.0,
			BestUseCase:      "Low latency, custom operations",
		},
	}
}

// PrintComparison prints a comparison table
func PrintComparison() string {
	specs := CompareAccelerators()
	result := fmt.Sprintf("\n%-20s %-25s %12s %12s %12s %12s %12s\n",
		"Type", "Name", "TFLOPS", "BW (GB/s)", "Power (W)", "Eff (TF/W)", "Cost ($)")
	result += "------------------------------------------------------------------------------------------------------------------------\n"

	typeNames := map[AcceleratorType]string{
		AcceleratorCPU: "CPU", AcceleratorGPU: "GPU",
		AcceleratorTPU: "TPU", AcceleratorNPU: "NPU",
		AcceleratorFPGA: "FPGA",
	}

	for _, spec := range specs {
		result += fmt.Sprintf("%-20s %-25s %12.1f %12.1f %12.1f %12.2f %12.0f\n",
			typeNames[spec.Type],
			spec.Name,
			spec.PeakPerformance,
			spec.MemoryBandwidth,
			spec.PowerConsumption,
			spec.Efficiency,
			spec.Cost)
	}

	return result
}
