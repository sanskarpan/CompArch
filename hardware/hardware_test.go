package hardware

import (
	"testing"
)

func TestTPUCreation(t *testing.T) {
	tpu := NewTPUv4()

	if tpu.Name != "Google TPU v4" {
		t.Errorf("Expected Google TPU v4, got %s", tpu.Name)
	}
	if tpu.MatrixSize != 128 {
		t.Errorf("Expected matrix size 128, got %d", tpu.MatrixSize)
	}
	if tpu.PeakTFLOPS <= 0 {
		t.Error("Peak TFLOPS should be positive")
	}
}

func TestTPUVersions(t *testing.T) {
	v4 := NewTPUv4()
	v5 := NewTPUv5()

	// v5 should be faster than v4
	if v5.PeakTFLOPS <= v4.PeakTFLOPS {
		t.Error("TPU v5 should have higher peak performance than v4")
	}
}

func TestTPUMatrixMultiply(t *testing.T) {
	tpu := NewTPUv4()

	// Small matrix multiplication
	duration := tpu.MatrixMultiply(256, 256, 256)

	if duration <= 0 {
		t.Error("Duration should be positive")
	}
	if tpu.MatrixOps != 1 {
		t.Errorf("Expected 1 matrix operation, got %d", tpu.MatrixOps)
	}
}

func TestSystolicArrayCreation(t *testing.T) {
	sa := NewSystolicArray(128, 128)

	if sa.Rows != 128 || sa.Cols != 128 {
		t.Errorf("Expected 128x128 array, got %dx%d", sa.Rows, sa.Cols)
	}
	if len(sa.Cells) != 128 {
		t.Error("Systolic array cells not properly initialized")
	}
}

func TestNPUCreation(t *testing.T) {
	npu := NewAppleNeuralEngine()

	if npu.Name != "Apple Neural Engine" {
		t.Errorf("Expected Apple Neural Engine, got %s", npu.Name)
	}
	if npu.TOPs <= 0 {
		t.Error("TOPs should be positive")
	}
}

func TestNPUInference(t *testing.T) {
	npu := NewAppleNeuralEngine()

	ops := uint64(1e9) // 1 billion operations
	duration := npu.RunInference(ops)

	if duration <= 0 {
		t.Error("Duration should be positive")
	}
	if npu.InferenceCount != 1 {
		t.Errorf("Expected 1 inference, got %d", npu.InferenceCount)
	}
	if npu.TotalOps != ops {
		t.Errorf("Expected %d ops, got %d", ops, npu.TotalOps)
	}
}

func TestNPUEfficiency(t *testing.T) {
	apple := NewAppleNeuralEngine()
	qualcomm := NewQualcommHexagon()

	appleEff := apple.TOPs / apple.PowerConsumption
	qualcommEff := qualcomm.TOPs / qualcomm.PowerConsumption

	if appleEff <= 0 || qualcommEff <= 0 {
		t.Error("Efficiency should be positive")
	}
}

func TestFPGACreation(t *testing.T) {
	fpga := NewXilinxVersal()

	if fpga.Name != "Xilinx Versal AI Core" {
		t.Errorf("Expected Xilinx Versal AI Core, got %s", fpga.Name)
	}
	if fpga.LogicCells <= 0 {
		t.Error("Logic cells should be positive")
	}
	if !fpga.Reconfigurable {
		t.Error("FPGA should be reconfigurable")
	}
}

func TestFPGAConfiguration(t *testing.T) {
	fpga := NewXilinxVersal()

	configTime := fpga.Configure("MatrixMultiply")

	if configTime <= 0 {
		t.Error("Configuration time should be positive")
	}
	if fpga.CurrentConfig != "MatrixMultiply" {
		t.Errorf("Expected current config 'MatrixMultiply', got '%s'", fpga.CurrentConfig)
	}
}

func TestFPGAExecution(t *testing.T) {
	fpga := NewXilinxVersal()

	fpga.Configure("MatrixMultiply")
	execTime := fpga.Execute()

	if execTime <= 0 {
		t.Error("Execution time should be positive")
	}
	if execTime != fpga.Latency {
		t.Error("Execution time should match configured latency")
	}
}

func TestEdgeDeviceCreation(t *testing.T) {
	mobile := NewMobileDevice()

	if mobile.Type != "Mobile" {
		t.Errorf("Expected Mobile type, got %s", mobile.Type)
	}
	if mobile.MemorySize <= 0 {
		t.Error("Memory size should be positive")
	}
}

func TestEdgeDeviceInference(t *testing.T) {
	mobile := NewMobileDevice()

	modelSize := uint64(50 * 1024 * 1024) // 50 MB
	ops := uint64(1e9)                     // 1 billion ops

	duration, power := mobile.RunInference(modelSize, ops)

	if duration <= 0 {
		t.Error("Duration should be positive")
	}
	if power <= 0 {
		t.Error("Power should be positive")
	}
}

func TestEdgeDeviceMemoryConstraint(t *testing.T) {
	iot := NewIoTDevice()

	// Model too large for IoT device
	largeModel := iot.MemorySize * 1024 * 1024 * 2 // 2x available memory
	ops := uint64(1e9)

	duration, power := iot.RunInference(largeModel, ops)

	if duration != 0 || power != 0 {
		t.Error("Should reject model that's too large")
	}
}

func TestAcceleratorComparison(t *testing.T) {
	specs := CompareAccelerators()

	if len(specs) == 0 {
		t.Error("Should return accelerator specs")
	}

	for _, spec := range specs {
		if spec.PeakPerformance <= 0 {
			t.Errorf("%s: Peak performance should be positive", spec.Name)
		}
		if spec.MemoryBandwidth <= 0 {
			t.Errorf("%s: Memory bandwidth should be positive", spec.Name)
		}
		if spec.PowerConsumption <= 0 {
			t.Errorf("%s: Power consumption should be positive", spec.Name)
		}
		if spec.Efficiency <= 0 {
			t.Errorf("%s: Efficiency should be positive", spec.Name)
		}
	}
}

func TestAcceleratorEfficiency(t *testing.T) {
	specs := CompareAccelerators()

	// Find TPU and GPU specs
	var tpuSpec, gpuSpec AcceleratorSpec
	for _, spec := range specs {
		if spec.Type == AcceleratorTPU && spec.Name == "Google TPU v5e" {
			tpuSpec = spec
		}
		if spec.Type == AcceleratorGPU && spec.Name == "NVIDIA A100" {
			gpuSpec = spec
		}
	}

	// TPUs should generally be more efficient (TFLOPS/W)
	if tpuSpec.Efficiency <= 0 || gpuSpec.Efficiency <= 0 {
		t.Error("Efficiency calculations incorrect")
	}
}

func TestPrintComparison(t *testing.T) {
	output := PrintComparison()

	if len(output) == 0 {
		t.Error("Should produce comparison output")
	}

	// Check that output contains expected accelerator types
	if !contains(output, "GPU") {
		t.Error("Output should contain GPU information")
	}
	if !contains(output, "TPU") {
		t.Error("Output should contain TPU information")
	}
}

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestTPUvsCPU(t *testing.T) {
	specs := CompareAccelerators()

	var cpu, tpu AcceleratorSpec
	for _, spec := range specs {
		if spec.Type == AcceleratorCPU {
			cpu = spec
		}
		if spec.Type == AcceleratorTPU && spec.Name == "Google TPU v4" {
			tpu = spec
		}
	}

	// TPU should have higher peak performance than CPU
	if tpu.PeakPerformance <= cpu.PeakPerformance {
		t.Error("TPU should have higher peak performance than CPU")
	}
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkTPUMatrixMultiply(b *testing.B) {
	tpu := NewTPUv5()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tpu.MatrixMultiply(512, 512, 512)
	}
}

func BenchmarkNPUInference(b *testing.B) {
	npu := NewAppleNeuralEngine()
	ops := uint64(1e9)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		npu.RunInference(ops)
	}
}

func BenchmarkFPGAConfigAndExecute(b *testing.B) {
	fpga := NewXilinxVersal()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fpga.Configure("MatrixMultiply")
		fpga.Execute()
	}
}
