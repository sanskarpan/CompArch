package cpu

import (
	"math"
	"testing"
)

// =============================================================================
// DVFSPoint Tests
// =============================================================================

func TestPredefinedDVFS(t *testing.T) {
	if len(PredefinedDVFS) == 0 {
		t.Fatal("PredefinedDVFS should not be empty")
	}
	// Verify eco < nominal < turbo in terms of frequency
	eco := PredefinedDVFS[0]
	nominal := PredefinedDVFS[1]
	turbo := PredefinedDVFS[3]

	if eco.FreqGHz >= nominal.FreqGHz {
		t.Error("eco frequency should be less than nominal")
	}
	if nominal.FreqGHz >= turbo.FreqGHz {
		t.Error("nominal frequency should be less than turbo")
	}
}

// =============================================================================
// CorePowerModel Tests
// =============================================================================

func TestDynamicPower(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	p := m.DynamicPower()
	// P = α·C·V²·f
	// = 0.10 * 5e-9 * 0.9^2 * 2.4e9
	expected := cfg.ActivityFactor * cfg.EffectiveCapacitance *
		m.CurrentDVFS.VoltageV * m.CurrentDVFS.VoltageV *
		m.CurrentDVFS.FreqGHz * 1e9

	if math.Abs(p-expected) > 1e-9 {
		t.Errorf("DynamicPower: got %.6f, want %.6f", p, expected)
	}
	if p <= 0 {
		t.Error("dynamic power must be positive")
	}
}

func TestStaticPower(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	// At ambient temperature, T_factor = exp(0) = 1
	p := m.StaticPower()
	expectedBase := cfg.LeakageCurrent * m.CurrentDVFS.VoltageV
	if math.Abs(p-expectedBase) > 1e-9 {
		t.Errorf("StaticPower at ambient: got %.6f, want %.6f", p, expectedBase)
	}
}

func TestStaticPowerIncreasesWithTemp(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	pCold := m.StaticPower()

	// Artificially raise temperature
	m.CurrentTempC = cfg.AmbientTempC + 20.0
	pHot := m.StaticPower()

	if pHot <= pCold {
		t.Error("static power should increase with temperature")
	}
}

func TestTotalPower(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	total := m.TotalPower()
	sum := m.DynamicPower() + m.StaticPower()
	if math.Abs(total-sum) > 1e-12 {
		t.Errorf("TotalPower: got %.6f, want %.6f", total, sum)
	}
}

func TestTickAdvancesCycles(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	m.Tick()
	if m.TotalCycles != 1 {
		t.Errorf("TotalCycles after Tick: got %d, want 1", m.TotalCycles)
	}
	if m.TotalEnergyJ <= 0 {
		t.Error("TotalEnergyJ should be positive after Tick")
	}
	if m.CurrentPowerW <= 0 {
		t.Error("CurrentPowerW should be positive after Tick")
	}
}

func TestTickNAdvancesCycles(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	m.TickN(1000)
	if m.TotalCycles != 1000 {
		t.Errorf("TotalCycles after TickN(1000): got %d, want 1000", m.TotalCycles)
	}
	if m.TotalEnergyJ <= 0 {
		t.Error("TotalEnergyJ should be positive after TickN")
	}
}

func TestTickVsTickN(t *testing.T) {
	cfg := DefaultPowerConfig()

	m1 := NewCorePowerModel(cfg)
	for i := 0; i < 1000; i++ {
		m1.Tick()
	}

	m2 := NewCorePowerModel(cfg)
	m2.TickN(1000)

	// Both should accumulate the same energy (small tolerance for floating point)
	diff := math.Abs(m1.TotalEnergyJ - m2.TotalEnergyJ)
	tolerance := m1.TotalEnergyJ * 0.001 // within 0.1%
	if diff > tolerance {
		t.Errorf("Tick vs TickN energy diverged: %.6f vs %.6f (diff %.6f > tol %.6f)",
			m1.TotalEnergyJ, m2.TotalEnergyJ, diff, tolerance)
	}
}

func TestTemperatureRises(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	initialTemp := m.CurrentTempC
	m.TickN(1_000_000) // Run a million cycles
	if m.CurrentTempC <= initialTemp {
		t.Error("temperature should rise under sustained load")
	}
}

func TestTemperatureEquilibrium(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	// Run long enough to approach steady-state temperature
	for i := 0; i < 100; i++ {
		m.TickN(10_000_000)
	}

	// Steady-state: T_ss = T_amb + P * Rth
	power := m.TotalPower()
	expectedTss := cfg.AmbientTempC + power*cfg.ThermalResistance
	diff := math.Abs(m.CurrentTempC - expectedTss)
	if diff > 1.0 { // within 1°C
		t.Errorf("temperature not at steady state: got %.2f°C, want ~%.2f°C",
			m.CurrentTempC, expectedTss)
	}
}

func TestSetDVFS(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	turbo := PredefinedDVFS[3]
	m.SetDVFS(turbo)

	if m.CurrentDVFS.Name != turbo.Name {
		t.Errorf("SetDVFS: got %q, want %q", m.CurrentDVFS.Name, turbo.Name)
	}
	// Turbo should draw more dynamic power than nominal
	nominalModel := NewCorePowerModel(cfg)
	if m.DynamicPower() <= nominalModel.DynamicPower() {
		t.Error("turbo should have higher dynamic power than nominal")
	}
}

func TestThrottling(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)

	// Should not be throttling at ambient
	if m.IsThrottling() {
		t.Error("should not throttle at ambient temperature")
	}

	// Force temperature above 100°C
	m.CurrentTempC = 110.0
	if !m.IsThrottling() {
		t.Error("should throttle above 100°C")
	}
}

func TestGetStats(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)
	m.TickN(100)

	stats := m.GetStats()
	if stats.FreqGHz != m.CurrentDVFS.FreqGHz {
		t.Errorf("FreqGHz: got %.2f, want %.2f", stats.FreqGHz, m.CurrentDVFS.FreqGHz)
	}
	if stats.Cycles != 100 {
		t.Errorf("Cycles: got %d, want 100", stats.Cycles)
	}
	if stats.EnergyJ <= 0 {
		t.Error("EnergyJ should be positive")
	}
	_ = stats.String() // Ensure String() doesn't panic
}

func TestDVFSLowerVoltageReducesPower(t *testing.T) {
	cfg := DefaultPowerConfig()

	mEco := NewCorePowerModel(cfg)
	mEco.SetDVFS(PredefinedDVFS[0]) // eco

	mTurbo := NewCorePowerModel(cfg)
	mTurbo.SetDVFS(PredefinedDVFS[3]) // turbo

	if mEco.DynamicPower() >= mTurbo.DynamicPower() {
		t.Error("eco dynamic power should be less than turbo")
	}
}

// =============================================================================
// SystemPowerModel Tests
// =============================================================================

func TestSystemPowerModelCreation(t *testing.T) {
	sp := NewSystemPowerModel(4)
	if len(sp.Cores) != 4 {
		t.Errorf("cores: got %d, want 4", len(sp.Cores))
	}
	if sp.UncoreWatts <= 0 {
		t.Error("uncore power should be positive")
	}
}

func TestSystemTotalPower(t *testing.T) {
	sp := NewSystemPowerModel(2)

	total := sp.TotalPowerW()
	if total <= sp.UncoreWatts {
		t.Error("system power should exceed uncore power")
	}
}

func TestSystemTickAll(t *testing.T) {
	sp := NewSystemPowerModel(2)
	sp.TickAll(500)

	for i, c := range sp.Cores {
		if c.TotalCycles != 500 {
			t.Errorf("core %d cycles: got %d, want 500", i, c.TotalCycles)
		}
	}
}

func TestSystemGetCoreStats(t *testing.T) {
	sp := NewSystemPowerModel(2)
	sp.TickAll(100)

	stats := sp.GetCoreStats(0)
	if stats.Cycles != 100 {
		t.Errorf("core 0 cycles: got %d, want 100", stats.Cycles)
	}

	// Out-of-range core → zero stats
	empty := sp.GetCoreStats(99)
	if empty.Cycles != 0 {
		t.Errorf("out-of-range core should return zero stats")
	}
}

func TestSystemPrintReport(t *testing.T) {
	sp := NewSystemPowerModel(2)
	sp.TickAll(1000)

	report := sp.PrintReport()
	if len(report) == 0 {
		t.Error("PrintReport should not return empty string")
	}
}

// =============================================================================
// PowerStats String
// =============================================================================

func TestPowerStatsString(t *testing.T) {
	cfg := DefaultPowerConfig()
	m := NewCorePowerModel(cfg)
	m.TickN(10000)
	stats := m.GetStats()
	str := stats.String()
	if len(str) == 0 {
		t.Error("PowerStats.String() should not be empty")
	}
}
