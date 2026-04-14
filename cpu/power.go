/*
CPU Power Model
===============

Simulates power consumption using a DVFS-aware model with:
- Dynamic power (switching activity: P = α·C·V²·f)
- Static power (leakage: P_leak = V·I_leak)
- Thermal model (simple RC lumped model)
- DVFS operating points (voltage/frequency scaling)

Applications:
- Energy-efficiency analysis
- Thermal throttling simulation
- Power-performance trade-off studies
*/

package cpu

import (
	"fmt"
	"math"
	"sync"
)

// =============================================================================
// DVFS Operating Points
// =============================================================================

// DVFSPoint represents one voltage/frequency operating point
type DVFSPoint struct {
	Name      string
	FreqGHz   float64 // clock frequency in GHz
	VoltageV  float64 // supply voltage in Volts
	RelPower  float64 // relative total power vs. baseline (informational)
}

// PredefinedDVFS lists common operating points (based on Intel Alder Lake data)
var PredefinedDVFS = []DVFSPoint{
	{Name: "eco",      FreqGHz: 1.0, VoltageV: 0.70, RelPower: 0.15},
	{Name: "nominal",  FreqGHz: 2.4, VoltageV: 0.90, RelPower: 0.50},
	{Name: "boost",    FreqGHz: 3.6, VoltageV: 1.00, RelPower: 0.75},
	{Name: "turbo",    FreqGHz: 4.8, VoltageV: 1.10, RelPower: 1.00},
	{Name: "extreme",  FreqGHz: 5.5, VoltageV: 1.20, RelPower: 1.35},
}

// =============================================================================
// Power Model
// =============================================================================

// PowerConfig holds the static characteristics of the chip
type PowerConfig struct {
	// Process technology node in nm (affects leakage)
	ProcessNm int

	// Activity factor (fraction of transistors switching per cycle, 0..1)
	ActivityFactor float64

	// Effective capacitance (Farads) – scales dynamic power
	EffectiveCapacitance float64

	// Static (leakage) current at 1 V (Amperes)
	LeakageCurrent float64

	// Thermal resistance (°C/W) – chip to ambient
	ThermalResistance float64

	// Thermal capacitance (J/°C) – affects thermal transients
	ThermalCapacitance float64

	// Ambient temperature (°C)
	AmbientTempC float64
}

// DefaultPowerConfig returns a reasonable config for a ~12 nm core
func DefaultPowerConfig() PowerConfig {
	return PowerConfig{
		ProcessNm:            12,
		ActivityFactor:       0.10,  // 10% transistors active per cycle
		EffectiveCapacitance: 5e-9,  // 5 nF
		LeakageCurrent:       0.05,  // 50 mA at 1 V
		ThermalResistance:    0.5,   // 0.5 °C/W (modern heatsink)
		ThermalCapacitance:   2.0,   // 2 J/°C
		AmbientTempC:         25.0,
	}
}

// CorePowerModel models power and thermal state for a single CPU core
type CorePowerModel struct {
	Config      PowerConfig
	CurrentDVFS DVFSPoint

	// Running totals
	TotalEnergyJ    float64 // Joules
	TotalCycles     uint64
	CurrentPowerW   float64 // instantaneous power in Watts
	CurrentTempC    float64 // current die temperature

	mu sync.Mutex
}

// NewCorePowerModel creates a power model initialised at nominal DVFS
func NewCorePowerModel(cfg PowerConfig) *CorePowerModel {
	nominal := PredefinedDVFS[1] // "nominal"
	return &CorePowerModel{
		Config:       cfg,
		CurrentDVFS:  nominal,
		CurrentTempC: cfg.AmbientTempC,
	}
}

// DynamicPower computes dynamic power for the current operating point.
// P_dyn = α · C · V² · f
func (m *CorePowerModel) DynamicPower() float64 {
	alpha := m.Config.ActivityFactor
	c := m.Config.EffectiveCapacitance
	v := m.CurrentDVFS.VoltageV
	f := m.CurrentDVFS.FreqGHz * 1e9
	return alpha * c * v * v * f
}

// StaticPower computes leakage power for the current voltage.
// P_static = V · I_leak · T_factor
// where T_factor grows exponentially with temperature (Arrhenius approximation)
func (m *CorePowerModel) StaticPower() float64 {
	v := m.CurrentDVFS.VoltageV
	iLeak := m.Config.LeakageCurrent
	// Simplified Arrhenius: leakage doubles every ~10°C
	tFactor := math.Exp((m.CurrentTempC - m.Config.AmbientTempC) * math.Log(2) / 10.0)
	return v * iLeak * tFactor
}

// TotalPower returns the sum of dynamic and static power in Watts
func (m *CorePowerModel) TotalPower() float64 {
	return m.DynamicPower() + m.StaticPower()
}

// Tick advances the power model by one simulated cycle, updating energy and temperature.
func (m *CorePowerModel) Tick() {
	m.mu.Lock()
	defer m.mu.Unlock()

	freq := m.CurrentDVFS.FreqGHz * 1e9 // Hz
	dt := 1.0 / freq                      // seconds per cycle

	power := m.DynamicPower() + m.StaticPower()
	m.CurrentPowerW = power
	m.TotalEnergyJ += power * dt
	m.TotalCycles++

	// Thermal model: dT/dt = (P - (T-T_amb)/Rth) / Cth
	// Euler integration
	tAmb := m.Config.AmbientTempC
	rth := m.Config.ThermalResistance
	cth := m.Config.ThermalCapacitance
	dT := (power - (m.CurrentTempC-tAmb)/rth) / cth * dt
	m.CurrentTempC += dT
}

// TickN advances by n cycles (efficient for large cycle counts)
func (m *CorePowerModel) TickN(n uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	freq := m.CurrentDVFS.FreqGHz * 1e9
	dt := float64(n) / freq

	power := m.DynamicPower() + m.StaticPower()
	m.CurrentPowerW = power
	m.TotalEnergyJ += power * dt
	m.TotalCycles += n

	tAmb := m.Config.AmbientTempC
	rth := m.Config.ThermalResistance
	cth := m.Config.ThermalCapacitance
	// Steady-state temperature: T_ss = T_amb + P * Rth
	// Transient with time constant τ = Rth * Cth
	tauInv := 1.0 / (rth * cth)
	tss := tAmb + power*rth
	m.CurrentTempC = tss + (m.CurrentTempC-tss)*math.Exp(-tauInv*dt)
}

// SetDVFS switches the operating point
func (m *CorePowerModel) SetDVFS(point DVFSPoint) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CurrentDVFS = point
}

// IsThrottling returns true if the temperature exceeds the safe threshold (100°C)
func (m *CorePowerModel) IsThrottling() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.CurrentTempC > 100.0
}

// GetStats returns the current power/thermal state
func (m *CorePowerModel) GetStats() PowerStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	return PowerStats{
		FreqGHz:       m.CurrentDVFS.FreqGHz,
		VoltageV:      m.CurrentDVFS.VoltageV,
		DynamicPowerW: m.DynamicPower(),
		StaticPowerW:  m.StaticPower(),
		TotalPowerW:   m.TotalPower(),
		TempC:         m.CurrentTempC,
		EnergyJ:       m.TotalEnergyJ,
		Cycles:        m.TotalCycles,
	}
}

// PowerStats contains point-in-time power/thermal metrics
type PowerStats struct {
	FreqGHz       float64
	VoltageV      float64
	DynamicPowerW float64
	StaticPowerW  float64
	TotalPowerW   float64
	TempC         float64
	EnergyJ       float64
	Cycles        uint64
}

// String returns a human-readable power report
func (s PowerStats) String() string {
	energyMJ := s.EnergyJ * 1000
	return fmt.Sprintf(`Power Statistics:
  Frequency:       %.2f GHz
  Voltage:         %.2f V
  Dynamic Power:   %.2f W
  Static Power:    %.2f W
  Total Power:     %.2f W
  Temperature:     %.1f °C
  Total Energy:    %.4f mJ
  Total Cycles:    %d`,
		s.FreqGHz,
		s.VoltageV,
		s.DynamicPowerW,
		s.StaticPowerW,
		s.TotalPowerW,
		s.TempC,
		energyMJ,
		s.Cycles)
}

// =============================================================================
// System Power Model – aggregates all cores
// =============================================================================

// SystemPowerModel models power across all cores plus uncore components
type SystemPowerModel struct {
	Cores         []*CorePowerModel
	UncoreWatts   float64 // memory controller, PCIe, etc.
	mu            sync.Mutex
}

// NewSystemPowerModel creates a system power model for numCores cores
func NewSystemPowerModel(numCores int) *SystemPowerModel {
	cfg := DefaultPowerConfig()
	cores := make([]*CorePowerModel, numCores)
	for i := range cores {
		cores[i] = NewCorePowerModel(cfg)
	}
	return &SystemPowerModel{
		Cores:       cores,
		UncoreWatts: 5.0, // typical uncore baseline
	}
}

// TotalPowerW returns the instantaneous total system power
func (sp *SystemPowerModel) TotalPowerW() float64 {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	total := sp.UncoreWatts
	for _, c := range sp.Cores {
		total += c.TotalPower()
	}
	return total
}

// TickAll advances all core power models by n cycles
func (sp *SystemPowerModel) TickAll(cycles uint64) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	for _, c := range sp.Cores {
		c.TickN(cycles)
	}
}

// GetCoreStats returns power stats for a specific core
func (sp *SystemPowerModel) GetCoreStats(coreID int) PowerStats {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	if coreID >= 0 && coreID < len(sp.Cores) {
		return sp.Cores[coreID].GetStats()
	}
	return PowerStats{}
}

// PrintReport prints a full power report for all cores
func (sp *SystemPowerModel) PrintReport() string {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	var result string
	result += fmt.Sprintf("System Power Report  (uncore=%.1fW)\n", sp.UncoreWatts)
	result += "=============================================================\n"
	total := sp.UncoreWatts
	for i, c := range sp.Cores {
		s := c.GetStats()
		total += s.TotalPowerW
		result += fmt.Sprintf("Core %d: %.2fW @ %.2fGHz (%.1f°C)\n",
			i, s.TotalPowerW, s.FreqGHz, s.TempC)
	}
	result += fmt.Sprintf("-------------------------------------------------------------\n")
	result += fmt.Sprintf("Total:  %.2f W\n", total)
	return result
}
