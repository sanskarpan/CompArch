/*
Translation Lookaside Buffer (TLB)
====================================

Simulates the virtual-to-physical address translation hardware found in
modern processors.

Features:
- Fully-associative and set-associative TLB configurations
- VIPT (Virtually Indexed, Physically Tagged) L1 TLB
- Separate instruction (ITLB) and data (DTLB) TLBs
- TLB shootdown / flush (full and address-specific)
- 2-level page table walk simulation
- TLB statistics (hits, misses, page faults, walk cycles)

Applications:
- Understanding virtual memory overhead
- Optimising huge-page usage
- Analysing TLB thrashing in multi-threaded workloads
*/

package cpu

import (
	"fmt"
	"sync"
)

// =============================================================================
// Page Table Constants
// =============================================================================

const (
	PageSizeSmall  = 4096        // 4 KB – standard page
	PageSizeLarge  = 2097152     // 2 MB – huge page (x86 2MiB)
	PageSizeHuge   = 1073741824  // 1 GB – gigantic page
	VPNBits        = 36          // virtual page number bits (48-bit VA / 12-bit offset)
	PageTableWalkCycles = 100    // simulated cycles per page-table walk
)

// PagePermission represents page access permissions
type PagePermission uint8

const (
	PermNone    PagePermission = 0
	PermRead    PagePermission = 1 << 0
	PermWrite   PagePermission = 1 << 1
	PermExecute PagePermission = 1 << 2
	PermRWX     PagePermission = PermRead | PermWrite | PermExecute
	PermRX      PagePermission = PermRead | PermExecute
	PermRW      PagePermission = PermRead | PermWrite
)

// =============================================================================
// TLB Entry
// =============================================================================

// TLBEntry represents a single TLB entry
type TLBEntry struct {
	Valid       bool
	VPN         uint64         // Virtual Page Number
	PPN         uint64         // Physical Page Number
	ASID        uint16         // Address Space ID (for process isolation)
	PageSize    uint64         // page size this entry covers
	Permission  PagePermission // read/write/execute bits
	Dirty       bool           // page has been written
	Accessed    bool           // page has been accessed
	LRUCounter  uint64         // for LRU replacement
}

// =============================================================================
// TLB
// =============================================================================

// TLBConfig holds configuration for a TLB
type TLBConfig struct {
	NumSets      int    // number of sets (1 = fully associative)
	Associativity int   // ways per set
	HitLatency   int    // cycles for a TLB hit
	MissLatency  int    // cycles for a TLB miss (page-table walk)
}

// TLB represents a Translation Lookaside Buffer
type TLB struct {
	Config     TLBConfig
	Sets       [][]TLBEntry // [set][way]
	lruCounter uint64
	mu         sync.Mutex

	// Statistics
	Hits        uint64
	Misses      uint64
	PageFaults  uint64
	Flushes     uint64
	WalkCycles  uint64
}

// NewTLB creates a new TLB
func NewTLB(cfg TLBConfig) *TLB {
	if cfg.NumSets <= 0 {
		cfg.NumSets = 1
	}
	if cfg.Associativity <= 0 {
		cfg.Associativity = 64
	}
	if cfg.HitLatency <= 0 {
		cfg.HitLatency = 1
	}
	if cfg.MissLatency <= 0 {
		cfg.MissLatency = PageTableWalkCycles
	}

	sets := make([][]TLBEntry, cfg.NumSets)
	for i := range sets {
		sets[i] = make([]TLBEntry, cfg.Associativity)
	}

	return &TLB{
		Config: cfg,
		Sets:   sets,
	}
}

// NewL1DTLB creates a typical L1 data TLB (fully associative, 64 entries)
func NewL1DTLB() *TLB {
	return NewTLB(TLBConfig{
		NumSets:      1,
		Associativity: 64,
		HitLatency:   1,
		MissLatency:  PageTableWalkCycles,
	})
}

// NewL1ITLB creates a typical L1 instruction TLB
func NewL1ITLB() *TLB {
	return NewTLB(TLBConfig{
		NumSets:      1,
		Associativity: 128,
		HitLatency:   1,
		MissLatency:  PageTableWalkCycles,
	})
}

// NewL2TLB creates a shared L2 TLB (larger, slower)
func NewL2TLB() *TLB {
	return NewTLB(TLBConfig{
		NumSets:      4,
		Associativity: 256,
		HitLatency:   6,
		MissLatency:  PageTableWalkCycles,
	})
}

// setIndex returns the set index for a VPN in a set-associative TLB
func (t *TLB) setIndex(vpn uint64) int {
	return int(vpn) % t.Config.NumSets
}

// Lookup looks up a virtual address in the TLB.
// Returns (physicalAddr, hit, latencyCycles).
// If hit==false, the caller should invoke PageTableWalk.
func (t *TLB) Lookup(virtualAddr uint64, pageSize uint64, asid uint16) (physAddr uint64, hit bool, cycles int) {
	vpn := virtualAddr / pageSize
	pageOffset := virtualAddr % pageSize
	setIdx := t.setIndex(vpn)

	t.mu.Lock()
	defer t.mu.Unlock()

	set := t.Sets[setIdx]
	for i := range set {
		e := &set[i]
		if e.Valid && e.VPN == vpn && e.ASID == asid && e.PageSize == pageSize {
			// TLB hit
			t.Hits++
			t.lruCounter++
			e.LRUCounter = t.lruCounter
			e.Accessed = true
			physAddr = e.PPN*pageSize + pageOffset
			return physAddr, true, t.Config.HitLatency
		}
	}

	// TLB miss
	t.Misses++
	return 0, false, t.Config.MissLatency
}

// Install installs a new TLB entry after a page-table walk.
// If the set is full, evicts the LRU entry.
func (t *TLB) Install(virtualAddr, physAddr uint64, pageSize uint64, asid uint16, perm PagePermission) {
	vpn := virtualAddr / pageSize
	ppn := physAddr / pageSize
	setIdx := t.setIndex(vpn)

	t.mu.Lock()
	defer t.mu.Unlock()

	t.lruCounter++
	set := t.Sets[setIdx]

	// Find an invalid (free) slot first
	for i := range set {
		if !set[i].Valid {
			set[i] = TLBEntry{
				Valid:      true,
				VPN:        vpn,
				PPN:        ppn,
				ASID:       asid,
				PageSize:   pageSize,
				Permission: perm,
				Accessed:   true,
				LRUCounter: t.lruCounter,
			}
			return
		}
	}

	// Evict LRU entry
	lruIdx := 0
	for i := 1; i < len(set); i++ {
		if set[i].LRUCounter < set[lruIdx].LRUCounter {
			lruIdx = i
		}
	}
	set[lruIdx] = TLBEntry{
		Valid:      true,
		VPN:        vpn,
		PPN:        ppn,
		ASID:       asid,
		PageSize:   pageSize,
		Permission: perm,
		Accessed:   true,
		LRUCounter: t.lruCounter,
	}
}

// FlushAll invalidates all TLB entries (full context switch)
func (t *TLB) FlushAll() {
	t.mu.Lock()
	defer t.mu.Unlock()

	for i := range t.Sets {
		for j := range t.Sets[i] {
			t.Sets[i][j].Valid = false
		}
	}
	t.Flushes++
}

// FlushASID invalidates all entries for the given address space (process exit)
func (t *TLB) FlushASID(asid uint16) {
	t.mu.Lock()
	defer t.mu.Unlock()

	for i := range t.Sets {
		for j := range t.Sets[i] {
			if t.Sets[i][j].ASID == asid {
				t.Sets[i][j].Valid = false
			}
		}
	}
	t.Flushes++
}

// FlushPage invalidates the specific virtual page in all sets
func (t *TLB) FlushPage(virtualAddr uint64, pageSize uint64, asid uint16) {
	vpn := virtualAddr / pageSize
	setIdx := t.setIndex(vpn)

	t.mu.Lock()
	defer t.mu.Unlock()

	for j := range t.Sets[setIdx] {
		e := &t.Sets[setIdx][j]
		if e.Valid && e.VPN == vpn && e.ASID == asid {
			e.Valid = false
		}
	}
	t.Flushes++
}

// GetStats returns TLB statistics
func (t *TLB) GetStats() TLBStats {
	t.mu.Lock()
	defer t.mu.Unlock()

	total := t.Hits + t.Misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(t.Hits) / float64(total)
	}
	return TLBStats{
		Hits:       t.Hits,
		Misses:     t.Misses,
		PageFaults: t.PageFaults,
		Flushes:    t.Flushes,
		WalkCycles: t.WalkCycles,
		HitRate:    hitRate,
	}
}

// TLBStats contains TLB performance statistics
type TLBStats struct {
	Hits       uint64
	Misses     uint64
	PageFaults uint64
	Flushes    uint64
	WalkCycles uint64
	HitRate    float64
}

// String returns a human-readable representation
func (s TLBStats) String() string {
	return fmt.Sprintf(`TLB Statistics:
  Hits:        %d (%.2f%%)
  Misses:      %d
  Page Faults: %d
  Flushes:     %d
  Walk Cycles: %d`,
		s.Hits, 100.0*s.HitRate,
		s.Misses,
		s.PageFaults,
		s.Flushes,
		s.WalkCycles)
}

// =============================================================================
// Page Table (2-level simulation)
// =============================================================================

// PageTableEntry represents one entry in a page table
type PageTableEntry struct {
	Present    bool
	PPN        uint64         // Physical page number
	Permission PagePermission
	Dirty      bool
	Accessed   bool
}

// PageTable simulates a 2-level hierarchical page table
type PageTable struct {
	ASID    uint16
	L1Table map[uint64]map[uint64]*PageTableEntry // L1[vpn_high] -> L2[vpn_low] -> PTE
	mu      sync.Mutex
}

// NewPageTable creates a new page table for a given address space
func NewPageTable(asid uint16) *PageTable {
	return &PageTable{
		ASID:    asid,
		L1Table: make(map[uint64]map[uint64]*PageTableEntry),
	}
}

// MapPage maps a virtual page to a physical page
func (pt *PageTable) MapPage(virtualAddr, physAddr uint64, pageSize uint64, perm PagePermission) {
	vpn := virtualAddr / pageSize
	l1Idx := vpn >> 9  // top 9 bits of VPN
	l2Idx := vpn & 0x1FF // bottom 9 bits of VPN

	pt.mu.Lock()
	defer pt.mu.Unlock()

	if _, ok := pt.L1Table[l1Idx]; !ok {
		pt.L1Table[l1Idx] = make(map[uint64]*PageTableEntry)
	}
	pt.L1Table[l1Idx][l2Idx] = &PageTableEntry{
		Present:    true,
		PPN:        physAddr / pageSize,
		Permission: perm,
	}
}

// Walk simulates a page-table walk for the given virtual address.
// Returns (physAddr, present, permissionError).
func (pt *PageTable) Walk(virtualAddr uint64, pageSize uint64) (physAddr uint64, present bool, permErr bool) {
	vpn := virtualAddr / pageSize
	pageOffset := virtualAddr % pageSize
	l1Idx := vpn >> 9
	l2Idx := vpn & 0x1FF

	pt.mu.Lock()
	defer pt.mu.Unlock()

	l2, ok := pt.L1Table[l1Idx]
	if !ok {
		return 0, false, false // page fault
	}
	pte, ok := l2[l2Idx]
	if !ok || !pte.Present {
		return 0, false, false // page fault
	}
	pte.Accessed = true
	return pte.PPN*pageSize + pageOffset, true, false
}

// UnmapPage removes the mapping for a virtual page
func (pt *PageTable) UnmapPage(virtualAddr uint64, pageSize uint64) {
	vpn := virtualAddr / pageSize
	l1Idx := vpn >> 9
	l2Idx := vpn & 0x1FF

	pt.mu.Lock()
	defer pt.mu.Unlock()

	if l2, ok := pt.L1Table[l1Idx]; ok {
		delete(l2, l2Idx)
	}
}

// =============================================================================
// MMU – integrates TLBs + Page Table
// =============================================================================

// MMU represents the Memory Management Unit with L1 DTLB, L1 ITLB, and L2 TLB.
type MMU struct {
	DTLB  *TLB       // L1 data TLB
	ITLB  *TLB       // L1 instruction TLB
	L2TLB *TLB       // Shared L2 TLB
	PT    *PageTable  // Page table for address translation

	PageSize uint64 // Active page size (default 4 KB)
	ASID     uint16

	// Statistics
	DataTranslations uint64
	CodeTranslations uint64
	PageFaults       uint64
	mu               sync.Mutex
}

// NewMMU creates a new MMU for the given address space
func NewMMU(asid uint16) *MMU {
	return &MMU{
		DTLB:     NewL1DTLB(),
		ITLB:     NewL1ITLB(),
		L2TLB:    NewL2TLB(),
		PT:       NewPageTable(asid),
		PageSize: PageSizeSmall,
		ASID:     asid,
	}
}

// TranslateData translates a virtual data address to physical.
// Returns (physAddr, cycles, pageFault).
func (m *MMU) TranslateData(virtualAddr uint64) (physAddr uint64, cycles int, pageFault bool) {
	m.mu.Lock()
	m.DataTranslations++
	m.mu.Unlock()

	// L1 DTLB lookup
	pa, hit, lat := m.DTLB.Lookup(virtualAddr, m.PageSize, m.ASID)
	if hit {
		return pa, lat, false
	}
	cycles += lat

	// L2 TLB lookup
	pa, hit, lat = m.L2TLB.Lookup(virtualAddr, m.PageSize, m.ASID)
	if hit {
		// Install in L1 DTLB
		m.DTLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRW)
		return pa, cycles + lat, false
	}
	cycles += lat

	// Page-table walk
	pa, present, _ := m.PT.Walk(virtualAddr, m.PageSize)
	if !present {
		m.mu.Lock()
		m.PageFaults++
		m.mu.Unlock()
		return 0, cycles + PageTableWalkCycles, true
	}

	cycles += PageTableWalkCycles
	m.DTLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRW)
	m.L2TLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRW)
	return pa, cycles, false
}

// TranslateCode translates a virtual instruction address to physical.
func (m *MMU) TranslateCode(virtualAddr uint64) (physAddr uint64, cycles int, pageFault bool) {
	m.mu.Lock()
	m.CodeTranslations++
	m.mu.Unlock()

	pa, hit, lat := m.ITLB.Lookup(virtualAddr, m.PageSize, m.ASID)
	if hit {
		return pa, lat, false
	}
	cycles += lat

	pa, hit, lat = m.L2TLB.Lookup(virtualAddr, m.PageSize, m.ASID)
	if hit {
		m.ITLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRX)
		return pa, cycles + lat, false
	}
	cycles += lat

	pa, present, _ := m.PT.Walk(virtualAddr, m.PageSize)
	if !present {
		m.mu.Lock()
		m.PageFaults++
		m.mu.Unlock()
		return 0, cycles + PageTableWalkCycles, true
	}

	cycles += PageTableWalkCycles
	m.ITLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRX)
	m.L2TLB.Install(virtualAddr, pa, m.PageSize, m.ASID, PermRX)
	return pa, cycles, false
}

// MapPage maps a virtual page in the page table.
// Call this to set up the address space before executing code.
func (m *MMU) MapPage(virtualAddr, physAddr uint64, perm PagePermission) {
	m.PT.MapPage(virtualAddr, physAddr, m.PageSize, perm)
}

// FlushTLBs performs a full TLB shootdown (e.g. on context switch)
func (m *MMU) FlushTLBs() {
	m.DTLB.FlushAll()
	m.ITLB.FlushAll()
	m.L2TLB.FlushAll()
}

// GetStats returns combined MMU statistics
func (m *MMU) GetStats() MMUStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	return MMUStats{
		DataTranslations: m.DataTranslations,
		CodeTranslations: m.CodeTranslations,
		PageFaults:       m.PageFaults,
		DTLB:             m.DTLB.GetStats(),
		ITLB:             m.ITLB.GetStats(),
		L2TLB:            m.L2TLB.GetStats(),
	}
}

// MMUStats contains aggregated MMU statistics
type MMUStats struct {
	DataTranslations uint64
	CodeTranslations uint64
	PageFaults       uint64
	DTLB             TLBStats
	ITLB             TLBStats
	L2TLB            TLBStats
}

// String returns a human-readable MMU report
func (s MMUStats) String() string {
	return fmt.Sprintf(`MMU Statistics:
  Data Translations: %d
  Code Translations: %d
  Page Faults:       %d

  L1 DTLB: %s
  L1 ITLB: %s
  L2  TLB: %s`,
		s.DataTranslations,
		s.CodeTranslations,
		s.PageFaults,
		s.DTLB.String(),
		s.ITLB.String(),
		s.L2TLB.String())
}
