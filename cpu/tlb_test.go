package cpu

import (
	"sync"
	"testing"
)

// =============================================================================
// TLB Tests
// =============================================================================

func TestTLBHitMiss(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall
	const asid = uint16(1)

	// First lookup – must miss
	_, hit, cycles := tlb.Lookup(0x1000, pageSize, asid)
	if hit {
		t.Fatal("expected TLB miss on cold lookup")
	}
	if cycles != PageTableWalkCycles {
		t.Errorf("miss latency: got %d, want %d", cycles, PageTableWalkCycles)
	}

	// Install the translation
	tlb.Install(0x1000, 0xA000, pageSize, asid, PermRW)

	// Second lookup – must hit
	physAddr, hit, cycles := tlb.Lookup(0x1000, pageSize, asid)
	if !hit {
		t.Fatal("expected TLB hit after Install")
	}
	if cycles != tlb.Config.HitLatency {
		t.Errorf("hit latency: got %d, want %d", cycles, tlb.Config.HitLatency)
	}
	// Physical address must preserve page offset (0x1000 % pageSize == 0)
	expectedPhys := uint64(0xA000)
	if physAddr != expectedPhys {
		t.Errorf("physAddr: got 0x%X, want 0x%X", physAddr, expectedPhys)
	}
}

func TestTLBPageOffset(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall
	const asid = uint16(0)
	const offset = uint64(0x123)

	virtualAddr := uint64(0x5000) + offset
	physBase := uint64(0x9000)

	tlb.Install(virtualAddr, physBase, pageSize, asid, PermRW)

	physAddr, hit, _ := tlb.Lookup(virtualAddr, pageSize, asid)
	if !hit {
		t.Fatal("expected TLB hit")
	}
	expectedPhys := physBase + offset
	if physAddr != expectedPhys {
		t.Errorf("offset not preserved: got 0x%X, want 0x%X", physAddr, expectedPhys)
	}
}

func TestTLBASIDIsolation(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall

	tlb.Install(0x2000, 0xB000, pageSize, 1, PermRW)

	// ASID 2 should NOT see ASID 1's entry
	_, hit, _ := tlb.Lookup(0x2000, pageSize, 2)
	if hit {
		t.Error("ASID isolation violated: ASID 2 hit ASID 1's entry")
	}

	// ASID 1 should still hit
	_, hit, _ = tlb.Lookup(0x2000, pageSize, 1)
	if !hit {
		t.Error("ASID 1 lost its own entry")
	}
}

func TestTLBFlushAll(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall
	const asid = uint16(1)

	tlb.Install(0x3000, 0xC000, pageSize, asid, PermRW)
	tlb.Install(0x4000, 0xD000, pageSize, asid, PermRW)

	tlb.FlushAll()

	_, hit, _ := tlb.Lookup(0x3000, pageSize, asid)
	if hit {
		t.Error("entry still present after FlushAll")
	}
	stats := tlb.GetStats()
	if stats.Flushes != 1 {
		t.Errorf("flush count: got %d, want 1", stats.Flushes)
	}
}

func TestTLBFlushASID(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall

	tlb.Install(0x5000, 0xE000, pageSize, 1, PermRW)
	tlb.Install(0x6000, 0xF000, pageSize, 2, PermRW)

	tlb.FlushASID(1)

	_, hit, _ := tlb.Lookup(0x5000, pageSize, 1)
	if hit {
		t.Error("ASID 1 entry still present after FlushASID(1)")
	}
	_, hit, _ = tlb.Lookup(0x6000, pageSize, 2)
	if !hit {
		t.Error("ASID 2 entry incorrectly flushed")
	}
}

func TestTLBFlushPage(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall
	const asid = uint16(1)

	tlb.Install(0x7000, 0x10000, pageSize, asid, PermRW)
	tlb.Install(0x8000, 0x11000, pageSize, asid, PermRW)

	tlb.FlushPage(0x7000, pageSize, asid)

	_, hit, _ := tlb.Lookup(0x7000, pageSize, asid)
	if hit {
		t.Error("page 0x7000 still present after FlushPage")
	}
	_, hit, _ = tlb.Lookup(0x8000, pageSize, asid)
	if !hit {
		t.Error("page 0x8000 incorrectly removed")
	}
}

func TestTLBLRUEviction(t *testing.T) {
	// Use a tiny fully-associative TLB with only 2 ways
	tlb := NewTLB(TLBConfig{NumSets: 1, Associativity: 2, HitLatency: 1})
	const pageSize = PageSizeSmall
	const asid = uint16(0)

	// Fill both ways
	tlb.Install(0x1000, 0xA000, pageSize, asid, PermRW)
	tlb.Install(0x2000, 0xB000, pageSize, asid, PermRW)

	// Touch 0x1000 to make 0x2000 the LRU
	tlb.Lookup(0x1000, pageSize, asid)

	// Install a third page – should evict 0x2000 (LRU)
	tlb.Install(0x3000, 0xC000, pageSize, asid, PermRW)

	_, hit0x1000, _ := tlb.Lookup(0x1000, pageSize, asid)
	_, hit0x2000, _ := tlb.Lookup(0x2000, pageSize, asid)
	_, hit0x3000, _ := tlb.Lookup(0x3000, pageSize, asid)

	if !hit0x1000 {
		t.Error("recently-used 0x1000 should still be in TLB")
	}
	if hit0x2000 {
		t.Error("LRU 0x2000 should have been evicted")
	}
	if !hit0x3000 {
		t.Error("newly installed 0x3000 should be in TLB")
	}
}

func TestTLBStats(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall
	const asid = uint16(0)

	// 3 misses
	for i := uint64(0); i < 3; i++ {
		tlb.Lookup(i*pageSize, pageSize, asid)
		tlb.Install(i*pageSize, i*pageSize+0xF000, pageSize, asid, PermRW)
	}
	// 3 hits
	for i := uint64(0); i < 3; i++ {
		tlb.Lookup(i*pageSize, pageSize, asid)
	}

	stats := tlb.GetStats()
	if stats.Hits != 3 {
		t.Errorf("hits: got %d, want 3", stats.Hits)
	}
	if stats.Misses != 3 {
		t.Errorf("misses: got %d, want 3", stats.Misses)
	}
	if stats.HitRate != 0.5 {
		t.Errorf("hit rate: got %.2f, want 0.50", stats.HitRate)
	}
	_ = stats.String() // Ensure String() doesn't panic
}

func TestTLBHugePage(t *testing.T) {
	tlb := NewL1DTLB()
	const asid = uint16(0)

	virtualAddr := uint64(0x200000) // 2 MB aligned
	physBase := uint64(0x400000)
	offset := uint64(0x1234)

	tlb.Install(virtualAddr, physBase, PageSizeLarge, asid, PermRW)

	physAddr, hit, _ := tlb.Lookup(virtualAddr+offset, PageSizeLarge, asid)
	if !hit {
		t.Fatal("huge page TLB miss after Install")
	}
	expectedPhys := physBase + offset
	if physAddr != expectedPhys {
		t.Errorf("huge page addr: got 0x%X, want 0x%X", physAddr, expectedPhys)
	}
}

// =============================================================================
// TLB Concurrency
// =============================================================================

func TestTLBConcurrentAccess(t *testing.T) {
	tlb := NewL1DTLB()
	const pageSize = PageSizeSmall

	var wg sync.WaitGroup
	for goroutine := 0; goroutine < 8; goroutine++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			asid := uint16(g)
			base := uint64(g) * pageSize * 10
			for i := uint64(0); i < 10; i++ {
				va := base + i*pageSize
				tlb.Install(va, va+0x80000, pageSize, asid, PermRW)
				tlb.Lookup(va, pageSize, asid)
			}
		}(goroutine)
	}
	wg.Wait()
}

// =============================================================================
// PageTable Tests
// =============================================================================

func TestPageTableMapWalk(t *testing.T) {
	pt := NewPageTable(1)
	const pageSize = PageSizeSmall

	virtualAddr := uint64(0x10000)
	physAddr := uint64(0x80000)

	pt.MapPage(virtualAddr, physAddr, pageSize, PermRW)

	pa, present, permErr := pt.Walk(virtualAddr, pageSize)
	if !present {
		t.Fatal("page not present after MapPage")
	}
	if permErr {
		t.Error("unexpected permission error")
	}
	if pa != physAddr {
		t.Errorf("physAddr: got 0x%X, want 0x%X", pa, physAddr)
	}
}

func TestPageTableWalkWithOffset(t *testing.T) {
	pt := NewPageTable(0)
	const pageSize = PageSizeSmall

	const offset = uint64(0xABC)
	virtualAddr := uint64(0x20000) + offset
	physBase := uint64(0xF0000)

	pt.MapPage(virtualAddr, physBase, pageSize, PermRW)

	pa, present, _ := pt.Walk(virtualAddr, pageSize)
	if !present {
		t.Fatal("page not present")
	}
	if pa != physBase+offset {
		t.Errorf("offset preserved: got 0x%X, want 0x%X", pa, physBase+offset)
	}
}

func TestPageTableFault(t *testing.T) {
	pt := NewPageTable(0)
	_, present, _ := pt.Walk(0x50000, PageSizeSmall)
	if present {
		t.Error("unmapped page should not be present")
	}
}

func TestPageTableUnmap(t *testing.T) {
	pt := NewPageTable(0)
	const pageSize = PageSizeSmall
	virtualAddr := uint64(0x30000)
	pt.MapPage(virtualAddr, 0x70000, pageSize, PermRW)
	pt.UnmapPage(virtualAddr, pageSize)

	_, present, _ := pt.Walk(virtualAddr, pageSize)
	if present {
		t.Error("page still present after UnmapPage")
	}
}

// =============================================================================
// MMU Tests
// =============================================================================

func TestMMUTranslateData(t *testing.T) {
	mmu := NewMMU(1)
	const offset = uint64(0x200)
	virtualAddr := uint64(0x1000) + offset
	physBase := uint64(0x9000)

	mmu.MapPage(0x1000, physBase, PermRW)

	pa, cycles, pageFault := mmu.TranslateData(virtualAddr)
	if pageFault {
		t.Fatal("unexpected page fault")
	}
	if pa != physBase+offset {
		t.Errorf("data translate: got 0x%X, want 0x%X", pa, physBase+offset)
	}
	if cycles <= 0 {
		t.Error("cycles should be > 0")
	}
}

func TestMMUTranslateCode(t *testing.T) {
	mmu := NewMMU(2)
	virtualAddr := uint64(0x4000)
	physBase := uint64(0xA000)

	mmu.MapPage(virtualAddr, physBase, PermRX)

	pa, _, pageFault := mmu.TranslateCode(virtualAddr)
	if pageFault {
		t.Fatal("unexpected page fault for code translation")
	}
	if pa != physBase {
		t.Errorf("code translate: got 0x%X, want 0x%X", pa, physBase)
	}
}

func TestMMUPageFault(t *testing.T) {
	mmu := NewMMU(3)

	_, _, pageFault := mmu.TranslateData(0xDEAD0000)
	if !pageFault {
		t.Error("expected page fault for unmapped address")
	}

	stats := mmu.GetStats()
	if stats.PageFaults == 0 {
		t.Error("page fault counter not incremented")
	}
}

func TestMMUTLBCaching(t *testing.T) {
	mmu := NewMMU(4)
	virtualAddr := uint64(0x5000)

	mmu.MapPage(virtualAddr, 0xB000, PermRW)

	// First translation: DTLB miss → page-table walk
	mmu.TranslateData(virtualAddr)

	// Second translation: should hit L1 DTLB
	_, _, pageFault := mmu.TranslateData(virtualAddr)
	if pageFault {
		t.Fatal("unexpected page fault on second translation")
	}

	stats := mmu.GetStats()
	if stats.DTLB.Hits == 0 {
		t.Error("L1 DTLB should have at least one hit after second translation")
	}
	_ = stats.String() // Ensure String() doesn't panic
}

func TestMMUFlushTLBs(t *testing.T) {
	mmu := NewMMU(5)
	virtualAddr := uint64(0x6000)
	mmu.MapPage(virtualAddr, 0xC000, PermRW)

	// Warm the TLBs
	mmu.TranslateData(virtualAddr)
	mmu.TranslateData(virtualAddr)

	// Context switch
	mmu.FlushTLBs()

	// Next access should miss again (but page is still in page table)
	_, _, pageFault := mmu.TranslateData(virtualAddr)
	if pageFault {
		t.Fatal("page fault after flush – page should still be in page table")
	}

	stats := mmu.GetStats()
	if stats.DTLB.Flushes == 0 {
		t.Error("DTLB flush counter not incremented")
	}
}

func TestMMUStats(t *testing.T) {
	mmu := NewMMU(6)
	mmu.MapPage(0x7000, 0xD000, PermRW)
	mmu.MapPage(0x8000, 0xE000, PermRX)

	mmu.TranslateData(0x7000)
	mmu.TranslateCode(0x8000)

	stats := mmu.GetStats()
	if stats.DataTranslations != 1 {
		t.Errorf("DataTranslations: got %d, want 1", stats.DataTranslations)
	}
	if stats.CodeTranslations != 1 {
		t.Errorf("CodeTranslations: got %d, want 1", stats.CodeTranslations)
	}
}

func TestTLBStatStringZeroDiv(t *testing.T) {
	// Empty TLB stats should not panic / produce NaN
	tlb := NewL1DTLB()
	stats := tlb.GetStats()
	if stats.HitRate != 0 {
		t.Errorf("empty TLB hit rate should be 0, got %.2f", stats.HitRate)
	}
	_ = stats.String()
}

func TestNewTLBDefaults(t *testing.T) {
	// Zero-value config should use safe defaults
	tlb := NewTLB(TLBConfig{})
	if tlb.Config.NumSets <= 0 {
		t.Error("NumSets should default to >= 1")
	}
	if tlb.Config.Associativity <= 0 {
		t.Error("Associativity should default to >= 1")
	}
	if tlb.Config.HitLatency <= 0 {
		t.Error("HitLatency should default to >= 1")
	}
	if tlb.Config.MissLatency <= 0 {
		t.Error("MissLatency should default to >= 1")
	}
}
