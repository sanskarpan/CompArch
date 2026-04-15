package memory

import (
	"sync"
	"testing"
)

// =============================================================================
// PrefetchQueue Tests
// =============================================================================

func TestPrefetchQueueEnqueueDequeue(t *testing.T) {
	q := NewPrefetchQueue(4)

	if !q.Enqueue(0x1000, 1) {
		t.Fatal("first enqueue should succeed")
	}
	if !q.Enqueue(0x2000, 3) {
		t.Fatal("second enqueue should succeed")
	}
	if !q.Enqueue(0x3000, 2) {
		t.Fatal("third enqueue should succeed")
	}

	// Dequeue should return highest priority first
	req, ok := q.Dequeue()
	if !ok {
		t.Fatal("dequeue should succeed")
	}
	if req.Addr != 0x2000 {
		t.Errorf("highest priority: got 0x%X, want 0x2000", req.Addr)
	}

	req, ok = q.Dequeue()
	if !ok || req.Addr != 0x3000 {
		t.Errorf("second priority: got 0x%X ok=%v", req.Addr, ok)
	}
}

func TestPrefetchQueueFull(t *testing.T) {
	q := NewPrefetchQueue(2)

	q.Enqueue(0x1000, 1)
	q.Enqueue(0x2000, 2)

	if q.Enqueue(0x3000, 3) {
		t.Error("enqueue to full queue should return false")
	}
	if q.Len() != 2 {
		t.Errorf("len: got %d, want 2", q.Len())
	}
}

func TestPrefetchQueueDuplicate(t *testing.T) {
	q := NewPrefetchQueue(8)
	q.Enqueue(0x1000, 1)

	// Second enqueue of same address should be rejected
	if q.Enqueue(0x1000, 2) {
		t.Error("duplicate enqueue should return false")
	}
	if q.Len() != 1 {
		t.Errorf("len after duplicate: got %d, want 1", q.Len())
	}
}

func TestPrefetchQueueEmptyDequeue(t *testing.T) {
	q := NewPrefetchQueue(4)
	_, ok := q.Dequeue()
	if ok {
		t.Error("dequeue from empty queue should return ok=false")
	}
}

func TestPrefetchQueueLen(t *testing.T) {
	q := NewPrefetchQueue(10)
	for i := 0; i < 5; i++ {
		q.Enqueue(uint64(i)*0x1000, i)
	}
	if q.Len() != 5 {
		t.Errorf("len: got %d, want 5", q.Len())
	}
}

// =============================================================================
// SequentialPrefetcher Tests
// =============================================================================

func TestSequentialPrefetcherOnAccess(t *testing.T) {
	const lineSize = uint64(64)
	p := NewSequentialPrefetcher(lineSize, 4)

	fetches := p.OnAccess(0x1000)
	if len(fetches) != 4 {
		t.Errorf("degree-4 should produce 4 prefetches, got %d", len(fetches))
	}

	// Verify addresses are sequential cache lines
	for i, addr := range fetches {
		expected := uint64(0x1000) + uint64(i+1)*lineSize
		if addr != expected {
			t.Errorf("prefetch[%d]: got 0x%X, want 0x%X", i, addr, expected)
		}
	}
}

func TestSequentialPrefetcherCounter(t *testing.T) {
	const lineSize = uint64(64)
	p := NewSequentialPrefetcher(lineSize, 2)

	p.OnAccess(0x0000)
	p.OnAccess(0x1000)

	stats := p.GetStats()
	if stats.Prefetches < 2 {
		t.Errorf("prefetch count: got %d, want >= 2", stats.Prefetches)
	}
}

func TestSequentialPrefetcherStats(t *testing.T) {
	const lineSize = uint64(64)
	p := NewSequentialPrefetcher(lineSize, 2)

	p.OnAccess(0x0000)
	stats := p.GetStats()

	if stats.Name != "Sequential" {
		t.Errorf("name: got %q, want %q", stats.Name, "Sequential")
	}
	_ = stats.String()
}

func TestSequentialPrefetcherAligned(t *testing.T) {
	const lineSize = uint64(64)
	p := NewSequentialPrefetcher(lineSize, 1)

	// Access in the middle of a cache line
	fetches := p.OnAccess(0x1010) // Not aligned to 64 bytes

	// First prefetch should be the next aligned line
	expected := (uint64(0x1010)/lineSize + 1) * lineSize
	if len(fetches) > 0 && fetches[0] != expected {
		t.Errorf("should prefetch aligned line: got 0x%X, want 0x%X", fetches[0], expected)
	}
}

// =============================================================================
// StridePrefetcher Tests
// =============================================================================

func TestStridePrefetcherDetectsStride(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStridePrefetcher(lineSize, 2, 64)

	const pc = uint64(0x100)
	const stride = int64(256)
	addr := uint64(0x1000)

	// First access: initialises entry
	p.OnAccess(pc, addr)
	// Second access: detects stride, still transient
	p.OnAccess(pc, uint64(int64(addr)+stride))
	// Third access: confirms stride (steady state) → first prefetch
	fetches := p.OnAccess(pc, uint64(int64(addr)+2*stride))

	if len(fetches) == 0 {
		t.Error("stride prefetcher should issue prefetches after stride confirmation")
	}
}

func TestStridePrefetcherIrregularStride(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStridePrefetcher(lineSize, 2, 64)

	const pc = uint64(0x200)
	// Irregular pattern: no consistent stride
	p.OnAccess(pc, 0x1000)
	p.OnAccess(pc, 0x1100)
	fetches := p.OnAccess(pc, 0x1050) // Stride changed

	// Should not issue prefetches in irregular case
	if len(fetches) != 0 {
		t.Errorf("irregular stride should not prefetch, got %d fetches", len(fetches))
	}
}

func TestStridePrefetcherStats(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStridePrefetcher(lineSize, 2, 64)

	const pc = uint64(0x300)
	const stride = int64(128)
	addr := uint64(0x2000)
	for i := 0; i < 5; i++ {
		p.OnAccess(pc, uint64(int64(addr)+int64(i)*stride))
	}

	stats := p.GetStats()
	if stats.Name != "Stride" {
		t.Errorf("name: got %q, want %q", stats.Name, "Stride")
	}
	_ = stats.String()
}

func TestStridePrefetcherNewPC(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStridePrefetcher(lineSize, 2, 64)

	// Fresh PC: first call should return nil (no entry yet)
	result := p.OnAccess(0x400, 0x3000)
	if result != nil {
		t.Error("first access for new PC should return nil")
	}
}

// =============================================================================
// StreamPrefetcher Tests
// =============================================================================

func TestStreamPrefetcherDetectsStream(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStreamPrefetcher(lineSize, 4, 4)

	// Sequential accesses to establish a stream
	addr := uint64(0x5000)
	p.OnAccess(addr)

	// Next line: matches stream
	fetches := p.OnAccess(addr + lineSize)
	if len(fetches) == 0 {
		t.Error("should have prefetched ahead in detected stream")
	}
}

func TestStreamPrefetcherMultipleStreams(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStreamPrefetcher(lineSize, 4, 2)

	// Establish two streams in different regions
	p.OnAccess(0x1000)
	p.OnAccess(0x1000 + lineSize) // stream 1

	p.OnAccess(0x9000)
	fetches := p.OnAccess(0x9000 + lineSize) // stream 2

	if len(fetches) == 0 {
		t.Error("second stream should also trigger prefetches")
	}
}

func TestStreamPrefetcherStats(t *testing.T) {
	const lineSize = uint64(64)
	p := NewStreamPrefetcher(lineSize, 4, 4)

	for i := uint64(0); i < 10; i++ {
		p.OnAccess(i * lineSize)
	}

	stats := p.GetStats()
	if stats.Name != "Stream" {
		t.Errorf("name: got %q, want %q", stats.Name, "Stream")
	}
	_ = stats.String()
}

// =============================================================================
// PrefetcherStats Tests
// =============================================================================

func TestPrefetcherStatsString(t *testing.T) {
	s := PrefetcherStats{
		Name:       "Test",
		Prefetches: 100,
		Useful:     80,
		Useless:    20,
		Accuracy:   0.8,
		QueueDepth: 5,
	}
	str := s.String()
	if len(str) == 0 {
		t.Error("PrefetcherStats.String() should not be empty")
	}
}

func TestPrefetcherStatsZeroAccuracy(t *testing.T) {
	s := PrefetcherStats{
		Name:       "Empty",
		Prefetches: 0,
	}
	str := s.String()
	if len(str) == 0 {
		t.Error("zero-prefetch stats String() should not be empty")
	}
}

// =============================================================================
// PrefetchEngine Tests
// =============================================================================

func TestPrefetchEngineOnAccess(t *testing.T) {
	const lineSize = uint64(64)
	e := NewPrefetchEngine(lineSize)

	addrs := e.OnAccess(0x100, 0x1000)
	// Sequential prefetcher should at least produce some addresses
	if len(addrs) == 0 {
		t.Error("combined engine should issue at least one prefetch on access")
	}

	// All returned addresses should be unique (de-duplicated)
	seen := make(map[uint64]bool)
	for _, a := range addrs {
		if seen[a] {
			t.Errorf("duplicate prefetch address 0x%X in combined engine output", a)
		}
		seen[a] = true
	}
}

func TestPrefetchEngineDisabled(t *testing.T) {
	e := NewPrefetchEngine(64)
	e.Enabled = false

	addrs := e.OnAccess(0x100, 0x1000)
	if len(addrs) != 0 {
		t.Error("disabled prefetch engine should return no addresses")
	}
}

func TestPrefetchEngineGetStats(t *testing.T) {
	e := NewPrefetchEngine(64)
	e.OnAccess(0x100, 0x1000)
	e.OnAccess(0x104, 0x2000)

	stats := e.GetStats()
	if len(stats) != 3 {
		t.Errorf("engine should return 3 sub-prefetcher stats, got %d", len(stats))
	}
	// Check sub-prefetcher names
	names := map[string]bool{}
	for _, s := range stats {
		names[s.Name] = true
	}
	for _, name := range []string{"Sequential", "Stride", "Stream"} {
		if !names[name] {
			t.Errorf("missing stats for %q prefetcher", name)
		}
	}
}

func TestPrefetchEngineStridePattern(t *testing.T) {
	const lineSize = uint64(64)
	e := NewPrefetchEngine(lineSize)
	const stride = int64(256)

	// Train the stride prefetcher with a regular pattern
	pc := uint64(0x1000)
	addr := uint64(0x8000)
	for i := 0; i < 5; i++ {
		e.OnAccess(pc, uint64(int64(addr)+int64(i)*stride))
	}

	stats := e.GetStats()
	var strideStats PrefetcherStats
	for _, s := range stats {
		if s.Name == "Stride" {
			strideStats = s
			break
		}
	}
	if strideStats.Prefetches == 0 {
		t.Error("stride prefetcher should have issued prefetches for regular stride pattern")
	}
}

// =============================================================================
// Concurrency Tests
// =============================================================================

func TestPrefetchQueueConcurrent(t *testing.T) {
	q := NewPrefetchQueue(1000)
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				q.Enqueue(uint64(base*1000+j)*64, j)
			}
		}(i)
	}
	wg.Wait()

	if q.Len() == 0 {
		t.Error("queue should have entries after concurrent enqueues")
	}
}

func TestSequentialPrefetcherConcurrent(t *testing.T) {
	p := NewSequentialPrefetcher(64, 4)
	var wg sync.WaitGroup

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				p.OnAccess(uint64(base*0x10000 + j*64))
			}
		}(i)
	}
	wg.Wait()

	stats := p.GetStats()
	if stats.Prefetches == 0 {
		t.Error("should have prefetches after concurrent accesses")
	}
}
