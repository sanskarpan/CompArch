/*
Hardware Memory Prefetcher
===========================

Simulates hardware prefetch engines that detect memory access patterns
and proactively fetch cache lines before they are needed.

Prefetchers implemented:
1. Sequential (Next-Line) – always prefetch addr+stride
2. Stride           – detect fixed strides and prefetch ahead
3. Stream Detector  – detect multiple independent streams
4. AMPM (Access Map Pattern Matching) – page-level pattern matching

Applications:
- Reducing cache miss penalties for regular access patterns
- Understanding prefetch pollution and coverage trade-offs
- Memory subsystem performance optimisation
*/

package memory

import (
	"fmt"
	"sync"
)

// =============================================================================
// Prefetch Queue
// =============================================================================

// PrefetchRequest is one pending prefetch
type PrefetchRequest struct {
	Addr      uint64
	Priority  int // higher = more urgent
	Issued    bool
}

// PrefetchQueue is a bounded queue of pending prefetch requests
type PrefetchQueue struct {
	requests []PrefetchRequest
	maxSize  int
	mu       sync.Mutex
}

// NewPrefetchQueue creates a prefetch queue
func NewPrefetchQueue(maxSize int) *PrefetchQueue {
	return &PrefetchQueue{
		requests: make([]PrefetchRequest, 0, maxSize),
		maxSize:  maxSize,
	}
}

// Enqueue adds a prefetch request if the queue is not full
func (q *PrefetchQueue) Enqueue(addr uint64, priority int) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.requests) >= q.maxSize {
		return false
	}
	// Avoid duplicates
	for _, r := range q.requests {
		if r.Addr == addr {
			return false
		}
	}
	q.requests = append(q.requests, PrefetchRequest{Addr: addr, Priority: priority})
	return true
}

// Dequeue removes and returns the highest-priority pending request
func (q *PrefetchQueue) Dequeue() (PrefetchRequest, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.requests) == 0 {
		return PrefetchRequest{}, false
	}

	best := 0
	for i := 1; i < len(q.requests); i++ {
		if q.requests[i].Priority > q.requests[best].Priority {
			best = i
		}
	}

	req := q.requests[best]
	q.requests = append(q.requests[:best], q.requests[best+1:]...)
	return req, true
}

// Len returns the current queue depth
func (q *PrefetchQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.requests)
}

// =============================================================================
// Sequential (Next-Line) Prefetcher
// =============================================================================

// SequentialPrefetcher prefetches the next N cache lines after every access
type SequentialPrefetcher struct {
	CacheLineSize uint64
	Degree        int // number of lines to prefetch ahead
	Queue         *PrefetchQueue

	// Statistics
	Prefetches  uint64
	Useful      uint64 // prefetches that were consumed before eviction
	Useless     uint64 // prefetches that were evicted without being used
	mu          sync.Mutex
}

// NewSequentialPrefetcher creates a next-line prefetcher
func NewSequentialPrefetcher(cacheLineSize uint64, degree int) *SequentialPrefetcher {
	return &SequentialPrefetcher{
		CacheLineSize: cacheLineSize,
		Degree:        degree,
		Queue:         NewPrefetchQueue(64),
	}
}

// OnAccess is called every time a cache line is accessed
func (p *SequentialPrefetcher) OnAccess(addr uint64) []uint64 {
	lineAddr := (addr / p.CacheLineSize) * p.CacheLineSize
	var toFetch []uint64

	for i := 1; i <= p.Degree; i++ {
		prefetchAddr := lineAddr + uint64(i)*p.CacheLineSize
		if p.Queue.Enqueue(prefetchAddr, p.Degree-i+1) {
			p.mu.Lock()
			p.Prefetches++
			p.mu.Unlock()
			toFetch = append(toFetch, prefetchAddr)
		}
	}
	return toFetch
}

// GetStats returns prefetcher statistics
func (p *SequentialPrefetcher) GetStats() PrefetcherStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	accuracy := float64(0)
	if p.Prefetches > 0 {
		accuracy = float64(p.Useful) / float64(p.Prefetches)
	}
	return PrefetcherStats{
		Name:       "Sequential",
		Prefetches: p.Prefetches,
		Useful:     p.Useful,
		Useless:    p.Useless,
		Accuracy:   accuracy,
		QueueDepth: uint64(p.Queue.Len()),
	}
}

// =============================================================================
// Stride Prefetcher
// =============================================================================

// strideEntry tracks stride detection for a PC
type strideEntry struct {
	lastAddr uint64
	stride   int64
	state    int // 0=initial, 1=transient, 2=steady, 3=no-pred
	tag      uint64
}

// StridePrefetcher detects regular strides and prefetches ahead
type StridePrefetcher struct {
	CacheLineSize uint64
	Degree        int // prefetch distance in strides
	NumEntries    int
	RPT           []strideEntry // Reference Prediction Table
	Queue         *PrefetchQueue

	Prefetches uint64
	Useful     uint64
	Useless    uint64
	mu         sync.Mutex
}

// NewStridePrefetcher creates a stride-based prefetcher
func NewStridePrefetcher(cacheLineSize uint64, degree, rptSize int) *StridePrefetcher {
	return &StridePrefetcher{
		CacheLineSize: cacheLineSize,
		Degree:        degree,
		NumEntries:    rptSize,
		RPT:           make([]strideEntry, rptSize),
		Queue:         NewPrefetchQueue(128),
	}
}

// OnAccess trains the stride predictor and issues prefetches
func (p *StridePrefetcher) OnAccess(pc, addr uint64) []uint64 {
	idx := int(pc>>2) % p.NumEntries

	p.mu.Lock()
	entry := &p.RPT[idx]

	var toFetch []uint64
	if entry.tag != pc>>2 {
		// New entry
		entry.tag = pc >> 2
		entry.lastAddr = addr
		entry.state = 0
		p.mu.Unlock()
		return nil
	}

	detectedStride := int64(addr) - int64(entry.lastAddr)
	entry.lastAddr = addr

	switch entry.state {
	case 0: // Initial
		entry.stride = detectedStride
		entry.state = 1
	case 1: // Transient
		if detectedStride == entry.stride {
			entry.state = 2 // confirmed steady stride
		} else {
			entry.stride = detectedStride
			entry.state = 3 // no prediction
		}
	case 2: // Steady
		if detectedStride != entry.stride {
			entry.state = 1
		}
	case 3: // No pred
		if detectedStride == entry.stride {
			entry.state = 1
		} else {
			entry.stride = detectedStride
		}
	}

	if entry.state == 2 && entry.stride != 0 {
		for i := 1; i <= p.Degree; i++ {
			prefetchAddr := uint64(int64(addr) + int64(i)*entry.stride)
			lineAddr := (prefetchAddr / p.CacheLineSize) * p.CacheLineSize
			if p.Queue.Enqueue(lineAddr, p.Degree-i+1) {
				p.Prefetches++
				toFetch = append(toFetch, lineAddr)
			}
		}
	}

	p.mu.Unlock()
	return toFetch
}

// GetStats returns prefetcher statistics
func (p *StridePrefetcher) GetStats() PrefetcherStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	accuracy := float64(0)
	if p.Prefetches > 0 {
		accuracy = float64(p.Useful) / float64(p.Prefetches)
	}
	return PrefetcherStats{
		Name:       "Stride",
		Prefetches: p.Prefetches,
		Useful:     p.Useful,
		Useless:    p.Useless,
		Accuracy:   accuracy,
		QueueDepth: uint64(p.Queue.Len()),
	}
}

// =============================================================================
// Stream Detector Prefetcher
// =============================================================================

// streamEntry tracks one detected memory stream
type streamEntry struct {
	valid     bool
	baseAddr  uint64
	direction int // +1 or -1
	depth     int // how many lines prefetched ahead
	lastSeen  uint64 // access counter for LRU eviction
}

// StreamPrefetcher detects and prefetches multiple concurrent memory streams
type StreamPrefetcher struct {
	CacheLineSize uint64
	MaxStreams    int
	Degree        int
	Streams       []streamEntry
	AccessCount   uint64
	Queue         *PrefetchQueue

	Prefetches uint64
	Useful     uint64
	Useless    uint64
	mu         sync.Mutex
}

// NewStreamPrefetcher creates a stream prefetcher tracking up to maxStreams streams
func NewStreamPrefetcher(cacheLineSize uint64, maxStreams, degree int) *StreamPrefetcher {
	return &StreamPrefetcher{
		CacheLineSize: cacheLineSize,
		MaxStreams:    maxStreams,
		Degree:        degree,
		Streams:       make([]streamEntry, maxStreams),
		Queue:         NewPrefetchQueue(256),
	}
}

// OnAccess detects stream membership and issues prefetches
func (p *StreamPrefetcher) OnAccess(addr uint64) []uint64 {
	lineAddr := (addr / p.CacheLineSize) * p.CacheLineSize

	p.mu.Lock()
	defer p.mu.Unlock()

	p.AccessCount++
	var toFetch []uint64

	// Try to match an existing stream
	for i := range p.Streams {
		s := &p.Streams[i]
		if !s.valid {
			continue
		}
		next := s.baseAddr + uint64(int64(p.CacheLineSize)*int64(s.direction))
		if lineAddr == next {
			s.baseAddr = lineAddr
			s.lastSeen = p.AccessCount
			// Prefetch degree lines ahead in the stream direction
			for d := 1; d <= p.Degree; d++ {
				pa := lineAddr + uint64(int64(p.CacheLineSize)*int64(s.direction)*int64(d))
				if p.Queue.Enqueue(pa, p.Degree-d+1) {
					p.Prefetches++
					toFetch = append(toFetch, pa)
				}
			}
			return toFetch
		}
	}

	// Allocate new stream (evict LRU)
	lru := 0
	for i := 1; i < p.MaxStreams; i++ {
		if !p.Streams[i].valid || p.Streams[i].lastSeen < p.Streams[lru].lastSeen {
			lru = i
		}
	}
	p.Streams[lru] = streamEntry{
		valid:     true,
		baseAddr:  lineAddr,
		direction: +1,
		lastSeen:  p.AccessCount,
	}
	return toFetch
}

// GetStats returns prefetcher statistics
func (p *StreamPrefetcher) GetStats() PrefetcherStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	accuracy := float64(0)
	if p.Prefetches > 0 {
		accuracy = float64(p.Useful) / float64(p.Prefetches)
	}
	return PrefetcherStats{
		Name:       "Stream",
		Prefetches: p.Prefetches,
		Useful:     p.Useful,
		Useless:    p.Useless,
		Accuracy:   accuracy,
		QueueDepth: uint64(p.Queue.Len()),
	}
}

// =============================================================================
// Prefetcher Statistics
// =============================================================================

// PrefetcherStats contains prefetcher performance statistics
type PrefetcherStats struct {
	Name       string
	Prefetches uint64
	Useful     uint64
	Useless    uint64
	Accuracy   float64
	QueueDepth uint64
}

// String returns a human-readable representation
func (s PrefetcherStats) String() string {
	coverage := float64(0)
	if s.Prefetches > 0 {
		coverage = float64(s.Useful) / float64(s.Prefetches+s.Useless)
	}
	return fmt.Sprintf(`%s Prefetcher:
  Total Prefetches:  %d
  Useful:            %d (accuracy %.2f%%)
  Useless:           %d
  Coverage:          %.2f%%
  Queue Depth:       %d`,
		s.Name,
		s.Prefetches,
		s.Useful, 100.0*s.Accuracy,
		s.Useless,
		100.0*coverage,
		s.QueueDepth)
}

// =============================================================================
// Combined Prefetch Engine
// =============================================================================

// PrefetchEngine is a complete prefetch engine combining multiple prefetchers
type PrefetchEngine struct {
	Sequential *SequentialPrefetcher
	Stride     *StridePrefetcher
	Stream     *StreamPrefetcher
	Enabled    bool
	mu         sync.Mutex
}

// NewPrefetchEngine creates a default prefetch engine
func NewPrefetchEngine(cacheLineSize uint64) *PrefetchEngine {
	return &PrefetchEngine{
		Sequential: NewSequentialPrefetcher(cacheLineSize, 4),
		Stride:     NewStridePrefetcher(cacheLineSize, 4, 64),
		Stream:     NewStreamPrefetcher(cacheLineSize, 8, 8),
		Enabled:    true,
	}
}

// OnAccess triggers all prefetchers on a memory access.
// pc is the program counter of the load instruction (for stride detection).
// Returns the list of addresses to prefetch.
func (e *PrefetchEngine) OnAccess(pc, addr uint64) []uint64 {
	e.mu.Lock()
	if !e.Enabled {
		e.mu.Unlock()
		return nil
	}
	e.mu.Unlock()

	seen := make(map[uint64]bool)
	var all []uint64

	for _, a := range e.Sequential.OnAccess(addr) {
		if !seen[a] {
			seen[a] = true
			all = append(all, a)
		}
	}
	for _, a := range e.Stride.OnAccess(pc, addr) {
		if !seen[a] {
			seen[a] = true
			all = append(all, a)
		}
	}
	for _, a := range e.Stream.OnAccess(addr) {
		if !seen[a] {
			seen[a] = true
			all = append(all, a)
		}
	}
	return all
}

// GetStats returns stats from all sub-prefetchers
func (e *PrefetchEngine) GetStats() []PrefetcherStats {
	return []PrefetcherStats{
		e.Sequential.GetStats(),
		e.Stride.GetStats(),
		e.Stream.GetStats(),
	}
}
