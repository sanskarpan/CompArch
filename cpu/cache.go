/*
Cache Hierarchy
===============

Multi-level cache implementation with various replacement policies.

Features:
- L1, L2, L3 cache levels
- Cache replacement policies (LRU, FIFO, Random, LFU)
- Cache write policies (write-through, write-back)
- Cache performance metrics (hit rate, miss rate)
- Set-associative and fully-associative caches

Applications:
- Understanding cache behavior
- Memory access optimization
- Performance analysis
*/

package cpu

import (
	"container/list"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// =============================================================================
// Cache Line and Set
// =============================================================================

// CacheLine represents a single cache line
type CacheLine struct {
	Valid       bool   // Valid bit
	Dirty       bool   // Dirty bit (for write-back)
	Tag         uint64 // Tag
	Data        []byte // Cache line data
	LastAccess  uint64 // For LRU
	AccessCount uint64 // For LFU
}

// CacheSet represents a set in a set-associative cache
type CacheSet struct {
	Lines []*CacheLine
	LRU   *list.List     // LRU queue
	Mutex sync.RWMutex   // For thread safety
}

// =============================================================================
// Cache Replacement Policies
// =============================================================================

// ReplacementPolicy defines the interface for cache replacement
type ReplacementPolicy interface {
	SelectVictim(set *CacheSet) int
	OnAccess(set *CacheSet, way int)
}

// LRUPolicy implements Least Recently Used
type LRUPolicy struct {
	timestamp uint64
	mu        sync.Mutex
}

func (p *LRUPolicy) SelectVictim(set *CacheSet) int {
	minTime := uint64(^uint64(0))
	victim := 0

	for i, line := range set.Lines {
		if !line.Valid {
			return i
		}
		if line.LastAccess < minTime {
			minTime = line.LastAccess
			victim = i
		}
	}

	return victim
}

func (p *LRUPolicy) OnAccess(set *CacheSet, way int) {
	p.mu.Lock()
	p.timestamp++
	set.Lines[way].LastAccess = p.timestamp
	p.mu.Unlock()
}

// FIFOPolicy implements First-In-First-Out
type FIFOPolicy struct {
	timestamp uint64
	mu        sync.Mutex
}

func (p *FIFOPolicy) SelectVictim(set *CacheSet) int {
	minTime := uint64(^uint64(0))
	victim := 0

	for i, line := range set.Lines {
		if !line.Valid {
			return i
		}
		if line.LastAccess < minTime {
			minTime = line.LastAccess
			victim = i
		}
	}

	return victim
}

func (p *FIFOPolicy) OnAccess(set *CacheSet, way int) {
	// Only set timestamp on first access (insertion)
	if !set.Lines[way].Valid {
		p.mu.Lock()
		p.timestamp++
		set.Lines[way].LastAccess = p.timestamp
		p.mu.Unlock()
	}
}

// RandomPolicy implements random replacement
type RandomPolicy struct {
	rng *rand.Rand
}

func NewRandomPolicy() *RandomPolicy {
	return &RandomPolicy{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (p *RandomPolicy) SelectVictim(set *CacheSet) int {
	// First check for invalid lines
	for i, line := range set.Lines {
		if !line.Valid {
			return i
		}
	}

	return p.rng.Intn(len(set.Lines))
}

func (p *RandomPolicy) OnAccess(set *CacheSet, way int) {
	// No state update needed
}

// LFUPolicy implements Least Frequently Used
type LFUPolicy struct{}

func (p *LFUPolicy) SelectVictim(set *CacheSet) int {
	minCount := uint64(^uint64(0))
	victim := 0

	for i, line := range set.Lines {
		if !line.Valid {
			return i
		}
		if line.AccessCount < minCount {
			minCount = line.AccessCount
			victim = i
		}
	}

	return victim
}

func (p *LFUPolicy) OnAccess(set *CacheSet, way int) {
	set.Lines[way].AccessCount++
}

// =============================================================================
// Cache
// =============================================================================

// CacheConfig contains cache configuration
type CacheConfig struct {
	Size         uint64 // Total cache size in bytes
	LineSize     uint64 // Cache line size in bytes
	Associativity uint64 // Ways per set (1 = direct-mapped, Size/LineSize = fully associative)
	WritePolicy  string // "write-through" or "write-back"
	Policy       ReplacementPolicy
}

// Cache represents a cache level
type Cache struct {
	Config     CacheConfig
	Sets       []CacheSet
	NumSets    uint64
	NextLevel  *Cache // Next level cache (L1 -> L2 -> L3)

	// Statistics
	Hits       uint64
	Misses     uint64
	Writes     uint64
	Writebacks uint64
	mu         sync.RWMutex
}

// NewCache creates a new cache
func NewCache(config CacheConfig) *Cache {
	numSets := config.Size / (config.LineSize * config.Associativity)

	cache := &Cache{
		Config:  config,
		NumSets: numSets,
		Sets:    make([]CacheSet, numSets),
	}

	// Initialize sets
	for i := range cache.Sets {
		cache.Sets[i].Lines = make([]*CacheLine, config.Associativity)
		cache.Sets[i].LRU = list.New()

		for j := range cache.Sets[i].Lines {
			cache.Sets[i].Lines[j] = &CacheLine{
				Data: make([]byte, config.LineSize),
			}
		}
	}

	return cache
}

// parseAddress parses an address into tag, set index, and offset
func (c *Cache) parseAddress(addr uint64) (tag uint64, setIndex uint64, offset uint64) {
	offsetBits := uint64(0)
	temp := c.Config.LineSize
	for temp > 1 {
		offsetBits++
		temp >>= 1
	}

	setBits := uint64(0)
	temp = c.NumSets
	for temp > 1 {
		setBits++
		temp >>= 1
	}

	offset = addr & ((1 << offsetBits) - 1)
	setIndex = (addr >> offsetBits) & ((1 << setBits) - 1)
	tag = addr >> (offsetBits + setBits)

	return
}

// Read reads data from the cache
func (c *Cache) Read(addr uint64, size int) ([]byte, bool) {
	tag, setIndex, offset := c.parseAddress(addr)

	c.mu.Lock()
	defer c.mu.Unlock()

	set := &c.Sets[setIndex]
	set.Mutex.Lock()
	defer set.Mutex.Unlock()

	// Check for hit
	for way, line := range set.Lines {
		if line.Valid && line.Tag == tag {
			// Cache hit
			c.Hits++
			c.Config.Policy.OnAccess(set, way)

			// Return data
			data := make([]byte, size)
			copy(data, line.Data[offset:offset+uint64(size)])
			return data, true
		}
	}

	// Cache miss
	c.Misses++

	// Try next level cache
	var data []byte
	if c.NextLevel != nil {
		data, _ = c.NextLevel.Read(addr, size)
	} else {
		// Simulate main memory access
		data = make([]byte, size)
		// In real system, would fetch from memory
	}

	// Allocate cache line
	victim := c.Config.Policy.SelectVictim(set)
	line := set.Lines[victim]

	// Writeback if dirty
	if line.Valid && line.Dirty && c.Config.WritePolicy == "write-back" {
		c.Writebacks++
		// In real system, would write to next level
	}

	// Load new line
	line.Valid = true
	line.Dirty = false
	line.Tag = tag
	copy(line.Data, data)
	c.Config.Policy.OnAccess(set, victim)

	result := make([]byte, size)
	copy(result, line.Data[offset:offset+uint64(size)])
	return result, false
}

// Write writes data to the cache
func (c *Cache) Write(addr uint64, data []byte) bool {
	tag, setIndex, offset := c.parseAddress(addr)

	c.mu.Lock()
	defer c.mu.Unlock()
	c.Writes++

	set := &c.Sets[setIndex]
	set.Mutex.Lock()
	defer set.Mutex.Unlock()

	// Check for hit
	for way, line := range set.Lines {
		if line.Valid && line.Tag == tag {
			// Cache hit
			c.Hits++
			copy(line.Data[offset:], data)

			if c.Config.WritePolicy == "write-back" {
				line.Dirty = true
			} else {
				// Write-through: write to next level
				if c.NextLevel != nil {
					c.NextLevel.Write(addr, data)
				}
			}

			c.Config.Policy.OnAccess(set, way)
			return true
		}
	}

	// Cache miss
	c.Misses++

	if c.Config.WritePolicy == "write-back" {
		// Allocate on write
		victim := c.Config.Policy.SelectVictim(set)
		line := set.Lines[victim]

		// Writeback if dirty
		if line.Valid && line.Dirty {
			c.Writebacks++
		}

		// Load new line
		line.Valid = true
		line.Dirty = true
		line.Tag = tag
		copy(line.Data[offset:], data)
		c.Config.Policy.OnAccess(set, victim)
	} else {
		// Write-through: write to next level
		if c.NextLevel != nil {
			c.NextLevel.Write(addr, data)
		}
	}

	return false
}

// Flush flushes all dirty lines to next level
func (c *Cache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, set := range c.Sets {
		set.Mutex.Lock()
		for _, line := range set.Lines {
			if line.Valid && line.Dirty {
				c.Writebacks++
				line.Dirty = false
			}
		}
		set.Mutex.Unlock()
	}
}

// GetHitRate returns the cache hit rate
func (c *Cache) GetHitRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.Hits + c.Misses
	if total == 0 {
		return 0
	}
	return float64(c.Hits) / float64(total)
}

// GetStats returns cache statistics
func (c *Cache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.Hits + c.Misses
	return CacheStats{
		Hits:       c.Hits,
		Misses:     c.Misses,
		Accesses:   total,
		HitRate:    c.GetHitRate(),
		MissRate:   1.0 - c.GetHitRate(),
		Writes:     c.Writes,
		Writebacks: c.Writebacks,
	}
}

// Reset resets cache statistics
func (c *Cache) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Hits = 0
	c.Misses = 0
	c.Writes = 0
	c.Writebacks = 0

	for i := range c.Sets {
		c.Sets[i].Mutex.Lock()
		for j := range c.Sets[i].Lines {
			c.Sets[i].Lines[j].Valid = false
			c.Sets[i].Lines[j].Dirty = false
			c.Sets[i].Lines[j].AccessCount = 0
			c.Sets[i].Lines[j].LastAccess = 0
		}
		c.Sets[i].Mutex.Unlock()
	}
}

// CacheStats contains cache performance statistics
type CacheStats struct {
	Hits       uint64
	Misses     uint64
	Accesses   uint64
	HitRate    float64
	MissRate   float64
	Writes     uint64
	Writebacks uint64
}

// String returns a string representation of the stats
func (s CacheStats) String() string {
	return fmt.Sprintf(`Cache Statistics:
  Accesses: %d
  Hits: %d (%.2f%%)
  Misses: %d (%.2f%%)
  Writes: %d
  Writebacks: %d`,
		s.Accesses,
		s.Hits, 100.0*s.HitRate,
		s.Misses, 100.0*s.MissRate,
		s.Writes,
		s.Writebacks)
}

// =============================================================================
// Memory Hierarchy
// =============================================================================

// MemoryHierarchy represents a complete cache hierarchy
type MemoryHierarchy struct {
	L1 *Cache
	L2 *Cache
	L3 *Cache
}

// NewMemoryHierarchy creates a typical 3-level cache hierarchy
func NewMemoryHierarchy() *MemoryHierarchy {
	// L1 Cache: 32KB, 64B lines, 8-way, write-back, LRU
	l1 := NewCache(CacheConfig{
		Size:          32 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	// L2 Cache: 256KB, 64B lines, 8-way, write-back, LRU
	l2 := NewCache(CacheConfig{
		Size:          256 * 1024,
		LineSize:      64,
		Associativity: 8,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	// L3 Cache: 8MB, 64B lines, 16-way, write-back, LRU
	l3 := NewCache(CacheConfig{
		Size:          8 * 1024 * 1024,
		LineSize:      64,
		Associativity: 16,
		WritePolicy:   "write-back",
		Policy:        &LRUPolicy{},
	})

	// Connect the hierarchy
	l1.NextLevel = l2
	l2.NextLevel = l3

	return &MemoryHierarchy{
		L1: l1,
		L2: l2,
		L3: l3,
	}
}

// Read reads from the memory hierarchy
func (mh *MemoryHierarchy) Read(addr uint64, size int) []byte {
	data, _ := mh.L1.Read(addr, size)
	return data
}

// Write writes to the memory hierarchy
func (mh *MemoryHierarchy) Write(addr uint64, data []byte) {
	mh.L1.Write(addr, data)
}

// GetStats returns statistics for all cache levels
func (mh *MemoryHierarchy) GetStats() map[string]CacheStats {
	return map[string]CacheStats{
		"L1": mh.L1.GetStats(),
		"L2": mh.L2.GetStats(),
		"L3": mh.L3.GetStats(),
	}
}
