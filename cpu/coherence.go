/*
Cache Coherence
===============

MESI protocol implementation for multi-core cache coherence.

States:
- Modified (M): Cache line is dirty and exclusive to this cache
- Exclusive (E): Cache line is clean and exclusive to this cache
- Shared (S): Cache line is clean and may be in other caches
- Invalid (I): Cache line is not valid

Applications:
- Multi-core processor cache coherence
- Understanding shared memory consistency
- Performance analysis of multi-core systems
*/

package cpu

import (
	"fmt"
	"sync"
)

// =============================================================================
// MESI Protocol States
// =============================================================================

// MESIState represents the MESI coherence state
type MESIState byte

const (
	Invalid   MESIState = 0 // Cache line is invalid
	Shared    MESIState = 1 // Cache line is shared (read-only)
	Exclusive MESIState = 2 // Cache line is exclusive (clean)
	Modified  MESIState = 3 // Cache line is modified (dirty)
)

func (s MESIState) String() string {
	switch s {
	case Invalid:
		return "Invalid"
	case Shared:
		return "Shared"
	case Exclusive:
		return "Exclusive"
	case Modified:
		return "Modified"
	default:
		return "Unknown"
	}
}

// =============================================================================
// Coherence Messages
// =============================================================================

// CoherenceMessage represents a message on the coherence bus
type CoherenceMessage struct {
	Type    MessageType
	Address uint64
	CoreID  int
	Data    []byte
}

// MessageType represents the type of coherence message
type MessageType byte

const (
	MsgRead         MessageType = 1  // Read request
	MsgReadX        MessageType = 2  // Read exclusive request
	MsgUpgrade      MessageType = 3  // Upgrade to modified
	MsgWriteback    MessageType = 4  // Writeback data
	MsgInvalidate   MessageType = 5  // Invalidate request
	MsgFlush        MessageType = 6  // Flush cache line
	MsgDataResponse MessageType = 10 // Data response
	MsgAck          MessageType = 11 // Acknowledgment
)

// =============================================================================
// Coherent Cache Line
// =============================================================================

// CoherentCacheLine represents a cache line with MESI state
type CoherentCacheLine struct {
	Valid bool
	Tag   uint64
	State MESIState
	Data  []byte
}

// =============================================================================
// Coherent Cache
// =============================================================================

// CoherentCache represents a cache with coherence protocol
type CoherentCache struct {
	ID       int
	Lines    map[uint64]*CoherentCacheLine
	LineSize uint64
	Bus      *CoherenceBus
	mu       sync.RWMutex

	// Statistics
	Reads            uint64
	Writes           uint64
	Invalidations    uint64
	Downgrades       uint64
	BusTransactions  uint64
}

// NewCoherentCache creates a new coherent cache
func NewCoherentCache(id int, lineSize uint64, bus *CoherenceBus) *CoherentCache {
	return &CoherentCache{
		ID:       id,
		Lines:    make(map[uint64]*CoherentCacheLine),
		LineSize: lineSize,
		Bus:      bus,
	}
}

// Read reads data from address
func (c *CoherentCache) Read(addr uint64) ([]byte, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Reads++
	lineAddr := addr / c.LineSize

	line, exists := c.Lines[lineAddr]

	// Cache hit in M, E, or S state
	if exists && line.Valid && line.State != Invalid {
		return line.Data, true
	}

	// Cache miss or invalid - need to request from bus
	c.BusTransactions++
	msg := CoherenceMessage{
		Type:    MsgRead,
		Address: lineAddr,
		CoreID:  c.ID,
	}

	// Send read request on bus
	responses := c.Bus.SendMessage(msg)

	// Check if any cache has the line in M state
	var data []byte
	sharedCopy := false

	for _, resp := range responses {
		if resp.Type == MsgDataResponse {
			data = resp.Data
			sharedCopy = true
		}
	}

	// If no cache provided data, fetch from memory
	if data == nil {
		data = c.Bus.ReadMemory(lineAddr)
	}

	// Allocate line in appropriate state
	line = &CoherentCacheLine{
		Valid: true,
		Tag:   lineAddr,
		Data:  make([]byte, c.LineSize),
	}
	copy(line.Data, data)

	if sharedCopy {
		line.State = Shared
	} else {
		line.State = Exclusive
	}

	c.Lines[lineAddr] = line
	return line.Data, false
}

// Write writes data to address
func (c *CoherentCache) Write(addr uint64, data []byte) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.Writes++
	lineAddr := addr / c.LineSize

	line, exists := c.Lines[lineAddr]

	if exists && line.Valid {
		switch line.State {
		case Modified:
			// Already have exclusive ownership
			copy(line.Data, data)
			return true

		case Exclusive:
			// Transition to Modified
			line.State = Modified
			copy(line.Data, data)
			return true

		case Shared:
			// Need to upgrade to Modified - invalidate other copies
			c.BusTransactions++
			c.Downgrades++
			msg := CoherenceMessage{
				Type:    MsgUpgrade,
				Address: lineAddr,
				CoreID:  c.ID,
			}
			c.Bus.SendMessage(msg)
			line.State = Modified
			copy(line.Data, data)
			return true

		case Invalid:
			// Fall through to cache miss
		}
	}

	// Cache miss - need read exclusive
	c.BusTransactions++
	msg := CoherenceMessage{
		Type:    MsgReadX,
		Address: lineAddr,
		CoreID:  c.ID,
	}

	responses := c.Bus.SendMessage(msg)

	// Get data if available
	var lineData []byte
	for _, resp := range responses {
		if resp.Type == MsgDataResponse {
			lineData = resp.Data
		}
	}

	if lineData == nil {
		lineData = c.Bus.ReadMemory(lineAddr)
	}

	// Allocate line in Modified state
	line = &CoherentCacheLine{
		Valid: true,
		Tag:   lineAddr,
		State: Modified,
		Data:  make([]byte, c.LineSize),
	}
	copy(line.Data, data)
	c.Lines[lineAddr] = line

	return false
}

// HandleBusMessage handles coherence messages from the bus
func (c *CoherentCache) HandleBusMessage(msg CoherenceMessage) *CoherenceMessage {
	// Don't respond to own messages
	if msg.CoreID == c.ID {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	line, exists := c.Lines[msg.Address]
	if !exists || !line.Valid || line.State == Invalid {
		return nil
	}

	switch msg.Type {
	case MsgRead:
		// Another cache is reading
		switch line.State {
		case Modified:
			// Provide data and downgrade to Shared
			c.Downgrades++
			response := &CoherenceMessage{
				Type:    MsgDataResponse,
				Address: msg.Address,
				CoreID:  c.ID,
				Data:    make([]byte, len(line.Data)),
			}
			copy(response.Data, line.Data)
			line.State = Shared
			return response

		case Exclusive:
			// Downgrade to Shared
			c.Downgrades++
			line.State = Shared
			return nil

		case Shared:
			// Already shared, no action needed
			return nil
		}

	case MsgReadX:
		// Another cache is requesting exclusive access
		switch line.State {
		case Modified:
			// Provide data and invalidate
			c.Invalidations++
			response := &CoherenceMessage{
				Type:    MsgDataResponse,
				Address: msg.Address,
				CoreID:  c.ID,
				Data:    make([]byte, len(line.Data)),
			}
			copy(response.Data, line.Data)
			line.State = Invalid
			return response

		case Exclusive, Shared:
			// Invalidate
			c.Invalidations++
			line.State = Invalid
			return nil
		}

	case MsgUpgrade:
		// Another cache is upgrading from Shared to Modified
		if line.State == Shared {
			c.Invalidations++
			line.State = Invalid
		}
		return nil

	case MsgInvalidate:
		// Explicit invalidation
		c.Invalidations++
		line.State = Invalid
		return nil
	}

	return nil
}

// Flush flushes dirty lines to memory
func (c *CoherentCache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for addr, line := range c.Lines {
		if line.State == Modified {
			c.Bus.WriteMemory(addr, line.Data)
			line.State = Exclusive
		}
	}
}

// GetStats returns cache coherence statistics
func (c *CoherentCache) GetStats() CoherenceStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return CoherenceStats{
		Reads:           c.Reads,
		Writes:          c.Writes,
		Invalidations:   c.Invalidations,
		Downgrades:      c.Downgrades,
		BusTransactions: c.BusTransactions,
	}
}

// CoherenceStats contains coherence statistics
type CoherenceStats struct {
	Reads           uint64
	Writes          uint64
	Invalidations   uint64
	Downgrades      uint64
	BusTransactions uint64
}

// String returns a string representation of the stats
func (s CoherenceStats) String() string {
	return fmt.Sprintf(`Coherence Statistics:
  Reads: %d
  Writes: %d
  Invalidations: %d
  Downgrades: %d
  Bus Transactions: %d`,
		s.Reads,
		s.Writes,
		s.Invalidations,
		s.Downgrades,
		s.BusTransactions)
}

// =============================================================================
// Coherence Bus
// =============================================================================

// CoherenceBus represents the system bus for coherence traffic
type CoherenceBus struct {
	Caches  []*CoherentCache
	Memory  map[uint64][]byte
	mu      sync.Mutex

	// Statistics
	Transactions uint64
	Broadcasts   uint64
}

// NewCoherenceBus creates a new coherence bus
func NewCoherenceBus() *CoherenceBus {
	return &CoherenceBus{
		Memory: make(map[uint64][]byte),
	}
}

// RegisterCache registers a cache with the bus
func (b *CoherenceBus) RegisterCache(cache *CoherentCache) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Caches = append(b.Caches, cache)
}

// SendMessage broadcasts a message to all caches
func (b *CoherenceBus) SendMessage(msg CoherenceMessage) []*CoherenceMessage {
	b.mu.Lock()
	b.Transactions++
	b.Broadcasts++
	b.mu.Unlock()

	var responses []*CoherenceMessage

	for _, cache := range b.Caches {
		if resp := cache.HandleBusMessage(msg); resp != nil {
			responses = append(responses, resp)
		}
	}

	return responses
}

// ReadMemory reads from main memory
func (b *CoherenceBus) ReadMemory(addr uint64) []byte {
	b.mu.Lock()
	defer b.mu.Unlock()

	data, exists := b.Memory[addr]
	if !exists {
		// Allocate and zero initialize
		data = make([]byte, 64) // Default cache line size
		b.Memory[addr] = data
	}

	result := make([]byte, len(data))
	copy(result, data)
	return result
}

// WriteMemory writes to main memory
func (b *CoherenceBus) WriteMemory(addr uint64, data []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.Memory[addr] = make([]byte, len(data))
	copy(b.Memory[addr], data)
}

// GetStats returns bus statistics
func (b *CoherenceBus) GetStats() BusStats {
	b.mu.Lock()
	defer b.mu.Unlock()

	return BusStats{
		Transactions: b.Transactions,
		Broadcasts:   b.Broadcasts,
	}
}

// BusStats contains bus statistics
type BusStats struct {
	Transactions uint64
	Broadcasts   uint64
}

// =============================================================================
// Multi-Core System with Coherence
// =============================================================================

// CoherentSystem represents a multi-core system with cache coherence
type CoherentSystem struct {
	Bus    *CoherenceBus
	Caches []*CoherentCache
	NumCores int
}

// NewCoherentSystem creates a new coherent multi-core system
func NewCoherentSystem(numCores int, lineSize uint64) *CoherentSystem {
	bus := NewCoherenceBus()
	caches := make([]*CoherentCache, numCores)

	for i := 0; i < numCores; i++ {
		caches[i] = NewCoherentCache(i, lineSize, bus)
		bus.RegisterCache(caches[i])
	}

	return &CoherentSystem{
		Bus:      bus,
		Caches:   caches,
		NumCores: numCores,
	}
}

// Read reads from a specific core's cache
func (s *CoherentSystem) Read(coreID int, addr uint64) []byte {
	if coreID >= 0 && coreID < s.NumCores {
		data, _ := s.Caches[coreID].Read(addr)
		return data
	}
	return nil
}

// Write writes from a specific core's cache
func (s *CoherentSystem) Write(coreID int, addr uint64, data []byte) {
	if coreID >= 0 && coreID < s.NumCores {
		s.Caches[coreID].Write(addr, data)
	}
}

// GetAllStats returns statistics for all cores
func (s *CoherentSystem) GetAllStats() map[string]interface{} {
	stats := make(map[string]interface{})

	for i, cache := range s.Caches {
		stats[fmt.Sprintf("Core%d", i)] = cache.GetStats()
	}

	stats["Bus"] = s.Bus.GetStats()

	return stats
}
