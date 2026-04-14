/*
Cache Coherence
===============

MESI protocol implementation for multi-core cache coherence.

States:
- Modified (M): Cache line is dirty and exclusive to this cache
- Exclusive (E): Cache line is clean and exclusive to this cache
- Shared (S): Cache line is clean and may be in other caches
- Invalid (I): Cache line is not valid

Deadlock prevention:
  Operations release their per-cache lock before broadcasting on the bus.
  The bus serialises all transactions with its own mutex so that no two
  cores can be simultaneously inside SendMessage – eliminating the classic
  "A waits for B while B waits for A" deadlock.

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

// CoherentCache represents a cache with MESI coherence protocol.
//
// Lock ordering to prevent deadlocks:
//   1. CoherenceBus.mu  (outer, held only briefly during SendMessage)
//   2. CoherentCache.mu (inner)
//
// CoherentCache.Read / Write always release their own lock before calling
// CoherenceBus.SendMessage, so the bus's callbacks to HandleBusMessage can
// safely acquire other caches' locks without circular waiting.
type CoherentCache struct {
	ID       int
	Lines    map[uint64]*CoherentCacheLine
	LineSize uint64
	Bus      *CoherenceBus
	mu       sync.Mutex // protects Lines and stats

	// Statistics
	Reads           uint64
	Writes          uint64
	Invalidations   uint64
	Downgrades      uint64
	BusTransactions uint64
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

// Read reads data from address.
//
// The per-cache lock is released before any bus transaction to prevent the
// classic two-cache deadlock (A holds A.mu, waits for B.mu; B holds B.mu,
// waits for A.mu).  After the bus transaction the lock is re-acquired and
// state is checked again (double-checked pattern).
func (c *CoherentCache) Read(addr uint64) ([]byte, bool) {
	lineAddr := addr / c.LineSize

	// --- Fast path: cache hit ---
	c.mu.Lock()
	c.Reads++
	line, exists := c.Lines[lineAddr]
	if exists && line.Valid && line.State != Invalid {
		data := make([]byte, len(line.Data))
		copy(data, line.Data)
		c.mu.Unlock()
		return data, true
	}
	c.BusTransactions++
	c.mu.Unlock() // Release BEFORE bus transaction to avoid deadlock

	// --- Slow path: cache miss – fetch via bus ---
	msg := CoherenceMessage{
		Type:    MsgRead,
		Address: lineAddr,
		CoreID:  c.ID,
	}
	responses := c.Bus.SendMessage(msg)

	var fetchedData []byte
	sharedCopy := false
	for _, resp := range responses {
		if resp.Type == MsgDataResponse {
			fetchedData = resp.Data
			sharedCopy = true
		}
	}
	if fetchedData == nil {
		fetchedData = c.Bus.ReadMemory(lineAddr, c.LineSize)
	}

	// --- Re-acquire lock to install the line ---
	c.mu.Lock()
	defer c.mu.Unlock()

	// Re-check: another goroutine might have already filled this line
	line, exists = c.Lines[lineAddr]
	if exists && line.Valid && line.State != Invalid {
		data := make([]byte, len(line.Data))
		copy(data, line.Data)
		return data, true
	}

	newLine := &CoherentCacheLine{
		Valid: true,
		Tag:   lineAddr,
		Data:  make([]byte, c.LineSize),
	}
	// Pad fetchedData to full line size
	copy(newLine.Data, fetchedData)
	if sharedCopy {
		newLine.State = Shared
	} else {
		newLine.State = Exclusive
	}
	c.Lines[lineAddr] = newLine

	result := make([]byte, len(newLine.Data))
	copy(result, newLine.Data)
	return result, false
}

// Write writes data to address.
// Same lock-release-before-bus pattern as Read.
func (c *CoherentCache) Write(addr uint64, data []byte) bool {
	lineAddr := addr / c.LineSize

	// --- Fast path: line present with write permission ---
	c.mu.Lock()
	c.Writes++
	line, exists := c.Lines[lineAddr]

	if exists && line.Valid {
		switch line.State {
		case Modified:
			// Already have exclusive-dirty ownership
			if len(data) <= len(line.Data) {
				copy(line.Data, data)
			}
			c.mu.Unlock()
			return true

		case Exclusive:
			// Upgrade to Modified (no bus transaction needed)
			line.State = Modified
			if len(data) <= len(line.Data) {
				copy(line.Data, data)
			}
			c.mu.Unlock()
			return true

		case Shared:
			// Need upgrade: invalidate other copies via bus
			c.BusTransactions++
			c.Downgrades++
			c.mu.Unlock() // Release BEFORE bus

			upgradeMsg := CoherenceMessage{
				Type:    MsgUpgrade,
				Address: lineAddr,
				CoreID:  c.ID,
			}
			c.Bus.SendMessage(upgradeMsg)

			c.mu.Lock()
			// Re-check the line still exists after lock release
			line, exists = c.Lines[lineAddr]
			if exists && line.Valid {
				line.State = Modified
				if len(data) <= len(line.Data) {
					copy(line.Data, data)
				}
				c.mu.Unlock()
				return true
			}
			c.mu.Unlock()
			// Fall through to miss path if evicted by another core
		}
	} else {
		c.mu.Unlock()
	}

	// --- Slow path: cache miss – read-exclusive via bus ---
	c.mu.Lock()
	c.BusTransactions++
	c.mu.Unlock()

	missMsg := CoherenceMessage{
		Type:    MsgReadX,
		Address: lineAddr,
		CoreID:  c.ID,
	}
	responses := c.Bus.SendMessage(missMsg)

	var lineData []byte
	for _, resp := range responses {
		if resp.Type == MsgDataResponse {
			lineData = resp.Data
		}
	}
	if lineData == nil {
		lineData = c.Bus.ReadMemory(lineAddr, c.LineSize)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Re-check
	line, exists = c.Lines[lineAddr]
	if exists && line.Valid && line.State == Modified {
		// Another goroutine raced us and already got ownership
		if len(data) <= len(line.Data) {
			copy(line.Data, data)
		}
		return true
	}

	newLine := &CoherentCacheLine{
		Valid: true,
		Tag:   lineAddr,
		State: Modified,
		Data:  make([]byte, c.LineSize),
	}
	copy(newLine.Data, lineData)
	// Write the new data on top
	if len(data) <= len(newLine.Data) {
		copy(newLine.Data, data)
	}
	c.Lines[lineAddr] = newLine
	return false
}

// HandleBusMessage handles coherence messages from the bus for other cores.
// Called by CoherenceBus.SendMessage (the bus serialises calls, so no two
// HandleBusMessage calls for the same cache run concurrently).
func (c *CoherentCache) HandleBusMessage(msg CoherenceMessage) *CoherenceMessage {
	if msg.CoreID == c.ID {
		return nil // Ignore own messages
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	line, exists := c.Lines[msg.Address]
	if !exists || !line.Valid || line.State == Invalid {
		return nil
	}

	switch msg.Type {
	case MsgRead:
		switch line.State {
		case Modified:
			// Provide data and downgrade to Shared; write back to memory
			c.Downgrades++
			resp := &CoherenceMessage{
				Type:    MsgDataResponse,
				Address: msg.Address,
				CoreID:  c.ID,
				Data:    make([]byte, len(line.Data)),
			}
			copy(resp.Data, line.Data)
			c.Bus.WriteMemory(msg.Address, line.Data) // writeback
			line.State = Shared
			return resp

		case Exclusive:
			c.Downgrades++
			line.State = Shared
			return nil

		case Shared:
			return nil // Already shared; no action
		}

	case MsgReadX:
		switch line.State {
		case Modified:
			// Provide data and invalidate
			c.Invalidations++
			resp := &CoherenceMessage{
				Type:    MsgDataResponse,
				Address: msg.Address,
				CoreID:  c.ID,
				Data:    make([]byte, len(line.Data)),
			}
			copy(resp.Data, line.Data)
			c.Bus.WriteMemory(msg.Address, line.Data) // writeback
			line.State = Invalid
			return resp

		case Exclusive, Shared:
			c.Invalidations++
			line.State = Invalid
			return nil
		}

	case MsgUpgrade:
		if line.State == Shared {
			c.Invalidations++
			line.State = Invalid
		}
		return nil

	case MsgInvalidate:
		c.Invalidations++
		line.State = Invalid
		return nil
	}

	return nil
}

// Flush writes all Modified lines back to memory and marks them Exclusive (clean).
func (c *CoherentCache) Flush() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for addr, line := range c.Lines {
		if line.State == Modified {
			c.Bus.WriteMemory(addr, line.Data)
			line.State = Exclusive // clean but still in this cache
		}
	}
}

// GetStats returns cache coherence statistics
func (c *CoherentCache) GetStats() CoherenceStats {
	c.mu.Lock()
	defer c.mu.Unlock()

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
  Reads:            %d
  Writes:           %d
  Invalidations:    %d
  Downgrades:       %d
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

// CoherenceBus represents the system bus for coherence traffic.
// Its mutex serialises all bus transactions so that only one
// SendMessage call proceeds at a time, preventing cache–cache deadlocks.
type CoherenceBus struct {
	Caches []*CoherentCache
	Memory map[uint64][]byte
	mu     sync.Mutex

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

// SendMessage serialises and broadcasts a coherence message to all other caches.
// Callers must NOT hold their own cache lock when calling this method.
func (b *CoherenceBus) SendMessage(msg CoherenceMessage) []*CoherenceMessage {
	// Serialise bus transactions to prevent concurrent SendMessage calls
	// from creating A→B / B→A lock cycles inside HandleBusMessage.
	b.mu.Lock()
	b.Transactions++
	b.Broadcasts++
	caches := make([]*CoherentCache, len(b.Caches)) // snapshot under lock
	copy(caches, b.Caches)
	b.mu.Unlock()

	var responses []*CoherenceMessage
	for _, cache := range caches {
		if resp := cache.HandleBusMessage(msg); resp != nil {
			responses = append(responses, resp)
		}
	}
	return responses
}

// ReadMemory reads lineSize bytes from main memory at the given line address.
func (b *CoherenceBus) ReadMemory(lineAddr uint64, lineSize uint64) []byte {
	b.mu.Lock()
	defer b.mu.Unlock()

	data, exists := b.Memory[lineAddr]
	if !exists {
		data = make([]byte, lineSize)
		b.Memory[lineAddr] = data
	}

	result := make([]byte, len(data))
	copy(result, data)
	return result
}

// WriteMemory writes data to main memory at the given line address.
func (b *CoherenceBus) WriteMemory(lineAddr uint64, data []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()

	stored := make([]byte, len(data))
	copy(stored, data)
	b.Memory[lineAddr] = stored
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
	Bus      *CoherenceBus
	Caches   []*CoherentCache
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

// GetAllStats returns statistics for all cores and the bus
func (s *CoherentSystem) GetAllStats() map[string]interface{} {
	stats := make(map[string]interface{})
	for i, cache := range s.Caches {
		stats[fmt.Sprintf("Core%d", i)] = cache.GetStats()
	}
	stats["Bus"] = s.Bus.GetStats()
	return stats
}
