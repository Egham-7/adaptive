package minions

import (
	"maps"
	"sync"
)

type MinionRegistry struct {
	mu      sync.RWMutex
	minions map[string]string
}

func NewMinionRegistry(capacity int) *MinionRegistry {
	return &MinionRegistry{
		minions: make(map[string]string, capacity), // Preallocate memory here
	}
}

func (mr *MinionRegistry) RegisterMinion(minionType, url string) {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	mr.minions[minionType] = url
}

func (mr *MinionRegistry) GetMinionURL(minionType string) (string, bool) {
	mr.mu.RLock()
	defer mr.mu.RUnlock()
	url, found := mr.minions[minionType]
	return url, found
}

func (mr *MinionRegistry) ListMinions() map[string]string {
	mr.mu.RLock()
	defer mr.mu.RUnlock()

	copyMap := make(map[string]string, len(mr.minions))
	maps.Copy(copyMap, mr.minions)
	return copyMap
}

func (mr *MinionRegistry) UnregisterMinion(minionType string) bool {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	_, found := mr.minions[minionType]
	if found {
		delete(mr.minions, minionType)
		return true
	}
	return false
}

/*
// Example of how you would use it in a main function or elsewhere:
func main() {
    // If you know you'll have about 10 task types initially, or maximum 10
    registry := NewMinionRegistry(10)

    registry.RegisterMinion("Open QA", "http://qa-service.com/open")
    registry.RegisterMinion("Closed QA", "http://qa-service.com/closed")
    // ... add more ...

    fmt.Printf("URL for 'Open QA': %s\n", registry.GetMinionURL("Open QA"))
}
*/
