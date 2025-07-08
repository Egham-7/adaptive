package minions

import (
	"maps"
	"sync"

	fiberlog "github.com/gofiber/fiber/v2/log"
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
	if minionType == "" || url == "" {
		fiberlog.Errorf("minionType and url cannot be empty: minionType=%s, url=%s", minionType, url)
		return
	}

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
