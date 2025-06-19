package main

import (
	"maps" // Import the new maps package
	"sync"
)

type MinionRegistry struct {
	mu      sync.RWMutex
	minions map[string]string
}

func NewMinionRegistry() *MinionRegistry {
	return &MinionRegistry{
		minions: make(map[string]string),
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
