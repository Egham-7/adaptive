package clientcache

import (
	"sync"

	"golang.org/x/sync/singleflight"
)

// Cache provides a type-safe client cache with singleflight to prevent duplicate creation
type Cache[T any] struct {
	cache   sync.Map
	sfGroup singleflight.Group
}

// NewCache creates a new type-safe client cache
func NewCache[T any]() *Cache[T] {
	return &Cache[T]{}
}

// GetOrCreate retrieves a cached client or creates a new one using the provided factory function
// The factory is only called once per key, even under concurrent load
func (c *Cache[T]) GetOrCreate(key string, factory func() (T, error)) (T, error) {
	// Try to get cached client
	if cached, ok := c.cache.Load(key); ok {
		return cached.(T), nil
	}

	// Use singleflight to prevent duplicate client creation under concurrent load
	v, err, _ := c.sfGroup.Do(key, func() (interface{}, error) {
		// Double-check cache after acquiring singleflight lock
		if cached, ok := c.cache.Load(key); ok {
			return cached.(T), nil
		}

		// Build new client using factory
		client, err := factory()
		if err != nil {
			var zero T
			return zero, err
		}

		// Cache the client for future reuse
		c.cache.Store(key, client)

		return client, nil
	})

	if err != nil {
		var zero T
		return zero, err
	}

	return v.(T), nil
}

// Delete removes a client from the cache
func (c *Cache[T]) Delete(key string) {
	c.cache.Delete(key)
}

// Clear removes all clients from the cache
func (c *Cache[T]) Clear() {
	c.cache.Range(func(key, value interface{}) bool {
		c.cache.Delete(key)
		return true
	})
}
