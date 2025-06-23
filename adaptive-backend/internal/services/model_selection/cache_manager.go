package model_selection

import (
	"adaptive-backend/internal/models"
	"log"

	"github.com/botirk38/semanticcache"
	lru "github.com/hashicorp/golang-lru/v2"
)

const (
	defaultGlobalCacheSize   = 2000
	defaultUserCachePoolSize = 200
	defaultUserCacheSize     = 150
	defaultSemanticThreshold = 0.85
)

type CacheManager struct {
	global      *semanticcache.SemanticCache[string, models.OrchestratorResponse]
	userPool    *lru.Cache[string, *semanticcache.SemanticCache[string, models.OrchestratorResponse]]
	embProv     semanticcache.EmbeddingProvider
	threshold   float32
	defaultSize int
}

func NewCacheManager(embProv semanticcache.EmbeddingProvider) (*CacheManager, error) {
	global, err := semanticcache.NewSemanticCache[string, models.OrchestratorResponse](
		defaultGlobalCacheSize, embProv, nil,
	)
	if err != nil {
		return nil, err
	}
	userPool, err := lru.New[string, *semanticcache.SemanticCache[string, models.OrchestratorResponse]](
		defaultUserCachePoolSize,
	)
	if err != nil {
		return nil, err
	}
	return &CacheManager{
		global:      global,
		userPool:    userPool,
		embProv:     embProv,
		threshold:   defaultSemanticThreshold,
		defaultSize: defaultUserCacheSize,
	}, nil
}

// Lookup returns (value, sourceTag, found)
func (c *CacheManager) Lookup(prompt, userID string) (models.OrchestratorResponse, string, bool) {
	key := prompt
	if uc := c.getUserCache(userID); uc != nil {
		if val, found, _ := uc.Lookup(key, c.threshold); found {
			return val, "user", true
		}
	}
	if val, found, _ := c.global.Lookup(key, c.threshold); found {
		return val, "global", true
	}
	return models.OrchestratorResponse{}, "", false
}

func (c *CacheManager) Store(prompt, userID string, resp models.OrchestratorResponse) {
	key := prompt
	_ = c.global.Set(key, key, resp)
	if uc := c.getUserCache(userID); uc != nil {
		_ = uc.Set(key, key, resp)
	}
}

func (c *CacheManager) getUserCache(userID string) *semanticcache.SemanticCache[string, models.OrchestratorResponse] {
	if uc, ok := c.userPool.Get(userID); ok {
		return uc
	}
	uc, err := semanticcache.NewSemanticCache[string, models.OrchestratorResponse](
		c.defaultSize, c.embProv, nil,
	)
	if err != nil {
		log.Printf("user cache init failed for %s: %v", userID, err)
		return nil
	}
	c.userPool.Add(userID, uc)
	return uc
}
