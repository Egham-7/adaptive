package protocol_manager

import (
	"adaptive-backend/internal/models"

	"github.com/botirk38/semanticcache"
	fiberlog "github.com/gofiber/fiber/v2/log"
	lru "github.com/hashicorp/golang-lru/v2"
)

const (
	defaultGlobalCacheSize   = 2000
	defaultUserCachePoolSize = 200
	defaultUserCacheSize     = 150
	defaultSemanticThreshold = 0.85
)

type CacheManager struct {
	global      *semanticcache.SemanticCache[string, models.ProtocolResponse]
	userPool    *lru.Cache[string, *semanticcache.SemanticCache[string, models.ProtocolResponse]]
	embProv     semanticcache.EmbeddingProvider
	threshold   float32
	defaultSize int
}

func NewCacheManager(embProv semanticcache.EmbeddingProvider) (*CacheManager, error) {
	global, err := semanticcache.NewSemanticCache[string, models.ProtocolResponse](
		defaultGlobalCacheSize, embProv, nil,
	)
	if err != nil {
		return nil, err
	}
	userPool, err := lru.New[string, *semanticcache.SemanticCache[string, models.ProtocolResponse]](
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
func (c *CacheManager) Lookup(prompt, userID string) (models.ProtocolResponse, string, bool) {
	key := prompt
	if uc := c.getUserCache(userID); uc != nil {
		if val, found, err := uc.Lookup(key, c.threshold); found {
			return val, "user", true
		} else if err != nil {
			fiberlog.Errorf("user cache lookup error for %s: %v", userID, err)
		}
	}
	if val, found, err := c.global.Lookup(key, c.threshold); found {
		return val, "global", true
	} else if err != nil {
		fiberlog.Errorf("global cache lookup error: %v", err)
	}
	return models.ProtocolResponse{}, "", false
}

func (c *CacheManager) Store(prompt, userID string, resp models.ProtocolResponse) {
	key := prompt
	_ = c.global.Set(key, key, resp)
	if uc := c.getUserCache(userID); uc != nil {
		_ = uc.Set(key, key, resp)
	}
}

func (c *CacheManager) getUserCache(userID string) *semanticcache.SemanticCache[string, models.ProtocolResponse] {
	if uc, ok := c.userPool.Get(userID); ok {
		return uc
	}
	uc, err := semanticcache.NewSemanticCache[string, models.ProtocolResponse](
		c.defaultSize, c.embProv, nil,
	)
	if err != nil {
		fiberlog.Errorf("user cache init failed for %s: %v", userID, err)
		return nil
	}
	c.userPool.Add(userID, uc)
	return uc
}
