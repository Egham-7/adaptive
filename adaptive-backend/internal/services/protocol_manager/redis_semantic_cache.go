package protocol_manager

import (
	"adaptive-backend/internal/models"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	fiberlog "github.com/gofiber/fiber/v2/log"
	"github.com/openai/openai-go"
	"github.com/redis/go-redis/v9"
)

const (
	globalCacheIndexName   = "semantic_cache_global_idx"
	userCacheIndexName     = "semantic_cache_user_idx"
	globalCachePrefix      = "semantic_cache:global:"
	userCachePrefix        = "semantic_cache:user:"
	defaultCacheTTL        = 24 * time.Hour
	defaultThreshold       = 0.85
	embeddingDimensions    = 1536 // OpenAI text-embedding-3-small dimension
	defaultGlobalCacheSize = 2000
	defaultUserCacheSize   = 150
	cleanupInterval        = 5 * time.Minute
)

// SemanticCache provides Redis-backed semantic caching using RediSearch vector search
type SemanticCache struct {
	redis     *redis.Client
	openai    *openai.Client
	threshold float32
	ttl       time.Duration
	ctx       context.Context
	cancel    context.CancelFunc
	ticker    *time.Ticker
}

// cacheDocument represents a cached item stored in Redis
type cacheDocument struct {
	Content   string                  `json:"content"`
	Response  models.ProtocolResponse `json:"response"`
	Timestamp time.Time               `json:"timestamp"`
	UserID    string                  `json:"user_id,omitempty"`
}

// NewSemanticCache creates a new Redis-backed semantic cache
func NewSemanticCache(redisClient *redis.Client, openaiClient *openai.Client) *SemanticCache {
	ctx, cancel := context.WithCancel(context.Background())
	ticker := time.NewTicker(cleanupInterval)

	cache := &SemanticCache{
		redis:     redisClient,
		openai:    openaiClient,
		threshold: defaultThreshold,
		ttl:       defaultCacheTTL,
		ctx:       ctx,
		cancel:    cancel,
		ticker:    ticker,
	}

	// Initialize vector search indices
	cache.initializeIndices()

	// Start periodic cleanup goroutine
	go cache.startPeriodicCleanup()

	return cache
}

// startPeriodicCleanup runs periodic cleanup for both cache prefixes
func (s *SemanticCache) startPeriodicCleanup() {
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-s.ticker.C:
			// Clean up both global and user caches
			s.cleanupOldEntries(globalCachePrefix)
			s.cleanupOldEntries(userCachePrefix)
		}
	}
}

// initializeIndices creates the Redis vector search indices if they don't exist
func (s *SemanticCache) initializeIndices() {
	// Create global cache index
	s.createVectorIndex(globalCacheIndexName, globalCachePrefix)

	// Create user cache index
	s.createVectorIndex(userCacheIndexName, userCachePrefix)
}

// createVectorIndex creates a vector search index for semantic caching
func (s *SemanticCache) createVectorIndex(indexName, prefix string) {
	// Drop existing index if it exists
	s.redis.FTDropIndex(s.ctx, indexName)

	// Create new index with vector field
	_, err := s.redis.FTCreate(s.ctx, indexName, &redis.FTCreateOptions{
		OnJSON: true,
		Prefix: []any{prefix},
	},
		&redis.FieldSchema{
			FieldName: "$.content",
			As:        "content",
			FieldType: redis.SearchFieldTypeText,
		},
		&redis.FieldSchema{
			FieldName: "$.user_id",
			As:        "user_id",
			FieldType: redis.SearchFieldTypeTag,
		},
		&redis.FieldSchema{
			FieldName: "$.embedding",
			As:        "embedding",
			FieldType: redis.SearchFieldTypeVector,
			VectorArgs: &redis.FTVectorArgs{
				HNSWOptions: &redis.FTHNSWOptions{
					Type:           "FLOAT64",
					Dim:            embeddingDimensions,
					DistanceMetric: "COSINE",
				},
			},
		},
	).Result()

	if err != nil {
		fiberlog.Warnf("Failed to create vector index %s: %v", indexName, err)
	} else {
		fiberlog.Infof("Created vector index: %s", indexName)
	}
}

// floatsToBytes converts a float64 slice to bytes for Redis storage
func floatsToBytes(fs []float64) []byte {
	buf := make([]byte, len(fs)*8)
	for i, f := range fs {
		binary.LittleEndian.PutUint64(buf[i*8:(i+1)*8], math.Float64bits(f))
	}
	return buf
}

// generateEmbedding generates embeddings using OpenAI API
func (s *SemanticCache) generateEmbedding(text string) ([]float64, error) {
	resp, err := s.openai.Embeddings.New(s.ctx, openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(text),
		},
		Model: openai.EmbeddingModelTextEmbedding3Small,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}

	return resp.Data[0].Embedding, nil
}

// Lookup searches for semantically similar entries in Redis using vector search
func (s *SemanticCache) Lookup(prompt, userID string) (models.ProtocolResponse, string, bool) {
	// Generate embedding for the query
	embedding, err := s.generateEmbedding(prompt)
	if err != nil {
		fiberlog.Errorf("failed to generate embedding for lookup: %v", err)
		return models.ProtocolResponse{}, "", false
	}

	// Convert embedding to bytes
	embeddingBytes := floatsToBytes(embedding)

	// 1. First check user cache
	if val, found := s.vectorSearch(userCacheIndexName, embeddingBytes, userID); found {
		return val, "user", true
	}

	// 2. Then check global cache
	if val, found := s.vectorSearch(globalCacheIndexName, embeddingBytes, ""); found {
		return val, "global", true
	}

	return models.ProtocolResponse{}, "", false
}

// Store saves the entry to both user and global Redis caches using JSON documents
func (s *SemanticCache) Store(prompt, userID string, resp models.ProtocolResponse) {
	// Generate embedding for the prompt
	embedding, err := s.generateEmbedding(prompt)
	if err != nil {
		fiberlog.Errorf("failed to generate embedding for storage: %v", err)
		return
	}

	doc := cacheDocument{
		Content:   prompt,
		Response:  resp,
		Timestamp: time.Now(),
		UserID:    userID,
	}

	// Store in global cache
	s.storeDocument(globalCachePrefix, doc, embedding)

	// Store in user cache
	s.storeDocument(userCachePrefix, doc, embedding)
}

// vectorSearch performs vector similarity search using Redis FT.SEARCH
func (s *SemanticCache) vectorSearch(indexName string, embeddingBytes []byte, userID string) (models.ProtocolResponse, bool) {
	// Build the query - find similar vectors within threshold
	query := "*=>[KNN 3 @embedding $vec AS vector_distance]"

	// Add user filter for user cache
	if userID != "" && indexName == userCacheIndexName {
		query = fmt.Sprintf("@user_id:{%s} => [KNN 3 @embedding $vec AS vector_distance]", userID)
	}

	// Perform vector search
	results, err := s.redis.FTSearchWithArgs(s.ctx, indexName, query, &redis.FTSearchOptions{
		Return: []redis.FTSearchReturn{
			{FieldName: "vector_distance"},
			{FieldName: "content"},
			{FieldName: "response"},
		},
		DialectVersion: 2,
		Params: map[string]any{
			"vec": embeddingBytes,
		},
	}).Result()
	if err != nil {
		fiberlog.Errorf("vector search error: %v", err)
		return models.ProtocolResponse{}, false
	}

	// Check if we have results and if the best match is within threshold
	if results.Total > 0 {
		doc := results.Docs[0]

		// Get the vector distance (lower is better for cosine similarity)
		distanceStr, ok := doc.Fields["vector_distance"]
		if !ok {
			return models.ProtocolResponse{}, false
		}

		distance, err := strconv.ParseFloat(distanceStr, 32)
		if err != nil {
			return models.ProtocolResponse{}, false
		}

		// Convert distance to similarity (1 - distance for cosine)
		similarity := 1.0 - distance

		// Check if similarity meets threshold
		if float32(similarity) >= s.threshold {
			// Parse the response from the cached document
			responseStr, ok := doc.Fields["response"]
			if !ok {
				return models.ProtocolResponse{}, false
			}

			var response models.ProtocolResponse
			if err := json.Unmarshal([]byte(responseStr), &response); err != nil {
				fiberlog.Errorf("failed to unmarshal cached response: %v", err)
				return models.ProtocolResponse{}, false
			}

			return response, true
		}
	}

	return models.ProtocolResponse{}, false
}

// storeDocument stores a document in Redis using JSON.SET with vector embedding
func (s *SemanticCache) storeDocument(prefix string, doc cacheDocument, embedding []float64) {
	// Generate unique key
	key := fmt.Sprintf("%s%d", prefix, time.Now().UnixNano())

	// Create document with embedding for vector search
	docWithEmbedding := map[string]any{
		"content":   doc.Content,
		"response":  doc.Response,
		"timestamp": doc.Timestamp.Unix(),
		"user_id":   doc.UserID,
		"embedding": embedding,
	}

	// Store as JSON document
	_, err := s.redis.JSONSet(s.ctx, key, "$", docWithEmbedding).Result()
	if err != nil {
		fiberlog.Errorf("failed to store document: %v", err)
		return
	}

	// Set TTL
	s.redis.Expire(s.ctx, key, s.ttl)
}

// cleanupOldEntries removes old entries to maintain cache size limits using SCAN
func (s *SemanticCache) cleanupOldEntries(prefix string) {
	// Determine cache size limits
	var maxSize int
	if prefix == globalCachePrefix {
		maxSize = defaultGlobalCacheSize
	} else {
		maxSize = defaultUserCacheSize
	}

	// Use SCAN to get all keys with this prefix
	pattern := prefix + "*"
	var keys []string
	var cursor uint64

	for {
		result, nextCursor, err := s.redis.Scan(s.ctx, cursor, pattern, 100).Result()
		if err != nil {
			fiberlog.Errorf("error scanning for cleanup: %v", err)
			return
		}

		keys = append(keys, result...)
		cursor = nextCursor
		if cursor == 0 {
			break
		}
	}

	// If we're over the limit, remove oldest entries
	if len(keys) > maxSize {
		// Get timestamps for all entries using pipeline
		pipe := s.redis.Pipeline()
		cmds := make([]*redis.JSONCmd, len(keys))

		for i, key := range keys {
			cmds[i] = pipe.JSONGet(s.ctx, key, "$.timestamp")
		}

		_, err := pipe.Exec(s.ctx)
		if err != nil {
			fiberlog.Errorf("error getting timestamps for cleanup: %v", err)
			return
		}

		// Collect entries with timestamps
		type keyWithTime struct {
			key       string
			timestamp int64
		}

		var entries []keyWithTime
		for i, cmd := range cmds {
			result, err := cmd.Result()
			if err != nil {
				continue
			}

			var timestamps []int64
			if err := json.Unmarshal([]byte(result), &timestamps); err != nil || len(timestamps) == 0 {
				continue
			}

			entries = append(entries, keyWithTime{
				key:       keys[i],
				timestamp: timestamps[0],
			})
		}

		// Sort by timestamp (oldest first)
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].timestamp < entries[j].timestamp
		})

		// Remove oldest entries
		toRemove := len(entries) - maxSize
		if toRemove > 0 {
			keysToDelete := make([]string, toRemove)
			for i := range toRemove {
				keysToDelete[i] = entries[i].key
			}
			s.redis.Del(s.ctx, keysToDelete...)
		}
	}
}

// Utility methods for cache management

// SetThreshold allows updating the semantic similarity threshold
func (s *SemanticCache) SetThreshold(threshold float32) {
	s.threshold = threshold
}

// SetTTL allows updating the cache TTL
func (s *SemanticCache) SetTTL(ttl time.Duration) {
	s.ttl = ttl
}

// ClearUserCache removes all entries for a specific user
func (s *SemanticCache) ClearUserCache(userID string) error {
	pattern := userCachePrefix + "*"
	return s.clearCacheByPattern(pattern, userID)
}

// ClearGlobalCache removes all global cache entries
func (s *SemanticCache) ClearGlobalCache() error {
	pattern := globalCachePrefix + "*"
	return s.clearCacheByPattern(pattern, "")
}

// clearCacheByPattern removes cache entries matching a pattern
func (s *SemanticCache) clearCacheByPattern(pattern, userID string) error {
	var keys []string
	var cursor uint64

	for {
		result, nextCursor, err := s.redis.Scan(s.ctx, cursor, pattern, 100).Result()
		if err != nil {
			return err
		}

		// Filter by user ID if specified
		if userID != "" {
			for _, key := range result {
				// Check if this key belongs to the user
				userIDResult, err := s.redis.JSONGet(s.ctx, key, "$.user_id").Result()
				if err != nil {
					continue
				}

				var userIDs []string
				if err := json.Unmarshal([]byte(userIDResult), &userIDs); err != nil || len(userIDs) == 0 {
					continue
				}

				if userIDs[0] == userID {
					keys = append(keys, key)
				}
			}
		} else {
			keys = append(keys, result...)
		}

		cursor = nextCursor
		if cursor == 0 {
			break
		}
	}

	if len(keys) > 0 {
		return s.redis.Del(s.ctx, keys...).Err()
	}
	return nil
}

// GetCacheStats returns cache statistics
func (s *SemanticCache) GetCacheStats() (globalCount, userCount int, err error) {
	// Count global cache entries
	globalCount, err = s.countKeysByPattern(globalCachePrefix + "*")
	if err != nil {
		return 0, 0, err
	}

	// Count all user cache entries
	userCount, err = s.countKeysByPattern(userCachePrefix + "*")
	if err != nil {
		return 0, 0, err
	}

	return globalCount, userCount, nil
}

// countKeysByPattern counts keys matching a pattern using SCAN
func (s *SemanticCache) countKeysByPattern(pattern string) (int, error) {
	var count int
	var cursor uint64

	for {
		result, nextCursor, err := s.redis.Scan(s.ctx, cursor, pattern, 100).Result()
		if err != nil {
			return 0, err
		}

		count += len(result)
		cursor = nextCursor
		if cursor == 0 {
			break
		}
	}

	return count, nil
}

// Close cancels the context and cleans up resources
func (s *SemanticCache) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.ticker != nil {
		s.ticker.Stop()
	}
	if s.redis != nil {
		return s.redis.Close()
	}
	return nil
}
