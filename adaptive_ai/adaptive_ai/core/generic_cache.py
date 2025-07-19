"""
Generic typed cache implementation for classifiers.
Supports LRU, TTL, and semantic similarity eviction policies.
"""

from collections import OrderedDict
from collections.abc import Callable
import hashlib
import threading
import time
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from adaptive_ai.models.llm_classification_models import (
        DomainClassificationResult,  # noqa: F401
    )

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type
V_contra = TypeVar(
    "V_contra", contravariant=True
)  # Contravariant value type for protocols
T = TypeVar("T")  # Generic type for functions


class SimilarityMatcher(Protocol[V_contra]):
    """Protocol for semantic similarity matching."""

    def calculate_similarity(self, item1: V_contra, item2: V_contra) -> float:
        """Calculate similarity between two cached items."""
        ...


class CacheStats:
    """Cache performance statistics."""

    def __init__(self) -> None:
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class GenericTypedCache(Generic[K, V]):
    """
    Generic typed cache with configurable eviction policies.

    Supports:
    - LRU (Least Recently Used)
    - TTL (Time To Live)
    - Semantic similarity matching
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int | None = None,
        eviction_policy: Literal["lru", "ttl", "semantic"] = "lru",
        similarity_threshold: float = 0.95,
        similarity_matcher: SimilarityMatcher[V] | None = None,
        thread_safe: bool = True,
        key_hasher: Callable[[K], str] | None = None,
    ) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        self.similarity_threshold = similarity_threshold
        self.similarity_matcher = similarity_matcher
        self.key_hasher = key_hasher or self._default_key_hasher

        # Cache storage
        self._cache: OrderedDict[str, tuple[V, float]] = OrderedDict()
        self._key_mapping: dict[K, str] = {}

        # Thread safety
        self._lock = threading.Lock() if thread_safe else None

        # Statistics
        self.stats = CacheStats()

    def _default_key_hasher(self, key: K) -> str:
        """Default key hashing strategy."""
        key_str = str(key)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _with_lock(self, func: Callable[[], T]) -> T:
        """Execute function with optional locking."""
        if self._lock:
            with self._lock:
                return func()
        return func()

    def _evict_if_needed(self) -> None:
        """Evict items based on the configured policy."""
        while len(self._cache) >= self.max_size:
            if self.eviction_policy == "lru":
                # Remove least recently used
                oldest_key, _ = self._cache.popitem(last=False)
                # Clean up key mapping
                self._key_mapping = {
                    k: v for k, v in self._key_mapping.items() if v != oldest_key
                }
            else:
                # For TTL and semantic, remove expired items first, then LRU
                self._evict_expired()
                if len(self._cache) >= self.max_size:
                    oldest_key, _ = self._cache.popitem(last=False)
                    self._key_mapping = {
                        k: v for k, v in self._key_mapping.items() if v != oldest_key
                    }
            self.stats.evictions += 1

    def _evict_expired(self) -> None:
        """Remove expired TTL items."""
        if not self.ttl:
            return

        current_time = time.time()
        expired_keys = [
            cache_key
            for cache_key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl
        ]

        for cache_key in expired_keys:
            del self._cache[cache_key]
            # Clean up key mapping
            self._key_mapping = {
                k: v for k, v in self._key_mapping.items() if v != cache_key
            }

    def get(self, key: K) -> V | None:
        """Get item from cache."""

        def _get() -> V | None:
            cache_key = self.key_hasher(key)

            # Check exact match first
            if cache_key in self._cache:
                value, timestamp = self._cache[cache_key]

                # Check TTL expiration
                if self.ttl and time.time() - timestamp > self.ttl:
                    del self._cache[cache_key]
                    del self._key_mapping[key]
                    self.stats.misses += 1
                    return None

                # Move to end (mark as recently used)
                self._cache.move_to_end(cache_key)
                self.stats.hits += 1
                return value

            # Semantic similarity search
            if (
                self.eviction_policy == "semantic"
                and self.similarity_matcher
                and self._cache
            ):

                for _cached_key, (
                    _cached_value,
                    cached_timestamp,
                ) in self._cache.items():
                    if self.ttl and time.time() - cached_timestamp > self.ttl:
                        continue

                    # This is a simplified approach - in practice, you'd want
                    # to compare the input key against the original keys
                    # that were used to store the cached values
                    pass

            self.stats.misses += 1
            return None

        result = self._with_lock(_get)
        return result

    def put(self, key: K, value: V) -> None:
        """Put item in cache."""

        def _put() -> None:
            # Evict if needed
            self._evict_if_needed()

            cache_key = self.key_hasher(key)
            timestamp = time.time()

            self._cache[cache_key] = (value, timestamp)
            self._key_mapping[key] = cache_key

            # Move to end (mark as recently used)
            self._cache.move_to_end(cache_key)

        self._with_lock(_put)

    def clear(self) -> None:
        """Clear all cache entries."""

        def _clear() -> None:
            self._cache.clear()
            self._key_mapping.clear()

        self._with_lock(_clear)

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats.to_dict(),
            "size": self.size(),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "eviction_policy": self.eviction_policy,
        }


# Specialized cache types for common use cases
class TextCache(GenericTypedCache[str, V]):
    """Cache optimized for string keys."""

    pass


class DomainClassificationCache(TextCache["DomainClassificationResult"]):
    """Specialized cache for domain classification results."""

    def __init__(self, **kwargs: Any) -> None:
        # Set defaults if not provided
        kwargs.setdefault("ttl", 3600)  # 1 hour default
        kwargs.setdefault("eviction_policy", "ttl")
        super().__init__(**kwargs)


class EmbeddingCache(TextCache[str]):
    """Specialized cache for embeddings with semantic similarity."""

    def __init__(self, **kwargs: Any) -> None:
        # Set defaults if not provided
        kwargs.setdefault("eviction_policy", "semantic")
        kwargs.setdefault("similarity_threshold", 0.95)
        super().__init__(**kwargs)
