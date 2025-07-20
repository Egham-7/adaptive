"""
Cache implementations using cachetools library.
Provides typed cache classes for domain classification and other use cases.
"""

from typing import TYPE_CHECKING, Any, TypeVar

import cachetools

if TYPE_CHECKING:
    from adaptive_ai.models.llm_classification_models import (
        DomainClassificationResult,
    )

V = TypeVar("V")  # Value type


class DomainClassificationCache:
    """Specialized cache for domain classification results using cachetools.TTLCache."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        thread_safe: bool = True,
        **kwargs: Any,
    ) -> None:
        if thread_safe:
            self._cache: cachetools.TTLCache[str, DomainClassificationResult] = (
                cachetools.TTLCache(maxsize=max_size, ttl=ttl)
            )
        else:
            # Use a regular TTLCache without thread safety
            self._cache = cachetools.TTLCache(maxsize=max_size, ttl=ttl)

        self._thread_safe = thread_safe

    def get(self, key: str) -> "DomainClassificationResult | None":
        """Get item from cache."""
        result = self._cache.get(key)
        return result

    def put(self, key: str, value: "DomainClassificationResult") -> None:
        """Put item in cache."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        # Note: cachetools doesn't provide hit/miss stats by default
        # We return basic info about the cache state
        return {
            "size": len(self._cache),
            "max_size": self._cache.maxsize,
            "ttl": self._cache.ttl,
            "hits": 0,  # cachetools doesn't track this by default
            "misses": 0,  # cachetools doesn't track this by default
            "evictions": 0,  # cachetools doesn't track this by default
            "hit_rate": 0.0,  # cachetools doesn't track this by default
        }


class EmbeddingCache:
    """Cache for embeddings using cachetools.LRUCache."""

    def __init__(
        self, max_size: int = 1000, thread_safe: bool = True, **kwargs: Any
    ) -> None:
        if thread_safe:
            self._cache: cachetools.LRUCache[str, str] = cachetools.LRUCache(
                maxsize=max_size
            )
        else:
            self._cache = cachetools.LRUCache(maxsize=max_size)

        self._thread_safe = thread_safe

    def get(self, key: str) -> str | None:
        """Get item from cache."""
        result = self._cache.get(key)
        return result

    def put(self, key: str, value: str) -> None:
        """Put item in cache."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
