from collections import OrderedDict
import json
import threading
from typing import Protocol
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: object) -> None: ...


class EmbeddingCache:
    def __init__(
        self,
        embeddings_model: HuggingFaceEmbeddings,
        similarity_threshold: float = 0.95,
        max_size: int = 1000,
        thread_safe: bool = True,
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:
        self.vectorstore = InMemoryVectorStore(embeddings_model)
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.thread_safe = thread_safe
        self._exact_match_ids: dict[str, str] = {}
        self._lru_order: OrderedDict[str, None] = OrderedDict()
        self._cached_size = 0
        self._lock = threading.Lock() if thread_safe else None
        self.lit_logger = lit_logger
        self.log(
            "embedding_cache_init",
            {
                "model": getattr(embeddings_model, "model_name", str(embeddings_model)),
                "similarity_threshold": similarity_threshold,
                "max_size": max_size,
                "thread_safe": thread_safe,
            },
        )

    def log(self, key: str, value: object) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _classification_result_to_json_string(
        self, result: ClassificationResult
    ) -> str:
        return json.dumps(result.model_dump(exclude_unset=True), sort_keys=True)

    def _evict_lru_if_needed(self) -> None:
        """Evict least recently used items if cache exceeds max_size."""
        while self._cached_size >= self.max_size and self._lru_order:
            # Get least recently used item
            oldest_json_string = next(iter(self._lru_order))
            oldest_doc_id = self._exact_match_ids.get(oldest_json_string)

            if oldest_doc_id:
                try:
                    self.vectorstore.delete(ids=[oldest_doc_id])
                    del self._exact_match_ids[oldest_json_string]
                    del self._lru_order[oldest_json_string]
                    self._cached_size -= 1
                    self.log("embedding_cache_evicted", {"doc_id": oldest_doc_id})
                except Exception as e:
                    self.log(
                        "embedding_cache_eviction_error",
                        {"doc_id": oldest_doc_id, "error": str(e)},
                    )
                    # Force remove from tracking to prevent infinite loop
                    self._exact_match_ids.pop(oldest_json_string, None)
                    self._lru_order.pop(oldest_json_string, None)
                    self._cached_size = max(0, self._cached_size - 1)
            else:
                # Inconsistent state - remove from LRU order
                del self._lru_order[oldest_json_string]

    def add_to_cache(
        self,
        classification_result: ClassificationResult,
        orchestrator_response: OrchestratorResponse,
    ) -> None:
        json_string = self._classification_result_to_json_string(classification_result)

        def _add_operation() -> None:
            doc_id: str
            is_update = False

            # Check if item already exists
            if json_string in self._exact_match_ids:
                doc_id = self._exact_match_ids[json_string]
                is_update = True
                self.log("embedding_cache_update", {"doc_id": doc_id})

                try:
                    # Atomic operation: delete old, prepare new
                    self.vectorstore.delete(ids=[doc_id])
                    # Update LRU order (move to end)
                    self._lru_order.move_to_end(json_string)
                except Exception as e:
                    self.log(
                        "embedding_cache_update_error",
                        {"doc_id": doc_id, "error": str(e)},
                    )
                    # Recovery: treat as new item
                    is_update = False
                    doc_id = uuid.uuid4().hex
            else:
                doc_id = uuid.uuid4().hex
                self.log("embedding_cache_add", {"doc_id": doc_id})

                # Evict LRU items if needed (only for new items)
                self._evict_lru_if_needed()

            try:
                # Add new document
                document = Document(
                    page_content=json_string,
                    metadata={
                        "orchestrator_response": orchestrator_response.model_dump()
                    },
                    id=doc_id,
                )
                self.vectorstore.add_documents([document])

                # Update tracking structures atomically
                self._exact_match_ids[json_string] = doc_id
                self._lru_order[json_string] = None
                self._lru_order.move_to_end(json_string)  # Mark as most recently used

                if not is_update:
                    self._cached_size += 1

                self.log("embedding_cache_size", self._cached_size)

            except Exception as e:
                # Rollback on failure
                self.log(
                    "embedding_cache_add_failed", {"doc_id": doc_id, "error": str(e)}
                )
                self._exact_match_ids.pop(json_string, None)
                self._lru_order.pop(json_string, None)
                if not is_update:
                    self._cached_size = max(0, self._cached_size - 1)
                raise

        if self._lock:
            with self._lock:
                _add_operation()
        else:
            _add_operation()

    def search_cache(
        self, query_classification_result: ClassificationResult
    ) -> OrchestratorResponse | None:
        def _search_operation() -> OrchestratorResponse | None:
            if self._cached_size == 0:
                self.log("embedding_cache_empty", 1)
                return None

            query_json_string: str = self._classification_result_to_json_string(
                query_classification_result
            )
            self.log(
                "embedding_cache_lookup",
                {"query": query_json_string, "cache_size": self._cached_size},
            )

            # Check exact match first
            if query_json_string in self._exact_match_ids:
                try:
                    exact_doc = self.vectorstore.get_by_ids(
                        [self._exact_match_ids[query_json_string]]
                    )[0]

                    # Update LRU order on cache hit
                    if query_json_string in self._lru_order:
                        self._lru_order.move_to_end(query_json_string)

                    self.log(
                        "embedding_cache_hit",
                        {
                            "type": "exact",
                            "doc_id": self._exact_match_ids[query_json_string],
                        },
                    )
                    return OrchestratorResponse.model_validate(
                        exact_doc.metadata["orchestrator_response"]
                    )
                except Exception as e:
                    self.log(
                        "embedding_cache_exact_error",
                        {
                            "doc_id": self._exact_match_ids[query_json_string],
                            "error": str(e),
                        },
                    )
                    # Clean up inconsistent state
                    self._exact_match_ids.pop(query_json_string, None)
                    self._lru_order.pop(query_json_string, None)
                    self._cached_size = max(0, self._cached_size - 1)

            # Semantic search fallback
            try:
                results: list[tuple[Document, float]] = (
                    self.vectorstore.similarity_search_with_score(
                        query=query_json_string, k=1
                    )
                )

                if results:
                    most_similar_doc, score = results[0]
                    if score >= self.similarity_threshold:
                        # Find and update LRU order for semantic match
                        for json_str, doc_id in self._exact_match_ids.items():
                            if doc_id == most_similar_doc.id:
                                if json_str in self._lru_order:
                                    self._lru_order.move_to_end(json_str)
                                break

                        self.log(
                            "embedding_cache_hit",
                            {
                                "type": "semantic",
                                "score": score,
                                "threshold": self.similarity_threshold,
                                "doc_id": most_similar_doc.id,
                            },
                        )
                        return OrchestratorResponse.model_validate(
                            most_similar_doc.metadata["orchestrator_response"]
                        )
            except Exception as e:
                self.log("embedding_cache_semantic_error", {"error": str(e)})

            self.log("embedding_cache_hit", {"type": "miss"})
            return None

        if self._lock:
            with self._lock:
                return _search_operation()
        else:
            return _search_operation()

    def get_cache_size(self) -> int:
        if self._lock:
            with self._lock:
                return self._cached_size
        else:
            return self._cached_size

    def clear_cache(self) -> None:
        def _clear_operation() -> None:
            self.vectorstore.store.clear()
            self._exact_match_ids.clear()
            self._lru_order.clear()
            self._cached_size = 0
            self.log("embedding_cache_cleared", 1)

        if self._lock:
            with self._lock:
                _clear_operation()
        else:
            _clear_operation()
