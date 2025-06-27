import json
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
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:
        self.vectorstore = InMemoryVectorStore(embeddings_model)
        self.similarity_threshold = similarity_threshold
        self._exact_match_ids: dict[str, str] = {}
        self.lit_logger = lit_logger
        self.log(
            "embedding_cache_init",
            {
                "model": getattr(embeddings_model, "model_name", str(embeddings_model)),
                "similarity_threshold": similarity_threshold,
            },
        )

    def log(self, key: str, value: object) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _classification_result_to_json_string(
        self, result: ClassificationResult
    ) -> str:
        return json.dumps(result.model_dump(exclude_unset=True), sort_keys=True)

    def add_to_cache(
        self,
        classification_result: ClassificationResult,
        orchestrator_response: OrchestratorResponse,
    ) -> None:
        json_string = self._classification_result_to_json_string(classification_result)

        doc_id: str
        if json_string in self._exact_match_ids:
            doc_id = self._exact_match_ids[json_string]
            self.log("embedding_cache_update", {"doc_id": doc_id})
            try:
                self.vectorstore.delete(ids=[doc_id])
            except Exception as e:
                self.log(
                    "embedding_cache_delete_error", {"doc_id": doc_id, "error": str(e)}
                )
        else:
            doc_id = uuid.uuid4().hex
            self.log("embedding_cache_add", {"doc_id": doc_id})

        document = Document(
            page_content=json_string,
            metadata={"orchestrator_response": orchestrator_response.model_dump()},
            id=doc_id,
        )
        self.vectorstore.add_documents([document])
        self._exact_match_ids[json_string] = doc_id

        cache_size = self.get_cache_size()
        self.log("embedding_cache_size", cache_size)

    def search_cache(
        self, query_classification_result: ClassificationResult
    ) -> OrchestratorResponse | None:
        cache_size: int = self.get_cache_size()
        if cache_size == 0:
            self.log("embedding_cache_empty", 1)
            return None

        query_json_string: str = self._classification_result_to_json_string(
            query_classification_result
        )
        self.log(
            "embedding_cache_lookup",
            {"query": query_json_string, "cache_size": cache_size},
        )

        if query_json_string in self._exact_match_ids:
            try:
                exact_doc = self.vectorstore.get_by_ids(
                    [self._exact_match_ids[query_json_string]]
                )[0]
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

        results: list[tuple[Document, float]] = (
            self.vectorstore.similarity_search_with_score(query=query_json_string, k=1)
        )

        if results:
            most_similar_doc, score = results[0]
            if score >= self.similarity_threshold:
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

        self.log("embedding_cache_hit", {"type": "miss"})
        return None

    def get_cache_size(self) -> int:
        size: int = len(self.vectorstore.store)
        self.log("embedding_cache_size_check", size)
        return size

    def clear_cache(self) -> None:
        self.vectorstore.store.clear()
        self._exact_match_ids.clear()
        self.log("embedding_cache_cleared", 1)
