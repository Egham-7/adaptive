import json
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_orchestration_models import OrchestratorResponse


class EmbeddingCache:
    def __init__(
        self,
        embeddings_model: HuggingFaceEmbeddings,
        similarity_threshold: float = 0.95,
    ) -> None:
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(
        self,
        embeddings_model: HuggingFaceEmbeddings,
        similarity_threshold: float = 0.95,
    ) -> None:
        logger.info(
            f"Initializing EmbeddingCache with LangChain InMemoryVectorStore and model: {embeddings_model.model_name}..."
        )
        self.vectorstore = InMemoryVectorStore(embeddings_model)
        self.similarity_threshold = similarity_threshold
        self._exact_match_ids: dict[str, str] = {}
        logger.info("Embedding cache initialized.")

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
            print(
                f"Updating existing classification result in cache (exact match). Doc ID: {doc_id}"
            )
            try:
                self.vectorstore.delete(ids=[doc_id])
            except Exception as e:
                logger.warning(f"Failed to delete document {doc_id}: {e}")
        else:
            doc_id = uuid.uuid4().hex
            print(f"Adding new classification result to cache. Doc ID: {doc_id}")

        document = Document(
            page_content=json_string,
            metadata={"orchestrator_response": orchestrator_response.model_dump()},
            id=doc_id,
        )

        self.vectorstore.add_documents([document])
        self._exact_match_ids[json_string] = doc_id

        print(f"Current cache size (documents): {self.get_cache_size()}")

    def search_cache(
        self, query_classification_result: ClassificationResult
    ) -> OrchestratorResponse | None:
        if self.get_cache_size() == 0:
            return None

        query_json_string = self._classification_result_to_json_string(
            query_classification_result
        )

        if query_json_string in self._exact_match_ids:
            try:
                exact_doc = self.vectorstore.get_by_ids(
                    [self._exact_match_ids[query_json_string]]
                )[0]
                print("Cache HIT (exact string match from internal map)!")
                return OrchestratorResponse.model_validate(
                    exact_doc.metadata["orchestrator_response"]
                )
            except Exception as e:
                logger.warning(
                    f"Error retrieving exact match document by ID '{self._exact_match_ids[query_json_string]}': {e}. Falling back to semantic search."
                )

        results: list[tuple[Document, float]] = (
            self.vectorstore.similarity_search_with_score(query=query_json_string, k=1)
        )

        if results:
            most_similar_doc, score = results[0]
            if score >= self.similarity_threshold:
                print(
                    f"Cache HIT (semantic match: score={score:.4f}, threshold={self.similarity_threshold})!"
                )
                return OrchestratorResponse.model_validate(
                    most_similar_doc.metadata["orchestrator_response"]
                )

        print("Cache MISS (no match above threshold).")
        return None

    def get_cache_size(self) -> int:
        return len(self.vectorstore.store)

    def clear_cache(self) -> None:
        print("Clearing embedding cache...")
        self.vectorstore.store.clear()
        self._exact_match_ids.clear()
        print("Embedding cache cleared.")
