# mypy: disable-error-code=import
from collections import OrderedDict
import hashlib
import time
from typing import Any

from huggingface_hub import PyTorchModelHubMixin
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from adaptive_ai.models.llm_classification_models import (
    DomainClassificationResult,
    DomainType,
)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """NVIDIA Domain Classifier following official documentation."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))
        self.config = config

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        features = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def process_logits(self, logits: torch.Tensor) -> list[DomainClassificationResult]:
        """Process logits and return domain classification results."""
        batch_size = logits.shape[0]
        # logits are already softmax'd in forward pass
        probabilities = logits

        results = []
        for i in range(batch_size):
            sample_probs = probabilities[i].detach().cpu().numpy()

            # Get top domain
            top_domain_idx = np.argmax(sample_probs)
            top_domain = self.config["id2label"][str(top_domain_idx)]
            confidence = float(sample_probs[top_domain_idx])

            # Create probability dictionary using official labels
            domain_probabilities = {
                self.config["id2label"][str(idx)]: float(prob)
                for idx, prob in enumerate(sample_probs)
            }

            # Convert to DomainType enum
            try:
                domain_enum = DomainType(top_domain)
            except ValueError:
                # Fallback to a default domain if enum conversion fails
                domain_enum = DomainType.REFERENCE

            result = DomainClassificationResult(
                domain=domain_enum,
                confidence=confidence,
                domain_probabilities=domain_probabilities,
            )
            results.append(result)

        return results


class DomainClassifier:
    def __init__(
        self, lit_logger: Any = None, cache_size: int = 1000, cache_ttl: int = 3600
    ) -> None:
        self.lit_logger: Any = lit_logger
        self.cache_size: int = cache_size
        self.cache_ttl: int = cache_ttl
        self.cache: OrderedDict[str, tuple[DomainClassificationResult, float]] = (
            OrderedDict()
        )  # LRU cache for domain classification results
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Load NVIDIA domain classifier following official documentation
        try:
            self.config = AutoConfig.from_pretrained("nvidia/domain-classifier")
            self.tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")

            # Load the model using the CustomModel class from official docs with safetensors
            self.model = CustomModel.from_pretrained(
                "nvidia/domain-classifier", use_safetensors=True
            )
            self.model.eval()

            self.log(
                "domain_classifier_weights_loaded",
                {"status": "success", "source": "nvidia/domain-classifier"},
            )
        except Exception as e:
            self.log("domain_classifier_weights_load_failed", {"error": str(e)})
            # Fallback to custom implementation if NVIDIA loading fails
            self.log(
                "domain_classifier_fallback", {"status": "using_custom_implementation"}
            )
            # Use the original custom implementation as fallback
            self._setup_custom_fallback(lit_logger)

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _setup_custom_fallback(self, lit_logger: Any) -> None:
        """Setup custom domain classifier as fallback when NVIDIA model fails."""
        # Create a proper config for the fallback model
        domain_types: list[str] = [e.value for e in DomainType]
        fallback_config: dict[str, Any] = {
            "id2label": {str(i): domain_types[i] for i in range(len(domain_types))},
            "label2id": {domain_types[i]: str(i) for i in range(len(domain_types))},
            "num_labels": len(domain_types),
        }

        # Create a custom model that mimics the NVIDIA architecture
        class CustomDomainClassifier(nn.Module):
            def __init__(self, config: dict[str, Any], lit_logger: Any = None) -> None:
                super().__init__()
                self.backbone = AutoModel.from_pretrained(
                    "microsoft/deberta-v3-base", use_safetensors=True
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(
                    self.backbone.config.hidden_size, config["num_labels"]
                )
                self.config = config
                self.lit_logger = lit_logger

            def forward(
                self, input_ids: torch.Tensor, attention_mask: torch.Tensor
            ) -> torch.Tensor:
                outputs = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # Use first token like NVIDIA implementation
                features = outputs.last_hidden_state[:, 0, :]
                logits = self.fc(features)
                return torch.softmax(logits, dim=1)

            def process_logits(
                self, logits: torch.Tensor
            ) -> list[DomainClassificationResult]:
                """Process logits and return domain classification results."""
                batch_size = logits.shape[0]
                probabilities = logits

                results = []
                for i in range(batch_size):
                    sample_probs = probabilities[i].detach().cpu().numpy()

                    # Get top domain
                    top_domain_idx = np.argmax(sample_probs)
                    top_domain = self.config["id2label"][str(top_domain_idx)]
                    confidence = float(sample_probs[top_domain_idx])

                    # Create probability dictionary using official labels
                    domain_probabilities = {
                        self.config["id2label"][str(idx)]: float(prob)
                        for idx, prob in enumerate(sample_probs)
                    }

                    # Convert to DomainType enum
                    try:
                        domain_enum = DomainType(top_domain)
                    except ValueError:
                        # Fallback to a default domain if enum conversion fails
                        domain_enum = DomainType.REFERENCE
                        if self.lit_logger:
                            self.lit_logger.log(
                                "domain_enum_conversion_failed", {"domain": top_domain}
                            )

                    result = DomainClassificationResult(
                        domain=domain_enum,
                        confidence=confidence,
                        domain_probabilities=domain_probabilities,
                    )
                    results.append(result)

                return results

        self.model = CustomDomainClassifier(fallback_config, lit_logger)  # type: ignore
        self.config = fallback_config

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, text: str) -> DomainClassificationResult | None:
        """Get domain classification result from cache if available and not expired."""
        cache_key: str = self._get_cache_key(text)

        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]

            # Check if cache entry is still valid
            if time.time() - timestamp < self.cache_ttl:
                # Move to end to mark as recently used
                self.cache.move_to_end(cache_key)
                self.cache_hits += 1
                return cached_result
            else:
                # Remove expired entry
                del self.cache[cache_key]

        self.cache_misses += 1
        return None

    def _add_to_cache(self, text: str, result: DomainClassificationResult) -> None:
        """Add domain classification result to cache."""
        cache_key: str = self._get_cache_key(text)

        # Remove oldest entries if cache is full
        while len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)

        self.cache[cache_key] = (result, time.time())

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests: int = self.cache_hits + self.cache_misses
        hit_rate: float = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
        }

    def classify_domains(self, texts: list[str]) -> list[DomainClassificationResult]:
        """
        Classify multiple texts into domains using batch processing with caching.

        Args:
            texts: List of texts to classify

        Returns:
            List of domain classification results, one per text
        """
        if not texts:
            self.log("domain_classification_empty_batch", {"batch_size": 0})
            return []

        self.log("domain_classification_batch_start", {"batch_size": len(texts)})

        try:
            # Check cache for each text
            results: list[DomainClassificationResult | None] = []
            texts_to_classify: list[str] = []
            cache_indices: list[int] = []

            for i, text in enumerate(texts):
                cached_result = self._get_from_cache(text)
                if cached_result:
                    results.append(cached_result)
                    cache_indices.append(i)
                else:
                    results.append(None)  # Placeholder
                    texts_to_classify.append(text)

            # Log cache performance
            cache_stats: dict[str, Any] = self.get_cache_stats()
            self.log("domain_classification_cache_stats", cache_stats)

            # Classify uncached texts
            if texts_to_classify:
                # Tokenize texts
                encoded_texts = self.tokenizer(
                    texts_to_classify,
                    padding=True,
                    truncation=True,
                    max_length=512,  # NVIDIA domain classifier context length
                    return_tensors="pt",
                )

                # Run inference
                with torch.no_grad():
                    logits = self.model(
                        encoded_texts["input_ids"], encoded_texts["attention_mask"]
                    )
                    new_results = self.model.process_logits(logits)

                # Add new results to cache and fill in placeholder positions
                new_result_idx = 0
                for i, result in enumerate(results):
                    if result is None:  # This was a cache miss
                        new_result = new_results[new_result_idx]
                        results[i] = new_result
                        # Cache the result
                        self._add_to_cache(texts[i], new_result)
                        new_result_idx += 1

            self.log(
                "domain_classification_batch_complete",
                {
                    "batch_size": len(results),
                    "cache_hits": len(cache_indices),
                    "cache_misses": len(texts_to_classify),
                    "hit_rate": cache_stats["hit_rate"],
                },
            )

            # Log domain distribution for debugging
            if results:
                domain_counts: dict[str, int] = {}
                for result in results:
                    if result is not None:
                        domain = result.domain.value
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1

                self.log(
                    "domain_classification_distribution",
                    {
                        "domain_counts": domain_counts,
                        "avg_confidence": sum(
                            r.confidence for r in results if r is not None
                        )
                        / len([r for r in results if r is not None]),
                    },
                )

            return [r for r in results if r is not None]

        except Exception as e:
            self.log(
                "domain_classification_error",
                {"error": str(e), "batch_size": len(texts)},
            )
            # Return fallback results with REFERENCE domain
            fallback_results: list[DomainClassificationResult] = []
            for _text in texts:
                fallback_results.append(
                    DomainClassificationResult(
                        domain=DomainType.REFERENCE,
                        confidence=0.5,
                        domain_probabilities={
                            domain.value: 1.0 / len(DomainType) for domain in DomainType
                        },
                    )
                )
            return fallback_results


def get_domain_classifier(lit_logger: Any = None) -> DomainClassifier:
    """Factory function to create domain classifier instance."""
    return DomainClassifier(lit_logger=lit_logger)
