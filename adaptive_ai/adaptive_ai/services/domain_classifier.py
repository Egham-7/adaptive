# mypy: disable-error-code=import
from typing import Any

import cachetools
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

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        super().__init__()
        # Extract use_safetensors from kwargs if present
        use_safetensors = kwargs.pop("use_safetensors", True)
        self.model = AutoModel.from_pretrained(
            config["base_model"], use_safetensors=use_safetensors
        )
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
            domain_enum = DomainType(top_domain)

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
        self.cache: cachetools.TTLCache[str, DomainClassificationResult] = (
            cachetools.TTLCache(maxsize=cache_size, ttl=cache_ttl)
        )

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
            # Re-raise the exception instead of using fallback
            raise RuntimeError(f"Failed to load NVIDIA domain classifier: {e!s}") from e

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

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
                cached_result = self.cache.get(text)
                if cached_result:
                    results.append(cached_result)
                    cache_indices.append(i)
                else:
                    results.append(None)  # Placeholder
                    texts_to_classify.append(text)

            # Log cache performance
            cache_stats: dict[str, Any] = {
                "size": len(self.cache),
                "max_size": self.cache.maxsize,
                "ttl": self.cache.ttl,
                "hits": 0,  # cachetools doesn't track this by default
                "misses": 0,  # cachetools doesn't track this by default
                "evictions": 0,  # cachetools doesn't track this by default
                "hit_rate": 0.0,  # cachetools doesn't track this by default
            }
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
                        self.cache[texts[i]] = new_result
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
            # Re-raise the exception instead of returning fallback results
            raise RuntimeError(f"Domain classification failed: {e!s}") from e


def get_domain_classifier(lit_logger: Any = None) -> DomainClassifier:
    """Factory function to create domain classifier instance."""
    return DomainClassifier(lit_logger=lit_logger)
