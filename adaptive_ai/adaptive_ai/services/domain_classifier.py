from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import PyTorchModelHubMixin

from adaptive_ai.models.llm_classification_models import (
    DomainClassificationResult,
    DomainType,
)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """NVIDIA Domain Classifier following official documentation."""
    
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))
        self.config = config

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
                domain_probabilities=domain_probabilities
            )
            results.append(result)
        
        return results


class DomainClassifier:
    def __init__(self, lit_logger: Any = None) -> None:
        self.lit_logger = lit_logger
        
        # Load NVIDIA domain classifier following official documentation
        try:
            self.config = AutoConfig.from_pretrained("nvidia/domain-classifier")
            self.tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
            
            # Load the model using the CustomModel class from official docs with safetensors
            self.model = CustomModel.from_pretrained("nvidia/domain-classifier", use_safetensors=True)
            self.model.eval()
            
            self.log("domain_classifier_weights_loaded", {"status": "success", "source": "nvidia/domain-classifier"})
        except Exception as e:
            self.log("domain_classifier_weights_load_failed", {"error": str(e)})
            # Fallback to custom implementation if NVIDIA loading fails
            self.log("domain_classifier_fallback", {"status": "using_custom_implementation"})
            # Use the original custom implementation as fallback
            self._setup_custom_fallback(lit_logger)

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _setup_custom_fallback(self, lit_logger: Any) -> None:
        """Setup custom domain classifier as fallback when NVIDIA model fails."""
        # Create a custom model that mimics the NVIDIA architecture
        class CustomDomainClassifier(nn.Module):
            def __init__(self, config, lit_logger: Any = None):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(
                    "microsoft/deberta-v3-base", use_safetensors=True
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(self.backbone.config.hidden_size, len(config.id2label))
                self.config = config
                self.lit_logger = lit_logger
                
            def forward(self, input_ids, attention_mask):
                outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                # Use first token like NVIDIA implementation
                features = outputs.last_hidden_state[:, 0, :]
                logits = self.fc(features)
                return torch.softmax(logits, dim=1)
                
            def process_logits(self, logits: torch.Tensor) -> list[DomainClassificationResult]:
                """Process logits and return domain classification results."""
                batch_size = logits.shape[0]
                probabilities = logits
                
                results = []
                for i in range(batch_size):
                    sample_probs = probabilities[i].detach().cpu().numpy()
                    
                    # Get top domain
                    top_domain_idx = np.argmax(sample_probs)
                    top_domain = self.config.id2label[str(top_domain_idx)]
                    confidence = float(sample_probs[top_domain_idx])
                    
                    # Create probability dictionary using official labels
                    domain_probabilities = {
                        self.config.id2label[str(idx)]: float(prob) 
                        for idx, prob in enumerate(sample_probs)
                    }
                    
                    # Convert to DomainType enum
                    try:
                        domain_enum = DomainType(top_domain)
                    except ValueError:
                        # Fallback to a default domain if enum conversion fails
                        domain_enum = DomainType.REFERENCE
                        if self.lit_logger:
                            self.lit_logger.log("domain_enum_conversion_failed", {"domain": top_domain})
                    
                    result = DomainClassificationResult(
                        domain=domain_enum,
                        confidence=confidence,
                        domain_probabilities=domain_probabilities
                    )
                    results.append(result)
                
                return results
        
        self.model = CustomDomainClassifier(self.config, lit_logger)

    def classify_domains(self, texts: list[str]) -> list[DomainClassificationResult]:
        """
        Classify multiple texts into domains using batch processing.

        Args:
            texts: List of texts to classify

        Returns:
            List of domain classification results, one per text
        """
        if self.lit_logger:
            self.lit_logger.log(
                "domain_classification_batch_start", {"batch_size": len(texts)}
            )

        # Tokenize texts
        encoded_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # NVIDIA domain classifier context length
            return_tensors="pt",
        )

        # Run inference
        with torch.no_grad():
            logits = self.model(encoded_texts["input_ids"], encoded_texts["attention_mask"])
            results = self.model.process_logits(logits)

        if self.lit_logger:
            self.lit_logger.log(
                "domain_classification_batch_complete", {"batch_size": len(results)}
            )

        print(
            f"Domain classification complete: {len(results)} results for {len(texts)} texts"
        )
        return results


def get_domain_classifier(lit_logger: Any = None) -> DomainClassifier:
    """Factory function to create domain classifier instance."""
    return DomainClassifier(lit_logger=lit_logger)