"""Prompt classifier service with hybrid Modal + local processing architecture.

This module provides the prompt classification interface for the adaptive_ai service.
It implements a hybrid approach where NVIDIA model inference runs on Modal (GPU-accelerated)
while all classification result processing happens locally for faster development iteration.

Architecture:
- NVIDIA model inference: Modal GPU deployment for performance and cost optimization
- Classification processing: Local numpy-based computation for flexibility and speed  
- Raw logits transfer: Modal returns raw model outputs, processed locally into results
- Same interface maintained: Backward compatibility with existing adaptive_ai consumers
- JWT authentication: Secure communication between adaptive_ai and Modal services

Benefits:
- Faster development: Classification logic changes don't require Modal redeployment
- Cost optimization: Modal usage limited to GPU inference only
- Better separation: Clear boundaries between ML inference and business logic
- Local testing: Classification processing can be tested without Modal dependency
"""

import logging
import numpy as np
from typing import Any, Dict, List

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.services.modal_client import ModalPromptClassifier

logger = logging.getLogger(__name__)


# For backward compatibility, we maintain the same class name and interface
class PromptClassifier:
    """Hybrid prompt classifier with Modal inference and local processing.
    
    This class maintains the same interface as the original classifier while implementing
    a hybrid architecture: NVIDIA model inference runs on Modal (GPU-accelerated) and
    classification result processing happens locally using numpy-based computations.
    
    Architecture Benefits:
    - GPU acceleration: NVIDIA model runs on Modal's T4 GPU infrastructure
    - Fast development: Classification logic changes don't require Modal redeployment
    - Cost optimization: Pay only for GPU usage during actual model inference
    - Local flexibility: Result processing, complexity calculations, and mappings run locally
    - Better testing: Classification processing logic can be tested independently
    - Clean separation: Clear boundaries between ML inference and business logic
    """
    
    def __init__(self, lit_logger: Any = None) -> None:
        """Initialize hybrid prompt classifier with Modal API client and local processing.
        
        Sets up the Modal client for NVIDIA model inference and initializes local
        classification processing components including task mappings, weights, and
        complexity calculation logic.
        
        Args:
            lit_logger: Optional LitServe logger for compatibility and metrics
        """
        self.lit_logger = lit_logger
        self._modal_client = ModalPromptClassifier(lit_logger=lit_logger)
        
        # Initialize classification processing components
        self._setup_classification_mappings()
        
        logger.info("Initialized PromptClassifier with Modal API client and local processing")
        
        # Perform health check on initialization
        try:
            health = self._modal_client.health_check()
            if health.get("status") == "healthy":
                logger.info("Modal service health check passed")
            else:
                logger.warning(f"Modal service health check failed: {health}")
        except Exception as e:
            logger.error(f"Modal service health check error: {e}")

    def _setup_classification_mappings(self) -> None:
        """Setup task type mappings and weights for classification processing."""
        # Task type mapping (from original NVIDIA model)
        self.task_type_map = {
            "0": "Open QA", "1": "Closed QA", "2": "Summarization", "3": "Text Generation",
            "4": "Code Generation", "5": "Chatbot", "6": "Classification", "7": "Rewrite",
            "8": "Brainstorming", "9": "Extraction", "10": "Other"
        }
        
        # Weights mapping for different classification targets
        self.weights_map = {
            "creativity_scope": [0.0, 1.0, 2.0, 3.0, 4.0],
            "reasoning": [0.0, 1.0, 2.0, 3.0, 4.0], 
            "contextual_knowledge": [0.0, 1.0, 2.0, 3.0, 4.0],
            "number_of_few_shots": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "domain_knowledge": [0.0, 1.0, 2.0, 3.0, 4.0],
            "no_label_reason": [0.0, 1.0, 2.0, 3.0, 4.0],
            "constraint_ct": [0.0, 1.0, 2.0, 3.0, 4.0]
        }
        
        # Divisor mapping for score normalization
        self.divisor_map = {
            "creativity_scope": 4.0, "reasoning": 4.0, "contextual_knowledge": 4.0,
            "number_of_few_shots": 5.0, "domain_knowledge": 4.0, 
            "no_label_reason": 4.0, "constraint_ct": 4.0
        }
        
        # Task-specific complexity weights
        self.task_complexity_weights = {
            "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15],
            "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],
            "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2],
            "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],
            "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1],
            "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],
            "Classification": [0.1, 0.35, 0.25, 0.2, 0.1],
            "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],
            "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1],
            "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],
            "Other": [0.25, 0.25, 0.2, 0.15, 0.15]
        }

    def classify_prompts(self, prompts: List[str]) -> List[ClassificationResult]:
        """Classify multiple prompts using Modal-deployed NVIDIA model.
        
        This method maintains the same interface as the original local classifier
        but now makes authenticated requests to the Modal API.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            List of classification results, one per prompt
            
        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
            
        logger.info(f"Classifying {len(prompts)} prompts via Modal API")
        
        try:
            # Get raw logits from Modal client
            raw_logits = self._modal_client.classify_prompts_raw(prompts)
            
            # Process raw logits locally into ClassificationResult objects
            results = self._process_raw_modal_response(raw_logits)
            
            logger.info(f"Successfully classified {len(results)} prompts")
            return results
            
        except Exception as e:
            logger.error(f"Prompt classification failed: {e}")
            
            # Log to LitServe if available
            if self.lit_logger:
                self.lit_logger.log("prompt_classification_error", {"error": str(e)})
                
            # Re-raise the exception
            raise

    async def classify_prompts_async(self, prompts: List[str]) -> List[ClassificationResult]:
        """Classify multiple prompts using Modal-deployed NVIDIA model asynchronously.
        
        This method provides async support for the Modal API integration.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            List of classification results, one per prompt
            
        Raises:
            ValueError: If prompts list is empty or invalid
            RuntimeError: If Modal API request fails
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
            
        logger.info(f"Classifying {len(prompts)} prompts via Modal API (async)")
        
        try:
            # Get raw logits from Modal client (async)
            raw_logits = await self._modal_client.classify_prompts_raw_async(prompts)
            
            # Process raw logits locally into ClassificationResult objects
            results = self._process_raw_modal_response(raw_logits)
            
            logger.info(f"Successfully classified {len(results)} prompts (async)")
            return results
            
        except Exception as e:
            logger.error(f"Prompt classification failed (async): {e}")
            
            # Log to LitServe if available
            if self.lit_logger:
                self.lit_logger.log("prompt_classification_error", {"error": str(e)})
                
            # Re-raise the exception
            raise

    def health_check(self) -> dict[str, Any]:
        """Check Modal service health.
        
        Returns:
            Health status information from Modal service
        """
        return self._modal_client.health_check()

    def _compute_results(self, preds: List[List[float]], target: str, decimal: int = 4) -> Any:
        """Compute classification results from raw predictions.
        
        Args:
            preds: Raw prediction logits as nested lists
            target: Target classification type
            decimal: Decimal precision for rounding
            
        Returns:
            Processed results based on target type
        """
        if target == "task_type":
            # Convert to numpy for easier processing
            preds_array = np.array(preds)
            
            # Get top 2 predictions
            top2_indices = np.argsort(preds_array, axis=1)[:, -2:]
            top2_indices = np.flip(top2_indices, axis=1)  # Descending order
            
            # Apply softmax
            exp_preds = np.exp(preds_array - np.max(preds_array, axis=1, keepdims=True))
            softmax_probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            # Get probabilities for top 2
            top2_probs = []
            for i, indices in enumerate(top2_indices):
                top2_probs.append([softmax_probs[i][indices[0]], softmax_probs[i][indices[1]]])
            
            # Convert indices to task type strings
            top2_strings = []
            for indices in top2_indices:
                sample_strings = [self.task_type_map[str(idx)] for idx in indices]
                top2_strings.append(sample_strings)
            
            # Round probabilities and filter low confidence secondary predictions
            top2_prob_rounded = [[round(value, 3) for value in sublist] for sublist in top2_probs]
            
            for i, sublist in enumerate(top2_prob_rounded):
                if sublist[1] < 0.1:
                    top2_strings[i][1] = "NA"
            
            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
            
            return (task_type_1, task_type_2, task_type_prob)
        else:
            # Convert to numpy and apply softmax
            preds_array = np.array(preds)
            exp_preds = np.exp(preds_array - np.max(preds_array, axis=1, keepdims=True))
            softmax_probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            # Apply weights and compute weighted sum
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(softmax_probs * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            scores = [round(value, decimal) for value in scores]
            
            if target == "number_of_few_shots":
                int_scores = [max(0, round(x)) for x in scores]
                return int_scores
            return scores
    
    def _extract_classification_results(self, logits: List[List[List[float]]]) -> Dict[str, List]:
        """Extract classification results from raw model logits.
        
        Args:
            logits: Raw logits from Modal API (list of prediction arrays)
            
        Returns:
            Dictionary with extracted classification results
        """
        result = {}
        
        # Process task type (first logits array)
        task_type_logits = logits[0]
        task_type_results = self._compute_results(task_type_logits, target="task_type")
        if isinstance(task_type_results, tuple):
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]
        
        # Process other classifications
        classifications = [
            ("creativity_scope", logits[1]), ("reasoning", logits[2]),
            ("contextual_knowledge", logits[3]), ("number_of_few_shots", logits[4]),
            ("domain_knowledge", logits[5]), ("no_label_reason", logits[6]),
            ("constraint_ct", logits[7]),
        ]
        
        for target, target_logits in classifications:
            target_results = self._compute_results(target_logits, target=target)
            if isinstance(target_results, list):
                result[target] = target_results
        
        return result
    
    def _calculate_complexity_scores(self, results: Dict[str, List], task_types: List[str]) -> List[float]:
        """Calculate complexity scores based on task types and other metrics.
        
        Args:
            results: Extracted classification results
            task_types: List of primary task types
            
        Returns:
            List of complexity scores (0-1) for each prompt
        """
        complexity_scores = []
        
        for i, task_type in enumerate(task_types):
            weights = self.task_complexity_weights.get(task_type, [0.3, 0.3, 0.2, 0.1, 0.1])
            
            score = round(
                weights[0] * results.get("creativity_scope", [])[i] +
                weights[1] * results.get("reasoning", [])[i] +
                weights[2] * results.get("constraint_ct", [])[i] +
                weights[3] * results.get("domain_knowledge", [])[i] +
                weights[4] * results.get("contextual_knowledge", [])[i], 5
            )
            complexity_scores.append(score)
        
        return complexity_scores
    
    def _process_raw_modal_response(self, raw_logits: List[List[List[float]]]) -> List[ClassificationResult]:
        """Process raw logits from Modal into ClassificationResult objects.
        
        Args:
            raw_logits: Raw model logits from Modal API
            
        Returns:
            List of ClassificationResult objects
        """
        # Extract all classifications from logits
        batch_results = self._extract_classification_results(raw_logits)
        
        # Calculate complexity scores if we have task types
        if "task_type_1" in batch_results:
            task_types = batch_results["task_type_1"]
            complexity_scores = self._calculate_complexity_scores(batch_results, task_types)
            batch_results["prompt_complexity_score"] = complexity_scores
        
        # Convert batch results to individual ClassificationResult objects
        num_prompts = len(batch_results.get("task_type_1", []))
        results = []
        
        for i in range(num_prompts):
            # Extract single-prompt data from batch results
            single_result = {}
            for key, value in batch_results.items():
                if isinstance(value, list) and len(value) > i:
                    single_result[key] = [value[i]]  # Wrap in list for consistency
                else:
                    single_result[key] = value
            
            results.append(ClassificationResult(**single_result))
        
        return results


def get_prompt_classifier(lit_logger: Any = None) -> PromptClassifier:
    """Get prompt classifier instance.
    
    This function maintains the same interface as before but now returns
    a Modal API client-based classifier instead of the local GPU model.
    
    Args:
        lit_logger: Optional LitServe logger
        
    Returns:
        PromptClassifier instance using Modal API
    """
    return PromptClassifier(lit_logger=lit_logger)
