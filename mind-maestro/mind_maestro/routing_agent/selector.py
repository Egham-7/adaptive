"""Intelligent model selector with efficiency scoring using benchmark data."""

import math
from typing import List, Dict, Optional, Tuple
from ..data.model_data import llm_benchmarks
from .models import (
    PromptAnalysis,
    ModelPerformance, 
    ModelSelection,
    RoutingConfig
)


class ModelSelector:
    """Selects the most efficient model based on task requirements and benchmark data."""
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize the model selector.
        
        Args:
            config: Configuration for routing parameters and thresholds
        """
        self.config = config or RoutingConfig()
        self.models_data = llm_benchmarks["models"]
        
        # Pre-process model data for faster selection
        self._processed_models = self._process_model_data()
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models for AI agent routing.
        
        Returns:
            List of model data dictionaries
        """
        return self.models_data
    
    def create_model_selection(
        self,
        selected_model_data: Dict,
        reasoning: str,
        confidence: float,
        estimated_performance: float
    ) -> ModelSelection:
        """Create ModelSelection object from AI agent decision.
        
        Args:
            selected_model_data: Raw model data dictionary
            reasoning: AI agent's reasoning
            confidence: Confidence score
            estimated_performance: Performance estimate
            
        Returns:
            ModelSelection object
        """
        # Create ModelPerformance object
        selected_model = ModelPerformance(
            name=selected_model_data["name"],
            company=selected_model_data["company"],
            parameter_count=selected_model_data["parameter_count"],
            benchmarks=selected_model_data["benchmarks"],
            context_window=selected_model_data["context_window"],
            efficiency_score=0.8,  # AI agent makes the decision
            task_relevance_score=0.8,
            cost_efficiency=selected_model_data["benchmarks"]["mmlu"] / math.log(selected_model_data["parameter_count"])
        )
        
        # Get alternative models (next best options)
        alternatives = []
        for model_data in self.models_data[:3]:  # Top 3 alternatives
            if model_data["name"] != selected_model_data["name"]:
                alt_model = ModelPerformance(
                    name=model_data["name"],
                    company=model_data["company"],
                    parameter_count=model_data["parameter_count"],
                    benchmarks=model_data["benchmarks"],
                    context_window=model_data["context_window"],
                    efficiency_score=0.7,
                    task_relevance_score=0.7,
                    cost_efficiency=model_data["benchmarks"]["mmlu"] / math.log(model_data["parameter_count"])
                )
                alternatives.append(alt_model)
        
        return ModelSelection(
            selected_model=selected_model,
            selection_reasoning=reasoning,
            confidence_score=confidence,
            alternatives=alternatives[:3],
            estimated_performance=estimated_performance
        )
    
    def _process_model_data(self) -> List[Dict]:
        """Pre-process model data for efficient selection."""
        processed = []
        
        for model in self.models_data:
            # Add derived metrics
            processed_model = model.copy()
            
            # Calculate parameter efficiency (MMLU per billion parameters)
            param_efficiency = model["benchmarks"]["mmlu"] / model["parameter_count"]
            processed_model["param_efficiency"] = param_efficiency
            
            # Calculate average benchmark score
            benchmarks = model["benchmarks"]
            avg_benchmark = sum(benchmarks.values()) / len(benchmarks)
            processed_model["avg_benchmark"] = avg_benchmark
            
            # Add task-specific strengths
            processed_model["math_strength"] = benchmarks.get("math", 0) + benchmarks.get("gsm8k", 0)
            processed_model["coding_strength"] = benchmarks.get("humaneval", 0)
            processed_model["general_strength"] = benchmarks.get("mmlu", 0)
            
            processed.append(processed_model)
        
        return processed
    
    def _filter_by_context_window(
        self, 
        models: List[Dict], 
        required_context: int
    ) -> List[Dict]:
        """Filter models by context window requirements."""
        # Apply safety buffer
        required_with_buffer = int(required_context * self.config.context_buffer_ratio)
        
        return [
            model for model in models 
            if model["context_window"] >= required_with_buffer
        ]
    
    # Legacy methods removed - AI agent now handles all routing decisions