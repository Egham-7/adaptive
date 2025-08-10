"""Intelligent model selector with efficiency scoring using benchmark data."""

import math
from typing import List, Dict, Optional, Tuple
from ..data.model_data import llm_benchmarks
from .models import (
    PromptAnalysis,
    ModelPerformance, 
    ModelSelection,
    TaskType,
    ComplexityLevel,
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
    
    def select_optimal_model(
        self, 
        prompt_analysis: PromptAnalysis
    ) -> ModelSelection:
        """Select the most efficient model for the given prompt analysis.
        
        Args:
            prompt_analysis: Analysis results from PromptAnalyzer
            
        Returns:
            ModelSelection with chosen model and reasoning
        """
        # Filter models by context window requirements
        eligible_models = self._filter_by_context_window(
            self._processed_models, 
            prompt_analysis.context_length
        )
        
        if not eligible_models:
            raise ValueError(f"No models found with sufficient context window ({prompt_analysis.context_length})")
        
        # Calculate efficiency scores for each eligible model
        scored_models = []
        for model in eligible_models:
            efficiency_score = self._calculate_efficiency_score(model, prompt_analysis)
            task_relevance = self._calculate_task_relevance_score(model, prompt_analysis)
            
            model_performance = ModelPerformance(
                name=model["name"],
                company=model["company"],
                parameter_count=model["parameter_count"],
                benchmarks=model["benchmarks"],
                context_window=model["context_window"],
                efficiency_score=efficiency_score,
                task_relevance_score=task_relevance,
                cost_efficiency=model["benchmarks"]["mmlu"] / math.log(model["parameter_count"])
            )
            scored_models.append(model_performance)
        
        # Sort by efficiency score (higher is better)
        scored_models.sort(key=lambda m: m.efficiency_score, reverse=True)
        
        # Apply complexity-based filtering for simple tasks
        if (prompt_analysis.complexity == ComplexityLevel.SIMPLE and 
            self.config.prefer_smaller_models_for_simple):
            small_models = [m for m in scored_models 
                          if m.parameter_count <= self.config.max_parameters_for_simple]
            if small_models:
                scored_models = small_models
        
        # Select best model
        selected_model = scored_models[0]
        alternatives = scored_models[1:4]  # Top 3 alternatives
        
        # Find fallback model (smallest efficient model)
        fallback_model = None
        if self.config.fallback_to_efficient:
            fallback_candidates = sorted(scored_models, key=lambda m: m.parameter_count)
            for candidate in fallback_candidates:
                if self._meets_minimum_threshold(candidate, prompt_analysis.task_type):
                    fallback_model = candidate
                    break
        
        # Generate selection reasoning
        reasoning = self._generate_selection_reasoning(
            selected_model, prompt_analysis, scored_models
        )
        
        # Calculate confidence based on score gap
        confidence = self._calculate_selection_confidence(scored_models)
        
        # Estimate expected performance
        expected_performance = self._estimate_performance(selected_model, prompt_analysis)
        
        return ModelSelection(
            selected_model=selected_model,
            selection_reasoning=reasoning,
            confidence_score=confidence,
            alternatives=alternatives,
            fallback_model=fallback_model,
            estimated_performance=expected_performance
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
    
    def _calculate_efficiency_score(
        self, 
        model: Dict, 
        prompt_analysis: PromptAnalysis
    ) -> float:
        """Calculate the overall efficiency score for a model."""
        # Get task-specific benchmark weights
        task_weights = self.config.task_benchmark_weights.get(
            prompt_analysis.task_type,
            {"mmlu": 1.0}  # Default to MMLU if task not configured
        )
        
        # Calculate weighted performance score
        performance_score = 0.0
        total_weight = 0.0
        
        for benchmark, weight in task_weights.items():
            if benchmark in model["benchmarks"]:
                performance_score += model["benchmarks"][benchmark] * weight
                total_weight += weight
        
        if total_weight > 0:
            performance_score /= total_weight
        else:
            performance_score = model["benchmarks"]["mmlu"]  # Fallback
        
        # Normalize performance score (0-100 scale)
        performance_score = min(performance_score, 100) / 100.0
        
        # Calculate parameter efficiency (performance per parameter)
        param_log = math.log(model["parameter_count"])
        max_param_log = math.log(671.0)  # Largest model in dataset
        efficiency_score = performance_score / (param_log / max_param_log)
        
        # Apply complexity-based adjustments
        if prompt_analysis.complexity == ComplexityLevel.SIMPLE:
            # Boost smaller models for simple tasks
            if model["parameter_count"] < 20:
                efficiency_score *= 1.2
        elif prompt_analysis.complexity == ComplexityLevel.EXPERT:
            # Boost larger, more capable models for expert tasks  
            if model["parameter_count"] > 100:
                efficiency_score *= 1.1
        
        # Context window efficiency bonus
        if model["context_window"] > 500000:  # Very large context
            efficiency_score *= 1.05
        
        # Apply configuration weights
        final_score = (
            performance_score * self.config.performance_weight +
            efficiency_score * self.config.efficiency_weight +
            (1.0 if model["context_window"] >= prompt_analysis.context_length else 0.5) * self.config.context_weight +
            (1.0 / math.log(model["parameter_count"])) * self.config.speed_weight
        )
        
        return min(final_score, 1.0)
    
    def _calculate_task_relevance_score(
        self, 
        model: Dict, 
        prompt_analysis: PromptAnalysis
    ) -> float:
        """Calculate how well-suited a model is for the specific task type."""
        task_type = prompt_analysis.task_type
        benchmarks = model["benchmarks"]
        
        relevance_scores = {
            TaskType.MATH: (
                benchmarks.get("math", 0) * 0.6 + 
                benchmarks.get("gsm8k", 0) * 0.4
            ) / 100.0,
            
            TaskType.CODING: benchmarks.get("humaneval", 0) / 100.0,
            
            TaskType.REASONING: (
                benchmarks.get("math", 0) * 0.4 + 
                benchmarks.get("mmlu", 0) * 0.6
            ) / 100.0,
            
            TaskType.GENERAL_QA: (
                benchmarks.get("mmlu", 0) * 0.8 + 
                benchmarks.get("gsm8k", 0) * 0.2
            ) / 100.0,
            
            # Default to MMLU for other task types
            TaskType.CREATIVE: benchmarks.get("mmlu", 0) / 100.0,
            TaskType.TECHNICAL: benchmarks.get("mmlu", 0) / 100.0,
            TaskType.ANALYSIS: benchmarks.get("mmlu", 0) / 100.0,
            TaskType.CONVERSATIONAL: benchmarks.get("mmlu", 0) / 100.0,
        }
        
        return relevance_scores.get(task_type, benchmarks.get("mmlu", 0) / 100.0)
    
    def _meets_minimum_threshold(
        self, 
        model: ModelPerformance, 
        task_type: TaskType
    ) -> bool:
        """Check if model meets minimum performance threshold for task type."""
        threshold = self.config.min_performance_thresholds.get(task_type, 60.0)
        
        # Check relevant benchmark based on task type
        if task_type == TaskType.MATH:
            return model.benchmarks.get("math", 0) >= threshold
        elif task_type == TaskType.CODING:
            return model.benchmarks.get("humaneval", 0) >= threshold
        else:
            return model.benchmarks.get("mmlu", 0) >= threshold
    
    def _generate_selection_reasoning(
        self, 
        selected_model: ModelPerformance,
        prompt_analysis: PromptAnalysis,
        all_scored_models: List[ModelPerformance]
    ) -> str:
        """Generate human-readable reasoning for model selection."""
        reasoning_parts = []
        
        # Task-specific reasoning
        task_reasons = {
            TaskType.MATH: f"Selected for strong mathematical reasoning (MATH: {selected_model.benchmarks.get('math', 'N/A')}, GSM8K: {selected_model.benchmarks.get('gsm8k', 'N/A')})",
            TaskType.CODING: f"Selected for superior code generation (HumanEval: {selected_model.benchmarks.get('humaneval', 'N/A')})", 
            TaskType.REASONING: f"Selected for advanced reasoning capabilities (MMLU: {selected_model.benchmarks.get('mmlu', 'N/A')})",
            TaskType.GENERAL_QA: f"Selected for balanced general knowledge (MMLU: {selected_model.benchmarks.get('mmlu', 'N/A')})"
        }
        
        if prompt_analysis.task_type in task_reasons:
            reasoning_parts.append(task_reasons[prompt_analysis.task_type])
        
        # Efficiency reasoning
        if selected_model.parameter_count < 50:
            reasoning_parts.append(f"Highly efficient with only {selected_model.parameter_count}B parameters")
        elif selected_model.parameter_count > 200:
            reasoning_parts.append(f"Powerful {selected_model.parameter_count}B parameter model for complex tasks")
        
        # Complexity reasoning
        if prompt_analysis.complexity == ComplexityLevel.SIMPLE:
            reasoning_parts.append("Optimized for simple tasks without unnecessary overhead")
        elif prompt_analysis.complexity == ComplexityLevel.EXPERT:
            reasoning_parts.append("Capable of handling expert-level complexity")
        
        # Context reasoning
        if prompt_analysis.context_length > 100000:
            reasoning_parts.append(f"Supports large context window ({selected_model.context_window:,} tokens)")
        
        # Performance gap reasoning
        if len(all_scored_models) > 1:
            score_gap = selected_model.efficiency_score - all_scored_models[1].efficiency_score
            if score_gap > 0.1:
                reasoning_parts.append("Clear performance advantage over alternatives")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_selection_confidence(
        self, 
        scored_models: List[ModelPerformance]
    ) -> float:
        """Calculate confidence in the selection based on score distribution."""
        if len(scored_models) < 2:
            return 0.8
        
        top_score = scored_models[0].efficiency_score
        second_score = scored_models[1].efficiency_score
        
        # Larger gap means higher confidence
        score_gap = top_score - second_score
        confidence = 0.7 + min(score_gap * 2, 0.25)  # Scale gap to confidence
        
        return min(confidence, 0.95)
    
    def _estimate_performance(
        self, 
        model: ModelPerformance,
        prompt_analysis: PromptAnalysis
    ) -> float:
        """Estimate expected performance for the specific task."""
        base_performance = model.task_relevance_score
        
        # Adjust based on complexity match
        if prompt_analysis.complexity == ComplexityLevel.SIMPLE and model.parameter_count < 20:
            base_performance *= 1.05  # Small models good for simple tasks
        elif prompt_analysis.complexity == ComplexityLevel.EXPERT and model.parameter_count > 100:
            base_performance *= 1.1  # Large models good for complex tasks
        
        # Factor in reasoning requirements
        if prompt_analysis.reasoning_steps_required > 3:
            # Models with strong math/reasoning scores handle multi-step better
            reasoning_bonus = (model.benchmarks.get("math", 70) / 100) * 0.1
            base_performance += reasoning_bonus
        
        return min(base_performance, 1.0)