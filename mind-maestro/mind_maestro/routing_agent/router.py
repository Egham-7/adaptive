"""Main ModelRouter class orchestrating intelligent LLM selection."""

import time
from datetime import datetime
from typing import Dict, Any, Optional

from .analyzer import PromptAnalyzer
from .selector import ModelSelector
from .models import (
    RoutingDecision,
    PromptAnalysis,
    ModelSelection,
    RoutingConfig
)


class ModelRouter:
    """Intelligent LLM routing agent that selects optimal models for prompts."""
    
    def __init__(
        self, 
        config: Optional[RoutingConfig] = None,
        use_embeddings: bool = True,
        version: str = "1.0"
    ):
        """Initialize the ModelRouter.
        
        Args:
            config: Configuration for routing parameters
            use_embeddings: Whether to use semantic embeddings in analysis
            version: Version of the routing algorithm
        """
        self.config = config or RoutingConfig()
        self.version = version
        
        # Initialize components
        self.analyzer = PromptAnalyzer(use_embeddings=use_embeddings)
        self.selector = ModelSelector(config=self.config)
        
        # Statistics tracking
        self._routing_stats = {
            "total_routes": 0,
            "avg_processing_time": 0.0,
            "task_type_distribution": {},
            "model_selection_frequency": {},
            "complexity_distribution": {}
        }
    
    def route_prompt(self, prompt: str) -> RoutingDecision:
        """Route a prompt to the optimal model.
        
        Args:
            prompt: The input prompt to route
            
        Returns:
            RoutingDecision with complete analysis and model selection
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze the prompt
            prompt_analysis = self.analyzer.analyze_prompt(prompt)
            
            # Step 2: Select optimal model
            model_selection = self.selector.select_optimal_model(prompt_analysis)
            
            # Step 3: Create routing decision
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            routing_decision = RoutingDecision(
                prompt_analysis=prompt_analysis,
                model_selection=model_selection,
                routing_timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                routing_version=self.version
            )
            
            # Update statistics
            self._update_stats(routing_decision)
            
            return routing_decision
            
        except Exception as e:
            # Handle routing errors gracefully
            return self._create_fallback_decision(prompt, str(e), time.time() - start_time)
    
    def route_batch(self, prompts: list[str]) -> list[RoutingDecision]:
        """Route multiple prompts efficiently.
        
        Args:
            prompts: List of prompts to route
            
        Returns:
            List of RoutingDecisions
        """
        return [self.route_prompt(prompt) for prompt in prompts]
    
    def get_model_recommendation(
        self, 
        prompt: str,
        return_reasoning: bool = True
    ) -> Dict[str, Any]:
        """Get a simple model recommendation for a prompt.
        
        Args:
            prompt: The input prompt
            return_reasoning: Whether to include detailed reasoning
            
        Returns:
            Dictionary with model recommendation and optional reasoning
        """
        routing_decision = self.route_prompt(prompt)
        
        result = {
            "recommended_model": routing_decision.model_selection.selected_model.name,
            "confidence": routing_decision.model_selection.confidence_score,
            "task_type": routing_decision.prompt_analysis.task_type.value,
            "complexity": routing_decision.prompt_analysis.complexity.value,
            "processing_time_ms": routing_decision.processing_time_ms
        }
        
        if return_reasoning:
            result.update({
                "reasoning": routing_decision.model_selection.selection_reasoning,
                "alternatives": [
                    alt.name for alt in routing_decision.model_selection.alternatives
                ],
                "expected_performance": routing_decision.model_selection.estimated_performance,
                "model_details": {
                    "company": routing_decision.model_selection.selected_model.company,
                    "parameters": f"{routing_decision.model_selection.selected_model.parameter_count}B",
                    "context_window": routing_decision.model_selection.selected_model.context_window,
                    "efficiency_score": routing_decision.model_selection.selected_model.efficiency_score
                }
            })
        
        return result
    
    def explain_routing_decision(self, routing_decision: RoutingDecision) -> str:
        """Generate a detailed explanation of the routing decision.
        
        Args:
            routing_decision: The routing decision to explain
            
        Returns:
            Human-readable explanation string
        """
        analysis = routing_decision.prompt_analysis
        selection = routing_decision.model_selection
        
        explanation = f"""
🎯 **Routing Decision Analysis**

**Prompt Analysis:**
• Task Type: {analysis.task_type.value.title()} (confidence: {analysis.confidence_score:.1%})
• Complexity: {analysis.complexity.value.title()}
• Domain: {analysis.domain.value.title()}
• Context Required: {analysis.context_length:,} tokens
• Reasoning Steps: {analysis.reasoning_steps_required}

**Selected Model:** {selection.selected_model.name} ({selection.selected_model.company})
• Parameters: {selection.selected_model.parameter_count}B
• Efficiency Score: {selection.selected_model.efficiency_score:.3f}
• Task Relevance: {selection.selected_model.task_relevance_score:.3f}
• Expected Performance: {selection.estimated_performance:.1%}

**Selection Reasoning:**
{selection.selection_reasoning}

**Alternative Models:**
{chr(10).join(f"• {alt.name} (score: {alt.efficiency_score:.3f})" for alt in selection.alternatives[:3])}

**Processing Time:** {routing_decision.processing_time_ms:.1f}ms
        """.strip()
        
        return explanation
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics.
        
        Returns:
            Dictionary with routing statistics
        """
        return {
            "total_routes_processed": self._routing_stats["total_routes"],
            "average_processing_time_ms": self._routing_stats["avg_processing_time"],
            "task_type_distribution": self._routing_stats["task_type_distribution"],
            "most_selected_models": dict(
                sorted(
                    self._routing_stats["model_selection_frequency"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "complexity_distribution": self._routing_stats["complexity_distribution"],
            "routing_version": self.version
        }
    
    def update_config(self, new_config: RoutingConfig) -> None:
        """Update the routing configuration.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        self.selector = ModelSelector(config=new_config)
    
    def benchmark_routing_speed(self, test_prompts: list[str]) -> Dict[str, float]:
        """Benchmark routing speed with test prompts.
        
        Args:
            test_prompts: List of test prompts to benchmark with
            
        Returns:
            Dictionary with performance metrics
        """
        start_time = time.time()
        
        routing_times = []
        for prompt in test_prompts:
            prompt_start = time.time()
            self.route_prompt(prompt)
            routing_times.append((time.time() - prompt_start) * 1000)
        
        total_time = time.time() - start_time
        
        return {
            "total_prompts": len(test_prompts),
            "total_time_seconds": total_time,
            "average_time_per_prompt_ms": sum(routing_times) / len(routing_times),
            "min_routing_time_ms": min(routing_times),
            "max_routing_time_ms": max(routing_times),
            "throughput_prompts_per_second": len(test_prompts) / total_time
        }
    
    def _create_fallback_decision(
        self, 
        prompt: str, 
        error_message: str,
        processing_time: float
    ) -> RoutingDecision:
        """Create a fallback routing decision when analysis fails.
        
        Args:
            prompt: Original prompt
            error_message: Error that occurred
            processing_time: Time taken before error
            
        Returns:
            Fallback RoutingDecision
        """
        from .models import TaskType, ComplexityLevel, Domain, ModelPerformance
        
        # Create minimal analysis
        fallback_analysis = PromptAnalysis(
            task_type=TaskType.GENERAL_QA,
            complexity=ComplexityLevel.MEDIUM,
            domain=Domain.GENERAL,
            context_length=8192,
            confidence_score=0.3
        )
        
        # Select a reliable fallback model (GPT-4o or similar)
        fallback_model_data = None
        for model in self.selector.models_data:
            if "GPT-4o" in model["name"] or "Claude" in model["name"]:
                fallback_model_data = model
                break
        
        if not fallback_model_data:
            fallback_model_data = self.selector.models_data[0]  # First available
        
        fallback_model = ModelPerformance(
            name=fallback_model_data["name"],
            company=fallback_model_data["company"],
            parameter_count=fallback_model_data["parameter_count"],
            benchmarks=fallback_model_data["benchmarks"],
            context_window=fallback_model_data["context_window"],
            efficiency_score=0.5,
            task_relevance_score=0.5,
            cost_efficiency=0.5
        )
        
        fallback_selection = ModelSelection(
            selected_model=fallback_model,
            selection_reasoning=f"Fallback selection due to routing error: {error_message}",
            confidence_score=0.3,
            estimated_performance=0.7
        )
        
        return RoutingDecision(
            prompt_analysis=fallback_analysis,
            model_selection=fallback_selection,
            routing_timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time * 1000,
            routing_version=f"{self.version}-fallback"
        )
    
    def _update_stats(self, routing_decision: RoutingDecision) -> None:
        """Update internal routing statistics.
        
        Args:
            routing_decision: The routing decision to record
        """
        stats = self._routing_stats
        stats["total_routes"] += 1
        
        # Update average processing time
        current_avg = stats["avg_processing_time"]
        new_time = routing_decision.processing_time_ms
        stats["avg_processing_time"] = (
            (current_avg * (stats["total_routes"] - 1) + new_time) / 
            stats["total_routes"]
        )
        
        # Update task type distribution
        task_type = routing_decision.prompt_analysis.task_type.value
        stats["task_type_distribution"][task_type] = (
            stats["task_type_distribution"].get(task_type, 0) + 1
        )
        
        # Update model selection frequency
        model_name = routing_decision.model_selection.selected_model.name
        stats["model_selection_frequency"][model_name] = (
            stats["model_selection_frequency"].get(model_name, 0) + 1
        )
        
        # Update complexity distribution
        complexity = routing_decision.prompt_analysis.complexity.value
        stats["complexity_distribution"][complexity] = (
            stats["complexity_distribution"].get(complexity, 0) + 1
        )