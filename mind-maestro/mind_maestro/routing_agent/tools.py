"""LangGraph tools and utilities for the routing agent."""

from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .router import ModelRouter
from .models import RoutingDecision, RoutingConfig
from .graph import ModelRoutingGraph


class PromptRoutingInput(BaseModel):
    """Input schema for prompt routing tool."""
    prompt: str = Field(description="The prompt to route to an optimal model")
    include_reasoning: bool = Field(default=True, description="Whether to include detailed reasoning")
    include_alternatives: bool = Field(default=True, description="Whether to include alternative models")


class BatchRoutingInput(BaseModel):
    """Input schema for batch routing tool."""
    prompts: List[str] = Field(description="List of prompts to route")
    max_concurrent: int = Field(default=5, description="Maximum concurrent routing operations")


class ModelRecommendationTool(BaseTool):
    """Tool for getting model recommendations using the routing agent."""
    
    name: str = "model_recommendation"
    description: str = """
    Get an intelligent model recommendation for a given prompt.
    Analyzes the prompt to determine task type, complexity, and requirements,
    then selects the most efficient model from available options.
    """
    args_schema = PromptRoutingInput
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        super().__init__()
        self.router = ModelRouter(config=config)
    
    def _run(self, prompt: str, include_reasoning: bool = True, include_alternatives: bool = True) -> Dict[str, Any]:
        """Execute the model recommendation."""
        try:
            routing_decision = self.router.route_prompt(prompt)
            
            result = {
                "recommended_model": routing_decision.model_selection.selected_model.name,
                "confidence": routing_decision.model_selection.confidence_score,
                "task_type": routing_decision.prompt_analysis.task_type.value,
                "complexity": routing_decision.prompt_analysis.complexity.value,
                "processing_time_ms": routing_decision.processing_time_ms,
                "success": True
            }
            
            if include_reasoning:
                result["reasoning"] = routing_decision.model_selection.selection_reasoning
                result["model_details"] = {
                    "company": routing_decision.model_selection.selected_model.company,
                    "parameters": f"{routing_decision.model_selection.selected_model.parameter_count}B",
                    "context_window": routing_decision.model_selection.selected_model.context_window,
                    "efficiency_score": routing_decision.model_selection.selected_model.efficiency_score,
                    "benchmarks": routing_decision.model_selection.selected_model.benchmarks
                }
            
            if include_alternatives:
                result["alternatives"] = [
                    {
                        "name": alt.name,
                        "efficiency_score": alt.efficiency_score,
                        "parameter_count": alt.parameter_count
                    }
                    for alt in routing_decision.model_selection.alternatives
                ]
                
                if routing_decision.model_selection.fallback_model:
                    result["fallback_model"] = routing_decision.model_selection.fallback_model.name
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "recommended_model": "GPT-4o",  # Safe fallback
                "success": False
            }


class BatchRoutingTool(BaseTool):
    """Tool for routing multiple prompts efficiently."""
    
    name: str = "batch_model_routing" 
    description: str = """
    Route multiple prompts to optimal models in batch.
    Efficiently processes multiple prompts and returns routing decisions for each.
    """
    args_schema = BatchRoutingInput
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        super().__init__()
        self.router = ModelRouter(config=config)
    
    def _run(self, prompts: List[str], max_concurrent: int = 5) -> Dict[str, Any]:
        """Execute batch routing."""
        try:
            routing_decisions = self.router.route_batch(prompts)
            
            results = []
            for i, decision in enumerate(routing_decisions):
                result = {
                    "prompt_index": i,
                    "prompt_preview": prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i],
                    "recommended_model": decision.model_selection.selected_model.name,
                    "task_type": decision.prompt_analysis.task_type.value,
                    "complexity": decision.prompt_analysis.complexity.value,
                    "confidence": decision.model_selection.confidence_score,
                    "processing_time_ms": decision.processing_time_ms
                }
                results.append(result)
            
            # Calculate batch statistics
            total_time = sum(r["processing_time_ms"] for r in results)
            model_counts = {}
            for r in results:
                model = r["recommended_model"]
                model_counts[model] = model_counts.get(model, 0) + 1
            
            return {
                "results": results,
                "batch_stats": {
                    "total_prompts": len(prompts),
                    "total_processing_time_ms": total_time,
                    "average_time_per_prompt_ms": total_time / len(prompts),
                    "model_distribution": model_counts,
                    "most_selected_model": max(model_counts.items(), key=lambda x: x[1])[0]
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }


class RoutingAnalysisTool(BaseTool):
    """Tool for analyzing routing performance and statistics."""
    
    name: str = "routing_analysis"
    description: str = """
    Analyze routing performance and get statistics about model selection patterns.
    Useful for understanding routing behavior and optimizing configurations.
    """
    
    def __init__(self, router: ModelRouter):
        super().__init__()
        self.router = router
    
    def _run(self) -> Dict[str, Any]:
        """Get routing analysis and statistics."""
        try:
            stats = self.router.get_routing_statistics()
            
            # Add additional analysis
            analysis = {
                "performance_summary": {
                    "total_routes": stats["total_routes_processed"],
                    "avg_response_time": f"{stats['average_processing_time_ms']:.1f}ms",
                    "routing_version": stats["routing_version"]
                },
                "task_analysis": {
                    "most_common_task": max(stats["task_type_distribution"].items(), key=lambda x: x[1])[0] if stats["task_type_distribution"] else "N/A",
                    "task_distribution": stats["task_type_distribution"],
                    "complexity_distribution": stats["complexity_distribution"]
                },
                "model_preferences": {
                    "most_selected": list(stats["most_selected_models"].keys())[:5],
                    "selection_counts": stats["most_selected_models"]
                },
                "efficiency_insights": self._generate_efficiency_insights(stats)
            }
            
            return {
                "analysis": analysis,
                "raw_statistics": stats,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def _generate_efficiency_insights(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Generate insights about routing efficiency."""
        insights = {}
        
        if stats["total_routes_processed"] > 0:
            avg_time = stats["average_processing_time_ms"]
            
            if avg_time < 50:
                insights["speed"] = "Excellent - Very fast routing"
            elif avg_time < 100:
                insights["speed"] = "Good - Fast routing"
            elif avg_time < 200:
                insights["speed"] = "Moderate - Acceptable routing speed"
            else:
                insights["speed"] = "Slow - Consider optimization"
            
            # Model diversity
            model_count = len(stats["most_selected_models"])
            if model_count > 5:
                insights["diversity"] = "High model diversity - Good coverage"
            elif model_count > 2:
                insights["diversity"] = "Moderate diversity - Reasonable coverage"  
            else:
                insights["diversity"] = "Low diversity - May indicate routing bias"
        
        return insights


class LangGraphRoutingTool(BaseTool):
    """Tool that uses LangGraph workflow for advanced routing."""
    
    name: str = "langgraph_routing"
    description: str = """
    Advanced routing using LangGraph workflow with state management.
    Provides detailed workflow execution information and enhanced error handling.
    """
    args_schema = PromptRoutingInput
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        super().__init__()
        self.routing_graph = ModelRoutingGraph(config)
    
    def _run(self, prompt: str, include_reasoning: bool = True, include_alternatives: bool = True) -> Dict[str, Any]:
        """Execute LangGraph-based routing."""
        try:
            routing_decision = self.routing_graph.route_prompt(prompt)
            
            result = {
                "recommended_model": routing_decision.model_selection.selected_model.name,
                "confidence": routing_decision.model_selection.confidence_score,
                "task_type": routing_decision.prompt_analysis.task_type.value,
                "complexity": routing_decision.prompt_analysis.complexity.value,
                "processing_time_ms": routing_decision.processing_time_ms,
                "routing_method": "langgraph_workflow",
                "success": True
            }
            
            if include_reasoning:
                result["reasoning"] = routing_decision.model_selection.selection_reasoning
                result["workflow_version"] = routing_decision.routing_version
                
            if include_alternatives:
                result["alternatives"] = [alt.name for alt in routing_decision.model_selection.alternatives]
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "recommended_model": "GPT-4o",
                "routing_method": "langgraph_workflow",
                "success": False
            }


class RoutingToolkit:
    """Collection of routing tools for easy integration."""
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize the routing toolkit.
        
        Args:
            config: Configuration for routing parameters
        """
        self.config = config or RoutingConfig()
        self.router = ModelRouter(config=self.config)
        
        # Initialize tools
        self.model_recommendation = ModelRecommendationTool(config)
        self.batch_routing = BatchRoutingTool(config)
        self.routing_analysis = RoutingAnalysisTool(self.router)
        self.langgraph_routing = LangGraphRoutingTool(config)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all available routing tools.
        
        Returns:
            List of routing tools
        """
        return [
            self.model_recommendation,
            self.batch_routing, 
            self.routing_analysis,
            self.langgraph_routing
        ]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        tools = self.get_all_tools()
        return {tool.name: tool.description for tool in tools}
    
    def quick_route(self, prompt: str) -> str:
        """Quick routing that returns just the model name.
        
        Args:
            prompt: The prompt to route
            
        Returns:
            Name of the recommended model
        """
        try:
            decision = self.router.route_prompt(prompt)
            return decision.model_selection.selected_model.name
        except Exception:
            return "GPT-4o"  # Safe fallback
    
    def explain_routing(self, prompt: str) -> str:
        """Get a detailed explanation of routing decision.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Human-readable routing explanation
        """
        try:
            decision = self.router.route_prompt(prompt)
            return self.router.explain_routing_decision(decision)
        except Exception as e:
            return f"Routing explanation failed: {str(e)}"