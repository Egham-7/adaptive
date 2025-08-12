"""Simplified models for AI agent routing decisions."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PromptAnalysis(BaseModel):
    """AI agent analysis of prompt for routing decisions."""
    
    analysis_reasoning: str = Field(description="AI agent's reasoning about the prompt")
    context_length: int = Field(description="Required context window size")
    requires_multimodal: bool = Field(default=False, description="Whether task needs multimodal capabilities")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in analysis")


class ModelPerformance(BaseModel):
    """Model performance metrics for selection algorithm."""
    
    name: str
    company: str
    parameter_count: float
    benchmarks: Dict[str, float]
    context_window: int
    pricing: Optional[Dict[str, float]] = Field(default=None, description="Pricing per 1M tokens")
    efficiency_score: float = Field(description="Calculated efficiency for this task")
    task_relevance_score: float = Field(description="How well suited for the specific task")
    cost_efficiency: float = Field(description="Performance per cost ratio")


class ModelSelection(BaseModel):
    """Selected model with reasoning and alternatives."""
    
    selected_model: ModelPerformance
    selection_reasoning: str = Field(description="Why this model was chosen")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in selection")
    alternatives: List[ModelPerformance] = Field(default_factory=list, description="Next best options")
    fallback_model: Optional[ModelPerformance] = Field(default=None, description="Emergency fallback")
    estimated_performance: float = Field(description="Expected performance for this task")


class RoutingDecision(BaseModel):
    """Complete routing decision with analysis and selection."""
    
    prompt_analysis: PromptAnalysis
    model_selection: ModelSelection
    routing_timestamp: str = Field(description="When routing decision was made")
    processing_time_ms: float = Field(description="Time taken for routing decision")
    routing_version: str = Field(default="1.0", description="Version of routing algorithm used")


class RoutingConfig(BaseModel):
    """Simple configuration for AI agent routing."""
    
    # Model selection preferences
    prefer_efficiency: bool = Field(default=True, description="Prefer efficient models when possible")
    context_buffer_ratio: float = Field(default=1.2, description="Safety margin for context")
    min_context_window: int = Field(default=4096)
    routing_model: str = Field(default="gpt-4o-mini", description="Model to use for routing decisions")


class RoutingState(BaseModel):
    """State management for LangGraph workflow."""
    
    original_prompt: str
    prompt_analysis: Optional[PromptAnalysis] = None
    candidate_models: List[ModelPerformance] = Field(default_factory=list)
    model_selection: Optional[ModelSelection] = None
    routing_decision: Optional[RoutingDecision] = None
    error_message: Optional[str] = None
    workflow_stage: str = Field(default="initialized")
    debug_info: Dict[str, Any] = Field(default_factory=dict)