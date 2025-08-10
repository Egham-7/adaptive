"""Pydantic models for routing agent state management and configuration."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task type classification for prompt analysis."""
    MATH = "math"
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL_QA = "general_qa"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


class ComplexityLevel(str, Enum):
    """Complexity level for task difficulty assessment."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class Domain(str, Enum):
    """Domain classification for specialized routing."""
    ACADEMIC = "academic"
    TECHNICAL = "technical" 
    BUSINESS = "business"
    CREATIVE = "creative"
    GENERAL = "general"
    SCIENTIFIC = "scientific"


class PromptAnalysis(BaseModel):
    """Results of prompt analysis for routing decisions."""
    
    task_type: TaskType
    complexity: ComplexityLevel
    domain: Domain
    context_length: int = Field(description="Required context window size")
    keywords: List[str] = Field(default_factory=list, description="Key terms found in prompt")
    semantic_embedding: Optional[List[float]] = Field(default=None, description="Semantic embedding vector")
    requires_multimodal: bool = Field(default=False, description="Whether task needs multimodal capabilities")
    reasoning_steps_required: int = Field(default=1, description="Estimated reasoning complexity")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in classification")


class ModelPerformance(BaseModel):
    """Model performance metrics for selection algorithm."""
    
    name: str
    company: str
    parameter_count: float
    benchmarks: Dict[str, float]
    context_window: int
    efficiency_score: float = Field(description="Calculated efficiency for this task")
    task_relevance_score: float = Field(description="How well suited for the specific task")
    cost_efficiency: float = Field(description="Performance per parameter ratio")


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
    """Configuration parameters for the routing system."""
    
    # Performance weights for efficiency scoring
    performance_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    efficiency_weight: float = Field(default=0.3, ge=0.0, le=1.0) 
    context_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    speed_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Task-specific benchmark priorities
    task_benchmark_weights: Dict[TaskType, Dict[str, float]] = Field(
        default_factory=lambda: {
            TaskType.MATH: {"math": 0.5, "gsm8k": 0.3, "mmlu": 0.2},
            TaskType.CODING: {"humaneval": 0.6, "mmlu": 0.4},
            TaskType.REASONING: {"math": 0.4, "mmlu": 0.6},
            TaskType.GENERAL_QA: {"mmlu": 0.8, "gsm8k": 0.2},
        }
    )
    
    # Minimum performance thresholds per task type
    min_performance_thresholds: Dict[TaskType, float] = Field(
        default_factory=lambda: {
            TaskType.MATH: 70.0,
            TaskType.CODING: 75.0, 
            TaskType.REASONING: 80.0,
            TaskType.GENERAL_QA: 65.0,
        }
    )
    
    # Model preferences for different scenarios  
    prefer_smaller_models_for_simple: bool = Field(default=True)
    max_parameters_for_simple: float = Field(default=20.0)  # Billion parameters
    fallback_to_efficient: bool = Field(default=True)
    
    # Context window requirements
    context_buffer_ratio: float = Field(default=1.2, description="Safety margin for context")
    min_context_window: int = Field(default=4096)
    
    # Classification parameters
    complexity_length_thresholds: Dict[ComplexityLevel, int] = Field(
        default_factory=lambda: {
            ComplexityLevel.SIMPLE: 100,
            ComplexityLevel.MEDIUM: 500, 
            ComplexityLevel.COMPLEX: 2000,
            ComplexityLevel.EXPERT: float('inf')
        }
    )


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