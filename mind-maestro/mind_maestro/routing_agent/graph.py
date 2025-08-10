"""LangGraph workflow for intelligent model routing with state management."""

from typing import Dict, Any, TypedDict, Annotated, Literal
import time
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from .models import (
    RoutingState,
    PromptAnalysis, 
    ModelSelection,
    RoutingDecision,
    RoutingConfig
)
from .analyzer import PromptAnalyzer
from .selector import ModelSelector


class RoutingGraphState(TypedDict):
    """State schema for the LangGraph routing workflow."""
    
    # Input/Output
    prompt: str
    routing_decision: Dict[str, Any]
    
    # Intermediate state  
    analysis_result: Dict[str, Any]
    candidate_models: list[Dict[str, Any]]
    selection_result: Dict[str, Any]
    
    # Metadata
    workflow_stage: str
    processing_start_time: float
    error_message: str
    debug_info: Dict[str, Any]
    
    # LangGraph message support
    messages: Annotated[list[BaseMessage], add_messages]


class ModelRoutingGraph:
    """LangGraph-powered routing workflow for model selection."""
    
    def __init__(self, config: RoutingConfig = None):
        """Initialize the routing graph.
        
        Args:
            config: Configuration for routing parameters
        """
        self.config = config or RoutingConfig()
        self.analyzer = PromptAnalyzer(use_embeddings=True)
        self.selector = ModelSelector(config=self.config)
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for routing."""
        # Define the workflow graph
        workflow = StateGraph(RoutingGraphState)
        
        # Add nodes
        workflow.add_node("analyze_prompt", self._analyze_prompt_node)
        workflow.add_node("select_model", self._select_model_node) 
        workflow.add_node("validate_selection", self._validate_selection_node)
        workflow.add_node("format_response", self._format_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_prompt")
        
        # Conditional routing based on analysis success
        workflow.add_conditional_edges(
            "analyze_prompt",
            self._should_continue_after_analysis,
            {
                "continue": "select_model",
                "error": "handle_error"
            }
        )
        
        # Conditional routing based on selection success
        workflow.add_conditional_edges(
            "select_model", 
            self._should_continue_after_selection,
            {
                "continue": "validate_selection", 
                "error": "handle_error"
            }
        )
        
        # Validation routing
        workflow.add_conditional_edges(
            "validate_selection",
            self._should_continue_after_validation,
            {
                "continue": "format_response",
                "retry": "select_model", 
                "error": "handle_error"
            }
        )
        
        # End points
        workflow.add_edge("format_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def route_prompt(self, prompt: str) -> RoutingDecision:
        """Route a prompt through the LangGraph workflow.
        
        Args:
            prompt: The input prompt to route
            
        Returns:
            RoutingDecision with complete routing information
        """
        # Initialize state
        initial_state = RoutingGraphState(
            prompt=prompt,
            routing_decision={},
            analysis_result={},
            candidate_models=[],
            selection_result={},
            workflow_stage="initialized",
            processing_start_time=time.time(),
            error_message="",
            debug_info={},
            messages=[]
        )
        
        # Execute the workflow
        final_state = self.graph.invoke(initial_state)
        
        # Extract routing decision
        if final_state["routing_decision"]:
            return self._state_to_routing_decision(final_state)
        else:
            # Return error state
            return self._create_error_routing_decision(
                prompt, 
                final_state.get("error_message", "Unknown routing error"),
                time.time() - final_state["processing_start_time"]
            )
    
    def _analyze_prompt_node(self, state: RoutingGraphState) -> RoutingGraphState:
        """Node: Analyze the input prompt."""
        try:
            state["workflow_stage"] = "analyzing_prompt"
            
            # Perform prompt analysis
            prompt_analysis = self.analyzer.analyze_prompt(state["prompt"])
            
            # Store results in state
            state["analysis_result"] = {
                "task_type": prompt_analysis.task_type.value,
                "complexity": prompt_analysis.complexity.value,
                "domain": prompt_analysis.domain.value,
                "context_length": prompt_analysis.context_length,
                "keywords": prompt_analysis.keywords,
                "requires_multimodal": prompt_analysis.requires_multimodal,
                "reasoning_steps_required": prompt_analysis.reasoning_steps_required,
                "confidence_score": prompt_analysis.confidence_score,
                "semantic_embedding": prompt_analysis.semantic_embedding
            }
            
            state["debug_info"]["analysis_time"] = time.time() - state["processing_start_time"]
            
        except Exception as e:
            state["error_message"] = f"Prompt analysis failed: {str(e)}"
            state["workflow_stage"] = "analysis_error"
        
        return state
    
    def _select_model_node(self, state: RoutingGraphState) -> RoutingGraphState:
        """Node: Select the optimal model based on analysis."""
        try:
            state["workflow_stage"] = "selecting_model"
            
            # Reconstruct PromptAnalysis from state
            analysis_data = state["analysis_result"]
            prompt_analysis = PromptAnalysis(
                task_type=analysis_data["task_type"],
                complexity=analysis_data["complexity"], 
                domain=analysis_data["domain"],
                context_length=analysis_data["context_length"],
                keywords=analysis_data["keywords"],
                requires_multimodal=analysis_data["requires_multimodal"],
                reasoning_steps_required=analysis_data["reasoning_steps_required"],
                confidence_score=analysis_data["confidence_score"],
                semantic_embedding=analysis_data.get("semantic_embedding")
            )
            
            # Perform model selection
            model_selection = self.selector.select_optimal_model(prompt_analysis)
            
            # Store results
            state["selection_result"] = {
                "selected_model": {
                    "name": model_selection.selected_model.name,
                    "company": model_selection.selected_model.company,
                    "parameter_count": model_selection.selected_model.parameter_count,
                    "benchmarks": model_selection.selected_model.benchmarks,
                    "context_window": model_selection.selected_model.context_window,
                    "efficiency_score": model_selection.selected_model.efficiency_score,
                    "task_relevance_score": model_selection.selected_model.task_relevance_score,
                    "cost_efficiency": model_selection.selected_model.cost_efficiency
                },
                "selection_reasoning": model_selection.selection_reasoning,
                "confidence_score": model_selection.confidence_score,
                "alternatives": [
                    {
                        "name": alt.name,
                        "efficiency_score": alt.efficiency_score,
                        "parameter_count": alt.parameter_count
                    } for alt in model_selection.alternatives
                ],
                "fallback_model": {
                    "name": model_selection.fallback_model.name,
                    "parameter_count": model_selection.fallback_model.parameter_count
                } if model_selection.fallback_model else None,
                "estimated_performance": model_selection.estimated_performance
            }
            
            state["debug_info"]["selection_time"] = time.time() - state["processing_start_time"]
            
        except Exception as e:
            state["error_message"] = f"Model selection failed: {str(e)}"
            state["workflow_stage"] = "selection_error"
        
        return state
    
    def _validate_selection_node(self, state: RoutingGraphState) -> RoutingGraphState:
        """Node: Validate the model selection meets requirements."""
        try:
            state["workflow_stage"] = "validating_selection"
            
            selection = state["selection_result"]
            analysis = state["analysis_result"]
            
            # Validation checks
            validation_results = {
                "context_sufficient": True,
                "performance_adequate": True,
                "complexity_match": True,
                "validation_passed": True
            }
            
            # Check context window
            selected_model = selection["selected_model"]
            required_context = analysis["context_length"]
            if selected_model["context_window"] < required_context:
                validation_results["context_sufficient"] = False
                validation_results["validation_passed"] = False
            
            # Check minimum performance threshold
            task_type = analysis["task_type"]
            min_threshold = self.config.min_performance_thresholds.get(task_type, 60.0)
            
            relevant_benchmark = "mmlu"  # Default
            if task_type == "math":
                relevant_benchmark = "math"
            elif task_type == "coding":
                relevant_benchmark = "humaneval"
            
            if selected_model["benchmarks"].get(relevant_benchmark, 0) < min_threshold:
                validation_results["performance_adequate"] = False
                validation_results["validation_passed"] = False
            
            # Store validation results
            state["debug_info"]["validation_results"] = validation_results
            
            if not validation_results["validation_passed"]:
                state["error_message"] = "Selected model failed validation checks"
                state["workflow_stage"] = "validation_failed"
            
        except Exception as e:
            state["error_message"] = f"Validation failed: {str(e)}"
            state["workflow_stage"] = "validation_error"
        
        return state
    
    def _format_response_node(self, state: RoutingGraphState) -> RoutingGraphState:
        """Node: Format the final routing decision."""
        try:
            state["workflow_stage"] = "formatting_response"
            
            processing_time = (time.time() - state["processing_start_time"]) * 1000
            
            # Create final routing decision
            routing_decision = {
                "prompt_analysis": state["analysis_result"],
                "model_selection": state["selection_result"],
                "routing_timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time,
                "routing_version": "1.0-langgraph",
                "workflow_debug": state["debug_info"]
            }
            
            state["routing_decision"] = routing_decision
            state["workflow_stage"] = "completed"
            
        except Exception as e:
            state["error_message"] = f"Response formatting failed: {str(e)}"
            state["workflow_stage"] = "formatting_error"
        
        return state
    
    def _handle_error_node(self, state: RoutingGraphState) -> RoutingGraphState:
        """Node: Handle errors and create fallback response."""
        state["workflow_stage"] = "handling_error"
        
        # Create minimal fallback decision
        processing_time = (time.time() - state["processing_start_time"]) * 1000
        
        fallback_decision = {
            "error": state["error_message"],
            "fallback_model": "GPT-4o",  # Safe fallback
            "processing_time_ms": processing_time,
            "routing_version": "1.0-langgraph-error",
            "workflow_stage": state["workflow_stage"]
        }
        
        state["routing_decision"] = fallback_decision
        return state
    
    def _should_continue_after_analysis(self, state: RoutingGraphState) -> Literal["continue", "error"]:
        """Conditional edge: Check if analysis was successful."""
        if state.get("error_message") or not state.get("analysis_result"):
            return "error"
        return "continue"
    
    def _should_continue_after_selection(self, state: RoutingGraphState) -> Literal["continue", "error"]:
        """Conditional edge: Check if selection was successful."""
        if state.get("error_message") or not state.get("selection_result"):
            return "error"
        return "continue"
    
    def _should_continue_after_validation(self, state: RoutingGraphState) -> Literal["continue", "retry", "error"]:
        """Conditional edge: Check validation results."""
        if state.get("error_message"):
            return "error"
        
        validation_results = state.get("debug_info", {}).get("validation_results", {})
        if validation_results.get("validation_passed", True):
            return "continue"
        
        # Could implement retry logic here
        return "error"
    
    def _state_to_routing_decision(self, state: RoutingGraphState) -> RoutingDecision:
        """Convert graph state to RoutingDecision model."""
        decision_data = state["routing_decision"]
        
        # Reconstruct models
        from .models import TaskType, ComplexityLevel, Domain, ModelPerformance, ModelSelection
        
        analysis_data = decision_data["prompt_analysis"]
        prompt_analysis = PromptAnalysis(
            task_type=TaskType(analysis_data["task_type"]),
            complexity=ComplexityLevel(analysis_data["complexity"]),
            domain=Domain(analysis_data["domain"]),
            context_length=analysis_data["context_length"],
            keywords=analysis_data["keywords"],
            requires_multimodal=analysis_data["requires_multimodal"],
            reasoning_steps_required=analysis_data["reasoning_steps_required"],
            confidence_score=analysis_data["confidence_score"],
            semantic_embedding=analysis_data.get("semantic_embedding")
        )
        
        selection_data = decision_data["model_selection"]
        selected_model_data = selection_data["selected_model"]
        selected_model = ModelPerformance(
            name=selected_model_data["name"],
            company=selected_model_data["company"],
            parameter_count=selected_model_data["parameter_count"],
            benchmarks=selected_model_data["benchmarks"],
            context_window=selected_model_data["context_window"],
            efficiency_score=selected_model_data["efficiency_score"],
            task_relevance_score=selected_model_data["task_relevance_score"],
            cost_efficiency=selected_model_data["cost_efficiency"]
        )
        
        # Create alternatives
        alternatives = []
        for alt_data in selection_data.get("alternatives", []):
            # Create minimal ModelPerformance for alternatives
            alt_model = ModelPerformance(
                name=alt_data["name"],
                company="",  # Not stored in state
                parameter_count=alt_data["parameter_count"],
                benchmarks={},  # Not stored in state  
                context_window=128000,  # Default
                efficiency_score=alt_data["efficiency_score"],
                task_relevance_score=0.0,  # Not stored
                cost_efficiency=0.0  # Not stored
            )
            alternatives.append(alt_model)
        
        model_selection = ModelSelection(
            selected_model=selected_model,
            selection_reasoning=selection_data["selection_reasoning"],
            confidence_score=selection_data["confidence_score"],
            alternatives=alternatives,
            estimated_performance=selection_data["estimated_performance"]
        )
        
        return RoutingDecision(
            prompt_analysis=prompt_analysis,
            model_selection=model_selection,
            routing_timestamp=decision_data["routing_timestamp"],
            processing_time_ms=decision_data["processing_time_ms"],
            routing_version=decision_data["routing_version"]
        )
    
    def _create_error_routing_decision(
        self,
        prompt: str,
        error_message: str, 
        processing_time: float
    ) -> RoutingDecision:
        """Create fallback routing decision for errors."""
        from .models import TaskType, ComplexityLevel, Domain, ModelPerformance, ModelSelection
        
        # Minimal analysis
        prompt_analysis = PromptAnalysis(
            task_type=TaskType.GENERAL_QA,
            complexity=ComplexityLevel.MEDIUM,
            domain=Domain.GENERAL,
            context_length=8192,
            confidence_score=0.1
        )
        
        # Fallback model
        fallback_model = ModelPerformance(
            name="GPT-4o",
            company="OpenAI",
            parameter_count=175.0,
            benchmarks={"mmlu": 88.7},
            context_window=128000,
            efficiency_score=0.5,
            task_relevance_score=0.5,
            cost_efficiency=0.5
        )
        
        model_selection = ModelSelection(
            selected_model=fallback_model,
            selection_reasoning=f"Fallback selection due to error: {error_message}",
            confidence_score=0.2,
            estimated_performance=0.7
        )
        
        return RoutingDecision(
            prompt_analysis=prompt_analysis,
            model_selection=model_selection,
            routing_timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time * 1000,
            routing_version="1.0-langgraph-error"
        )