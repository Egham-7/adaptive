"""AI-powered routing agent that uses LLM to make smart routing decisions."""

import json
from typing import List, Dict, Any
from openai import OpenAI

from .models import PromptAnalysis


class AIRoutingAgent:
    """Uses an LLM to analyze prompts and make intelligent routing decisions."""
    
    def __init__(self, routing_model: str = "gpt-4o-mini", api_key: str = None):
        """Initialize the AI routing agent.
        
        Args:
            routing_model: Which model to use for routing decisions
            api_key: OpenAI API key
        """
        self.routing_model = routing_model
        self.client = OpenAI(api_key=api_key)
        
    def analyze_prompt(self, prompt: str, available_models: List[Dict[str, Any]]) -> PromptAnalysis:
        """Analyze prompt using AI agent to determine routing requirements.
        
        Args:
            prompt: The user's prompt to analyze
            available_models: List of available models with their capabilities
            
        Returns:
            PromptAnalysis with AI agent's reasoning and requirements
        """
        
        # Create model summary for the AI agent
        model_summary = self._create_model_summary(available_models)
        
        system_prompt = f"""You are an expert AI routing agent. Your job is to analyze user prompts and determine routing requirements.

Available models:
{model_summary}

For each prompt, analyze:
1. What type of task this is and what capabilities are needed
2. Estimated context window requirements (be realistic about prompt + response)
3. Whether multimodal capabilities are needed
4. Your confidence in this analysis (0.0 to 1.0)

Respond with JSON in this exact format:
{{
    "analysis_reasoning": "Your detailed reasoning about the prompt and requirements",
    "context_length": 8192,
    "requires_multimodal": false,
    "confidence_score": 0.85
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.routing_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this prompt:\n\n{prompt}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the JSON response
            analysis_json = json.loads(response.choices[0].message.content)
            
            return PromptAnalysis(
                analysis_reasoning=analysis_json["analysis_reasoning"],
                context_length=analysis_json["context_length"],
                requires_multimodal=analysis_json.get("requires_multimodal", False),
                confidence_score=analysis_json["confidence_score"]
            )
            
        except Exception as e:
            # Fallback analysis if AI fails
            return PromptAnalysis(
                analysis_reasoning=f"Failed to analyze with AI agent: {str(e)}. Using basic analysis.",
                context_length=8192,
                requires_multimodal=False,
                confidence_score=0.3
            )
    
    def select_best_model(
        self, 
        prompt: str, 
        analysis: PromptAnalysis, 
        available_models: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use AI agent to select the best model for the prompt.
        
        Args:
            prompt: Original user prompt
            analysis: Previous analysis results
            available_models: Available models with capabilities
            
        Returns:
            Dictionary with selected model and reasoning
        """
        
        model_summary = self._create_model_summary(available_models)
        
        system_prompt = f"""You are an expert AI model selector. Based on the prompt analysis, select the best model.

Available models:
{model_summary}

Previous analysis: {analysis.analysis_reasoning}
Context needed: {analysis.context_length}
Multimodal needed: {analysis.requires_multimodal}

Consider:
- Model capabilities for this specific task
- Context window requirements
- Efficiency vs performance trade-offs
- Cost considerations

Respond with JSON in this exact format:
{{
    "selected_model": "exact model name from the list",
    "selection_reasoning": "Detailed explanation of why this model is best",
    "confidence_score": 0.85,
    "estimated_performance": 0.88
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.routing_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Select best model for:\n\n{prompt}"}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            # Fallback to first suitable model
            suitable_models = [
                m for m in available_models 
                if m.get("context_window", 0) >= analysis.context_length
            ]
            fallback_model = suitable_models[0] if suitable_models else available_models[0]
            
            return {
                "selected_model": fallback_model["name"],
                "selection_reasoning": f"AI selection failed: {str(e)}. Using fallback model.",
                "confidence_score": 0.3,
                "estimated_performance": 0.7
            }
    
    def _create_model_summary(self, models: List[Dict[str, Any]]) -> str:
        """Create a concise summary of available models for the AI agent."""
        summary_parts = []
        
        for model in models:
            summary = f"- {model['name']} ({model.get('company', 'Unknown')})"
            summary += f" | {model.get('parameter_count', 'Unknown')}B params"
            summary += f" | {model.get('context_window', 'Unknown')} context"
            
            # Add key benchmarks if available
            if 'benchmarks' in model and model['benchmarks']:
                benchmarks = []
                for bench, score in model['benchmarks'].items():
                    if score and score > 0:
                        benchmarks.append(f"{bench}: {score:.1f}")
                if benchmarks:
                    summary += f" | {', '.join(benchmarks[:3])}"
            
            summary_parts.append(summary)
        
        return "\n".join(summary_parts)