#!/usr/bin/env python3
"""
AI Model Capability Analysis Agent

This agent analyzes AI models to automatically determine their optimal task types,
complexity capabilities, and performance characteristics. It enriches the basic
model YAML files with intelligent capability mappings.

Features:
- LLM-powered model analysis and capability assessment
- Task type optimization (code, math, creative, reasoning, etc.)
- Complexity rating (low, medium, high, expert)
- Cost-performance analysis and recommendations
- Automated provider configuration generation
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

import aiohttp
import click
import yaml
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class TaskCapability:
    """Task-specific capability information."""
    task_type: str
    suitability_score: float  # 0.0 to 1.0
    complexity_levels: List[str]  # ["low", "medium", "high", "expert"]
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    benchmarks: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPerformanceProfile:
    """Comprehensive model performance profile."""
    model_id: str
    provider: str
    
    # Task capabilities
    task_capabilities: Dict[str, TaskCapability] = field(default_factory=dict)
    
    # Overall ratings
    overall_complexity: str = "medium"  # low, medium, high, expert
    cost_efficiency: str = "balanced"   # budget, balanced, premium
    latency_tier: str = "medium"       # very_low, low, medium, high
    
    # Specialized capabilities
    specializations: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Performance metrics
    estimated_quality_score: float = 0.0
    cost_per_quality_ratio: float = 0.0
    
    # Metadata
    analysis_timestamp: str = ""
    confidence_score: float = 0.0
    analysis_version: str = "1.0"


class ModelCapabilityAgent:
    """AI agent for analyzing model capabilities."""
    
    def __init__(self, analysis_model: str = "gpt-4o-mini"):
        self.analysis_model = analysis_model
        self.session: Optional[ClientSession] = None
        
        # Model knowledge base (can be expanded)
        self.model_knowledge = self._load_model_knowledge()
        
        # Task definitions
        self.task_definitions = self._get_task_definitions()
        
        # Complexity framework
        self.complexity_framework = self._get_complexity_framework()
    
    def _load_model_knowledge(self) -> Dict[str, Dict]:
        """Load existing knowledge about model capabilities."""
        return {
            # OpenAI Models
            "gpt-4": {
                "strengths": ["reasoning", "analysis", "complex_tasks"],
                "known_benchmarks": {"mmlu": 86.4, "hellaswag": 95.3},
                "typical_use_cases": ["complex_reasoning", "expert_analysis", "research"]
            },
            "gpt-4-turbo": {
                "strengths": ["reasoning", "speed", "long_context"],
                "known_benchmarks": {"mmlu": 86.4},
                "typical_use_cases": ["document_analysis", "complex_reasoning", "code_review"]
            },
            "gpt-4o": {
                "strengths": ["multimodal", "reasoning", "speed"],
                "known_benchmarks": {"mmlu": 88.7},
                "typical_use_cases": ["vision_tasks", "multimodal_analysis", "general_ai"]
            },
            "gpt-4o-mini": {
                "strengths": ["speed", "cost_efficiency", "general_tasks"],
                "known_benchmarks": {"mmlu": 82.0},
                "typical_use_cases": ["chatbots", "content_generation", "simple_analysis"]
            },
            "gpt-3.5-turbo": {
                "strengths": ["speed", "cost_efficiency"],
                "known_benchmarks": {"mmlu": 70.0},
                "typical_use_cases": ["chatbots", "simple_tasks", "content_generation"]
            },
            
            # Anthropic Models
            "claude-3-5-haiku": {
                "strengths": ["speed", "efficiency", "general_tasks"],
                "typical_use_cases": ["fast_responses", "simple_analysis", "content_creation"]
            },
            "claude-sonnet-4": {
                "strengths": ["reasoning", "analysis", "complex_tasks", "tool_use"],
                "typical_use_cases": ["complex_reasoning", "research", "advanced_analysis"]
            },
            "claude-opus-4": {
                "strengths": ["highest_capability", "complex_reasoning", "creative_tasks"],
                "typical_use_cases": ["expert_analysis", "creative_writing", "complex_problem_solving"]
            },
            
            # Google Models
            "gemini-2.5-flash-lite": {
                "strengths": ["speed", "efficiency", "simple_tasks"],
                "typical_use_cases": ["quick_responses", "simple_qa", "basic_tasks"]
            },
            "gemini-2.5-flash": {
                "strengths": ["balanced", "multimodal", "general_purpose"],
                "typical_use_cases": ["general_ai", "multimodal_tasks", "balanced_performance"]
            },
            "gemini-2.5-pro": {
                "strengths": ["advanced_reasoning", "long_context", "complex_tasks"],
                "typical_use_cases": ["complex_analysis", "long_document_processing", "expert_tasks"]
            },
            
            # Other providers
            "llama-3.1-70b": {
                "strengths": ["open_source", "balanced", "code_generation"],
                "typical_use_cases": ["code_tasks", "general_purpose", "local_deployment"]
            },
            "deepseek-chat": {
                "strengths": ["coding", "math", "reasoning"],
                "typical_use_cases": ["programming_tasks", "mathematical_reasoning", "technical_analysis"]
            },
            "deepseek-reasoner": {
                "strengths": ["step_by_step_reasoning", "complex_problem_solving"],
                "typical_use_cases": ["complex_reasoning", "mathematical_proofs", "logical_analysis"]
            },
        }
    
    def _get_task_definitions(self) -> Dict[str, Dict]:
        """Define task types and their characteristics - matching existing TaskType enum."""
        return {
            "Open QA": {
                "description": "Open-ended question answering without specific context",
                "complexity_factors": ["domain_expertise", "reasoning_depth", "factual_accuracy"],
                "evaluation_criteria": ["knowledge_breadth", "answer_quality", "reasoning_clarity"]
            },
            "Closed QA": {
                "description": "Question answering with specific context or documents",
                "complexity_factors": ["context_understanding", "information_extraction", "synthesis"],
                "evaluation_criteria": ["context_accuracy", "relevance", "completeness"]
            },
            "Summarization": {
                "description": "Condensing long text into key points and insights",
                "complexity_factors": ["text_length", "domain_complexity", "abstraction_level"],
                "evaluation_criteria": ["information_retention", "clarity", "conciseness"]
            },
            "Text Generation": {
                "description": "General text generation and creative writing",
                "complexity_factors": ["creativity_level", "style_requirements", "content_depth"],
                "evaluation_criteria": ["fluency", "creativity", "coherence", "appropriateness"]
            },
            "Code Generation": {
                "description": "Writing, debugging, and explaining code",
                "complexity_factors": ["algorithm_complexity", "language_expertise", "debugging_depth"],
                "evaluation_criteria": ["correctness", "efficiency", "readability", "best_practices"]
            },
            "Chatbot": {
                "description": "Natural conversational interactions and dialogue",
                "complexity_factors": ["context_retention", "personality_consistency", "topic_handling"],
                "evaluation_criteria": ["naturalness", "helpfulness", "engagement", "safety"]
            },
            "Classification": {
                "description": "Categorizing and labeling text or data",
                "complexity_factors": ["category_complexity", "nuance_detection", "accuracy_requirements"],
                "evaluation_criteria": ["accuracy", "consistency", "edge_case_handling"]
            },
            "Rewrite": {
                "description": "Reformulating and improving existing text",
                "complexity_factors": ["style_transformation", "content_preservation", "target_audience"],
                "evaluation_criteria": ["style_improvement", "meaning_preservation", "readability"]
            },
            "Brainstorming": {
                "description": "Creative ideation and concept generation",
                "complexity_factors": ["creativity_scope", "domain_knowledge", "innovation_level"],
                "evaluation_criteria": ["creativity", "feasibility", "diversity", "relevance"]
            },
            "Extraction": {
                "description": "Extracting specific information from text",
                "complexity_factors": ["data_complexity", "extraction_precision", "format_requirements"],
                "evaluation_criteria": ["accuracy", "completeness", "format_compliance"]
            },
            "Other": {
                "description": "General-purpose tasks not fitting specific categories",
                "complexity_factors": ["task_novelty", "multi_domain_knowledge", "adaptability"],
                "evaluation_criteria": ["task_completion", "adaptability", "general_quality"]
            }
        }
    
    def _get_complexity_framework(self) -> Dict[str, Dict]:
        """Define complexity levels and their characteristics."""
        return {
            "low": {
                "description": "Simple, straightforward tasks with clear parameters",
                "characteristics": ["single_step", "well_defined", "common_knowledge", "minimal_context"],
                "examples": ["basic_qa", "simple_formatting", "standard_templates"]
            },
            "medium": {
                "description": "Multi-step tasks requiring some reasoning and domain knowledge",
                "characteristics": ["multi_step", "some_reasoning", "domain_knowledge", "moderate_context"],
                "examples": ["code_explanation", "content_analysis", "structured_responses"]
            },
            "high": {
                "description": "Complex tasks requiring deep reasoning and expertise",
                "characteristics": ["deep_reasoning", "expert_knowledge", "complex_relationships", "long_context"],
                "examples": ["algorithm_design", "research_analysis", "complex_problem_solving"]
            },
            "expert": {
                "description": "Highly specialized tasks requiring cutting-edge capabilities",
                "characteristics": ["specialized_expertise", "novel_problems", "interdisciplinary", "creative_solutions"],
                "examples": ["research_breakthroughs", "novel_architectures", "complex_proofs"]
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = ClientTimeout(total=60, connect=15)
        self.session = ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _analyze_model_with_llm(self, model_info: Dict) -> Dict:
        """Use LLM to analyze model capabilities."""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Get existing knowledge
        model_id = model_info.get("id", "")
        existing_knowledge = self.model_knowledge.get(model_id, {})
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(model_info, existing_knowledge)
        
        # Make API call to analysis model
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            # Fallback to rule-based analysis if no API key
            return self._rule_based_analysis(model_info, existing_knowledge)
        
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.analysis_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI model capability analyst. Analyze AI models and provide structured assessments of their capabilities, optimal use cases, and performance characteristics."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    logger.warning(f"LLM analysis failed: {response.status}, falling back to rule-based")
                    return self._rule_based_analysis(model_info, existing_knowledge)
                
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Parse LLM response
                return self._parse_llm_analysis(content, model_info, existing_knowledge)
                
        except Exception as e:
            logger.warning(f"LLM analysis error: {e}, falling back to rule-based")
            return self._rule_based_analysis(model_info, existing_knowledge)
    
    def _create_analysis_prompt(self, model_info: Dict, existing_knowledge: Dict) -> str:
        """Create analysis prompt for the LLM."""
        model_id = model_info.get("id", "")
        provider = model_info.get("provider", "")
        context_length = model_info.get("context_length", "unknown")
        supports_function_calling = model_info.get("supports_function_calling", False)
        supports_vision = model_info.get("supports_vision", False)
        
        prompt = f"""
Analyze the AI model "{model_id}" from provider "{provider}" and provide a structured assessment.

Model Information:
- ID: {model_id}
- Provider: {provider}
- Context Length: {context_length}
- Function Calling: {supports_function_calling}
- Vision Support: {supports_vision}

Existing Knowledge:
{json.dumps(existing_knowledge, indent=2) if existing_knowledge else "No existing knowledge available"}

Task Types to Analyze:
{json.dumps(list(self.task_definitions.keys()), indent=2)}

Please provide analysis in this JSON format:
{{
    "task_capabilities": {{
        "code_generation": {{
            "suitability_score": 0.85,
            "complexity_levels": ["low", "medium", "high"],
            "reasoning": "Strong code generation based on model architecture",
            "recommended_params": {{"temperature": 0.1, "max_tokens": 2048}}
        }},
        // ... analyze all task types
    }},
    "overall_complexity": "high",
    "cost_efficiency": "balanced",
    "latency_tier": "medium",
    "specializations": ["reasoning", "analysis"],
    "limitations": ["very_long_context", "real_time_data"],
    "estimated_quality_score": 0.87,
    "confidence_score": 0.92
}}

Focus on:
1. Realistic capability assessment based on model characteristics
2. Appropriate complexity levels for each task type
3. Practical parameter recommendations
4. Known strengths and limitations
5. Cost-performance positioning
"""
        return prompt
    
    def _parse_llm_analysis(self, content: str, model_info: Dict, existing_knowledge: Dict) -> Dict:
        """Parse LLM analysis response."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                return analysis_data
            else:
                logger.warning("Could not parse LLM response as JSON, using rule-based fallback")
                return self._rule_based_analysis(model_info, existing_knowledge)
        except Exception as e:
            logger.warning(f"Error parsing LLM analysis: {e}, using rule-based fallback")
            return self._rule_based_analysis(model_info, existing_knowledge)
    
    def _rule_based_analysis(self, model_info: Dict, existing_knowledge: Dict) -> Dict:
        """Rule-based analysis as fallback."""
        model_id = model_info.get("id", "")
        provider = model_info.get("provider", "")
        
        # Determine model tier based on name patterns
        model_tier = self._determine_model_tier(model_id)
        
        # Base capabilities by tier
        tier_capabilities = {
            "flagship": {
                "overall_complexity": "expert",
                "cost_efficiency": "premium",
                "latency_tier": "medium",
                "estimated_quality_score": 0.90
            },
            "advanced": {
                "overall_complexity": "high", 
                "cost_efficiency": "balanced",
                "latency_tier": "medium",
                "estimated_quality_score": 0.80
            },
            "standard": {
                "overall_complexity": "medium",
                "cost_efficiency": "balanced",
                "latency_tier": "low",
                "estimated_quality_score": 0.70
            },
            "efficient": {
                "overall_complexity": "low",
                "cost_efficiency": "budget",
                "latency_tier": "very_low",
                "estimated_quality_score": 0.60
            }
        }
        
        base_config = tier_capabilities.get(model_tier, tier_capabilities["standard"])
        
        # Task-specific analysis
        task_capabilities = {}
        for task_type in self.task_definitions.keys():
            capability = self._analyze_task_capability(model_id, provider, task_type, model_tier, existing_knowledge)
            task_capabilities[task_type] = capability
        
        return {
            "task_capabilities": task_capabilities,
            "overall_complexity": base_config["overall_complexity"],
            "cost_efficiency": base_config["cost_efficiency"], 
            "latency_tier": base_config["latency_tier"],
            "specializations": existing_knowledge.get("strengths", []),
            "limitations": self._determine_limitations(model_id, provider),
            "estimated_quality_score": base_config["estimated_quality_score"],
            "confidence_score": 0.75  # Rule-based has lower confidence
        }
    
    def _determine_model_tier(self, model_id: str) -> str:
        """Determine model tier based on ID patterns."""
        model_id_lower = model_id.lower()
        
        # Flagship models
        if any(pattern in model_id_lower for pattern in [
            "gpt-4o", "claude-opus", "claude-sonnet-4", "gemini-2.5-pro", "deepseek-reasoner"
        ]):
            return "flagship"
        
        # Advanced models
        elif any(pattern in model_id_lower for pattern in [
            "gpt-4", "claude-3.5", "claude-sonnet", "gemini-2.5-flash", "llama-3.1-70b"
        ]):
            return "advanced"
        
        # Efficient models  
        elif any(pattern in model_id_lower for pattern in [
            "mini", "lite", "haiku", "3.5-turbo", "small", "flash-lite"
        ]):
            return "efficient"
        
        # Standard models (default)
        else:
            return "standard"
    
    def _analyze_task_capability(self, model_id: str, provider: str, task_type: str, model_tier: str, existing_knowledge: Dict) -> Dict:
        """Analyze capability for a specific task type."""
        
        # Base suitability by tier
        tier_base_scores = {
            "flagship": 0.90,
            "advanced": 0.80,
            "standard": 0.70,
            "efficient": 0.60
        }
        
        base_score = tier_base_scores.get(model_tier, 0.70)
        
        # Task-specific adjustments
        task_adjustments = {
            "Code Generation": {
                "gpt-4": 0.05, "claude": 0.03, "deepseek": 0.10
            },
            "Open QA": {
                "gpt-4": 0.05, "claude": 0.03, "deepseek": 0.05, "reasoner": 0.08
            },
            "Closed QA": {
                "gpt-4": 0.05, "claude": 0.03, "deepseek": 0.05
            },
            "Text Generation": {
                "claude": 0.10, "gpt-4": 0.05, "opus": 0.15
            },
            "Chatbot": {
                "gpt-3.5": 0.05, "haiku": 0.08, "mini": 0.10
            },
            "Summarization": {
                "claude": 0.05, "gpt-4": 0.03, "haiku": 0.08
            },
            "Brainstorming": {
                "claude": 0.10, "gpt-4": 0.05, "opus": 0.15, "grok": 0.08
            },
            "Classification": {
                "gpt-4": 0.03, "mini": 0.05, "deepseek": 0.05
            },
            "Rewrite": {
                "claude": 0.08, "gpt-4": 0.05, "sonnet": 0.10
            },
            "Extraction": {
                "gpt-4": 0.05, "claude": 0.03, "mini": 0.08
            }
        }
        
        # Apply adjustments
        adjustment = 0.0
        model_lower = model_id.lower()
        provider_lower = provider.lower()
        
        for pattern, boost in task_adjustments.get(task_type, {}).items():
            if pattern in model_lower or pattern in provider_lower:
                adjustment = max(adjustment, boost)
        
        final_score = min(1.0, base_score + adjustment)
        
        # Determine complexity levels
        complexity_levels = []
        if final_score >= 0.90:
            complexity_levels = ["low", "medium", "high", "expert"]
        elif final_score >= 0.75:
            complexity_levels = ["low", "medium", "high"]
        elif final_score >= 0.60:
            complexity_levels = ["low", "medium"]
        else:
            complexity_levels = ["low"]
        
        # Recommended parameters by task
        param_recommendations = {
            "Code Generation": {"temperature": 0.1, "max_tokens": 2048},
            "Text Generation": {"temperature": 0.8, "max_tokens": 1024},
            "Open QA": {"temperature": 0.3, "max_tokens": 1024},
            "Closed QA": {"temperature": 0.2, "max_tokens": 1024},
            "Chatbot": {"temperature": 0.7, "max_tokens": 512},
            "Summarization": {"temperature": 0.3, "max_tokens": 1024},
            "Classification": {"temperature": 0.1, "max_tokens": 256},
            "Rewrite": {"temperature": 0.5, "max_tokens": 1024},
            "Brainstorming": {"temperature": 0.8, "max_tokens": 1024},
            "Extraction": {"temperature": 0.2, "max_tokens": 512},
            "Other": {"temperature": 0.5, "max_tokens": 1024}
        }
        
        return {
            "suitability_score": round(final_score, 2),
            "complexity_levels": complexity_levels,
            "reasoning": f"Based on {model_tier} tier model with task-specific adjustments",
            "recommended_params": param_recommendations.get(task_type, {"temperature": 0.5, "max_tokens": 1024})
        }
    
    def _determine_limitations(self, model_id: str, provider: str) -> List[str]:
        """Determine model limitations."""
        limitations = []
        
        model_lower = model_id.lower()
        
        if "mini" in model_lower or "lite" in model_lower or "3.5" in model_lower:
            limitations.extend(["complex_reasoning", "very_long_context", "specialized_domains"])
        
        if provider.lower() in ["groq", "deepseek"]:
            limitations.append("real_time_data")
        
        if not any(vision_indicator in model_lower for vision_indicator in ["4o", "vision", "claude-3"]):
            limitations.append("multimodal_tasks")
        
        return limitations
    
    async def analyze_model(self, model_info: Dict) -> ModelPerformanceProfile:
        """Analyze a single model and create performance profile."""
        logger.info(f"Analyzing model: {model_info.get('id', 'unknown')}")
        
        # Get analysis from LLM or rule-based fallback
        analysis_data = await self._analyze_model_with_llm(model_info)
        
        # Create task capabilities
        task_capabilities = {}
        for task_type, task_data in analysis_data.get("task_capabilities", {}).items():
            task_capabilities[task_type] = TaskCapability(
                task_type=task_type,
                suitability_score=task_data.get("suitability_score", 0.5),
                complexity_levels=task_data.get("complexity_levels", ["low"]),
                recommended_params=task_data.get("recommended_params", {}),
                reasoning=task_data.get("reasoning", ""),
                benchmarks=task_data.get("benchmarks", {})
            )
        
        # Calculate cost per quality ratio
        cost_per_quality = 0.0
        if "cost_per_1m_input_tokens" in model_info and analysis_data.get("estimated_quality_score", 0) > 0:
            input_cost = model_info["cost_per_1m_input_tokens"]
            quality = analysis_data["estimated_quality_score"]
            cost_per_quality = input_cost / quality
        
        # Create performance profile
        profile = ModelPerformanceProfile(
            model_id=model_info.get("id", ""),
            provider=model_info.get("provider", ""),
            task_capabilities=task_capabilities,
            overall_complexity=analysis_data.get("overall_complexity", "medium"),
            cost_efficiency=analysis_data.get("cost_efficiency", "balanced"),
            latency_tier=analysis_data.get("latency_tier", "medium"),
            specializations=analysis_data.get("specializations", []),
            limitations=analysis_data.get("limitations", []),
            estimated_quality_score=analysis_data.get("estimated_quality_score", 0.0),
            cost_per_quality_ratio=cost_per_quality,
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=analysis_data.get("confidence_score", 0.75),
            analysis_version="1.0"
        )
        
        return profile
    
    async def enrich_yaml_files(self, models_dir: Path) -> None:
        """Enrich all YAML files with capability analysis."""
        logger.info(f"Enriching YAML files in {models_dir}")
        
        # Process each provider directory
        for provider_dir in models_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            
            logger.info(f"Processing provider: {provider_dir.name}")
            
            # Find YAML file
            yaml_files = list(provider_dir.glob("*.yaml"))
            if not yaml_files:
                logger.warning(f"No YAML files found in {provider_dir}")
                continue
            
            yaml_file = yaml_files[0]
            
            # Load existing data
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Analyze each model
            enriched_models = []
            models = data.get("models", [])
            
            for model_info in models:
                try:
                    # Analyze model
                    profile = await self.analyze_model(model_info)
                    
                    # Merge original info with analysis
                    enriched_model = {**model_info}
                    enriched_model.update({
                        "task_capabilities": {
                            task_type: asdict(capability) 
                            for task_type, capability in profile.task_capabilities.items()
                        },
                        "performance_profile": {
                            "overall_complexity": profile.overall_complexity,
                            "cost_efficiency": profile.cost_efficiency,
                            "latency_tier": profile.latency_tier,
                            "specializations": profile.specializations,
                            "limitations": profile.limitations,
                            "estimated_quality_score": profile.estimated_quality_score,
                            "cost_per_quality_ratio": profile.cost_per_quality_ratio,
                            "analysis_timestamp": profile.analysis_timestamp,
                            "confidence_score": profile.confidence_score,
                            "analysis_version": profile.analysis_version
                        }
                    })
                    
                    enriched_models.append(enriched_model)
                    
                except Exception as e:
                    logger.error(f"Error analyzing model {model_info.get('id', 'unknown')}: {e}")
                    enriched_models.append(model_info)  # Keep original if analysis fails
            
            # Update data
            data["models"] = enriched_models
            data["provider"]["enriched_at"] = datetime.now().isoformat()
            data["provider"]["analysis_version"] = "1.0"
            
            # Save enriched YAML
            enriched_file = provider_dir / f"{provider_dir.name}_enriched_models.yaml"
            with open(enriched_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=120
                )
            
            logger.info(f"Saved enriched models to {enriched_file}")
    
    def generate_provider_config(self, enriched_yaml_dir: Path, output_file: Path) -> None:
        """Generate Python provider configuration from enriched YAML files."""
        logger.info("Generating provider configuration...")
        
        config_data = {}
        
        # Process each provider
        for provider_dir in enriched_yaml_dir.iterdir():
            if not provider_dir.is_dir():
                continue
            
            # Find enriched YAML files
            enriched_files = list(provider_dir.glob("*_enriched_models.yaml"))
            if not enriched_files:
                continue
                
            yaml_file = enriched_files[0]
            
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            provider_id = data["provider"]["id"]
            models = data.get("models", [])
            
            # Convert to provider config format  
            model_capabilities = []
            for model in models:
                # Create ModelCapability-like structure
                capability = {
                    "description": f"AI model with {model.get('performance_profile', {}).get('overall_complexity', 'medium')} complexity",
                    "provider": provider_id.upper(),
                    "model_name": model["id"],
                    "cost_per_1m_input_tokens": model.get("input_cost_per_1m_tokens", 0.0),
                    "cost_per_1m_output_tokens": model.get("output_cost_per_1m_tokens", 0.0),
                    "max_context_tokens": model.get("context_length", 4096),
                    "max_output_tokens": model.get("max_tokens", 4096),
                    "supports_function_calling": model.get("supports_function_calling", False),
                    "languages_supported": ["en"],  # Default
                    "model_size_params": "Unknown",
                    "latency_tier": model.get("performance_profile", {}).get("latency_tier", "medium"),
                    "task_type": self._determine_primary_task_type(model.get("task_capabilities", {})),
                    "complexity": model.get("performance_profile", {}).get("overall_complexity", "medium"),
                }
                
                model_capabilities.append(capability)
            
            config_data[provider_id.upper()] = model_capabilities
        
        # Generate Python code
        python_code = self._generate_python_config(config_data)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(python_code)
        
        logger.info(f"Generated provider configuration: {output_file}")
    
    def _determine_primary_task_type(self, task_capabilities: Dict) -> str:
        """Determine primary task type based on capabilities."""
        if not task_capabilities:
            return "general"
        
        # Find task with highest suitability score
        best_task = "general"
        best_score = 0.0
        
        for task_type, capability in task_capabilities.items():
            score = capability.get("suitability_score", 0.0)
            if score > best_score:
                best_score = score
                best_task = task_type
        
        # Map to simplified categories that align with your existing system
        task_mapping = {
            "Code Generation": "code",
            "Open QA": "general",
            "Closed QA": "general", 
            "Text Generation": "creative",
            "Chatbot": "general",
            "Summarization": "analysis",
            "Classification": "analysis",
            "Rewrite": "general",
            "Brainstorming": "creative",
            "Extraction": "analysis",
            "Other": "general"
        }
        
        return task_mapping.get(best_task, "general")
    
    def _generate_python_config(self, config_data: Dict) -> str:
        """Generate Python configuration code."""
        imports = '''"""
Provider configurations and model capabilities for all supported providers.
Generated automatically by Model Capability Agent.
"""

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.llm_enums import ProviderType

# Model capabilities for all providers
provider_model_capabilities: dict[ProviderType, list[ModelCapability]] = {
'''
        
        provider_configs = []
        
        for provider_enum, models in config_data.items():
            model_entries = []
            
            for model in models:
                model_entry = f'''        ModelCapability(
            description="{model['description']}",
            provider=ProviderType.{provider_enum},
            model_name="{model['model_name']}",
            cost_per_1m_input_tokens={model['cost_per_1m_input_tokens']},
            cost_per_1m_output_tokens={model['cost_per_1m_output_tokens']},
            max_context_tokens={model['max_context_tokens']},
            max_output_tokens={model['max_output_tokens']},
            supports_function_calling={model['supports_function_calling']},
            languages_supported={model['languages_supported']},
            model_size_params="{model['model_size_params']}",
            latency_tier="{model['latency_tier']}",
            task_type="{model['task_type']}",
            complexity="{model['complexity']}",
        )'''
                model_entries.append(model_entry)
            
            model_entries_joined = ",\n".join(model_entries)
            provider_config = f'''    ProviderType.{provider_enum}: [
{model_entries_joined},
    ]'''
            provider_configs.append(provider_config)
        
        config_body = ",\n".join(provider_configs)
        
        return f'''{imports}{config_body}
}}
'''


@click.command()
@click.option(
    "--models-dir",
    default="models",
    help="Directory containing extracted model YAML files"
)
@click.option(
    "--output-dir", 
    default="enriched_models",
    help="Output directory for enriched YAML files"
)
@click.option(
    "--generate-config",
    is_flag=True,
    help="Generate Python provider configuration file"
)
@click.option(
    "--config-output",
    default="generated_providers.py",
    help="Output file for generated provider configuration"
)
@click.option(
    "--analysis-model",
    default="gpt-4o-mini",
    help="Model to use for capability analysis"
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    models_dir: str,
    output_dir: str, 
    generate_config: bool,
    config_output: str,
    analysis_model: str,
    verbose: bool
) -> None:
    """Analyze and enrich AI model capabilities with intelligent task mappings."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not models_path.exists():
        logger.error(f"Models directory {models_path} does not exist")
        sys.exit(1)
    
    async def run_analysis():
        async with ModelCapabilityAgent(analysis_model) as agent:
            logger.info("Starting model capability analysis...")
            
            # Copy original files to output directory
            import shutil
            for item in models_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
            
            # Enrich YAML files
            await agent.enrich_yaml_files(output_path)
            
            # Generate provider configuration if requested
            if generate_config:
                config_path = Path(config_output)
                agent.generate_provider_config(output_path, config_path)
            
            logger.info("✅ Model capability analysis complete!")
    
    try:
        asyncio.run(run_analysis())
    except KeyboardInterrupt:
        logger.info("Analysis cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()