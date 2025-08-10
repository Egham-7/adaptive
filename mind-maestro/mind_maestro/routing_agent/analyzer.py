"""Intelligent prompt analyzer for task classification and complexity assessment."""

import re
import math
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from .models import (
    PromptAnalysis, 
    TaskType, 
    ComplexityLevel, 
    Domain
)


class PromptAnalyzer:
    """Analyzes prompts to determine task type, complexity, and routing requirements."""
    
    # Task classification patterns
    TASK_PATTERNS = {
        TaskType.MATH: [
            r'solve|calculate|compute|integral|derivative|equation|theorem|proof',
            r'∫|∂|∑|∏|√|±|≤|≥|≠|∞|π|θ|α|β|γ',
            r'\d+\s*[+\-*/^]\s*\d+|x\^?\d+|\d*x\b',
            r'algebra|calculus|geometry|trigonometry|statistics|probability'
        ],
        TaskType.CODING: [
            r'write\s+(a\s+)?(function|class|script|program|code)',
            r'implement|debug|refactor|optimize|algorithm',
            r'python|javascript|java|c\+\+|sql|html|css|react',
            r'def\s+\w+|class\s+\w+|function\s+\w+|<.*>|{.*}',
            r'api|database|framework|library|package|import'
        ],
        TaskType.REASONING: [
            r'analyze|reason|explain why|deduce|infer|conclude',
            r'logical|philosophy|ethics|argument|premise|conclusion',
            r'cause and effect|if.*then|because|therefore|thus',
            r'compare|contrast|evaluate|assess|judge|critique'
        ],
        TaskType.CREATIVE: [
            r'write a story|poem|essay|creative|imagine|fiction',
            r'brainstorm|generate ideas|creative writing|narrative',
            r'character|plot|setting|dialogue|metaphor|imagery'
        ],
        TaskType.TECHNICAL: [
            r'technical|specification|documentation|architecture',
            r'system design|infrastructure|deployment|configuration',
            r'performance|optimization|scalability|security'
        ],
        TaskType.ANALYSIS: [
            r'analyze.*data|statistical analysis|trends|patterns',
            r'research|study|investigate|examine|interpret',
            r'findings|results|conclusions|insights|implications'
        ]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        ComplexityLevel.SIMPLE: [
            r'^what is|^who is|^when|^where|^define|^list',
            r'simple|basic|easy|straightforward|quick'
        ],
        ComplexityLevel.MEDIUM: [
            r'explain|describe|compare|how does|why does',
            r'step by step|process|method|approach'
        ],
        ComplexityLevel.COMPLEX: [
            r'analyze|evaluate|synthesize|design|create',
            r'multiple|complex|advanced|sophisticated|comprehensive'
        ],
        ComplexityLevel.EXPERT: [
            r'research|thesis|dissertation|academic|scholarly',
            r'peer.?review|publication|methodology|framework'
        ]
    }
    
    # Domain classification patterns
    DOMAIN_PATTERNS = {
        Domain.ACADEMIC: [
            r'research|study|paper|journal|academic|scholarly',
            r'thesis|dissertation|peer.?review|citation|bibliography'
        ],
        Domain.TECHNICAL: [
            r'technical|engineering|software|hardware|system',
            r'specification|architecture|protocol|api|database'
        ],
        Domain.BUSINESS: [
            r'business|marketing|finance|sales|revenue|profit',
            r'strategy|management|leadership|corporate|company'
        ],
        Domain.SCIENTIFIC: [
            r'experiment|hypothesis|theory|methodology|data',
            r'biology|chemistry|physics|medicine|research|lab'
        ],
        Domain.CREATIVE: [
            r'creative|art|design|story|poem|music|video',
            r'imagination|inspiration|aesthetic|beauty|style'
        ]
    }
    
    def __init__(self, use_embeddings: bool = True):
        """Initialize the prompt analyzer.
        
        Args:
            use_embeddings: Whether to use semantic embeddings for analysis
        """
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        
        if use_embeddings:
            try:
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.use_embeddings = False
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Analyze a prompt to determine routing parameters.
        
        Args:
            prompt: The input prompt to analyze
            
        Returns:
            PromptAnalysis with task type, complexity, and other routing info
        """
        prompt_lower = prompt.lower()
        
        # Extract keywords
        keywords = self._extract_keywords(prompt)
        
        # Classify task type
        task_type, task_confidence = self._classify_task_type(prompt_lower)
        
        # Assess complexity
        complexity, complexity_confidence = self._assess_complexity(prompt)
        
        # Determine domain
        domain, domain_confidence = self._classify_domain(prompt_lower)
        
        # Calculate context requirements
        context_length = self._estimate_context_length(prompt)
        
        # Check for multimodal requirements
        requires_multimodal = self._check_multimodal_requirements(prompt_lower)
        
        # Estimate reasoning steps
        reasoning_steps = self._estimate_reasoning_complexity(prompt_lower, complexity)
        
        # Get semantic embedding if available
        semantic_embedding = None
        if self.use_embeddings and self._embedding_model:
            try:
                semantic_embedding = self._embedding_model.encode(prompt).tolist()
            except Exception:
                pass
        
        # Calculate overall confidence
        confidence_score = (task_confidence + complexity_confidence + domain_confidence) / 3
        
        return PromptAnalysis(
            task_type=task_type,
            complexity=complexity,
            domain=domain,
            context_length=context_length,
            keywords=keywords,
            semantic_embedding=semantic_embedding,
            requires_multimodal=requires_multimodal,
            reasoning_steps_required=reasoning_steps,
            confidence_score=confidence_score
        )
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract important keywords from the prompt."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words, numbers, and special characters
        words = re.findall(r'\b\w+\b|[+\-*/=<>≤≥≠∫∂∑∏√π]', prompt.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return top 10 most meaningful keywords
        return keywords[:10]
    
    def _classify_task_type(self, prompt: str) -> tuple[TaskType, float]:
        """Classify the task type based on patterns and keywords."""
        task_scores = {}
        
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            task_scores[task_type] = score
        
        # Default to general QA if no strong patterns found
        if not any(task_scores.values()):
            return TaskType.GENERAL_QA, 0.5
        
        # Get task type with highest score
        best_task = max(task_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on score distribution
        total_score = sum(task_scores.values())
        confidence = best_task[1] / total_score if total_score > 0 else 0.5
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        return best_task[0], confidence
    
    def _assess_complexity(self, prompt: str) -> tuple[ComplexityLevel, float]:
        """Assess the complexity level of the prompt."""
        prompt_lower = prompt.lower()
        
        # Length-based initial assessment
        length = len(prompt)
        if length < 50:
            base_complexity = ComplexityLevel.SIMPLE
        elif length < 200:
            base_complexity = ComplexityLevel.MEDIUM  
        elif length < 1000:
            base_complexity = ComplexityLevel.COMPLEX
        else:
            base_complexity = ComplexityLevel.EXPERT
        
        # Pattern-based refinement
        complexity_scores = {}
        for complexity, patterns in self.COMPLEXITY_INDICATORS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            complexity_scores[complexity] = score
        
        # Combine length and pattern assessment
        if any(complexity_scores.values()):
            pattern_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
            # Average the assessments with slight bias toward patterns
            complexities = [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]
            base_idx = complexities.index(base_complexity)
            pattern_idx = complexities.index(pattern_complexity)
            final_idx = int((base_idx + pattern_idx * 1.2) / 2.2)
            final_complexity = complexities[min(final_idx, 3)]
        else:
            final_complexity = base_complexity
        
        # Calculate confidence
        total_patterns = sum(complexity_scores.values())
        confidence = min(0.8 + (total_patterns * 0.05), 0.95)
        
        return final_complexity, confidence
    
    def _classify_domain(self, prompt: str) -> tuple[Domain, float]:
        """Classify the domain of the prompt."""
        domain_scores = {}
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            domain_scores[domain] = score
        
        # Default to general if no specific domain detected
        if not any(domain_scores.values()):
            return Domain.GENERAL, 0.6
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence
        total_score = sum(domain_scores.values())
        confidence = best_domain[1] / total_score if total_score > 0 else 0.6
        confidence = min(confidence, 0.9)
        
        return best_domain[0], confidence
    
    def _estimate_context_length(self, prompt: str) -> int:
        """Estimate required context window size."""
        # Basic estimation: prompt length + expected response + safety margin
        prompt_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        expected_response = max(100, min(prompt_tokens * 2, 2000))  # Adaptive response size
        safety_margin = (prompt_tokens + expected_response) * 0.3
        
        total_estimated = int(prompt_tokens + expected_response + safety_margin)
        
        # Round up to common context window sizes
        context_windows = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
        for window in context_windows:
            if total_estimated <= window:
                return window
        
        return context_windows[-1]  # Return largest if exceeds all
    
    def _check_multimodal_requirements(self, prompt: str) -> bool:
        """Check if the prompt requires multimodal capabilities."""
        multimodal_indicators = [
            r'image|picture|photo|visual|diagram|chart|graph',
            r'audio|voice|sound|music|speech',
            r'video|movie|animation|clip',
            r'draw|sketch|illustrate|visualize'
        ]
        
        for pattern in multimodal_indicators:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        
        return False
    
    def _estimate_reasoning_complexity(self, prompt: str, complexity: ComplexityLevel) -> int:
        """Estimate the number of reasoning steps required."""
        base_steps = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MEDIUM: 2, 
            ComplexityLevel.COMPLEX: 4,
            ComplexityLevel.EXPERT: 8
        }
        
        # Look for reasoning indicators
        reasoning_patterns = [
            r'step.by.step|steps?|process|method',
            r'first.*then|next|finally|conclusion',
            r'because|therefore|thus|hence|consequently',
            r'analyze|evaluate|compare|contrast'
        ]
        
        reasoning_multiplier = 1
        for pattern in reasoning_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                reasoning_multiplier += 0.5
        
        return max(1, int(base_steps[complexity] * reasoning_multiplier))