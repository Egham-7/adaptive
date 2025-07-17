"""Model selection logic for MinionS protocol."""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class TaskType(Enum):
    """Task types for model selection."""
    OPEN_QA = "open_qa"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text_generation"
    CHATBOT = "chatbot"
    CLASSIFICATION = "classification"
    CLOSED_QA = "closed_qa"
    REWRITE = "rewrite"
    BRAINSTORMING = "brainstorming"
    EXTRACTION = "extraction"
    OTHER = "other"


class ModelSelector:
    """Model selection for MinionS protocol."""
    
    def __init__(self) -> None:
        """Initialize model selector with HuggingFace models."""
        self.model_mappings: Dict[TaskType, str] = {
            TaskType.OPEN_QA: "deepset/roberta-base-squad2",
            TaskType.CODE_GENERATION: "microsoft/codebert-base",
            TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
            TaskType.TEXT_GENERATION: "HuggingFaceTB/SmolLM3-3B",
            TaskType.CHATBOT: "HuggingFaceTB/SmolLM3-3B",
            TaskType.CLASSIFICATION: "cardiffnlp/twitter-roberta-base-sentiment",
            TaskType.CLOSED_QA: "google/tapas-base-finetuned-wtq",
            TaskType.REWRITE: "facebook/bart-large",
            TaskType.BRAINSTORMING: "HuggingFaceTB/SmolLM3-3B",
            TaskType.EXTRACTION: "dslim/bert-base-NER",
            TaskType.OTHER: "HuggingFaceTB/SmolLM3-3B",
        }
    
    def classify_task(self, text: str) -> TaskType:
        """Classify task type based on input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Classified task type
        """
        text_lower = text.lower()
        
        # Code generation keywords
        if any(keyword in text_lower for keyword in [
            "code", "program", "function", "class", "import", "def", "var",
            "javascript", "python", "java", "c++", "html", "css", "sql"
        ]):
            return TaskType.CODE_GENERATION
        
        # Summarization keywords
        if any(keyword in text_lower for keyword in [
            "summarize", "summary", "tldr", "brief", "outline", "key points"
        ]):
            return TaskType.SUMMARIZATION
        
        # Question answering keywords
        if any(keyword in text_lower for keyword in [
            "what", "how", "why", "when", "where", "who", "which", "?"
        ]):
            if "table" in text_lower or "data" in text_lower:
                return TaskType.CLOSED_QA
            return TaskType.OPEN_QA
        
        # Classification keywords
        if any(keyword in text_lower for keyword in [
            "classify", "categorize", "label", "sentiment", "positive", "negative"
        ]):
            return TaskType.CLASSIFICATION
        
        # Extraction keywords
        if any(keyword in text_lower for keyword in [
            "extract", "find", "identify", "locate", "names", "entities"
        ]):
            return TaskType.EXTRACTION
        
        # Rewrite keywords
        if any(keyword in text_lower for keyword in [
            "rewrite", "rephrase", "paraphrase", "translate", "improve"
        ]):
            return TaskType.REWRITE
        
        # Brainstorming keywords
        if any(keyword in text_lower for keyword in [
            "brainstorm", "ideas", "suggestions", "alternatives", "creative"
        ]):
            return TaskType.BRAINSTORMING
        
        # Chatbot keywords
        if any(keyword in text_lower for keyword in [
            "hello", "hi", "help", "assist", "chat", "talk"
        ]):
            return TaskType.CHATBOT
        
        # Default to text generation
        return TaskType.TEXT_GENERATION
    
    def select_model(self, task_type: TaskType) -> str:
        """Select appropriate model for task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Selected model name
        """
        return self.model_mappings.get(task_type, self.model_mappings[TaskType.OTHER])
    
    def analyze_conversation(self, conversation: List[Dict[str, str]]) -> Tuple[TaskType, str]:
        """Analyze conversation and select model.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Tuple of (task_type, selected_model)
        """
        if not conversation:
            return TaskType.OTHER, self.model_mappings[TaskType.OTHER]
        
        # Analyze first user message
        first_message = conversation[0]
        user_input = first_message.get("content", "")
        
        task_type = self.classify_task(user_input)
        selected_model = self.select_model(task_type)
        
        return task_type, selected_model
    
    def get_model_statistics(self, conversations: List[List[Dict[str, str]]]) -> Dict[str, int]:
        """Get statistics on model selection for conversations.
        
        Args:
            conversations: List of conversations
            
        Returns:
            Dictionary mapping models to usage counts
        """
        model_counts: Dict[str, int] = {}
        
        for conversation in conversations:
            _, selected_model = self.analyze_conversation(conversation)
            model_counts[selected_model] = model_counts.get(selected_model, 0) + 1
        
        return model_counts