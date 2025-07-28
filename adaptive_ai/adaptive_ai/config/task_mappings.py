"""
Task-based model mappings for standard protocol.
Maps task types to ranked model choices for standard LLM protocol.
"""

from adaptive_ai.models.llm_core_models import ModelEntry, TaskModelMapping
from adaptive_ai.models.llm_enums import ProviderType, TaskType

# Task-based model mappings for standard protocol
# Models are ordered with budget-friendly options first, premium options last
# This allows cost_bias to dynamically reorder based on cost-performance preference
task_model_mappings_data = {
    TaskType.OPEN_QA: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-mini"),
            # Premium Tier
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3"),
        ]
    ),
    TaskType.CODE_GENERATION: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-mini"),
            # Premium Tier
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
        ]
    ),
    TaskType.SUMMARIZATION: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
        ]
    ),
    TaskType.TEXT_GENERATION: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-mini"),
            # Premium Tier
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
        ]
    ),
    TaskType.CHATBOT: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            # Premium Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3"),
        ]
    ),
    TaskType.CLASSIFICATION: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
        ]
    ),
    TaskType.CLOSED_QA: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            # Premium Tier
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
        ]
    ),
    TaskType.REWRITE: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-mini"),
            # Premium Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
        ]
    ),
    TaskType.BRAINSTORMING: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            # Premium Tier
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
        ]
    ),
    TaskType.EXTRACTION: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
        ]
    ),
    TaskType.OTHER: TaskModelMapping(
        model_entries=[
            # Ultra Budget Tier
            ModelEntry(providers=[ProviderType.DEEPSEEK], model_name="deepseek-chat"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-nano"),
            ModelEntry(providers=[ProviderType.GROK], model_name="grok-3-mini"),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o-mini"),
            ModelEntry(
                providers=[ProviderType.GROQ], model_name="llama-3.1-70b-versatile"
            ),
            # Budget Tier
            ModelEntry(
                providers=[ProviderType.ANTHROPIC], model_name="claude-3-haiku-20240307"
            ),
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4.1-mini"),
            # Premium Tier
            ModelEntry(providers=[ProviderType.OPENAI], model_name="gpt-4o"),
            ModelEntry(
                providers=[ProviderType.ANTHROPIC],
                model_name="claude-3-5-sonnet-20241022",
            ),
        ]
    ),
}
