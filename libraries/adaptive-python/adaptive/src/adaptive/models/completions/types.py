"""
Common type definitions used across the provider package.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel  # type: ignore

# Type alias for message roles
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """
    Represents a single chat message in a conversation.
    """

    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """
    Standard request format for chat completions.
    """

    messages: List[Message]
