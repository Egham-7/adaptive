"""Utilities for working with OpenAI request structures."""


from openai.types.chat import CompletionCreateParams


def extract_last_message_content(
    chat_completion_request: CompletionCreateParams,
) -> str:
    """
    Extract the content from the last message in an OpenAI chat completion request.

    Args:
        chat_completion_request: The OpenAI chat completion request dict

    Returns:
        The text content of the last message, or empty string if no messages
    """
    messages = list(chat_completion_request.get("messages", []))
    if not messages:
        return ""

    last_msg = messages[-1]
    content = last_msg.get("content", "")

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle multimodal content, extract text parts
        text_parts = [
            item.get("text", "")
            for item in content
            if item.get("type") == "text" and item.get("text")
        ]
        return " ".join(text_parts)
    else:
        return ""


def has_tools(chat_completion_request: CompletionCreateParams) -> bool:
    """
    Check if the OpenAI chat completion request has tools defined.

    Args:
        chat_completion_request: The OpenAI chat completion request dict

    Returns:
        True if tools are present, False otherwise
    """
    tools = chat_completion_request.get("tools")
    return tools is not None and len(list(tools)) > 0
