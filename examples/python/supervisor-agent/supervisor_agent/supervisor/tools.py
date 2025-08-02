"""Handoff tools for agent coordination in the supervisor system."""

import json
from typing import Literal

from langchain_core.tools import tool


def _create_handoff_message(agent_name: str, agent_id: str, task: str) -> str:
    """Create a structured handoff message in JSON format.
    
    Args:
        agent_name: Human-readable agent name
        agent_id: Internal agent identifier
        task: Task description
        
    Returns:
        JSON-structured handoff message
    """
    message_data = {
        "action": "handoff",
        "target_agent": agent_id,
        "agent_name": agent_name,
        "task": task
    }
    return f"HANDOFF_MESSAGE:{json.dumps(message_data)}"


def _create_completion_message(result: str) -> str:
    """Create a structured completion message.
    
    Args:
        result: Task result
        
    Returns:
        JSON-structured completion message
    """
    message_data = {
        "action": "complete",
        "result": result
    }
    return f"COMPLETION_MESSAGE:{json.dumps(message_data)}"


@tool
def handoff_to_code_agent(task: str) -> str:
    """Hand off task to the Code Agent for code-related work.
    
    Args:
        task: Description of the coding task to perform
    
    Returns:
        Structured handoff message for Code Agent
    """
    return _create_handoff_message("Code Agent", "code_agent", task)


@tool
def handoff_to_data_agent(task: str) -> str:
    """Hand off task to the Data Agent for data analysis work.
    
    Args:
        task: Description of the data analysis task to perform
    
    Returns:
        Structured handoff message for Data Agent
    """
    return _create_handoff_message("Data Agent", "data_agent", task)


@tool
def handoff_to_file_agent(task: str) -> str:
    """Hand off task to the File Agent for file operations.
    
    Args:
        task: Description of the file operation task to perform
    
    Returns:
        Structured handoff message for File Agent
    """
    return _create_handoff_message("File Agent", "file_agent", task)


@tool
def complete_task(result: str) -> str:
    """Mark the current task as completed and return final result.
    
    Args:
        result: Final result or output of the completed task
    
    Returns:
        Structured completion message
    """
    return _create_completion_message(result)


def parse_structured_message(message_content: str) -> dict:
    """Parse structured messages from handoff tools.
    
    Args:
        message_content: The message content to parse
        
    Returns:
        Dictionary with parsed message data, or empty dict if not a structured message
    """
    try:
        if message_content.startswith("HANDOFF_MESSAGE:"):
            json_str = message_content[len("HANDOFF_MESSAGE:"):]
            return json.loads(json_str)
        elif message_content.startswith("COMPLETION_MESSAGE:"):
            json_str = message_content[len("COMPLETION_MESSAGE:"):]
            return json.loads(json_str)
        else:
            # Not a structured message, return empty dict
            return {}
    except (json.JSONDecodeError, ValueError):
        # Malformed JSON, return empty dict
        return {}


def extract_agent_from_message(message_content: str) -> str:
    """Extract target agent from a handoff message.
    
    Args:
        message_content: The message content to parse
        
    Returns:
        Agent identifier or empty string if not found
    """
    parsed = parse_structured_message(message_content)
    if parsed.get("action") == "handoff":
        return parsed.get("target_agent", "")
    return ""


def is_completion_message(message_content: str) -> bool:
    """Check if message is a task completion message.
    
    Args:
        message_content: The message content to check
        
    Returns:
        True if this is a completion message
    """
    parsed = parse_structured_message(message_content)
    return parsed.get("action") == "complete"


def extract_completion_result(message_content: str) -> str:
    """Extract result from a completion message.
    
    Args:
        message_content: The message content to parse
        
    Returns:
        Completion result or empty string if not found
    """
    parsed = parse_structured_message(message_content)
    if parsed.get("action") == "complete":
        return parsed.get("result", "")
    return ""


def create_handoff_tool(agent_name: str):
    """Create a handoff tool for a specific agent.
    
    Args:
        agent_name: Name of the agent to create handoff tool for
        
    Returns:
        Handoff tool function for the specified agent
    """
    agent_map = {
        "code": handoff_to_code_agent,
        "data": handoff_to_data_agent,
        "file": handoff_to_file_agent
    }
    
    return agent_map.get(agent_name.lower())


# All available handoff tools
HANDOFF_TOOLS = [
    handoff_to_code_agent,
    handoff_to_data_agent,
    handoff_to_file_agent,
    complete_task,
]