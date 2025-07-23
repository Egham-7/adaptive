"""Handoff tools for agent coordination in the supervisor system."""

from typing import Literal

from langchain_core.tools import tool


@tool
def handoff_to_code_agent(task: str) -> str:
    """Hand off task to the Code Agent for code-related work.
    
    Args:
        task: Description of the coding task to perform
    
    Returns:
        Confirmation that task was handed off to Code Agent
    """
    return f"Task handed off to Code Agent: {task}"


@tool
def handoff_to_data_agent(task: str) -> str:
    """Hand off task to the Data Agent for data analysis work.
    
    Args:
        task: Description of the data analysis task to perform
    
    Returns:
        Confirmation that task was handed off to Data Agent
    """
    return f"Task handed off to Data Agent: {task}"


@tool
def handoff_to_file_agent(task: str) -> str:
    """Hand off task to the File Agent for file operations.
    
    Args:
        task: Description of the file operation task to perform
    
    Returns:
        Confirmation that task was handed off to File Agent
    """
    return f"Task handed off to File Agent: {task}"


@tool
def complete_task(result: str) -> str:
    """Mark the current task as completed and return final result.
    
    Args:
        result: Final result or output of the completed task
    
    Returns:
        Confirmation that task is completed
    """
    return f"Task completed. Result: {result}"


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