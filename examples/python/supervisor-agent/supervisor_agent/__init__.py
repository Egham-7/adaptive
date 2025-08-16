"""Supervisor Agent - Multi-Agent System with LangGraph."""

__version__ = "0.1.0"

from supervisor_agent.supervisor.simple_supervisor import SimpleSupervisorAgent as SupervisorAgent
from supervisor_agent.supervisor.supervisor import SupervisorAgent as LangGraphSupervisorAgent

__all__ = ["SupervisorAgent", "LangGraphSupervisorAgent"]