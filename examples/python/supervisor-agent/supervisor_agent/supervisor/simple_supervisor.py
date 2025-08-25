"""Simplified Supervisor Agent - Direct coordination without complex graphs."""

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from supervisor_agent.agents.code_agent import CodeAgent
from supervisor_agent.agents.data_agent import DataAgent
from supervisor_agent.agents.file_agent import FileAgent
from supervisor_agent.utils.config import get_config


class SimpleSupervisorAgent:
    """Simplified supervisor that directly routes to appropriate agents."""
    
    def __init__(self):
        """Initialize the supervisor and all agents."""
        self.config = get_config()
        
        # Initialize specialized agents
        self.code_agent = CodeAgent()
        self.data_agent = DataAgent()
        self.file_agent = FileAgent()
        
        # Initialize supervisor LLM
        self.supervisor_llm = ChatOpenAI(
            model=self.config.openai_model,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key
        )
        
        # Create agent executors
        self.code_executor = create_react_agent(
            self.supervisor_llm,
            self.code_agent.tools
        )
        
        self.data_executor = create_react_agent(
            self.supervisor_llm,
            self.data_agent.tools
        )
        
        self.file_executor = create_react_agent(
            self.supervisor_llm,
            self.file_agent.tools
        )
    
    def _route_request(self, user_input: str) -> str:
        """Route request to the most appropriate agent."""
        capabilities = self.can_handle_request(user_input)
        
        # Find the best agent (prefer agents that can handle the request)
        capable_agents = [agent for agent, can_handle in capabilities.items() if can_handle]
        
        if not capable_agents:
            # If no agent specifically matches, use a simple keyword approach
            user_lower = user_input.lower()
            if any(keyword in user_lower for keyword in ["code", "function", "debug", "test", "program"]):
                return "code_agent"
            elif any(keyword in user_lower for keyword in ["data", "calculate", "chart", "statistics", "math"]):
                return "data_agent"
            elif any(keyword in user_lower for keyword in ["file", "read", "write", "directory", "folder"]):
                return "file_agent"
            else:
                # Default to code agent for general requests
                return "code_agent"
        
        # Return the first capable agent (could be made smarter)
        return capable_agents[0]
    
    def process_request(self, user_input: str) -> str:
        """Process a user request through the appropriate agent.
        
        Args:
            user_input: User's request or question
            
        Returns:
            Final response from the agent system
        """
        try:
            # Route to appropriate agent
            selected_agent = self._route_request(user_input)
            
            # Prepare messages with appropriate system message
            if selected_agent == "code_agent":
                system_msg = SystemMessage(content=self.code_agent.get_system_message())
                executor = self.code_executor
            elif selected_agent == "data_agent":
                system_msg = SystemMessage(content=self.data_agent.get_system_message())
                executor = self.data_executor
            elif selected_agent == "file_agent":
                system_msg = SystemMessage(content=self.file_agent.get_system_message())
                executor = self.file_executor
            else:
                # Fallback to code agent
                system_msg = SystemMessage(content=self.code_agent.get_system_message())
                executor = self.code_executor
            
            # Execute the request
            messages = [system_msg, HumanMessage(content=user_input)]
            result = executor.invoke({"messages": messages})
            
            # Extract the response
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
            
            return "Task completed successfully."
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    def can_handle_request(self, user_input: str) -> Dict[str, bool]:
        """Analyze which agents can handle the request.
        
        Args:
            user_input: User's request
            
        Returns:
            Dictionary showing which agents can handle the request
        """
        return {
            "code_agent": self.code_agent.can_handle(user_input),
            "data_agent": self.data_agent.can_handle(user_input),
            "file_agent": self.file_agent.can_handle(user_input)
        }
    
    def get_agent_capabilities(self) -> Dict[str, str]:
        """Get a summary of each agent's capabilities.
        
        Returns:
            Dictionary with agent capabilities
        """
        return {
            "code_agent": self.code_agent.description,
            "data_agent": self.data_agent.description,
            "file_agent": self.file_agent.description,
            "supervisor": "Routes tasks to the most appropriate specialized agent"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of all agents.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test supervisor LLM
            test_response = self.supervisor_llm.invoke([HumanMessage(content="Hello")])
            supervisor_healthy = bool(test_response.content)
        except Exception:
            supervisor_healthy = False
        
        return {
            "supervisor": supervisor_healthy,
            "code_agent": len(self.code_agent.tools) > 0,
            "data_agent": len(self.data_agent.tools) > 0,
            "file_agent": len(self.file_agent.tools) > 0,
            "config_valid": bool(self.config.openai_api_key),
            "total_tools": (
                len(self.code_agent.tools) + 
                len(self.data_agent.tools) + 
                len(self.file_agent.tools)
            )
        }