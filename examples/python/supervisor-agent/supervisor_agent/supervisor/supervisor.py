"""Supervisor Agent - Coordinates multiple specialized agents using LangGraph."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

from supervisor_agent.agents.code_agent import CodeAgent
from supervisor_agent.agents.data_agent import DataAgent
from supervisor_agent.agents.file_agent import FileAgent
from supervisor_agent.supervisor.tools import (
    HANDOFF_TOOLS, 
    extract_agent_from_message, 
    is_completion_message,
    extract_completion_result
)
from supervisor_agent.utils.config import get_config


class AgentState(TypedDict):
    """State shared between agents in the supervisor system."""
    messages: List[BaseMessage]
    current_agent: str
    task_completed: bool
    result: Optional[str]


class SupervisorAgent:
    """Main supervisor that coordinates specialized agents."""
    
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
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph for agent coordination."""
        
        # Create individual agent nodes with proper system messages
        code_agent_node = create_react_agent(
            self.supervisor_llm,
            self.code_agent.tools,
            state_modifier=SystemMessage(content=self.code_agent.get_system_message())
        )
        
        data_agent_node = create_react_agent(
            self.supervisor_llm,
            self.data_agent.tools,
            state_modifier=SystemMessage(content=self.data_agent.get_system_message())
        )
        
        file_agent_node = create_react_agent(
            self.supervisor_llm,
            self.file_agent.tools,
            state_modifier=SystemMessage(content=self.file_agent.get_system_message())
        )
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("code_agent", self._create_agent_node(code_agent_node, "code_agent"))
        workflow.add_node("data_agent", self._create_agent_node(data_agent_node, "data_agent"))
        workflow.add_node("file_agent", self._create_agent_node(file_agent_node, "file_agent"))
        
        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "code_agent": "code_agent",
                "data_agent": "data_agent", 
                "file_agent": "file_agent",
                "FINISH": END
            }
        )
        
        # All agents return to supervisor
        workflow.add_edge("code_agent", "supervisor")
        workflow.add_edge("data_agent", "supervisor")
        workflow.add_edge("file_agent", "supervisor")
        
        return workflow.compile()
    
    def _get_supervisor_system_message(self) -> str:
        """Get the system message for the supervisor."""
        return """You are the Supervisor Agent that coordinates a team of specialized agents.

Available agents:
- Code Agent: Handles code generation, debugging, testing, and code analysis
- Data Agent: Handles data analysis, statistics, visualizations, and mathematical calculations  
- File Agent: Handles file operations, directory management, and system tasks

Your responsibilities:
1. Analyze incoming user requests
2. Determine which agent(s) can best handle the task
3. Route tasks to appropriate agents using handoff tools
4. Coordinate multi-agent workflows when needed
5. Aggregate results and provide final responses

When you receive a user request:
1. Analyze what type of task it is
2. Use the appropriate handoff tool to delegate to the right agent
3. If the task requires multiple agents, coordinate the workflow
4. Once all work is complete, use complete_task with the final result

Available handoff tools:
- handoff_to_code_agent: For code-related tasks
- handoff_to_data_agent: For data analysis and math tasks
- handoff_to_file_agent: For file and system operations
- complete_task: To mark the task as finished with final results

Always be helpful and ensure tasks are routed to the most appropriate agent."""
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Process messages in the supervisor node."""
        messages = state["messages"]
        
        # Create supervisor agent with handoff tools
        supervisor_agent = create_react_agent(
            self.supervisor_llm,
            HANDOFF_TOOLS,
            state_modifier=SystemMessage(content=self._get_supervisor_system_message())
        )
        
        # Process the messages
        result = supervisor_agent.invoke({"messages": messages})
        
        # Update state
        new_state = state.copy()
        new_state["messages"] = result["messages"]
        
        # Check if task was completed or handed off using robust parsing
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            message_content = last_message.content
            
            # Check for task completion
            if is_completion_message(message_content):
                new_state["task_completed"] = True
                completion_result = extract_completion_result(message_content)
                new_state["result"] = completion_result if completion_result else message_content
            else:
                # Check for agent handoff
                target_agent = extract_agent_from_message(message_content)
                if target_agent:
                    new_state["current_agent"] = target_agent
        
        return new_state
    
    def _create_agent_node(self, agent_node, agent_name: str):
        """Create a wrapper for agent nodes."""
        def agent_wrapper(state: AgentState) -> AgentState:
            result = agent_node.invoke({"messages": state["messages"]})
            new_state = state.copy()
            new_state["messages"] = result["messages"]
            new_state["current_agent"] = "supervisor"  # Return to supervisor
            return new_state
        
        return agent_wrapper
    
    def _route_to_agent(self, state: AgentState) -> Literal["code_agent", "data_agent", "file_agent", "FINISH"]:
        """Route to the appropriate agent based on state."""
        if state.get("task_completed", False):
            return "FINISH"
        
        current_agent = state.get("current_agent", "supervisor")
        if current_agent in ["code_agent", "data_agent", "file_agent"]:
            return current_agent
        
        return "FINISH"
    
    def process_request(self, user_input: str) -> str:
        """Process a user request through the multi-agent system.
        
        Args:
            user_input: User's request or question
            
        Returns:
            Final response from the agent system
        """
        try:
            # Initialize state
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "current_agent": "supervisor",
                "task_completed": False,
                "result": None
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract final result
            if final_state.get("result"):
                return final_state["result"]
            
            # If no specific result, return the last message
            if final_state.get("messages"):
                last_message = final_state["messages"][-1]
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
            "supervisor": "Coordinates and routes tasks between specialized agents"
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
                len(self.file_agent.tools) + 
                len(HANDOFF_TOOLS)
            )
        }