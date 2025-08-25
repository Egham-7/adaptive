"""Tests for the supervisor agent."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from supervisor_agent.supervisor.supervisor import SupervisorAgent
from supervisor_agent.supervisor.tools import (
    HANDOFF_TOOLS, 
    parse_structured_message, 
    extract_agent_from_message,
    is_completion_message,
    extract_completion_result
)


class TestSupervisorAgent:
    """Test cases for Supervisor Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('supervisor_agent.supervisor.supervisor.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini", 
                temperature=0.1
            )
            # Mock the LLM and agents
            with patch('supervisor_agent.supervisor.supervisor.ChatOpenAI'), \
                 patch('supervisor_agent.supervisor.supervisor.CodeAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.DataAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.FileAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.create_react_agent'):
                self.supervisor = SupervisorAgent()
    
    def test_initialization(self):
        """Test supervisor initialization."""
        assert hasattr(self.supervisor, 'code_agent')
        assert hasattr(self.supervisor, 'data_agent')
        assert hasattr(self.supervisor, 'file_agent')
        assert hasattr(self.supervisor, 'supervisor_llm')
        assert hasattr(self.supervisor, 'graph')
    
    def test_get_agent_capabilities(self):
        """Test getting agent capabilities."""
        # Mock agent descriptions
        self.supervisor.code_agent.description = "Code generation and debugging"
        self.supervisor.data_agent.description = "Data analysis and visualization" 
        self.supervisor.file_agent.description = "File operations and management"
        
        capabilities = self.supervisor.get_agent_capabilities()
        
        assert "code_agent" in capabilities
        assert "data_agent" in capabilities
        assert "file_agent" in capabilities
        assert "supervisor" in capabilities
        
        assert capabilities["code_agent"] == "Code generation and debugging"
        assert capabilities["data_agent"] == "Data analysis and visualization"
        assert capabilities["file_agent"] == "File operations and management"
    
    def test_can_handle_request(self):
        """Test request handling analysis."""
        # Mock agent can_handle methods
        self.supervisor.code_agent.can_handle = Mock(return_value=True)
        self.supervisor.data_agent.can_handle = Mock(return_value=False)
        self.supervisor.file_agent.can_handle = Mock(return_value=False)
        
        result = self.supervisor.can_handle_request("Generate Python code")
        
        assert result["code_agent"] is True
        assert result["data_agent"] is False
        assert result["file_agent"] is False
        
        # Verify methods were called
        self.supervisor.code_agent.can_handle.assert_called_once_with("Generate Python code")
        self.supervisor.data_agent.can_handle.assert_called_once_with("Generate Python code")
        self.supervisor.file_agent.can_handle.assert_called_once_with("Generate Python code")
    
    def test_health_check(self):
        """Test health check functionality."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "test response"
        self.supervisor.supervisor_llm.invoke = Mock(return_value=mock_response)
        
        # Mock agent tools
        self.supervisor.code_agent.tools = ["tool1", "tool2"]
        self.supervisor.data_agent.tools = ["tool3"]
        self.supervisor.file_agent.tools = ["tool4", "tool5"]
        
        # Mock config
        self.supervisor.config.openai_api_key = "test-key"
        
        health = self.supervisor.health_check()
        
        assert health["supervisor"] is True
        assert health["code_agent"] is True
        assert health["data_agent"] is True
        assert health["file_agent"] is True
        assert health["config_valid"] is True
        assert health["total_tools"] == len(HANDOFF_TOOLS) + 5  # 5 mocked tools + handoff tools
    
    def test_health_check_llm_failure(self):
        """Test health check when LLM fails."""
        # Mock LLM failure
        self.supervisor.supervisor_llm.invoke = Mock(side_effect=Exception("API Error"))
        
        # Mock other components as healthy
        self.supervisor.code_agent.tools = ["tool1"]
        self.supervisor.data_agent.tools = ["tool2"]
        self.supervisor.file_agent.tools = ["tool3"]
        self.supervisor.config.openai_api_key = "test-key"
        
        health = self.supervisor.health_check()
        
        assert health["supervisor"] is False
        assert health["code_agent"] is True
        assert health["data_agent"] is True
        assert health["file_agent"] is True
        assert health["config_valid"] is True


class TestSupervisorTools:
    """Test supervisor handoff tools."""
    
    def test_handoff_tools_available(self):
        """Test that all handoff tools are available."""
        assert len(HANDOFF_TOOLS) > 0
        
        tool_names = [tool.name for tool in HANDOFF_TOOLS]
        expected_tools = [
            "handoff_to_code_agent",
            "handoff_to_data_agent", 
            "handoff_to_file_agent",
            "complete_task"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing handoff tool: {tool}"
    
    def test_handoff_tool_execution(self):
        """Test handoff tool execution."""
        from supervisor_agent.supervisor.tools import handoff_to_code_agent
        
        result = handoff_to_code_agent("Generate a function")
        assert "Task handed off to Code Agent" in result
        assert "Generate a function" in result


class TestSupervisorSystemMessage:
    """Test supervisor system message generation."""
    
    def test_supervisor_system_message(self):
        """Test supervisor system message content."""
        with patch('supervisor_agent.supervisor.supervisor.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            with patch('supervisor_agent.supervisor.supervisor.ChatOpenAI'), \
                 patch('supervisor_agent.supervisor.supervisor.CodeAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.DataAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.FileAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.create_react_agent'):
                supervisor = SupervisorAgent()
        
        message = supervisor._get_supervisor_system_message()
        
        # Check that message contains key components
        assert "Supervisor Agent" in message
        assert "Code Agent" in message
        assert "Data Agent" in message
        assert "File Agent" in message
        assert "handoff" in message.lower()
        assert "coordinate" in message.lower() or "routing" in message.lower()


class TestSupervisorIntegration:
    """Integration tests for supervisor functionality."""
    
    @patch('supervisor_agent.supervisor.supervisor.create_react_agent')
    @patch('supervisor_agent.supervisor.supervisor.ChatOpenAI')
    @patch('supervisor_agent.supervisor.supervisor.get_config')
    def test_process_request_mock(self, mock_config, mock_llm, mock_create_agent):
        """Test request processing with mocked components."""
        # Setup mocks
        mock_config.return_value = Mock(
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
            temperature=0.1
        )
        
        # Mock the graph invoke method
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "task_completed": True,
            "result": "Task completed successfully",
            "messages": [Mock(content="Test response")]
        }
        
        with patch('supervisor_agent.supervisor.supervisor.CodeAgent'), \
             patch('supervisor_agent.supervisor.supervisor.DataAgent'), \
             patch('supervisor_agent.supervisor.supervisor.FileAgent'):
            supervisor = SupervisorAgent()
            supervisor.graph = mock_graph
        
        # Test request processing
        response = supervisor.process_request("Test request")
        
        assert response == "Task completed successfully"
        mock_graph.invoke.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in request processing."""
        with patch('supervisor_agent.supervisor.supervisor.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            
            with patch('supervisor_agent.supervisor.supervisor.ChatOpenAI'), \
                 patch('supervisor_agent.supervisor.supervisor.CodeAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.DataAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.FileAgent'), \
                 patch('supervisor_agent.supervisor.supervisor.create_react_agent'):
                supervisor = SupervisorAgent()
                
                # Mock graph to raise exception
                supervisor.graph = Mock()
                supervisor.graph.invoke.side_effect = Exception("Test error")
                
                response = supervisor.process_request("Test request")
                
                assert "Error processing request" in response
                assert "Test error" in response


class TestStructuredMessageParsing:
    """Test cases for structured message parsing functions."""
    
    def test_parse_handoff_message(self):
        """Test parsing valid handoff messages."""
        message = 'HANDOFF_MESSAGE:{"action":"handoff","target_agent":"code_agent","agent_name":"Code Agent","task":"Generate code"}'
        result = parse_structured_message(message)
        
        assert result["action"] == "handoff"
        assert result["target_agent"] == "code_agent"
        assert result["agent_name"] == "Code Agent"
        assert result["task"] == "Generate code"
    
    def test_parse_completion_message(self):
        """Test parsing valid completion messages."""
        message = 'COMPLETION_MESSAGE:{"action":"complete","result":"Task finished successfully"}'
        result = parse_structured_message(message)
        
        assert result["action"] == "complete"
        assert result["result"] == "Task finished successfully"
    
    def test_parse_invalid_message(self):
        """Test parsing invalid or non-structured messages."""
        # Non-structured message
        result = parse_structured_message("Just a regular message")
        assert result == {}
        
        # Malformed JSON
        result = parse_structured_message("HANDOFF_MESSAGE:{invalid json}")
        assert result == {}
        
        # Empty message
        result = parse_structured_message("")
        assert result == {}
    
    def test_extract_agent_from_handoff_message(self):
        """Test extracting agent from handoff messages."""
        message = 'HANDOFF_MESSAGE:{"action":"handoff","target_agent":"data_agent","agent_name":"Data Agent","task":"Analyze data"}'
        agent = extract_agent_from_message(message)
        assert agent == "data_agent"
        
        # Non-handoff message
        message = 'COMPLETION_MESSAGE:{"action":"complete","result":"Done"}'
        agent = extract_agent_from_message(message)
        assert agent == ""
        
        # Invalid message
        agent = extract_agent_from_message("Not a structured message")
        assert agent == ""
    
    def test_is_completion_message(self):
        """Test completion message detection."""
        # Valid completion message
        message = 'COMPLETION_MESSAGE:{"action":"complete","result":"Task done"}'
        assert is_completion_message(message) is True
        
        # Handoff message
        message = 'HANDOFF_MESSAGE:{"action":"handoff","target_agent":"code_agent","task":"Code task"}'
        assert is_completion_message(message) is False
        
        # Non-structured message
        assert is_completion_message("Regular message") is False
    
    def test_extract_completion_result(self):
        """Test extracting results from completion messages."""
        message = 'COMPLETION_MESSAGE:{"action":"complete","result":"The final result"}'
        result = extract_completion_result(message)
        assert result == "The final result"
        
        # Non-completion message
        message = 'HANDOFF_MESSAGE:{"action":"handoff","target_agent":"file_agent"}'
        result = extract_completion_result(message)
        assert result == ""
        
        # Invalid message
        result = extract_completion_result("Not structured")
        assert result == ""