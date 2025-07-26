"""Tests for individual agents."""

import pytest
from unittest.mock import Mock, patch

from supervisor_agent.agents.code_agent import CodeAgent
from supervisor_agent.agents.data_agent import DataAgent
from supervisor_agent.agents.file_agent import FileAgent


class TestCodeAgent:
    """Test cases for Code Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('supervisor_agent.agents.code_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            self.agent = CodeAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.name == "Code Agent"
        assert "code generation" in self.agent.description.lower()
        assert len(self.agent.tools) > 0
    
    def test_can_handle_code_requests(self):
        """Test that agent can identify code-related requests."""
        code_requests = [
            "Generate a Python function",
            "Debug this code",
            "Create unit tests",
            "Explain this algorithm",
            "Fix syntax error"
        ]
        
        for request in code_requests:
            assert self.agent.can_handle(request), f"Should handle: {request}"
    
    def test_cannot_handle_non_code_requests(self):
        """Test that agent rejects non-code requests."""
        non_code_requests = [
            "Analyze this dataset",
            "Create a chart",
            "List files in directory",
            "Calculate statistics"
        ]
        
        for request in non_code_requests:
            assert not self.agent.can_handle(request), f"Should not handle: {request}"
    
    def test_system_message(self):
        """Test system message generation."""
        message = self.agent.get_system_message()
        assert "Code Agent" in message
        assert "code generation" in message.lower() or "debugging" in message.lower()


class TestDataAgent:
    """Test cases for Data Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('supervisor_agent.agents.data_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            self.agent = DataAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.name == "Data Agent"
        assert "data" in self.agent.description.lower()
        assert len(self.agent.tools) > 0
    
    def test_can_handle_data_requests(self):
        """Test that agent can identify data-related requests."""
        data_requests = [
            "Analyze this data",
            "Calculate statistics",
            "Create a chart",
            "Process CSV file",
            "Calculate mean and median"
        ]
        
        for request in data_requests:
            assert self.agent.can_handle(request), f"Should handle: {request}"
    
    def test_cannot_handle_non_data_requests(self):
        """Test that agent rejects non-data requests."""
        non_data_requests = [
            "Generate Python code",
            "Debug this function",
            "Read a file",
            "List directory contents"
        ]
        
        for request in non_data_requests:
            assert not self.agent.can_handle(request), f"Should not handle: {request}"
    
    def test_system_message(self):
        """Test system message generation."""
        message = self.agent.get_system_message()
        assert "Data Agent" in message
        assert "data" in message.lower() or "analysis" in message.lower()


class TestFileAgent:
    """Test cases for File Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('supervisor_agent.agents.file_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            self.agent = FileAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.name == "File Agent"
        assert "file" in self.agent.description.lower()
        assert len(self.agent.tools) > 0
    
    def test_can_handle_file_requests(self):
        """Test that agent can identify file-related requests."""
        file_requests = [
            "Read a file",
            "List directory contents",
            "Create a folder",
            "Delete this file",
            "Search in files"
        ]
        
        for request in file_requests:
            assert self.agent.can_handle(request), f"Should handle: {request}"
    
    def test_cannot_handle_non_file_requests(self):
        """Test that agent rejects non-file requests."""
        non_file_requests = [
            "Generate Python code",
            "Calculate statistics",
            "Create a chart",
            "Debug this function"
        ]
        
        for request in non_file_requests:
            assert not self.agent.can_handle(request), f"Should not handle: {request}"
    
    def test_system_message(self):
        """Test system message generation."""
        message = self.agent.get_system_message()
        assert "File Agent" in message
        assert "file" in message.lower() or "directory" in message.lower()


class TestAgentTools:
    """Test individual agent tools."""
    
    def test_code_agent_tools(self):
        """Test that code agent has expected tools."""
        with patch('supervisor_agent.agents.code_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            agent = CodeAgent()
        
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = [
            "generate_code",
            "validate_python_syntax",
            "explain_code",
            "generate_tests",
            "debug_code",
            "run_python_code"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
    
    def test_data_agent_tools(self):
        """Test that data agent has expected tools."""
        with patch('supervisor_agent.agents.data_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            agent = DataAgent()
        
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = [
            "analyze_data",
            "calculate_statistics", 
            "create_visualization",
            "process_csv_data"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
    
    def test_file_agent_tools(self):
        """Test that file agent has expected tools."""
        with patch('supervisor_agent.agents.file_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            agent = FileAgent()
        
        tool_names = [tool.name for tool in agent.tools]
        expected_tools = [
            "read_file",
            "write_file",
            "list_directory",
            "create_directory",
            "delete_file_or_directory"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"