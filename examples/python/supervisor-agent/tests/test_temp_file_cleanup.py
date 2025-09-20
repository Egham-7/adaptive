"""Tests for temporary file cleanup functionality in DataAgent."""

import os
import tempfile
from unittest.mock import patch, Mock

import pytest

from supervisor_agent.agents.data_agent import DataAgent, create_visualization, _cleanup_global_temp_files


class TestDataAgentTempFileCleanup:
    """Test cases for DataAgent temporary file cleanup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('supervisor_agent.agents.data_agent.get_config') as mock_config:
            mock_config.return_value = Mock(
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
                temperature=0.1
            )
            self.agent = DataAgent()
    
    def test_track_temp_file(self):
        """Test that temp files are tracked correctly."""
        import tempfile
        test_file = os.path.join(tempfile.gettempdir(), "test_file.png")
        
        # Initially no temp files
        assert len(self.agent._temp_files) == 0
        
        # Track a temp file
        self.agent._track_temp_file(test_file)
        assert len(self.agent._temp_files) == 1
        assert test_file in self.agent._temp_files
        
        # Don't track the same file twice
        self.agent._track_temp_file(test_file)
        assert len(self.agent._temp_files) == 1
    
    def test_get_temp_file_count(self):
        """Test temp file count reporting."""
        # Initially no temp files
        counts = self.agent.get_temp_file_count()
        assert counts["agent_temp_files"] == 0
        assert "global_temp_files" in counts
        assert "total_temp_files" in counts
        
        # Track some files
        self.agent._track_temp_file("/tmp/test1.png")
        self.agent._track_temp_file("/tmp/test2.png")
        
        counts = self.agent.get_temp_file_count()
        assert counts["agent_temp_files"] == 2
    
    def test_cleanup_temp_files(self):
        """Test that temp files are cleaned up properly."""
        # Create some real temporary files
        temp_files = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_files.append(temp_file.name)
            temp_file.close()
            self.agent._track_temp_file(temp_file.name)
        
        # Verify files exist
        for file_path in temp_files:
            assert os.path.exists(file_path)
        
        # Cleanup
        deleted_count = self.agent.cleanup_temp_files()
        
        # Verify cleanup
        assert deleted_count >= 3  # At least our 3 files
        assert len(self.agent._temp_files) == 0
        
        # Verify files are actually deleted
        for file_path in temp_files:
            assert not os.path.exists(file_path)
    
    def test_cleanup_handles_missing_files(self):
        """Test that cleanup handles missing files gracefully."""
        # Track a non-existent file
        self.agent._track_temp_file("/tmp/nonexistent.png")
        
        # Should not raise an exception
        deleted_count = self.agent.cleanup_temp_files()
        assert deleted_count >= 0  # Should not crash
        assert len(self.agent._temp_files) == 0
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tight_layout')
    def test_create_visualization_tracks_temp_file(self, mock_tight_layout, mock_title, 
                                                  mock_bar, mock_figure, mock_close, mock_savefig):
        """Test that create_visualization registers temp files for cleanup."""
        # Mock the temporary file creation
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test_chart.png"
            mock_temp.return_value = mock_temp_file
            
            # Call the visualization function using invoke
            result = create_visualization.invoke({
                "data": {"A": 1, "B": 2}, 
                "chart_type": "bar", 
                "title": "Test Chart"
            })
            
            # Should return the temp file path
            assert result == "/tmp/test_chart.png"
            
            # Should have registered the file for cleanup
            from supervisor_agent.agents.data_agent import _temp_file_registry
            assert "/tmp/test_chart.png" in _temp_file_registry
    
    def test_global_temp_file_cleanup(self):
        """Test global temp file cleanup functionality."""
        # Create some real temporary files and register them globally
        from supervisor_agent.agents.data_agent import _register_temp_file
        
        temp_files = []
        for i in range(2):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_files.append(temp_file.name)
            temp_file.close()
            _register_temp_file(temp_file.name)
        
        # Verify files exist
        for file_path in temp_files:
            assert os.path.exists(file_path)
        
        # Cleanup global files
        deleted_count = _cleanup_global_temp_files()
        
        # Verify cleanup
        assert deleted_count == 2
        
        # Verify files are actually deleted
        for file_path in temp_files:
            assert not os.path.exists(file_path)
    
    def test_agent_cleanup_includes_global_files(self):
        """Test that agent cleanup also cleans global temp files."""
        from supervisor_agent.agents.data_agent import _register_temp_file
        
        # Create agent-specific temp file
        agent_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        agent_temp.close()
        self.agent._track_temp_file(agent_temp.name)
        
        # Create global temp file
        global_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        global_temp.close()
        _register_temp_file(global_temp.name)
        
        # Both files should exist
        assert os.path.exists(agent_temp.name)
        assert os.path.exists(global_temp.name)
        
        # Agent cleanup should clean both
        deleted_count = self.agent.cleanup_temp_files()
        assert deleted_count >= 2
        
        # Both files should be deleted
        assert not os.path.exists(agent_temp.name)
        assert not os.path.exists(global_temp.name)