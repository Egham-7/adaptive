"""Protocol testing for MinionS implementation using adaptive_ai service."""

from typing import Dict, List, Optional, Tuple, Any
import time
import pandas as pd
from dataclasses import dataclass
from .model_selector import ModelSelector, TaskType
from .adaptive_ai_client import AdaptiveAIClient, AdaptiveAIResponse


@dataclass
class ProtocolResult:
    """Result of protocol testing."""
    task_type: TaskType
    selected_model: str
    protocol: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    classification_result: Dict[str, Any] = None
    
    
class ProtocolTester:
    """Tester for MinionS protocol performance using adaptive_ai service."""
    
    def __init__(self, service_url: str = "http://localhost:8000") -> None:
        """Initialize protocol tester.
        
        Args:
            service_url: URL of the adaptive_ai service
        """
        self.client = AdaptiveAIClient(service_url)
        self.model_selector = ModelSelector()  # For local task classification
        self.results: List[ProtocolResult] = []
        self.service_url = service_url
    
    def test_service_connection(self) -> Dict[str, Any]:
        """Test connection to the adaptive_ai service.
        
        Returns:
            Connection test results
        """
        return self.client.test_connection()
    
    def test_model_selection(self, conversations: List[List[Dict[str, str]]]) -> List[ProtocolResult]:
        """Test model selection on conversations using adaptive_ai service.
        
        Args:
            conversations: List of conversations to test
            
        Returns:
            List of protocol results
        """
        results = []
        
        # Check service availability first
        if not self.client.health_check():
            raise ConnectionError(f"adaptive_ai service is not available at {self.service_url}")
        
        for i, conversation in enumerate(conversations):
            try:
                # Extract the user prompt from conversation
                if not conversation:
                    continue
                    
                user_prompt = conversation[0].get('content', '')
                if not user_prompt:
                    continue
                
                # Get local task classification for comparison
                local_task_type, _ = self.model_selector.analyze_conversation(conversation)
                
                # Make request to adaptive_ai service
                service_response: AdaptiveAIResponse = self.client.make_request(user_prompt)
                
                result = ProtocolResult(
                    task_type=local_task_type,  # Local classification for comparison
                    selected_model=service_response.selected_model,
                    protocol=service_response.protocol,
                    execution_time=service_response.execution_time,
                    success=service_response.success,
                    error_message=service_response.error_message,
                    classification_result=service_response.classification_result
                )
                
            except Exception as e:
                result = ProtocolResult(
                    task_type=TaskType.OTHER,
                    selected_model="unknown",
                    protocol="unknown",
                    execution_time=0.0,
                    success=False,
                    error_message=str(e)
                )
            
            results.append(result)
            
            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(conversations)} conversations")
        
        self.results = results
        return results
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics about the adaptive_ai service.
        
        Returns:
            Service statistics
        """
        connection_test = self.test_service_connection()
        
        stats = {
            "service_url": self.service_url,
            "service_available": connection_test["health_check"],
            "service_info": connection_test.get("service_info", {}),
            "test_request": connection_test.get("test_request", {})
        }
        
        return stats
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze protocol testing results.
        
        Returns:
            Analysis results dictionary
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Model usage statistics
        model_usage = {}
        protocol_usage = {}
        task_distribution = {}
        execution_times = []
        
        for result in self.results:
            if result.success:
                model_usage[result.selected_model] = model_usage.get(result.selected_model, 0) + 1
                protocol_usage[result.protocol] = protocol_usage.get(result.protocol, 0) + 1
                task_distribution[result.task_type.value] = task_distribution.get(result.task_type.value, 0) + 1
                execution_times.append(result.execution_time)
        
        # Performance metrics
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "model_usage": model_usage,
            "protocol_usage": protocol_usage,
            "task_distribution": task_distribution,
            "avg_execution_time": avg_execution_time,
            "execution_times": execution_times
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.
        
        Returns:
            Results as DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                "task_type": result.task_type.value,
                "selected_model": result.selected_model,
                "protocol": result.protocol,
                "execution_time": result.execution_time,
                "success": result.success,
                "error_message": result.error_message,
                "classification_result": str(result.classification_result)
            })
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """Generate text report of results.
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_results()
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        report = f"""
Adaptive AI Service Protocol Testing Report
==========================================

Service URL: {self.service_url}
Total Tests: {analysis['total_tests']}
Successful Tests: {analysis['successful_tests']}
Success Rate: {analysis['success_rate']:.2%}
Average Execution Time: {analysis['avg_execution_time']:.4f}s

Model Usage:
{'-' * 20}
"""
        
        for model, count in sorted(analysis['model_usage'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis['successful_tests']) * 100
            report += f"{model}: {count} ({percentage:.1f}%)\n"
        
        report += f"\nProtocol Usage:\n{'-' * 20}\n"
        
        for protocol, count in sorted(analysis['protocol_usage'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis['successful_tests']) * 100
            report += f"{protocol}: {count} ({percentage:.1f}%)\n"
        
        report += f"\nTask Distribution:\n{'-' * 20}\n"
        
        for task, count in sorted(analysis['task_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / analysis['successful_tests']) * 100
            report += f"{task}: {count} ({percentage:.1f}%)\n"
        
        return report