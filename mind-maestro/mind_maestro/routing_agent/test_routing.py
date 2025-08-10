"""Comprehensive test suite for validating routing agent accuracy and performance."""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .router import ModelRouter
from .models import RoutingConfig, TaskType, ComplexityLevel
from .config import PresetConfigs
from .tools import RoutingToolkit
from .graph import ModelRoutingGraph


@dataclass
class TestCase:
    """Test case for routing validation."""
    prompt: str
    expected_task_type: TaskType
    expected_complexity: ComplexityLevel
    expected_models: List[str]  # Acceptable model choices
    description: str


class RoutingTestSuite:
    """Comprehensive test suite for routing agent validation."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_cases = self._create_test_cases()
        self.router = ModelRouter()
        self.routing_graph = ModelRoutingGraph()
        self.toolkit = RoutingToolkit()
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering different scenarios."""
        return [
            # Math Problems
            TestCase(
                prompt="Solve the integral: ∫(x² + 2x + 1)dx",
                expected_task_type=TaskType.MATH,
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_models=["DeepSeek-R1", "o1-preview", "DeepSeek-V3"],
                description="Calculus integration problem"
            ),
            TestCase(
                prompt="If a train travels 120 miles in 2 hours, what is its average speed?",
                expected_task_type=TaskType.MATH,
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_models=["Phi-4", "o1-mini", "Qwen2.5-32B", "Llama-3.1-8B"],
                description="Simple arithmetic word problem"
            ),
            TestCase(
                prompt="Prove that the sum of angles in any triangle equals 180 degrees using geometric principles",
                expected_task_type=TaskType.MATH,
                expected_complexity=ComplexityLevel.COMPLEX,
                expected_models=["DeepSeek-R1", "o1-preview"],
                description="Complex mathematical proof"
            ),
            
            # Coding Tasks
            TestCase(
                prompt="Write a Python function to implement binary search on a sorted array",
                expected_task_type=TaskType.CODING,
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_models=["Claude-3.5-Sonnet", "o1-preview", "GPT-4o"],
                description="Standard algorithm implementation"
            ),
            TestCase(
                prompt="Debug this Python code: print('Hello World'",
                expected_task_type=TaskType.CODING,
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_models=["Claude-3.5-Sonnet", "Phi-4", "Llama-3.1-8B"],
                description="Simple syntax error fix"
            ),
            TestCase(
                prompt="Design and implement a distributed caching system with consistent hashing, fault tolerance, and automatic scaling capabilities",
                expected_task_type=TaskType.CODING,
                expected_complexity=ComplexityLevel.EXPERT,
                expected_models=["Claude-3.5-Sonnet", "o1-preview", "GPT-4o"],
                description="Complex system design"
            ),
            
            # General Q&A
            TestCase(
                prompt="What is the capital of France?",
                expected_task_type=TaskType.GENERAL_QA,
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_models=["Phi-4", "Gemma-2-9B", "Llama-3.1-8B"],
                description="Simple factual question"
            ),
            TestCase(
                prompt="Explain the causes and consequences of World War II",
                expected_task_type=TaskType.GENERAL_QA,
                expected_complexity=ComplexityLevel.COMPLEX,
                expected_models=["GPT-4o", "Claude-3.5-Sonnet", "Llama-3.1-405B"],
                description="Complex historical explanation"
            ),
            
            # Reasoning Tasks
            TestCase(
                prompt="If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning.",
                expected_task_type=TaskType.REASONING,
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_models=["o1-preview", "DeepSeek-R1", "Claude-3.5-Sonnet"],
                description="Logical reasoning problem"
            ),
            
            # Creative Tasks  
            TestCase(
                prompt="Write a short story about a robot learning to paint",
                expected_task_type=TaskType.CREATIVE,
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_models=["GPT-4o", "Claude-3.5-Sonnet", "Llama-3.1-405B"],
                description="Creative writing task"
            ),
            
            # Long Context Tests
            TestCase(
                prompt="Analyze this 50-page research paper on quantum computing applications in cryptography: " + "Lorem ipsum " * 1000,
                expected_task_type=TaskType.ANALYSIS,
                expected_complexity=ComplexityLevel.EXPERT,
                expected_models=["Gemini-1.5-Pro"],  # Only model with 1M context
                description="Long context analysis task"
            ),
        ]
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite with detailed results."""
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "task_classification_accuracy": 0.0,
            "complexity_accuracy": 0.0,
            "model_selection_accuracy": 0.0,
            "average_response_time": 0.0,
            "detailed_results": [],
            "failed_cases": []
        }
        
        total_time = 0.0
        task_correct = 0
        complexity_correct = 0
        model_correct = 0
        
        print("🧪 Running Comprehensive Routing Test Suite")
        print("=" * 60)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\nTest {i}/{len(self.test_cases)}: {test_case.description}")
            
            try:
                # Measure routing time
                start_time = time.time()
                routing_decision = self.router.route_prompt(test_case.prompt)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                total_time += response_time
                
                # Evaluate results
                analysis = routing_decision.prompt_analysis
                selection = routing_decision.model_selection
                
                task_match = analysis.task_type == test_case.expected_task_type
                complexity_match = analysis.complexity == test_case.expected_complexity
                model_match = selection.selected_model.name in test_case.expected_models
                
                if task_match:
                    task_correct += 1
                if complexity_match:
                    complexity_correct += 1
                if model_match:
                    model_correct += 1
                
                test_passed = task_match and complexity_match and model_match
                if test_passed:
                    results["passed"] += 1
                    status = "✅ PASS"
                else:
                    results["failed"] += 1
                    status = "❌ FAIL"
                    results["failed_cases"].append({
                        "test_case": test_case.description,
                        "expected": {
                            "task_type": test_case.expected_task_type.value,
                            "complexity": test_case.expected_complexity.value,
                            "models": test_case.expected_models
                        },
                        "actual": {
                            "task_type": analysis.task_type.value,
                            "complexity": analysis.complexity.value,
                            "model": selection.selected_model.name
                        }
                    })
                
                print(f"  {status}")
                print(f"  Task: {analysis.task_type.value} (expected: {test_case.expected_task_type.value}) {'✓' if task_match else '✗'}")
                print(f"  Complexity: {analysis.complexity.value} (expected: {test_case.expected_complexity.value}) {'✓' if complexity_match else '✗'}")
                print(f"  Model: {selection.selected_model.name} ({'✓' if model_match else '✗'})")
                print(f"  Response Time: {response_time:.1f}ms")
                
                # Store detailed results
                results["detailed_results"].append({
                    "test_case": test_case.description,
                    "passed": test_passed,
                    "task_correct": task_match,
                    "complexity_correct": complexity_match,
                    "model_correct": model_match,
                    "response_time_ms": response_time,
                    "selected_model": selection.selected_model.name,
                    "confidence": selection.confidence_score
                })
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                results["failed"] += 1
                results["failed_cases"].append({
                    "test_case": test_case.description,
                    "error": str(e)
                })
        
        # Calculate final metrics
        if len(self.test_cases) > 0:
            results["task_classification_accuracy"] = task_correct / len(self.test_cases)
            results["complexity_accuracy"] = complexity_correct / len(self.test_cases) 
            results["model_selection_accuracy"] = model_correct / len(self.test_cases)
            results["average_response_time"] = total_time / len(self.test_cases)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ({results['passed']/results['total_tests']*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/results['total_tests']*100:.1f}%)")
        print(f"\nAccuracy Metrics:")
        print(f"  Task Classification: {results['task_classification_accuracy']*100:.1f}%")
        print(f"  Complexity Assessment: {results['complexity_accuracy']*100:.1f}%") 
        print(f"  Model Selection: {results['model_selection_accuracy']*100:.1f}%")
        print(f"\nPerformance:")
        print(f"  Average Response Time: {results['average_response_time']:.1f}ms")
        
        if results["failed_cases"]:
            print(f"\n❌ Failed Cases: {len(results['failed_cases'])}")
            for case in results["failed_cases"][:3]:  # Show first 3 failures
                print(f"  - {case.get('test_case', 'Unknown')}")
        
        return results
    
    def test_preset_configurations(self) -> Dict[str, Any]:
        """Test different configuration presets."""
        print("\n🔧 Testing Configuration Presets")
        print("=" * 50)
        
        presets = {
            "speed": PresetConfigs.get_speed_optimized(),
            "quality": PresetConfigs.get_quality_optimized(),
            "cost": PresetConfigs.get_cost_optimized(),
            "research": PresetConfigs.get_research_optimized()
        }
        
        test_prompt = "Solve this calculus problem: find the derivative of x³ + 2x² - 5x + 1"
        
        results = {}
        
        for preset_name, config in presets.items():
            print(f"\nTesting {preset_name.upper()} preset...")
            
            router = ModelRouter(config=config)
            
            start_time = time.time()
            decision = router.route_prompt(test_prompt)
            response_time = (time.time() - start_time) * 1000
            
            results[preset_name] = {
                "selected_model": decision.model_selection.selected_model.name,
                "efficiency_score": decision.model_selection.selected_model.efficiency_score,
                "response_time_ms": response_time,
                "reasoning": decision.model_selection.selection_reasoning
            }
            
            print(f"  Model: {results[preset_name]['selected_model']}")
            print(f"  Efficiency: {results[preset_name]['efficiency_score']:.3f}")
            print(f"  Time: {response_time:.1f}ms")
        
        return results
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark routing performance with multiple iterations."""
        print(f"\n⚡ Performance Benchmark ({num_iterations} iterations)")
        print("=" * 50)
        
        test_prompts = [
            "What is 2+2?",
            "Write a Python function to sort a list",
            "Explain quantum computing",
            "Solve: ∫x²dx",
            "What are the causes of climate change?"
        ]
        
        # Test standard router
        router_times = []
        print("Testing ModelRouter...")
        for i in range(num_iterations):
            prompt = test_prompts[i % len(test_prompts)]
            start_time = time.time()
            self.router.route_prompt(prompt)
            router_times.append((time.time() - start_time) * 1000)
        
        # Test LangGraph workflow
        graph_times = []
        print("Testing LangGraph workflow...")
        for i in range(num_iterations):
            prompt = test_prompts[i % len(test_prompts)]
            start_time = time.time()
            self.routing_graph.route_prompt(prompt)
            graph_times.append((time.time() - start_time) * 1000)
        
        results = {
            "iterations": num_iterations,
            "router_performance": {
                "average_ms": sum(router_times) / len(router_times),
                "min_ms": min(router_times),
                "max_ms": max(router_times),
                "throughput_per_second": 1000 / (sum(router_times) / len(router_times))
            },
            "langgraph_performance": {
                "average_ms": sum(graph_times) / len(graph_times),
                "min_ms": min(graph_times),
                "max_ms": max(graph_times), 
                "throughput_per_second": 1000 / (sum(graph_times) / len(graph_times))
            }
        }
        
        print(f"\nModelRouter Results:")
        print(f"  Average: {results['router_performance']['average_ms']:.1f}ms")
        print(f"  Min: {results['router_performance']['min_ms']:.1f}ms")
        print(f"  Max: {results['router_performance']['max_ms']:.1f}ms")
        print(f"  Throughput: {results['router_performance']['throughput_per_second']:.1f} req/sec")
        
        print(f"\nLangGraph Results:")
        print(f"  Average: {results['langgraph_performance']['average_ms']:.1f}ms")
        print(f"  Min: {results['langgraph_performance']['min_ms']:.1f}ms")
        print(f"  Max: {results['langgraph_performance']['max_ms']:.1f}ms")
        print(f"  Throughput: {results['langgraph_performance']['throughput_per_second']:.1f} req/sec")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        all_results = {}
        
        print("🚀 Running Complete Routing Agent Test Suite")
        print("=" * 70)
        
        # Run main routing tests
        all_results["routing_accuracy"] = self.run_comprehensive_test()
        
        # Test configuration presets
        all_results["preset_tests"] = self.test_preset_configurations()
        
        # Performance benchmark
        all_results["performance_benchmark"] = self.benchmark_performance(50)
        
        # Overall assessment
        routing_accuracy = all_results["routing_accuracy"]
        overall_pass_rate = routing_accuracy["passed"] / routing_accuracy["total_tests"]
        
        print(f"\n🎯 OVERALL ASSESSMENT")
        print("=" * 30)
        if overall_pass_rate >= 0.8:
            print("🟢 EXCELLENT - Routing agent performs very well")
        elif overall_pass_rate >= 0.6:
            print("🟡 GOOD - Routing agent performs adequately") 
        else:
            print("🔴 NEEDS IMPROVEMENT - Routing agent requires optimization")
        
        print(f"Overall Pass Rate: {overall_pass_rate*100:.1f}%")
        
        return all_results


def run_quick_test():
    """Run a quick test with a few examples."""
    print("🔬 Quick Routing Test")
    print("=" * 30)
    
    router = ModelRouter()
    
    test_prompts = [
        "What is 15 × 23?",
        "Write Python code to reverse a string",
        "Explain the theory of relativity"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        decision = router.route_prompt(prompt)
        
        print(f"  → Task: {decision.prompt_analysis.task_type.value}")
        print(f"  → Complexity: {decision.prompt_analysis.complexity.value}")
        print(f"  → Selected: {decision.model_selection.selected_model.name}")
        print(f"  → Confidence: {decision.model_selection.confidence_score:.2f}")
        print(f"  → Time: {decision.processing_time_ms:.1f}ms")


if __name__ == "__main__":
    # Run quick test by default
    test_suite = RoutingTestSuite()
    test_suite.run_all_tests()