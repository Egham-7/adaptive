"""Demo script showing how to use the intelligent LLM routing agent."""

from typing import List

from . import (
    ModelRouter,
    RoutingToolkit, 
    ModelRoutingGraph,
    PresetConfigs,
    create_config_from_preset
)


def demo_basic_routing():
    """Demonstrate basic routing functionality."""
    print("🎯 Basic Routing Demo")
    print("=" * 40)
    
    # Initialize router
    router = ModelRouter()
    
    # Test different types of prompts
    test_prompts = [
        "What is the integral of x² + 3x + 2?",
        "Write a Python function to implement quicksort",
        "What is the capital of France?", 
        "Explain quantum entanglement in simple terms",
        "Debug this code: def hello() print('Hello')",
        "Write a creative story about a time-traveling cat"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt}")
        
        # Get routing recommendation
        recommendation = router.get_model_recommendation(prompt)
        
        print(f"✅ Recommended Model: {recommendation['recommended_model']}")
        print(f"📋 Task Type: {recommendation['task_type']}")
        print(f"📊 Complexity: {recommendation['complexity']}")
        print(f"🎯 Confidence: {recommendation['confidence']:.2f}")
        print(f"⚡ Processing Time: {recommendation['processing_time_ms']:.1f}ms")
        
        if recommendation.get('reasoning'):
            print(f"💡 Reasoning: {recommendation['reasoning']}")


def demo_configuration_presets():
    """Demonstrate different configuration presets."""
    print("\n🔧 Configuration Presets Demo") 
    print("=" * 40)
    
    # Test prompt
    test_prompt = "Solve this complex calculus problem: find the limit of (sin(x)/x) as x approaches 0"
    
    presets = ["balanced", "speed", "quality", "cost", "research"]
    
    for preset_name in presets:
        print(f"\n--- {preset_name.upper()} Configuration ---")
        
        # Create router with preset
        config = create_config_from_preset(preset_name)
        router = ModelRouter(config=config)
        
        # Get recommendation
        result = router.get_model_recommendation(test_prompt, return_reasoning=False)
        
        print(f"Selected Model: {result['recommended_model']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {result['processing_time_ms']:.1f}ms")


def demo_langgraph_workflow():
    """Demonstrate LangGraph workflow routing."""
    print("\n🔄 LangGraph Workflow Demo")
    print("=" * 40)
    
    # Initialize LangGraph routing
    routing_graph = ModelRoutingGraph()
    
    test_prompt = "Design a distributed system for processing millions of real-time transactions with fault tolerance and automatic scaling"
    
    print(f"Prompt: {test_prompt[:100]}...")
    
    # Route using LangGraph
    decision = routing_graph.route_prompt(test_prompt)
    
    print(f"\n📊 Routing Decision:")
    print(f"Selected Model: {decision.model_selection.selected_model.name}")
    print(f"Company: {decision.model_selection.selected_model.company}")
    print(f"Parameters: {decision.model_selection.selected_model.parameter_count}B")
    print(f"Task Type: {decision.prompt_analysis.task_type.value}")
    print(f"Complexity: {decision.prompt_analysis.complexity.value}")
    print(f"Context Required: {decision.prompt_analysis.context_length:,} tokens")
    print(f"Reasoning Steps: {decision.prompt_analysis.reasoning_steps_required}")
    print(f"Processing Time: {decision.processing_time_ms:.1f}ms")
    print(f"Routing Version: {decision.routing_version}")
    
    print(f"\n💭 Selection Reasoning:")
    print(f"{decision.model_selection.selection_reasoning}")


def demo_routing_toolkit():
    """Demonstrate the routing toolkit with various tools."""
    print("\n🛠️ Routing Toolkit Demo")
    print("=" * 40)
    
    # Initialize toolkit
    toolkit = RoutingToolkit()
    
    # Demo individual tools
    test_prompts = [
        "Calculate the derivative of e^(x²)",
        "Implement a REST API in Python using FastAPI",
        "What are the main theories of consciousness?"
    ]
    
    print("\n--- Model Recommendation Tool ---")
    for prompt in test_prompts[:1]:  # Just first one for demo
        result = toolkit.model_recommendation.invoke({
            "prompt": prompt,
            "include_reasoning": True,
            "include_alternatives": True
        })
        
        if result["success"]:
            print(f"Prompt: {prompt}")
            print(f"Recommended: {result['recommended_model']}")
            print(f"Alternatives: {', '.join([alt['name'] for alt in result.get('alternatives', [])][:3])}")
    
    print("\n--- Batch Routing Tool ---")
    batch_result = toolkit.batch_routing.invoke({
        "prompts": test_prompts,
        "max_concurrent": 3
    })
    
    if batch_result["success"]:
        print("Batch routing completed!")
        stats = batch_result["batch_stats"]
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Average time: {stats['average_time_per_prompt_ms']:.1f}ms")
        print(f"Most selected: {stats['most_selected_model']}")
    
    print("\n--- Quick Routing ---")
    for prompt in test_prompts:
        model = toolkit.quick_route(prompt)
        print(f"'{prompt[:50]}...' → {model}")


def demo_performance_comparison():
    """Compare performance of different routing methods."""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 40)
    
    import time
    
    # Initialize different routers
    standard_router = ModelRouter()
    speed_router = ModelRouter(config=PresetConfigs.get_speed_optimized())
    langgraph_router = ModelRoutingGraph()
    
    test_prompt = "Explain machine learning algorithms for natural language processing"
    
    # Test standard router
    start = time.time()
    result1 = standard_router.route_prompt(test_prompt)
    standard_time = (time.time() - start) * 1000
    
    # Test speed-optimized router
    start = time.time()
    result2 = speed_router.route_prompt(test_prompt)
    speed_time = (time.time() - start) * 1000
    
    # Test LangGraph router
    start = time.time()  
    result3 = langgraph_router.route_prompt(test_prompt)
    langgraph_time = (time.time() - start) * 1000
    
    print(f"Standard Router: {result1.model_selection.selected_model.name} ({standard_time:.1f}ms)")
    print(f"Speed Router: {result2.model_selection.selected_model.name} ({speed_time:.1f}ms)")
    print(f"LangGraph Router: {result3.model_selection.selected_model.name} ({langgraph_time:.1f}ms)")
    
    print(f"\nSpeed improvement: {((standard_time - speed_time) / standard_time * 100):.1f}%")


def main():
    """Run all demos."""
    print("🚀 Intelligent LLM Routing Agent Demo")
    print("=" * 50)
    
    # Run all demo functions
    demo_basic_routing()
    demo_configuration_presets()
    demo_langgraph_workflow()
    demo_routing_toolkit()
    demo_performance_comparison()
    
    print(f"\n✅ Demo completed! The routing agent successfully:")
    print(f"   • Analyzed prompts for task type and complexity")
    print(f"   • Selected optimal models based on efficiency scoring")
    print(f"   • Used benchmark data from model_data.py")
    print(f"   • Demonstrated different configuration presets")
    print(f"   • Showed LangGraph workflow capabilities")
    print(f"   • Provided comprehensive tooling and utilities")


if __name__ == "__main__":
    main()