"""Demo examples showing different use cases of the Supervisor Agent system."""

import asyncio
import os
from pathlib import Path

from supervisor_agent.supervisor.supervisor import SupervisorAgent
from supervisor_agent.utils.config import validate_config


def run_code_examples(supervisor: SupervisorAgent):
    """Demonstrate code-related capabilities."""
    print("\n🔧 CODE AGENT EXAMPLES")
    print("=" * 50)
    
    examples = [
        "Generate a Python function to calculate fibonacci numbers",
        "Debug this code: def add(a, b): return a + c",
        "Create unit tests for a function that sorts a list",
        "Explain what this code does: def quicksort(arr): return sorted(arr)",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example}")
        print("-" * 40)
        response = supervisor.process_request(example)
        print(response)
        print()


def run_data_examples(supervisor: SupervisorAgent):
    """Demonstrate data analysis capabilities."""
    print("\n📊 DATA AGENT EXAMPLES")
    print("=" * 50)
    
    examples = [
        "Calculate statistics for these numbers: [1, 2, 3, 4, 5, 10, 15, 20]",
        "Analyze this data: [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]",
        "Create a bar chart for this data: {'Jan': 100, 'Feb': 120, 'Mar': 90}",
        "Calculate the compound interest for principal=1000, rate=5%, time=3 years",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example}")
        print("-" * 40)
        response = supervisor.process_request(example)
        print(response)
        print()


def run_file_examples(supervisor: SupervisorAgent):
    """Demonstrate file operation capabilities."""
    print("\n📁 FILE AGENT EXAMPLES")
    print("=" * 50)
    
    # Create a temporary directory for demos
    demo_dir = Path("demo_files")
    demo_dir.mkdir(exist_ok=True)
    
    examples = [
        f"Create a directory called {demo_dir}/test_folder",
        f"Write 'Hello World!' to {demo_dir}/hello.txt",
        f"Read the contents of {demo_dir}/hello.txt",
        f"List all files in {demo_dir}",
        f"Get information about {demo_dir}/hello.txt",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example}")
        print("-" * 40)
        response = supervisor.process_request(example)
        print(response)
        print()
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree(demo_dir)
        print(f"\nCleaned up demo directory: {demo_dir}")
    except Exception as e:
        print(f"Cleanup warning: {e}")


def run_multi_agent_examples(supervisor: SupervisorAgent):
    """Demonstrate multi-agent coordination."""
    print("\n🤝 MULTI-AGENT EXAMPLES")
    print("=" * 50)
    
    examples = [
        "Generate Python code to analyze CSV data, then create a visualization",
        "Create a Python script that reads a config file and validates the settings",
        "Write a function to process data, test it, and save the results to a file",
        "Analyze the performance of different sorting algorithms with sample data",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example}")
        print("-" * 40)
        response = supervisor.process_request(example)
        print(response)
        print()


def show_agent_capabilities(supervisor: SupervisorAgent):
    """Display all agent capabilities."""
    print("\n🤖 AGENT CAPABILITIES")
    print("=" * 50)
    
    capabilities = supervisor.get_agent_capabilities()
    health = supervisor.health_check()
    
    for agent, description in capabilities.items():
        status = "✅" if health.get(agent, False) else "❌"
        print(f"{status} {agent.replace('_', ' ').title()}: {description}")
    
    print(f"\nTotal Tools Available: {health['total_tools']}")


def analyze_sample_requests(supervisor: SupervisorAgent):
    """Show which agents can handle different types of requests."""
    print("\n🎯 REQUEST ROUTING ANALYSIS") 
    print("=" * 50)
    
    sample_requests = [
        "Write a Python function",
        "Calculate the mean of a dataset",
        "Read a file from disk",
        "Debug my JavaScript code",
        "Create a scatter plot",
        "List directory contents",
        "Generate unit tests",
        "Analyze CSV data",
        "Copy files to backup folder",
        "Explain this algorithm"
    ]
    
    for request in sample_requests:
        capabilities = supervisor.can_handle_request(request)
        capable_agents = [agent for agent, can_handle in capabilities.items() if can_handle]
        
        print(f"\nRequest: '{request}'")
        print(f"Capable agents: {', '.join(capable_agents) if capable_agents else 'None detected'}")


def main():
    """Run all demo examples."""
    print("🚀 SUPERVISOR AGENT DEMO")
    print("=" * 60)
    
    # Check configuration
    if not validate_config():
        print("❌ Configuration Error: OpenAI API key not found")
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize supervisor
        print("Initializing Supervisor Agent...")
        supervisor = SupervisorAgent()
        print("✅ Supervisor Agent initialized successfully\n")
        
        # Show system status
        show_agent_capabilities(supervisor)
        
        # Run different demo categories
        print("\nChoose demo category:")
        print("1. Code Agent Examples")
        print("2. Data Agent Examples") 
        print("3. File Agent Examples")
        print("4. Multi-Agent Examples")
        print("5. Request Routing Analysis")
        print("6. Run All Examples")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            run_code_examples(supervisor)
        elif choice == "2":
            run_data_examples(supervisor)
        elif choice == "3":
            run_file_examples(supervisor)
        elif choice == "4":
            run_multi_agent_examples(supervisor)
        elif choice == "5":
            analyze_sample_requests(supervisor)
        elif choice == "6":
            show_agent_capabilities(supervisor)
            analyze_sample_requests(supervisor)
            run_code_examples(supervisor)
            run_data_examples(supervisor)
            run_file_examples(supervisor)
            run_multi_agent_examples(supervisor)
        else:
            print("Invalid choice. Running capability analysis...")
            show_agent_capabilities(supervisor)
            analyze_sample_requests(supervisor)
        
        print("\n✨ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    main()