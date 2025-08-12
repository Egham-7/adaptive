"""Test script for the new AI-powered routing system."""

import os
from mind_maestro.routing_agent.router import ModelRouter
from mind_maestro.routing_agent.models import RoutingConfig

def test_ai_routing():
    """Test the AI routing system with sample prompts."""
    
    # Set up OpenAI API key (you'll need to provide this)
    # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
    
    print("🚀 Testing AI-Powered Routing Agent\n")
    print("=" * 50)
    
    # Initialize router with AI agent
    config = RoutingConfig(routing_model="gpt-4o-mini")
    router = ModelRouter(config=config)
    
    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Write a Python function to calculate the factorial of a number",
        "Explain quantum entanglement in simple terms",
        "Create a comprehensive business plan for a SaaS startup",
        "Solve this integral: ∫(2x + 3)dx",
        "Write a creative short story about time travel"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt[:50]}...")
        print("-" * 50)
        
        try:
            # Get routing decision
            decision = router.route_prompt(prompt)
            
            # Get simple recommendation
            recommendation = router.get_model_recommendation(prompt, return_reasoning=True)
            
            print(f"✅ Recommended Model: {recommendation['recommended_model']}")
            print(f"🎯 Confidence: {recommendation['confidence']:.1%}")
            print(f"⏱️ Processing Time: {recommendation['processing_time_ms']:.1f}ms")
            
            if 'analysis_details' in recommendation:
                print(f"\n📊 AI Analysis:")
                print(recommendation['analysis_details'])
            
            if 'reasoning' in recommendation:
                print(f"\n🤔 Selection Reasoning:")
                print(recommendation['reasoning'])
            
            print(f"\n📈 Model Details:")
            details = recommendation.get('model_details', {})
            print(f"  Company: {details.get('company', 'N/A')}")
            print(f"  Parameters: {details.get('parameters', 'N/A')}")
            print(f"  Context Window: {details.get('context_window', 'N/A'):,} tokens")
            
            if recommendation.get('alternatives'):
                print(f"\n🔄 Alternative Models: {', '.join(recommendation['alternatives'][:3])}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n" + "=" * 50)
    
    print("\n✨ AI Routing Test Complete!")
    print("\nKey improvements:")
    print("• Removed complex regex pattern matching")
    print("• Uses LLM for intelligent prompt analysis")
    print("• Model makes routing decisions with reasoning")
    print("• Simplified and more flexible architecture")

if __name__ == "__main__":
    test_ai_routing()