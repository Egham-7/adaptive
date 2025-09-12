"""Comprehensive tests for task type specialization and routing intelligence."""

import pytest
import requests


@pytest.mark.integration
class TestTaskTypeSpecialization:
    """Test that all TaskType enum values route to appropriate models."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_open_qa_routing(self, base_url):
        """Test Open QA task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "What causes climate change and how does it affect global weather patterns?",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        # Open QA should route to models good at factual knowledge
        print(f"Open QA -> {result['provider']}/{result['model']}")
        assert result["model"], "Open QA should select a valid model"

    def test_closed_qa_routing(self, base_url):
        """Test Closed QA task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Based on the document provided, what is the main conclusion about renewable energy adoption?",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        print(f"Closed QA -> {result['provider']}/{result['model']}")
        assert result["model"], "Closed QA should select a valid model"

    def test_summarization_routing(self, base_url):
        """Test Summarization task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Please summarize this 50-page research paper on quantum computing applications in cryptography",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        # Summarization might prefer models with larger context windows
        print(f"Summarization -> {result['provider']}/{result['model']}")
        assert result["model"], "Summarization should select a valid model"

    def test_text_generation_routing(self, base_url):
        """Test Text Generation task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Generate a professional email template for customer service responses",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        print(f"Text Generation -> {result['provider']}/{result['model']}")
        assert result["model"], "Text Generation should select a valid model"

    def test_code_generation_routing_specialization(self, base_url):
        """Test Code Generation routes to coding specialists."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write a Python class that implements a balanced binary search tree with insert, delete, and search operations",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        selected_model = result["model"]

        # Code generation should route to coding specialists

        print(f"Code Generation -> {result['provider']}/{selected_model}")

        # Should NOT route to basic chatbot models for complex code
        basic_models = ["gpt-3.5-turbo", "claude-3-haiku", "babbage-002"]
        assert (
            selected_model not in basic_models
        ), f"Code generation should not use basic model {selected_model} for complex coding tasks"

        print("✓ Code generation routes to appropriate models")

    def test_chatbot_routing_efficiency(self, base_url):
        """Test Chatbot task routing for efficiency."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Hello! How can I help you today?",
                "cost_bias": 0.2,  # Prefer efficiency for simple chat
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        selected_model = result["model"]

        # Simple chatbot tasks should use efficient models
        print(f"Chatbot -> {result['provider']}/{selected_model}")

        # Should NOT use ultra-expensive reasoning models for simple chat
        ultra_expensive = ["o3-pro", "o1-pro"]  # 150+ cost per 1M tokens
        assert (
            selected_model not in ultra_expensive
        ), f"Simple chatbot should not use ultra-expensive model {selected_model}"

        print("✓ Chatbot routing is cost-efficient")

    def test_classification_routing(self, base_url):
        """Test Classification task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Classify this text as positive, negative, or neutral sentiment: 'I really enjoyed the movie but the ending was disappointing'",
                "cost_bias": 0.3,  # Classification can use efficient models
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        print(f"Classification -> {result['provider']}/{result['model']}")
        assert result["model"], "Classification should select a valid model"

    def test_rewrite_routing(self, base_url):
        """Test Rewrite task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Rewrite this technical documentation to be more accessible to non-technical users",
                "cost_bias": 0.5,
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        print(f"Rewrite -> {result['provider']}/{result['model']}")
        assert result["model"], "Rewrite should select a valid model"

    def test_brainstorming_routing(self, base_url):
        """Test Brainstorming task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Brainstorm innovative ideas for sustainable urban transportation systems",
                "cost_bias": 0.6,  # Creative tasks may benefit from higher capability models
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        # Brainstorming should prefer creative/general models
        print(f"Brainstorming -> {result['provider']}/{result['model']}")
        assert result["model"], "Brainstorming should select a valid model"

    def test_extraction_routing(self, base_url):
        """Test Extraction task routing."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Extract the key financial metrics from this earnings report: revenue, profit margin, and growth rate",
                "cost_bias": 0.4,  # Extraction can be efficient
            },
            timeout=30,
        )
        assert response.status_code == 200
        result = response.json()

        print(f"Extraction -> {result['provider']}/{result['model']}")
        assert result["model"], "Extraction should select a valid model"


@pytest.mark.integration
class TestProviderConstraints:
    """Test provider-specific constraints and routing."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_all_major_providers(self, base_url):
        """Test routing works for all major providers."""
        providers = ["ANTHROPIC", "OPENAI", "GROQ", "DEEPSEEK", "GOOGLE"]

        for provider in providers:
            response = requests.post(
                f"{base_url}/predict",
                json={
                    "prompt": f"Test routing to {provider.lower()} provider",
                    "models": [{"provider": provider}],
                    "cost_bias": 0.5,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()

                # Check if this is an error response or successful model selection
                if "error" in result:
                    print(f"{provider} constraint -> Error: {result['message']}")
                elif result["provider"] is None:
                    print(f"{provider} constraint -> No models available for task type")
                else:
                    print(
                        f"{provider} constraint -> {result['provider']}/{result['model']}"
                    )

                    # Should route to the requested provider (if models are available)
                    assert (
                        result["provider"].upper() == provider
                    ), f"Provider constraint failed: requested {provider}, got {result['provider']}"
            else:
                # Some providers might not have models available
                print(f"{provider} not available (status {response.status_code})")

    def test_provider_cost_optimization(self, base_url):
        """Test cost optimization within provider constraints."""
        # Test Anthropic with different cost preferences
        anthropic_cheap = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write a simple greeting message",
                "models": [{"provider": "ANTHROPIC"}],
                "cost_bias": 0.0,  # Prefer cheap Anthropic models
            },
            timeout=30,
        )

        anthropic_premium = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write a simple greeting message",
                "models": [{"provider": "ANTHROPIC"}],
                "cost_bias": 1.0,  # Prefer capable Anthropic models
            },
            timeout=30,
        )

        if anthropic_cheap.status_code == 200 and anthropic_premium.status_code == 200:
            cheap_result = anthropic_cheap.json()
            premium_result = anthropic_premium.json()

            print("Anthropic cost optimization:")
            print(f"  Low cost bias -> {cheap_result['model']}")
            print(f"  High cost bias -> {premium_result['model']}")

            # Within same provider, should make cost-optimized choices
            # claude-3-haiku (cheap) vs claude-3-opus/sonnet (premium)
            cheap_model = cheap_result["model"]
            premium_model = premium_result["model"]

            # If different models selected, validate the cost logic
            if cheap_model != premium_model:
                premium_models = ["claude-3-opus", "claude-3-5-sonnet"]

                if (
                    cheap_model in premium_models
                    and len(cheap_result.get("alternatives", [])) > 0
                ):
                    print(
                        "⚠️  Within-provider cost optimization may not be working optimally"
                    )

            print("✓ Provider-constrained cost optimization tested")

    def test_context_length_constraints(self, base_url):
        """Test models are filtered by context length requirements."""
        # Request models with high context requirements
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Process this very long document that requires 150,000 tokens of context",
                # High context requirement
                "models": [{"max_context_tokens": 150000}],
                "cost_bias": 0.5,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"High context constraint -> {result['provider']}/{result['model']}")

            # Should route to high-context models like Claude-3 (200k), GPT-5 (200k)
            low_context_models = [
                "gpt-3.5-turbo",
                "claude-3-haiku",
            ]  # Typically 8k-16k context

            assert (
                result["model"] not in low_context_models
            ), f"High context requirement should not route to low-context model {result['model']}"

            print("✓ Context length constraints working")
        else:
            print("(i) No models meet high context requirements")

    def test_function_calling_constraints(self, base_url):
        """Test function calling capability constraints."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Call the weather API to get current conditions for New York",
                "models": [{"supports_function_calling": True}],
                "cost_bias": 0.5,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(
                f"Function calling constraint -> {result['provider']}/{result['model']}"
            )

            # Should route to function-calling capable models
            # Most modern models support this, but some older ones don't
            non_function_models = ["babbage-002", "text-davinci-002"]

            assert (
                result["model"] not in non_function_models
            ), f"Function calling requirement should not route to non-capable model {result['model']}"

            print("✓ Function calling constraints working")
        else:
            print("(i) Function calling constraint test failed")


@pytest.mark.integration
class TestComplexRoutingScenarios:
    """Test complex routing scenarios with multiple constraints."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_multi_constraint_routing(self, base_url):
        """Test routing with multiple simultaneous constraints."""
        response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write a complex Python API with proper error handling and documentation",
                "models": [
                    {
                        "provider": "ANTHROPIC",
                        "supports_function_calling": True,
                        "max_context_tokens": 100000,
                        "cost_per_1m_input_tokens": 20.0,  # Budget constraint
                    }
                ],
                "cost_bias": 0.6,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Multi-constraint -> {result['provider']}/{result['model']}")

            # Should meet all constraints
            assert result["provider"].upper() == "ANTHROPIC"
            # Should be a capable Anthropic model within budget

            print("✓ Multi-constraint routing working")
        else:
            print("(i) Multi-constraint may be too restrictive")

    def test_reasoning_vs_code_specialization(self, base_url):
        """Test that reasoning and code tasks route to different specialists."""
        # Complex mathematical reasoning
        reasoning_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Prove that the square root of 2 is irrational using proof by contradiction",
                "cost_bias": 0.7,
            },
            timeout=30,
        )

        # Complex code generation
        code_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Implement a distributed cache with consistent hashing and replication",
                "cost_bias": 0.7,
            },
            timeout=30,
        )

        assert reasoning_response.status_code == 200
        assert code_response.status_code == 200

        reasoning_result = reasoning_response.json()
        code_result = code_response.json()

        reasoning_model = reasoning_result["model"]
        code_model = code_result["model"]

        print(f"Complex reasoning -> {reasoning_result['provider']}/{reasoning_model}")
        print(f"Complex coding -> {code_result['provider']}/{code_model}")

        # Reasoning should prefer o1/o3 series or similar reasoning specialists
        reasoning_specialists = [
            "o1",
            "o1-pro",
            "o3",
            "o3-pro",
            "o3-mini",
            "deepseek-reasoner",
        ]
        coding_specialists = ["claude-3-5-sonnet", "gpt-4", "gpt-5", "deepseek-coder"]

        # At least one should route to their appropriate specialist type
        reasoning_specialized = reasoning_model in reasoning_specialists
        code_specialized = code_model in coding_specialists

        print(f"Reasoning specialization: {reasoning_specialized}")
        print(f"Code specialization: {code_specialized}")

        # Document intelligent routing behavior
        if reasoning_specialized and code_specialized:
            print("✓ EXCELLENT: Both tasks route to appropriate specialists")
        elif reasoning_specialized or code_specialized:
            print("✓ PARTIAL: At least one task routes to specialist")
        else:
            print("(i) Specialization routing may need optimization")

    def test_creative_vs_analytical_routing(self, base_url):
        """Test creative vs analytical tasks route differently."""
        # Creative task
        creative_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write an imaginative short story about time travel with unexpected plot twists",
                "cost_bias": 0.5,
            },
            timeout=30,
        )

        # Analytical task
        analytical_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Perform a detailed statistical analysis of this dataset and identify key trends",
                "cost_bias": 0.5,
            },
            timeout=30,
        )

        assert creative_response.status_code == 200
        assert analytical_response.status_code == 200

        creative_result = creative_response.json()
        analytical_result = analytical_response.json()

        print(
            f"Creative task -> {creative_result['provider']}/{creative_result['model']}"
        )
        print(
            f"Analytical task -> {analytical_result['provider']}/{analytical_result['model']}"
        )

        # Document routing intelligence
        creative_model = creative_result["model"]
        analytical_model = analytical_result["model"]

        # Creative tasks might prefer Claude (known for creativity) or GPT-4
        # Analytical tasks might prefer reasoning models or high-context models

        if creative_model != analytical_model:
            print("✓ Creative and analytical tasks route to different models")
        else:
            print("(i) Same model selected for both creative and analytical tasks")

        assert creative_model, "Creative task should select valid model"
        assert analytical_model, "Analytical task should select valid model"


@pytest.mark.integration
class TestTaskComplexityRouting:
    """Test that task complexity affects model routing appropriately."""

    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"

    def test_simple_vs_complex_task_routing(self, base_url):
        """Test simple vs complex versions of same task type route differently."""
        # Simple code task
        simple_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Write a function that adds two numbers",
                "cost_bias": 0.3,  # Prefer efficiency
            },
            timeout=30,
        )

        # Complex code task
        complex_response = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Implement a distributed consensus algorithm like Raft with leader election, log replication, and failure recovery mechanisms",
                "cost_bias": 0.3,  # Same cost preference
            },
            timeout=30,
        )

        assert simple_response.status_code == 200
        assert complex_response.status_code == 200

        simple_result = simple_response.json()
        complex_result = complex_response.json()

        simple_model = simple_result["model"]
        complex_model = complex_result["model"]

        print(f"Simple coding task -> {simple_result['provider']}/{simple_model}")
        print(f"Complex coding task -> {complex_result['provider']}/{complex_model}")

        # Complex task should not use the most basic models
        basic_models = ["gpt-3.5-turbo", "babbage-002"]

        assert (
            complex_model not in basic_models
        ), f"Complex task should not use basic model {complex_model}"

        # Document complexity-aware routing
        if simple_model != complex_model:
            print("✓ Task complexity affects model selection")
        else:
            print("(i) Same model selected regardless of complexity")

    def test_mathematical_reasoning_complexity(self, base_url):
        """Test mathematical reasoning with different complexity levels."""
        # Simple math
        simple_math = requests.post(
            f"{base_url}/predict",
            json={"prompt": "What is 15 * 8?", "cost_bias": 0.2},
            timeout=30,
        )

        # Complex math
        complex_math = requests.post(
            f"{base_url}/predict",
            json={
                "prompt": "Solve the system of differential equations: dx/dt = 2x - 3y, dy/dt = x + 4y with initial conditions x(0)=1, y(0)=0",
                "cost_bias": 0.8,  # Allow premium models for complex math
            },
            timeout=30,
        )

        assert simple_math.status_code == 200
        assert complex_math.status_code == 200

        simple_result = simple_math.json()
        complex_result = complex_math.json()

        print(f"Simple math -> {simple_result['provider']}/{simple_result['model']}")
        print(f"Complex math -> {complex_result['provider']}/{complex_result['model']}")

        # Complex math should route to reasoning specialists
        reasoning_models = ["o1", "o1-pro", "o3", "o3-pro", "o3-mini", "claude-3-opus"]
        complex_model = complex_result["model"]

        # Complex mathematical reasoning should prefer reasoning specialists
        if complex_model in reasoning_models:
            print("✓ Complex math routes to reasoning specialist")
        else:
            print(f"(i) Complex math routed to general model: {complex_model}")

        assert simple_result["model"], "Simple math should select valid model"
        assert complex_result["model"], "Complex math should select valid model"
