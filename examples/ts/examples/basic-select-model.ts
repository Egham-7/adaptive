/**
 * Basic Select Model Example
 *
 * This example demonstrates how to use Adaptive's intelligent model selection
 * without doing inference. Perfect for testing routing decisions, cost planning,
 * and integrating with your own provider accounts.
 *
 * Features demonstrated:
 * - Basic model selection from multiple providers
 * - Cost vs performance optimization with cost_bias parameter
 * - Function calling model selection
 * - Provider-only selection (let Adaptive choose best model)
 * - Custom model specifications for local/enterprise models
 *
 * Use cases:
 * - Test routing decisions before implementing
 * - Cost planning and budget optimization
 * - Integration with your own provider accounts
 * - On-premise/local model routing
 * - A/B testing different routing strategies
 */

// Type definitions for the Select Model API response
interface SelectModelResponse {
  provider: string;
  model: string;
  alternatives?: Array<{
    provider: string;
    model: string;
  }>;
}

async function main() {
  const apiKey = process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key";
  const baseURL = "https://www.llmadaptive.uk/api/v1";

  try {
    // Example 1: Simple model selection with known providers
    // This demonstrates the basic functionality of the select-model endpoint.
    // We provide a list of available models and a prompt, and Adaptive chooses
    // the best model based on prompt complexity, cost, and performance.
    console.log("=== Example 1: Basic Model Selection ===");
    console.log(
      "Selecting optimal model from OpenAI, Anthropic, and Google providers...",
    );

    const basicResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { provider: "openai" },
          { provider: "anthropic" },
          { provider: "google" },
        ],
        prompt: "Hello, how are you today?",
      }),
    });

    if (!basicResponse.ok) {
      throw new Error(
        `HTTP ${basicResponse.status}: ${await basicResponse.text()}`,
      );
    }

    const basicResult = (await basicResponse.json()) as SelectModelResponse;
    console.log(`Selected: ${basicResult.provider}/${basicResult.model}`);
    if (basicResult.alternatives) {
      console.log(`Alternatives: ${JSON.stringify(basicResult.alternatives)}`);
    }
    console.log();

    // Example 2: Cost-optimized selection
    // The cost_bias parameter controls the balance between cost and performance.
    // A value of 0.9 heavily prioritizes cost savings, often selecting cheaper models
    // even for moderately complex tasks. This is perfect for budget-conscious applications.
    console.log("=== Example 2: Cost-Optimized Selection ===");
    console.log("Using cost_bias=0.1 to prioritize cost savings...");

    const costOptimizedResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { provider: "openai", model_name: "gpt-4o-mini" },
          { provider: "openai", model_name: "gpt-4o" },
          { provider: "anthropic", model_name: "claude-3-5-sonnet-20241022" },
        ],
        prompt: "Write a simple Python function to calculate fibonacci numbers",
        cost_bias: 0.1, // Prioritize cost savings
      }),
    });

    if (!costOptimizedResponse.ok) {
      throw new Error(
        `HTTP ${costOptimizedResponse.status}: ${await costOptimizedResponse.text()}`,
      );
    }

    const costOptimizedResult =
      (await costOptimizedResponse.json()) as SelectModelResponse;
    console.log(
      `Cost-optimized: ${costOptimizedResult.provider}/${costOptimizedResult.model}`,
    );
    console.log();

    // Example 3: Performance-focused selection
    // Setting cost_bias=0.1 prioritizes performance and quality over cost.
    // This is ideal for complex tasks requiring higher reasoning capabilities,
    // such as detailed analysis, code generation, or creative writing.
    console.log("=== Example 3: Performance-Focused Selection ===");
    console.log(
      "Using cost_bias=0.9 to prioritize performance for complex analysis...",
    );

    const performanceResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { model_name: "gpt-4o-mini" },
          { model_name: "gpt-4o" },
          { model_name: "claude-3-5-sonnet-20241022" },
        ],
        prompt:
          "Analyze this complex dataset and provide detailed insights on market trends, customer behavior patterns, and predictive analytics recommendations",
        cost_bias: 0.9, // Prioritize performance
      }),
    });

    if (!performanceResponse.ok) {
      throw new Error(
        `HTTP ${performanceResponse.status}: ${await performanceResponse.text()}`,
      );
    }

    const performanceResult =
      (await performanceResponse.json()) as SelectModelResponse;
    console.log(
      `Performance-focused: ${performanceResult.provider}/${performanceResult.model}`,
    );
    console.log();

    // Example 4: Function calling scenario
    // When tools are provided, Adaptive automatically prioritizes models that support
    // function calling. This ensures that your application can reliably use tools
    // and structured outputs, even if some of your available models don't support them.
    console.log("=== Example 4: Function Calling Selection ===");
    console.log(
      "Including tool definitions to demonstrate function calling prioritization...",
    );

    const functionCallingResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { provider: "openai", model_name: "gpt-4o-mini" },
          { provider: "anthropic", model_name: "claude-3-haiku-20240307" },
          { provider: "google", model_name: "gemini-1.5-flash" },
        ],
        prompt: "What's the current weather in San Francisco?",
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get current weather for a location",
              parameters: {
                type: "object",
                properties: {
                  location: {
                    type: "string",
                    description: "The city name to get weather for",
                  },
                },
                required: ["location"],
              },
            },
          },
        ],
      }),
    });

    if (!functionCallingResponse.ok) {
      throw new Error(
        `HTTP ${functionCallingResponse.status}: ${await functionCallingResponse.text()}`,
      );
    }

    const functionCallingResult =
      (await functionCallingResponse.json()) as SelectModelResponse;
    console.log(
      `Function calling: ${functionCallingResult.provider}/${functionCallingResult.model}`,
    );
    console.log("(Selected model supports function calling)");
    console.log();

    // Example 5: Provider-only selection (let Adaptive choose the best model)
    // Instead of specifying exact models, you can just specify providers and let
    // Adaptive choose the best model from each provider. This is useful when you
    // have access to multiple providers but want optimal model selection within each.
    console.log("=== Example 5: Provider-Only Selection ===");
    console.log("Letting Adaptive choose the best model from each provider...");

    const providerOnlyResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { provider: "openai" }, // Let Adaptive choose the best OpenAI model
          { provider: "anthropic" }, // Let Adaptive choose the best Anthropic model
          { provider: "google" }, // Let Adaptive choose the best Google model
        ],
        prompt: "Explain quantum computing in simple terms",
      }),
    });

    if (!providerOnlyResponse.ok) {
      throw new Error(
        `HTTP ${providerOnlyResponse.status}: ${await providerOnlyResponse.text()}`,
      );
    }

    const providerOnlyResult =
      (await providerOnlyResponse.json()) as SelectModelResponse;
    console.log(
      `Provider-only: ${providerOnlyResult.provider}/${providerOnlyResult.model}`,
    );
    console.log("(Adaptive chose the best model from available providers)");
    console.log();

    // Example 6: Custom model specification
    // This example shows how to mix known models (like GPT-4o-mini) with custom
    // models (like local or fine-tuned models). For custom models, you provide
    // full specifications including cost, context size, and capabilities.
    // This is perfect for on-premise deployments or enterprise custom models.
    console.log("=== Example 6: Custom Model Specification ===");
    console.log("Comparing known cloud models with custom local model...");

    const customModelResponse = await fetch(`${baseURL}/select-model`, {
      method: "POST",
      headers: {
        "X-Stainless-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        models: [
          { provider: "openai", model_name: "gpt-4o-mini" }, // Known model
          {
            // Custom model with full specification
            provider: "local",
            model_name: "my-custom-llama-fine-tune",
            cost_per_1m_input_tokens: 0.0, // Free since it's local
            cost_per_1m_output_tokens: 0.0,
            max_context_tokens: 4096,
            supports_function_calling: false,
            complexity: "medium",
            task_type: "Text Generation",
          },
        ],
        prompt: "Hello, how are you?",
        cost_bias: 0.1, // Should prefer the free local model
      }),
    });

    if (!customModelResponse.ok) {
      throw new Error(
        `HTTP ${customModelResponse.status}: ${await customModelResponse.text()}`,
      );
    }

    const customModelResult =
      (await customModelResponse.json()) as SelectModelResponse;
    console.log(
      `Custom model: ${customModelResult.provider}/${customModelResult.model}`,
    );
    console.log("(Comparison between known and custom models)");
    console.log();

    // Summary
    console.log("=== Summary ===");
    console.log("✅ All examples completed successfully!");
    console.log();
    console.log("Key takeaways:");
    console.log(
      "• Use cost_bias to balance cost vs performance (0.0 = cheapest, 1.0 = performance)",
    );
    console.log(
      "• Include tool definitions to prioritize function calling models",
    );
    console.log(
      "• Specify providers only to let Adaptive choose the best model",
    );
    console.log("• Mix known and custom models for hybrid deployments");
    console.log(
      "• Fast and cheap - no inference, just intelligent routing decisions",
    );
    console.log();
    console.log("💡 Next steps:");
    console.log("• Use these routing decisions in your own provider clients");
    console.log(
      "• Integrate with your existing OpenAI/Anthropic/etc. accounts",
    );
    console.log(
      "• Test different cost_bias values for your specific use cases",
    );
  } catch (error) {
    console.error("❌ Error:", error);
    process.exit(1);
  }
}

main();
