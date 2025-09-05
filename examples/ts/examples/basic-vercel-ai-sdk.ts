/**
 * Basic Vercel AI SDK Example
 * 
 * This example demonstrates how to use Adaptive with Vercel AI SDK's
 * modern AI patterns while getting intelligent model routing and cost optimization.
 * 
 * Features demonstrated:
 * - Vercel AI SDK compatibility with Adaptive provider
 * - Both streaming and non-streaming text generation
 * - Intelligent model routing and provider selection
 * - Modern AI patterns (generateText, streamText)
 * - Type-safe responses with Vercel AI SDK types
 * 
 * Key benefits:
 * - Use familiar Vercel AI SDK patterns
 * - Get automatic cost optimization and intelligent routing
 * - Seamless integration with React apps using ai/react
 * - Built-in streaming support for real-time UIs
 * 
 * Set ADAPTIVE_API_KEY environment variable to avoid hardcoding your API key.
 */

import { createAdaptive } from "@adaptive-llm/adaptive-ai-provider";
import { generateText, streamText } from "ai";

// Initialize the Adaptive provider for Vercel AI SDK
// This gives you intelligent routing while using Vercel AI SDK patterns
const adaptive = createAdaptive({
  apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
  baseURL: "https://llmadaptive.com/api/v1",
});

// Non-streaming example using generateText
async function nonStreamingExample() {
  console.log("=== Non-Streaming Vercel AI SDK Example ===");
  console.log("💬 Generating text: 'Explain quantum computing in simple terms'");
  console.log("🧠 Using intelligent routing via Adaptive provider...");
  console.log("⏳ Waiting for complete response...");
  
  try {
    const { text, usage, finishReason } = await generateText({
      model: adaptive(), // Adaptive automatically selects the best model
      prompt: "Explain quantum computing in simple terms",
      maxCompletionTokens: 150, // Limit response length for demo
    });

    if (!text) {
      console.log("❌ No response text received");
      return;
    }

    // Display the response
    console.log();
    console.log("✅ Response received:");
    console.log("📝 Content:", text);
    console.log("🏁 Finish reason:", finishReason);
    
    // Usage information (if available)
    if (usage) {
      console.log("📊 Usage:");
      console.log(`   • Prompt tokens: ${usage.promptTokens}`);
      console.log(`   • Completion tokens: ${usage.completionTokens}`);
      console.log(`   • Total tokens: ${usage.totalTokens}`);
    }
    
    console.log();
    console.log("✨ Non-streaming example completed successfully!");

  } catch (error) {
    console.error("❌ Non-streaming Error:", error);
    throw error;
  }
}

// Streaming example using streamText
async function streamingExample() {
  console.log("=== Streaming Vercel AI SDK Example ===");
  console.log("💬 Streaming text: 'Write a creative short story about AI and humanity'");
  console.log("🧠 Using intelligent routing with streaming...");
  console.log("📡 Response will appear in real-time:");
  console.log();

  try {
    const { textStream, usage, finishReason } = await streamText({
      model: adaptive(), // Adaptive automatically selects the best model
      prompt: "Write a creative short story about AI and humanity",
      maxTokens: 200, // Limit response length for demo
    });

    let fullContent = "";
    let chunkCount = 0;
    
    console.log("🔄 Streaming response:");
    process.stdout.write("📝 ");

    // Stream the text chunks
    for await (const textChunk of textStream) {
      process.stdout.write(textChunk);
      fullContent += textChunk;
      chunkCount++;
    }

    // Wait for final metadata
    const finalUsage = await usage;
    const finalFinishReason = await finishReason;
    
    console.log(); // New line after content
    console.log();
    console.log("✅ Streaming completed!");
    console.log("🏁 Finish reason:", finalFinishReason);
    
    console.log("📊 Streaming statistics:");
    console.log(`   • Text chunks received: ${chunkCount}`);
    console.log(`   • Total content length: ${fullContent.length} characters`);
    
    // Usage information
    if (finalUsage) {
      console.log("📊 Final usage:");
      console.log(`   • Prompt tokens: ${finalUsage.promptTokens}`);
      console.log(`   • Completion tokens: ${finalUsage.completionTokens}`);
      console.log(`   • Total tokens: ${finalUsage.totalTokens}`);
    }

    console.log();
    console.log("✨ Streaming example completed successfully!");

  } catch (error) {
    console.error("❌ Streaming Error:", error);
    throw error;
  }
}

// Advanced example with tool calling
async function toolCallingExample() {
  console.log("=== Tool Calling Example ===");
  console.log("💬 Testing tool calling: 'What's the weather like in San Francisco?'");
  console.log("🔧 Using tools with intelligent model selection...");
  console.log("⏳ Adaptive will choose a model that supports function calling...");

  try {
    const { text, toolCalls, finishReason } = await generateText({
      model: adaptive(), // Adaptive prioritizes models with function calling support
      prompt: "What's the weather like in San Francisco? Please use the weather tool.",
      tools: {
        getWeather: {
          description: 'Get current weather for a location',
          parameters: {
            type: 'object',
            properties: {
              location: {
                type: 'string',
                description: 'The city name to get weather for',
              },
            },
            required: ['location'],
          },
          execute: async ({ location }: { location: string }) => {
            // Simulate weather API call
            return `Weather in ${location}: Sunny, 72°F (22°C), light breeze`;
          },
        },
      },
      maxToolRoundtrips: 2, // Allow the model to call tools
    });

    console.log();
    console.log("✅ Tool calling completed:");
    console.log("📝 Final response:", text);
    console.log("🏁 Finish reason:", finishReason);
    
    if (toolCalls && toolCalls.length > 0) {
      console.log("🔧 Tools called:");
      toolCalls.forEach((call, index) => {
        console.log(`   ${index + 1}. ${call.toolName} with args:`, call.args);
      });
    }

    console.log();
    console.log("✨ Tool calling example completed successfully!");

  } catch (error) {
    console.error("❌ Tool calling Error:", error);
    throw error;
  }
}

async function main() {
  console.log("🚀 Starting Vercel AI SDK example with Adaptive...");
  console.log("📦 Using Adaptive provider with Vercel AI SDK");
  console.log("🔧 Modern AI patterns with intelligent routing");
  console.log();

  try {
    // Run non-streaming example first
    await nonStreamingExample();
    
    console.log();
    console.log("=" .repeat(50));
    console.log();
    
    // Run streaming example second
    await streamingExample();
    
    console.log();
    console.log("=" .repeat(50));
    console.log();
    
    // Run tool calling example
    await toolCallingExample();
    
    console.log();
    console.log("=" .repeat(50));
    console.log();
    console.log("🎉 All Vercel AI SDK examples completed successfully!");
    console.log("💰 You're getting intelligent routing and cost optimization.");
    console.log("🔄 Perfect for React apps with ai/react hooks.");
    console.log("🧠 Adaptive works seamlessly with Vercel AI SDK patterns.");
    console.log();
    console.log("💡 Next steps:");
    console.log("• Use these patterns in your React components");
    console.log("• Try the useChat, useCompletion hooks with Adaptive");
    console.log("• Leverage streaming for better user experience");

  } catch (error) {
    console.error("❌ Example failed:", error);
    process.exit(1);
  }
}

// Run the example
main();