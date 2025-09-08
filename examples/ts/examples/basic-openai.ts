/**
 * Basic OpenAI SDK Example
 *
 * This example demonstrates how to use Adaptive as a drop-in replacement
 * for the OpenAI API. Simply change the baseURL and you get intelligent
 * model routing, cost optimization, and provider failover.
 *
 * Features demonstrated:
 * - Drop-in OpenAI SDK compatibility
 * - Intelligent model routing (empty model string)
 * - Automatic cost optimization
 * - Provider failover and resilience
 *
 * Key benefits:
 * - 60-80% cost reduction through intelligent routing
 * - No code changes required beyond baseURL
 * - Automatic fallback if providers are unavailable
 * - Real-time model selection based on prompt complexity
 *
 * Set ADAPTIVE_API_KEY environment variable to avoid hardcoding your API key.
 */

import OpenAI from "openai";

// Initialize the OpenAI client with Adaptive's endpoint
// This is the only change needed to use Adaptive instead of OpenAI directly
const client = new OpenAI({
  apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
  baseURL: "https://www.llmadaptive.uk/api/v1", // Adaptive's endpoint
});

// Non-streaming example - get complete response at once
async function nonStreamingExample() {
  console.log("=== Non-Streaming Example ===");
  console.log("💬 Sending message: 'Hello!'");
  console.log("🧠 Using intelligent routing (empty model string)...");
  console.log("⏳ Waiting for complete response...");

  try {
    const response = await client.chat.completions.create({
      model: "", // Leave empty for intelligent routing - Adaptive chooses the best model
      messages: [{ role: "user", content: "Hello!" }],
      stream: false, // Explicitly disable streaming
    });

    // Type-safe response handling
    if (!response.choices?.[0]?.message?.content) {
      console.log("❌ No response content received");
      return;
    }

    // Display the response with provider information
    console.log();
    console.log("✅ Response received:");
    console.log("📝 Content:", response.choices[0].message.content);

    // Adaptive adds provider information to the response
    const adaptiveResponse =
      response as OpenAI.Chat.Completions.ChatCompletion & {
        provider?: string;
        model?: string;
      };

    if (adaptiveResponse.provider) {
      console.log("🏢 Provider used:", adaptiveResponse.provider);
    }
    if (adaptiveResponse.model) {
      console.log("🤖 Model used:", adaptiveResponse.model);
    }

    // Usage information
    if (response.usage) {
      console.log("📊 Usage:");
      console.log(`   • Input tokens: ${response.usage.prompt_tokens}`);
      console.log(`   • Output tokens: ${response.usage.completion_tokens}`);
      console.log(`   • Total tokens: ${response.usage.total_tokens}`);
    }

    console.log();
    console.log("✨ Non-streaming example completed successfully!");
  } catch (error) {
    console.error("❌ Non-streaming Error:", error);

    if (error instanceof OpenAI.APIError) {
      console.error("🔧 API Error Details:");
      console.error("   • Status:", error.status);
      console.error("   • Message:", error.message);
      if (error.code) {
        console.error("   • Code:", error.code);
      }
    }

    throw error;
  }
}

// Streaming example - get response as it's generated
async function streamingExample() {
  console.log("=== Streaming Example ===");
  console.log("💬 Sending message: 'Write a short poem about coding'");
  console.log("🧠 Using intelligent routing with streaming...");
  console.log("📡 Response will appear in real-time:");
  console.log();

  try {
    const stream = await client.chat.completions.create({
      model: "", // Leave empty for intelligent routing
      messages: [{ role: "user", content: "Write a short poem about coding" }],
      stream: true, // Enable streaming
    });

    let fullContent = "";
    let tokenCount = 0;

    console.log("🔄 Streaming response:");
    process.stdout.write("📝 ");

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;

      if (content) {
        process.stdout.write(content);
        fullContent += content;
        tokenCount++;
      }

      // Check if streaming is done
      if (chunk.choices[0]?.finish_reason) {
        console.log(); // New line after content
        console.log();
        console.log("✅ Streaming completed!");
        console.log("🏁 Finish reason:", chunk.choices[0].finish_reason);

        // Note: Streaming responses may not include provider info in every chunk
        // The final chunk might have additional metadata
        const finalChunk = chunk as any;
        if (finalChunk.provider) {
          console.log("🏢 Provider used:", finalChunk.provider);
        }
        if (finalChunk.model) {
          console.log("🤖 Model used:", finalChunk.model);
        }

        console.log("📊 Approximate chunks received:", tokenCount);
        console.log(
          "📏 Total content length:",
          fullContent.length,
          "characters",
        );
        break;
      }
    }

    console.log();
    console.log("✨ Streaming example completed successfully!");
  } catch (error) {
    console.error("❌ Streaming Error:", error);

    if (error instanceof OpenAI.APIError) {
      console.error("🔧 API Error Details:");
      console.error("   • Status:", error.status);
      console.error("   • Message:", error.message);
      if (error.code) {
        console.error("   • Code:", error.code);
      }
    }

    throw error;
  }
}

async function main() {
  console.log("🚀 Starting OpenAI SDK example with Adaptive...");
  console.log("📡 Using endpoint:", client.baseURL);
  console.log();

  try {
    // Run non-streaming example first
    await nonStreamingExample();

    console.log();
    console.log("=".repeat(50));
    console.log();

    // Run streaming example second
    await streamingExample();

    console.log();
    console.log("=".repeat(50));
    console.log();
    console.log("🎉 All examples completed successfully!");
    console.log(
      "💰 You're likely saving 60-80% compared to direct OpenAI usage.",
    );
    console.log(
      "🔄 Both streaming and non-streaming work seamlessly with Adaptive.",
    );
  } catch (error) {
    console.error("❌ Example failed:", error);
    process.exit(1);
  }
}

// Run the example
main();
