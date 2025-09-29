/**
 * Basic Anthropic SDK Example
 *
 * This example demonstrates how to use Adaptive with Anthropic's Messages API
 * format while getting intelligent model routing and cost optimization.
 *
 * Features demonstrated:
 * - Anthropic Messages API compatibility
 * - Both streaming and non-streaming responses
 * - Intelligent model routing (empty model string)
 * - Tool use (function calling) capabilities
 * - Automatic cost optimization and provider selection
 * - Proper TypeScript typing for Claude responses
 *
 * Key benefits:
 * - Keep using familiar Anthropic SDK patterns
 * - Get cost savings through intelligent routing
 * - Automatic fallback between providers
 * - Native Claude response format preserved
 *
 * Set ADAPTIVE_API_KEY environment variable to avoid hardcoding your API key.
 */

import Anthropic from "@anthropic-ai/sdk";

// Initialize the Anthropic client with Adaptive's endpoint
// This allows you to use Claude's Messages API while getting intelligent routing
const client = new Anthropic({
	apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
	baseURL: "https://www.llmadaptive.uk/api", // Adaptive's endpoint
});

// Type definitions for better TypeScript support
interface AdaptiveAnthropicResponse extends Anthropic.Message {
	provider?: string;
}

// Non-streaming example - get complete response at once
async function nonStreamingExample() {
	console.log("=== Non-Streaming Anthropic Example ===");
	console.log("💬 Sending message: 'Hello!'");
	console.log("🧠 Using intelligent routing (empty model string)...");
	console.log("⏳ Waiting for complete response...");

	try {
		const response = (await client.messages.create({
			model: "", // Leave empty for intelligent routing - Adaptive chooses the best model
			max_tokens: 1000,
			messages: [{ role: "user", content: "Hello!" }],
			stream: false, // Explicitly disable streaming
		})) as AdaptiveAnthropicResponse;

		// Type-safe response handling for Anthropic's format
		const firstContent = response.content[0];
		if (!firstContent || firstContent.type !== "text") {
			console.log("❌ No text response content received");
			return;
		}

		// Display the response with provider information
		console.log();
		console.log("✅ Response received:");
		console.log("📝 Content:", firstContent.text);

		// Adaptive adds provider information to the response
		if (response.provider) {
			console.log("🏢 Provider used:", response.provider);
		}
		// Model information is part of Anthropic's standard response
		console.log("🤖 Model used:", response.model);

		// Usage information (Anthropic format)
		if (response.usage) {
			console.log("📊 Usage:");
			console.log(`   • Input tokens: ${response.usage.input_tokens}`);
			console.log(`   • Output tokens: ${response.usage.output_tokens}`);
			console.log(
				`   • Total tokens: ${response.usage.input_tokens + response.usage.output_tokens}`,
			);
		}

		console.log("🔍 Response metadata:");
		console.log(`   • ID: ${response.id}`);
		console.log(`   • Role: ${response.role}`);
		console.log(`   • Stop reason: ${response.stop_reason}`);

		console.log();
		console.log("✨ Non-streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Non-streaming Error:", error);

		if (error instanceof Anthropic.APIError) {
			console.error("🔧 API Error Details:");
			console.error("   • Status:", error.status);
			console.error("   • Message:", error.message);
			if (error.error) {
				console.error("   • Error type:", error.error.type);
			}
		}

		throw error;
	}
}

// Streaming example - get response as it's generated
async function streamingExample() {
	console.log("=== Streaming Anthropic Example ===");
	console.log(
		"💬 Sending message: 'Write a short story about a robot learning to paint'",
	);
	console.log("🧠 Using intelligent routing with streaming...");
	console.log("📡 Response will appear in real-time:");
	console.log();

	try {
		const stream = client.messages.stream({
			model: "", // Leave empty for intelligent routing
			max_tokens: 200, // Limit response length for demo
			messages: [
				{
					role: "user",
					content: "Write a short poem about coding",
				},
			],
		});

		let fullContent = "";
		let chunkCount = 0;
		let streamError: Error | null = null;

		console.log("🔄 Streaming response:");
		process.stdout.write("📝 ");

		// Handle streaming events
		stream.on("text", (text) => {
			process.stdout.write(text);
			fullContent += text;
			chunkCount++;
		});

		stream.on("error", (error) => {
			console.error("\n❌ Stream error:", error);
			streamError = error;
			stream.abort();
		});

		// Wait for the final message
		const finalMessage = await stream.finalMessage();

		// Check if there was a captured stream error and rethrow it
		if (streamError) {
			throw streamError;
		}

		console.log(); // New line after content
		console.log();
		console.log("✅ Streaming completed!");
		console.log("🏁 Stop reason:", finalMessage.stop_reason);

		// Adaptive adds provider information
		const adaptiveFinalMessage = finalMessage as AdaptiveAnthropicResponse;
		if (adaptiveFinalMessage.provider) {
			console.log("🏢 Provider used:", adaptiveFinalMessage.provider);
		}
		// Model information is part of Anthropic's standard response
		console.log("🤖 Model used:", adaptiveFinalMessage.model);

		console.log("📊 Streaming statistics:");
		console.log(`   • Text chunks received: ${chunkCount}`);
		console.log(`   • Total content length: ${fullContent.length} characters`);

		// Usage information
		if (finalMessage.usage) {
			console.log("📊 Final usage:");
			console.log(`   • Input tokens: ${finalMessage.usage.input_tokens}`);
			console.log(`   • Output tokens: ${finalMessage.usage.output_tokens}`);
		}

		console.log();
		console.log("✨ Streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Streaming Error:", error);

		if (error instanceof Anthropic.APIError) {
			console.error("🔧 API Error Details:");
			console.error("   • Status:", error.status);
			console.error("   • Message:", error.message);
			if (error.error) {
				console.error("   • Error type:", error.error.type);
			}
		}

		throw error;
	}
}

// Function calling example - demonstrate Anthropic's tool use capabilities
async function functionCallingExample() {
	console.log("=== Function Calling Anthropic Example ===");
	console.log("💬 Testing tool use with weather lookup");
	console.log("🧠 Using intelligent routing with tool use...");
	console.log();

	try {
		// Define a tool for the model to use (Anthropic's tool format)
		const weatherTool = {
			name: "get_weather",
			description: "Get the current weather for a given location",
			input_schema: {
				type: "object" as const,
				properties: {
					location: {
						type: "string",
						description: "The city and state/country, e.g. San Francisco, CA",
					},
					unit: {
						type: "string",
						enum: ["celsius", "fahrenheit"],
						description: "The temperature unit to use",
					},
				},
				required: ["location"],
			},
		};

		const response = (await client.messages.create({
			model: "", // Leave empty for intelligent routing - Adaptive will choose a model that supports tools
			max_tokens: 1000,
			messages: [
				{
					role: "user",
					content:
						"What's the weather like in Tokyo, Japan? Please use Celsius.",
				},
			],
			tools: [weatherTool],
		})) as AdaptiveAnthropicResponse;

		console.log("🔄 Tool use response:");

		// Check if the model wants to use a tool
		const toolUseBlocks = response.content.filter(
			(block) => block.type === "tool_use",
		) as Anthropic.ToolUseBlock[];

		if (toolUseBlocks.length > 0) {
			console.log("🛠️ Tool use detected:");

			for (const toolUse of toolUseBlocks) {
				console.log(`   • Tool: ${toolUse.name}`);
				console.log(`   • Input:`, JSON.stringify(toolUse.input, null, 2));

				// Simulate tool execution (in real implementation, call actual weather API)
				const input = toolUse.input as { location?: string; unit?: string };
				const toolResponse = {
					location: input.location || "Unknown",
					temperature: "22°C",
					condition: "Partly cloudy",
					humidity: "65%",
					wind: "8 km/h NE",
				};

				console.log("📡 Tool response:", JSON.stringify(toolResponse, null, 2));

				// For a complete tool use flow, send the tool result back to the model
				const followUpResponse = (await client.messages.create({
					model: "", // Continue with intelligent routing
					max_tokens: 1000,
					messages: [
						{
							role: "user",
							content:
								"What's the weather like in Tokyo, Japan? Please use Celsius.",
						},
						{
							role: "assistant",
							content: response.content,
						},
						{
							role: "user",
							content: [
								{
									type: "tool_result",
									tool_use_id: toolUse.id,
									content: JSON.stringify(toolResponse),
								},
							],
						},
					],
					tools: [weatherTool],
				})) as AdaptiveAnthropicResponse;

				console.log();
				console.log("📝 Follow-up response:");
				const textContent = followUpResponse.content.find(
					(block) => block.type === "text",
				) as Anthropic.TextBlock;
				if (textContent) {
					console.log(textContent.text);
				}
			}
		} else {
			// No tool use, just regular response
			console.log("📝 Direct response:");
			const textContent = response.content.find(
				(block) => block.type === "text",
			) as Anthropic.TextBlock;
			if (textContent) {
				console.log(textContent.text);
			}
		}

		// Usage information
		if (response.usage) {
			console.log();
			console.log("📊 Usage:");
			console.log(`   • Input tokens: ${response.usage.input_tokens}`);
			console.log(`   • Output tokens: ${response.usage.output_tokens}`);
		}

		console.log();
		console.log("✨ Function calling example completed successfully!");
	} catch (error) {
		console.error("❌ Function Calling Error:", error);

		if (error instanceof Anthropic.APIError) {
			console.error("🔧 API Error Details:");
			console.error("   • Status:", error.status);
			console.error("   • Message:", error.message);
			if (error.error) {
				console.error("   • Error type:", error.error.type);
			}
		}

		throw error;
	}
}

async function main() {
	console.log("🚀 Starting Anthropic SDK example with Adaptive...");
	console.log("📡 Using endpoint:", client.baseURL);
	console.log("🏛️ Using Claude's Messages API format");
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

		// Run function calling example third
		await functionCallingExample();

		console.log();
		console.log("=".repeat(50));
		console.log();
		console.log("🎉 All Anthropic examples completed successfully!");
		console.log(
			"💰 You're getting cost optimization while keeping Claude's API format.",
		);
		console.log(
			"🔄 Streaming, non-streaming, and tool use all work with Anthropic SDK patterns.",
		);
		console.log(
			"🧠 Adaptive intelligently routes to the best available model.",
		);
		console.log();
		console.log("💡 Key features demonstrated:");
		console.log("   • Anthropic Messages API compatibility");
		console.log("   • Intelligent model routing with empty model string");
		console.log("   • Native Claude response format preservation");
		console.log("   • Automatic cost optimization and provider selection");
		console.log("   • Tool use (function calling) capabilities");
		console.log("   • Streaming and non-streaming responses");
	} catch (error) {
		console.error("❌ Example failed:", error);
		process.exit(1);
	}
}

// Run the example
main();
