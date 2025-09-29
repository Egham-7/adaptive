/**
 * Basic OpenAI SDK Example
 *
 * This example demonstrates how to use Adaptive as a drop-in replacement
 * for the OpenAI API. Simply change the baseURL and you get intelligent
 * model routing, cost optimization, and provider failover.
 *
 * Features demonstrated:
 * - Drop-in OpenAI SDK compatibility
 * - Both streaming and non-streaming responses
 * - Intelligent model routing (empty model string)
 * - Function calling capabilities
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

import OpenAI, { APIError } from "openai";

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

		if (error instanceof APIError) {
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

		if (error instanceof APIError) {
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

// Function calling example - demonstrate OpenAI's function calling capabilities
async function functionCallingExample() {
	console.log("=== Function Calling Example ===");
	console.log("💬 Testing function calling with weather lookup");
	console.log("🧠 Using intelligent routing with function tools...");
	console.log();

	try {
		// Define a function for the model to call
		const weatherFunction = {
			name: "getCurrentWeather",
			description: "Get the current weather for a given location",
			parameters: {
				type: "object",
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

		const response = await client.chat.completions.create({
			model: "", // Leave empty for intelligent routing - Adaptive will choose a model that supports function calling
			messages: [
				{
					role: "user",
					content:
						"What's the weather like in Tokyo, Japan? Please use Celsius.",
				},
			],
			tools: [
				{
					type: "function",
					function: weatherFunction,
				},
			],
			tool_choice: "auto", // Let the model decide when to call functions
		});

		console.log("🔄 Function calling response:");

		// Check if the model wants to call a function
		const message = response.choices[0]?.message;
		if (message?.tool_calls && message.tool_calls.length > 0) {
			console.log("🛠️ Function call detected:");

			for (const toolCall of message.tool_calls) {
				if (toolCall.type === "function") {
					console.log(`   • Function: ${toolCall.function.name}`);
					console.log(`   • Arguments:`, toolCall.function.arguments);

					// Parse the arguments
					const args = JSON.parse(toolCall.function.arguments);

					// Simulate function execution (in real implementation, call actual weather API)
					const functionResponse = {
						location: args.location || "Unknown",
						temperature: "22°C",
						condition: "Partly cloudy",
						humidity: "65%",
						wind: "8 km/h NE",
					};

					console.log(
						"📡 Function response:",
						JSON.stringify(functionResponse, null, 2),
					);

					// For a complete function calling flow, you would send the function result back to the model
					const followUpResponse = await client.chat.completions.create({
						model: "", // Continue with intelligent routing
						messages: [
							{
								role: "user",
								content:
									"What's the weather like in Tokyo, Japan? Please use Celsius.",
							},
							message, // Include the assistant's message with function call
							{
								role: "tool",
								content: JSON.stringify(functionResponse),
								tool_call_id: toolCall.id,
							},
						],
						tools: [
							{
								type: "function",
								function: weatherFunction,
							},
						],
					});

					console.log();
					console.log("📝 Follow-up response:");
					console.log(followUpResponse.choices[0]?.message?.content);
				}
			}
		} else {
			// No function call, just regular response
			console.log("📝 Direct response:");
			console.log(message?.content);
		}

		// Usage information
		if (response.usage) {
			console.log();
			console.log("📊 Usage:");
			console.log(`   • Input tokens: ${response.usage.prompt_tokens}`);
			console.log(`   • Output tokens: ${response.usage.completion_tokens}`);
			console.log(`   • Total tokens: ${response.usage.total_tokens}`);
		}

		console.log();
		console.log("✨ Function calling example completed successfully!");
	} catch (error) {
		console.error("❌ Function Calling Error:", error);

		if (error instanceof APIError) {
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

		// Run function calling example third
		await functionCallingExample();

		console.log();
		console.log("=".repeat(50));
		console.log();
		console.log("🎉 All examples completed successfully!");
		console.log(
			"💰 You're likely saving 60-80% compared to direct OpenAI usage.",
		);
		console.log(
			"🔄 Streaming, non-streaming, and function calling all work seamlessly with Adaptive.",
		);
		console.log(
			"🧠 Adaptive intelligently routes to the best available model.",
		);
		console.log();
		console.log("💡 Key features demonstrated:");
		console.log("   • OpenAI SDK drop-in compatibility");
		console.log("   • Intelligent model routing with empty model string");
		console.log("   • Native OpenAI response format preservation");
		console.log("   • Automatic cost optimization and provider selection");
		console.log("   • Function calling capabilities");
		console.log("   • Streaming and non-streaming responses");
	} catch (error) {
		console.error("❌ Example failed:", error);
		process.exit(1);
	}
}

// Run the example
main();
