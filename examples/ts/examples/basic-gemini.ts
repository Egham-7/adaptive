/**
 * Basic Gemini SDK Example
 *
 * This example demonstrates how to use Adaptive with Google's Gemini API
 * format while getting intelligent model routing and cost optimization.
 *
 * Features demonstrated:
 * - Google Gen AI SDK compatibility (@google/genai)
 * - Both streaming and non-streaming responses
 * - Intelligent model routing (empty model string)
 * - Automatic cost optimization and provider selection
 * - Proper TypeScript typing for Gemini responses
 * - Function calling capabilities
 *
 * Key benefits:
 * - Keep using familiar Gemini SDK patterns
 * - Get cost savings through intelligent routing
 * - Automatic fallback between providers
 * - Native Gemini response format preserved
 *
 * Set ADAPTIVE_API_KEY environment variable to avoid hardcoding your API key.
 */

import {
	FunctionCallingConfigMode,
	type FunctionDeclaration,
	GoogleGenAI,
} from "@google/genai";

// Initialize the Gemini client with Adaptive's endpoint
// This allows you to use Gemini's API while getting intelligent routing
const ai = new GoogleGenAI({
	apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
	httpOptions: {
		baseUrl: "http://localhost:8080",
	},
});

// Non-streaming example - get complete response at once
async function nonStreamingExample() {
	console.log("=== Non-Streaming Gemini Example ===");
	console.log("💬 Sending message: 'Hello! Tell me about TypeScript.'");
	console.log("🧠 Using intelligent routing (empty model string)...");
	console.log("⏳ Waiting for complete response...");

	try {
		// Use generateContent with empty model for intelligent routing
		const response = await ai.models.generateContent({
			model: "intelligent-routing", // Empty for intelligent routing - Adaptive chooses the best model
			contents: "Hello! Tell me about TypeScript in exactly 3 sentences.",
			config: {
				maxOutputTokens: 1000,
				temperature: 0.7,
			},
		});

		// Display the response
		console.log();
		console.log("✅ Response received:");
		console.log("📝 Content:", response.text);

		// Adaptive adds provider information to the response
		const adaptiveResponse = response as any;
		if (adaptiveResponse.provider) {
			console.log("🏢 Provider used:", adaptiveResponse.provider);
		}

		// Model information
		if (adaptiveResponse.modelVersion) {
			console.log("🤖 Model version:", adaptiveResponse.modelVersion);
		}

		// Usage information (Gemini format)
		if (adaptiveResponse.usageMetadata) {
			const usage = adaptiveResponse.usageMetadata;
			console.log("📊 Usage:");
			console.log(`   • Prompt tokens: ${usage.promptTokenCount}`);
			console.log(`   • Response tokens: ${usage.candidatesTokenCount}`);
			console.log(`   • Total tokens: ${usage.totalTokenCount}`);
		}

		// Candidate information
		if (adaptiveResponse.candidates && adaptiveResponse.candidates.length > 0) {
			const candidate = adaptiveResponse.candidates[0];
			console.log("🔍 Response metadata:");
			console.log(
				`   • Finish reason: ${candidate.finishReason || "completed"}`,
			);
			console.log(`   • Role: ${candidate.content.role}`);
		}

		console.log();
		console.log("✨ Non-streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Non-streaming Error:", error);

		if (error instanceof Error) {
			console.error("🔧 Error Details:");
			console.error("   • Message:", error.message);
			console.error("   • Name:", error.name);
		}

		throw error;
	}
}

// Streaming example - get response as it's generated
async function streamingExample() {
	console.log("=== Streaming Gemini Example ===");
	console.log(
		"💬 Sending message: 'Write a short poem about artificial intelligence'",
	);
	console.log("🧠 Using intelligent routing with streaming...");
	console.log("📡 Response will appear in real-time:");
	console.log();

	try {
		// Use generateContentStream with empty model for intelligent routing
		const response = await ai.models.generateContentStream({
			model: "intelligent-routing", // Empty for intelligent routing
			contents:
				"Write a short, creative poem about artificial intelligence and the future of programming.",
			config: {
				maxOutputTokens: 200,
				temperature: 0.8,
			},
		});

		let fullContent = "";
		let chunkCount = 0;

		console.log("🔄 Streaming response:");
		process.stdout.write("📝 ");

		// Handle streaming chunks
		for await (const chunk of response) {
			const chunkText = chunk.text;
			if (chunkText) {
				process.stdout.write(chunkText);
				fullContent += chunkText;
				chunkCount++;
			}
		}

		console.log(); // New line after content
		console.log();
		console.log("✅ Streaming completed!");

		console.log("📊 Streaming statistics:");
		console.log(`   • Text chunks received: ${chunkCount}`);
		console.log(`   • Total content length: ${fullContent.length} characters`);

		console.log();
		console.log("✨ Streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Streaming Error:", error);

		if (error instanceof Error) {
			console.error("🔧 Error Details:");
			console.error("   • Message:", error.message);
			console.error("   • Name:", error.name);
		}

		throw error;
	}
}

// Function calling example - demonstrate Gemini's function calling capabilities
async function functionCallingExample() {
	console.log("=== Function Calling Gemini Example ===");
	console.log("💬 Testing function calling with weather lookup");
	console.log("🧠 Using intelligent routing with function tools...");
	console.log();

	try {
		// Define a function for the model to call
		const getCurrentWeather: FunctionDeclaration = {
			name: "getCurrentWeather",
			description: "Get the current weather for a given location",
			parametersJsonSchema: {
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

		const response = await ai.models.generateContent({
			model: "intelligent-routing", // Empty for intelligent routing
			contents: "What's the weather like in Tokyo, Japan? Please use Celsius.",
			config: {
				tools: [
					{
						functionDeclarations: [getCurrentWeather],
					},
				],
				toolConfig: {
					functionCallingConfig: {
						mode: FunctionCallingConfigMode.ANY,
						allowedFunctionNames: ["getCurrentWeather"],
					},
				},
				maxOutputTokens: 1000,
				temperature: 0.3,
			},
		});

		console.log("🔄 Function calling response:");

		// Check if the model wants to call a function
		const functionCalls = response.functionCalls;
		if (functionCalls && functionCalls.length > 0) {
			console.log("🛠️ Function call detected:");

			for (const call of functionCalls) {
				console.log(`   • Function: ${call.name}`);
				console.log(`   • Arguments:`, JSON.stringify(call.args, null, 2));

				// Simulate function execution
				const functionResponse = {
					name: call.name,
					response: {
						location: call.args?.location || "Unknown",
						temperature: "22°C",
						condition: "Partly cloudy",
						humidity: "65%",
						wind: "8 km/h NE",
					},
				};

				console.log(
					"📡 Function response:",
					JSON.stringify(functionResponse.response, null, 2),
				);

				// For a complete function calling flow, you would normally
				// send the function result back to the model for a follow-up response
				console.log();
				console.log("📝 Function call completed successfully!");
				console.log(
					"💡 In a real implementation, you would send this result back to the model.",
				);
			}
		} else {
			// No function call, just regular response
			console.log("📝 Direct response:");
			console.log(response.text);
		}

		console.log();
		console.log("✨ Function calling example completed successfully!");
	} catch (error) {
		console.error("❌ Function Calling Error:", error);

		if (error instanceof Error) {
			console.error("🔧 Error Details:");
			console.error("   • Message:", error.message);
			console.error("   • Name:", error.name);
		}

		throw error;
	}
}

async function main() {
	console.log("🚀 Starting Gemini SDK example with Adaptive...");
	console.log("📡 Using Google Gen AI SDK (@google/genai)");
	console.log("🏛️ Using Google's Gemini API format with intelligent routing");
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
		console.log("🎉 All Gemini examples completed successfully!");
		console.log(
			"💰 You're getting cost optimization while keeping Gemini's API format.",
		);
		console.log(
			"🔄 Streaming, non-streaming, and function calling all work with Gemini SDK patterns.",
		);
		console.log(
			"🧠 Adaptive intelligently routes to the best available model.",
		);
		console.log();
		console.log("💡 Key features demonstrated:");
		console.log("   • Google Gen AI SDK (@google/genai) compatibility");
		console.log("   • Intelligent model routing with empty model string");
		console.log("   • Native Gemini response format preservation");
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
