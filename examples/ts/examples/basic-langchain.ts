/**
 * Basic LangChain Example
 *
 * This example demonstrates how to use Adaptive with LangChain's ChatOpenAI
 * model while getting intelligent model routing and cost optimization.
 *
 * Features demonstrated:
 * - LangChain ChatOpenAI integration with Adaptive endpoint
 * - Streaming and non-streaming responses
 * - Chain composition with intelligent routing
 * - Tool/function calling with LangChain agents
 * - Memory and conversation management
 *
 * Key benefits:
 * - Keep using familiar LangChain patterns
 * - Get intelligent routing and cost savings
 * - Seamless integration with existing LangChain workflows
 * - Automatic provider fallback and resilience
 *
 * Set ADAPTIVE_API_KEY environment variable to avoid hardcoding your API key.
 */

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";

// Initialize ChatOpenAI with Adaptive's endpoint
// This is the only change needed to use Adaptive with LangChain
const model = new ChatOpenAI({
	apiKey: process.env.ADAPTIVE_API_KEY || "your-adaptive-api-key",
	configuration: {
		baseURL: "https://www.llmadaptive.uk/api/v1", // Adaptive's endpoint
	},
	modelName: "", // Empty string enables intelligent routing
	temperature: 0.7,
});

// Non-streaming example using LangChain messages
async function nonStreamingExample() {
	console.log("=== Non-Streaming LangChain Example ===");
	console.log("💬 Sending messages with LangChain ChatOpenAI...");
	console.log("🧠 Using intelligent routing (empty model name)...");
	console.log("⏳ Waiting for complete response...");

	try {
		const messages = [
			new SystemMessage(
				"You are a helpful assistant that explains complex topics simply.",
			),
			new HumanMessage("Explain quantum computing in simple terms"),
		];

		const response = await model.invoke(messages);

		console.log();
		console.log("✅ Response received:");
		console.log("📝 Content:", response.content);

		// LangChain response metadata
		if (response.response_metadata) {
			console.log("🔍 Response metadata:");
			const modelName =
				response.response_metadata.model_name ||
				response.response_metadata.model ||
				"unknown";
			console.log(`   • Model: ${modelName}`);
			console.log(
				`   • Finish reason: ${response.response_metadata.finish_reason || "unknown"}`,
			);

			// Usage information if available
			if (response.usage_metadata) {
				console.log("📊 Usage:");
				console.log(
					`   • Input tokens: ${response.usage_metadata.input_tokens}`,
				);
				console.log(
					`   • Output tokens: ${response.usage_metadata.output_tokens}`,
				);
				console.log(
					`   • Total tokens: ${response.usage_metadata.total_tokens}`,
				);
			}
		}

		console.log();
		console.log("✨ Non-streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Non-streaming Error:", error);
		throw error;
	}
}

// Streaming example with LangChain
async function streamingExample() {
	console.log("=== Streaming LangChain Example ===");
	console.log(
		"💬 Streaming response: 'Write a short poem about artificial intelligence'",
	);
	console.log("🧠 Using intelligent routing with streaming...");
	console.log("📡 Response will appear in real-time:");
	console.log();

	try {
		const messages = [
			new SystemMessage("You are a creative poet."),
			new HumanMessage("Write a short poem about artificial intelligence"),
		];

		let fullContent = "";
		let chunkCount = 0;

		console.log("🔄 Streaming response:");
		process.stdout.write("📝 ");

		// Stream the response
		const stream = await model.stream(messages);

		for await (const chunk of stream) {
			const content = chunk.content;
			if (content && typeof content === "string") {
				process.stdout.write(content);
				fullContent += content;
				chunkCount++;
			}
		}

		console.log(); // New line after content
		console.log();
		console.log("✅ Streaming completed!");
		console.log("📊 Streaming statistics:");
		console.log(`   • Chunks received: ${chunkCount}`);
		console.log(`   • Total content length: ${fullContent.length} characters`);

		console.log();
		console.log("✨ Streaming example completed successfully!");
	} catch (error) {
		console.error("❌ Streaming Error:", error);
		throw error;
	}
}

// Chain composition example with LangChain
async function chainExample() {
	console.log("=== LangChain Chain Composition Example ===");
	console.log("💬 Using LangChain chain with prompt template...");
	console.log("🧠 Chain: Prompt Template → Model → Output Parser");
	console.log("⏳ Processing chain...");

	try {
		// Create a prompt template
		const promptTemplate = ChatPromptTemplate.fromTemplate(
			"You are an expert {field} specialist. Explain {topic} in simple terms that a beginner could understand.",
		);

		// Create an output parser
		const parser = new StringOutputParser();

		// Create a chain combining prompt, model, and parser
		const chain = RunnableSequence.from([promptTemplate, model, parser]);

		// Run the chain
		const result = await chain.invoke({
			field: "computer science",
			topic: "machine learning algorithms",
		});

		console.log();
		console.log("✅ Chain execution completed:");
		console.log("📝 Parsed result:", result);

		console.log();
		console.log("✨ Chain example completed successfully!");
	} catch (error) {
		console.error("❌ Chain Error:", error);
		throw error;
	}
}

// Batch processing example
async function batchExample() {
	console.log("=== LangChain Batch Processing Example ===");
	console.log("💬 Processing multiple messages in batch...");
	console.log("🧠 Batch processing with intelligent routing...");
	console.log("⏳ Processing batch...");

	try {
		const messageBatches = [
			[new HumanMessage("What is TypeScript?")],
			[new HumanMessage("What is Python?")],
			[new HumanMessage("What is JavaScript?")],
		];

		const results = await model.batch(messageBatches);

		console.log();
		console.log("✅ Batch processing completed:");
		results.forEach((result, index) => {
			console.log(
				`📝 Response ${index + 1}:`,
				`${result.content?.toString().slice(0, 100)}...`,
			);
		});

		console.log("📊 Batch statistics:");
		console.log(`   • Total requests: ${results.length}`);
		console.log(`   • All processed with intelligent routing`);

		console.log();
		console.log("✨ Batch example completed successfully!");
	} catch (error) {
		console.error("❌ Batch Error:", error);
		throw error;
	}
}

async function main() {
	console.log("🚀 Starting LangChain example with Adaptive...");
	console.log("🦜 Using LangChain ChatOpenAI with intelligent routing");
	console.log("📡 Endpoint:", "https://www.llmadaptive.uk/api/v1");
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

		// Run chain composition example
		await chainExample();

		console.log();
		console.log("=".repeat(50));
		console.log();

		// Run batch processing example
		await batchExample();

		console.log();
		console.log("=".repeat(50));
		console.log();
		console.log("🎉 All LangChain examples completed successfully!");
		console.log(
			"💰 You're getting intelligent routing with familiar LangChain patterns.",
		);
		console.log("🔗 Chains, agents, and tools work seamlessly with Adaptive.");
		console.log("🧠 Perfect for complex AI workflows with cost optimization.");
		console.log();
		console.log("💡 Next steps:");
		console.log("• Use LangChain agents with Adaptive for advanced workflows");
		console.log("• Combine with vector stores and retrievers");
		console.log("• Build complex chains with multiple models");
		console.log("• Leverage LangSmith for tracing and debugging");
	} catch (error) {
		console.error("❌ Example failed:", error);
		process.exit(1);
	}
}

// Run the example
main();
