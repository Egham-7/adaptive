import { PrismaClient } from "prisma/generated";

const prisma = new PrismaClient();

// Provider pricing data
const globalPricing = {
	openai: {
		displayName: "OpenAI",
		description: "Advanced AI models from OpenAI",
		models: {
			"gpt-3.5-turbo": {
				displayName: "GPT-3.5 Turbo",
				type: "chat",
				inputTokenCost: 0.5,
				outputTokenCost: 1.5,
			},
			"gpt-4o": {
				displayName: "GPT-4o",
				type: "chat",
				inputTokenCost: 3.0,
				outputTokenCost: 10.0,
			},
			"gpt-4o-mini": {
				displayName: "GPT-4o Mini",
				type: "chat",
				inputTokenCost: 0.15,
				outputTokenCost: 0.6,
			},
			"gpt-4": {
				displayName: "GPT-4",
				type: "chat",
				inputTokenCost: 30.0,
				outputTokenCost: 60.0,
			},
			"gpt-4-turbo": {
				displayName: "GPT-4 Turbo",
				type: "chat",
				inputTokenCost: 10.0,
				outputTokenCost: 30.0,
			},
		},
	},
	anthropic: {
		displayName: "Anthropic",
		description: "Constitutional AI models by Anthropic",
		models: {
			"claude-3.5-sonnet": {
				displayName: "Claude 3.5 Sonnet",
				type: "chat",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"claude-3.5-haiku": {
				displayName: "Claude 3.5 Haiku",
				type: "chat",
				inputTokenCost: 0.8,
				outputTokenCost: 4.0,
			},
			"claude-3-opus": {
				displayName: "Claude 3 Opus",
				type: "chat",
				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"claude-4-sonnet": {
				displayName: "Claude 4 Sonnet",
				type: "chat",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
		},
	},
	gemini: {
		displayName: "Gemini",
		description: "Google's advanced multimodal AI models",
		models: {
			"gemini-2.5-pro": {
				displayName: "Gemini 2.5 Pro",
				type: "chat",
				inputTokenCost: 1.25,
				outputTokenCost: 10.0,
			},
			"gemini-2.5-pro-large": {
				displayName: "Gemini 2.5 Pro Large",
				type: "chat",
				inputTokenCost: 2.5,
				outputTokenCost: 15.0,
			},
			"gemini-1.5-flash": {
				displayName: "Gemini 1.5 Flash",
				type: "chat",
				inputTokenCost: 1.25,
				outputTokenCost: 5.0,
			},
			"gemini-2.0-flash": {
				displayName: "Gemini 2.0 Flash",
				type: "chat",
				inputTokenCost: 0.1,
				outputTokenCost: 0.4,
			},
			"gemini-pro": {
				displayName: "Gemini Pro",
				type: "chat",
				inputTokenCost: 1.25,
				outputTokenCost: 5.0,
			},
		},
	},
	deepseek: {
		displayName: "DeepSeek",
		description: "Advanced reasoning AI models by DeepSeek",
		models: {
			"deepseek-chat": {
				displayName: "DeepSeek Chat",
				type: "chat",
				inputTokenCost: 0.27,
				outputTokenCost: 1.1,
			},
			"deepseek-reasoner": {
				displayName: "DeepSeek Reasoner",
				type: "chat",
				inputTokenCost: 0.55,
				outputTokenCost: 2.19,
			},
		},
	},
	groq: {
		displayName: "Groq",
		description: "High-performance inference for open-source models",
		models: {
			"llama-4-scout-17b-16e-instruct": {
				displayName: "Llama 4 Scout 17B",
				type: "chat",
				inputTokenCost: 0.11,
				outputTokenCost: 0.34,
			},
			"llama-4-maverick-17b-128e-instruct": {
				displayName: "Llama 4 Maverick 17B",
				type: "chat",
				inputTokenCost: 0.2,
				outputTokenCost: 0.6,
			},
			"llama-guard-4-12b": {
				displayName: "Llama Guard 4 12B",
				type: "chat",
				inputTokenCost: 0.2,
				outputTokenCost: 0.2,
			},
			"deepseek-r1-distill-llama-70b": {
				displayName: "DeepSeek R1 Distill Llama 70B",
				type: "chat",
				inputTokenCost: 0.75,
				outputTokenCost: 0.99,
			},
			"qwen-qwq-32b": {
				displayName: "Qwen QwQ 32B",
				type: "chat",
				inputTokenCost: 0.29,
				outputTokenCost: 0.39,
			},
			"mistral-saba-24b": {
				displayName: "Mistral Saba 24B",
				type: "chat",
				inputTokenCost: 0.79,
				outputTokenCost: 0.79,
			},
			"llama-3.3-70b-versatile": {
				displayName: "Llama 3.3 70B Versatile",
				type: "chat",
				inputTokenCost: 0.59,
				outputTokenCost: 0.79,
			},
			"llama-3.1-8b-instant": {
				displayName: "Llama 3.1 8B Instant",
				type: "chat",
				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"llama3-70b-8192": {
				displayName: "Llama 3 70B",
				type: "chat",
				inputTokenCost: 0.59,
				outputTokenCost: 0.79,
			},
			"llama3-8b-8192": {
				displayName: "Llama 3 8B",
				type: "chat",
				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"mixtral-8x7b-32768": {
				displayName: "Mixtral 8x7B",
				type: "chat",
				inputTokenCost: 0.27,
				outputTokenCost: 0.27,
			},
			"gemma-7b-it": {
				displayName: "Gemma 7B IT",
				type: "chat",
				inputTokenCost: 0.1,
				outputTokenCost: 0.1,
			},
			"gemma2-9b-it": {
				displayName: "Gemma 2 9B IT",
				type: "chat",
				inputTokenCost: 0.1,
				outputTokenCost: 0.1,
			},
		},
	},
	grok: {
		displayName: "Grok",
		description: "Advanced AI models by xAI",
		models: {
			"grok-3": {
				displayName: "Grok 3",
				type: "chat",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"grok-3-mini": {
				displayName: "Grok 3 Mini",
				type: "chat",
				inputTokenCost: 0.3,
				outputTokenCost: 0.5,
			},
			"grok-3-fast": {
				displayName: "Grok 3 Fast",
				type: "chat",
				inputTokenCost: 5.0,
				outputTokenCost: 25.0,
			},
			"grok-beta": {
				displayName: "Grok Beta",
				type: "chat",
				inputTokenCost: 38.15,
				outputTokenCost: 114.44,
			},
		},
	},
	adaptive: {
		displayName: "Adaptive",
		description: "Self-hosted adaptive AI models",
		models: {
			"meta-llama/Meta-Llama-3-8B-Instruct": {
				displayName: "Meta Llama 3 8B Instruct",
				type: "chat",
				inputTokenCost: 0.02,
				outputTokenCost: 0.04,
			},
			"google/codegemma-7b-it": {
				displayName: "CodeGemma 7B IT",
				type: "chat",
				inputTokenCost: 0.02,
				outputTokenCost: 0.04,
			},
			"Qwen/Qwen2.5-7B-Instruct": {
				displayName: "Qwen 2.5 7B Instruct",
				type: "chat",
				inputTokenCost: 0.02,
				outputTokenCost: 0.04,
			},
			"Qwen/Qwen2.5-Math-7B-Instruct": {
				displayName: "Qwen 2.5 Math 7B Instruct",
				type: "chat",
				inputTokenCost: 0.025,
				outputTokenCost: 0.05,
			},
			"microsoft/Phi-4-mini-reasoning": {
				displayName: "Phi 4 Mini Reasoning",
				type: "chat",
				inputTokenCost: 0.015,
				outputTokenCost: 0.03,
			},
			"HuggingFaceTB/SmolLM2-1.7B-Instruct": {
				displayName: "SmolLM2 1.7B Instruct",
				type: "chat",
				inputTokenCost: 0.005,
				outputTokenCost: 0.01,
			},
		},
	},
};

async function seedProviders() {
	console.log("🌱 Starting provider and model seeding...");

	try {
		// Clear existing data
		console.log("🧹 Clearing existing provider data...");
		await prisma.providerModel.deleteMany();
		await prisma.provider.deleteMany();

		// Seed providers and their models
		for (const [providerName, providerData] of Object.entries(globalPricing)) {
			console.log(`📦 Creating provider: ${providerData.displayName}`);

			const provider = await prisma.provider.create({
				data: {
					name: providerName,
					displayName: providerData.displayName,
					description: providerData.description,
					isActive: true,
				},
			});

			console.log(
				`  ✅ Created provider: ${provider.displayName} (${provider.id})`,
			);

			// Create models for this provider
			for (const [modelName, modelData] of Object.entries(
				providerData.models,
			)) {
				const model = await prisma.providerModel.create({
					data: {
						providerId: provider.id,
						name: modelName,
						displayName: modelData.displayName,
						type: modelData.type,
						inputTokenCost: modelData.inputTokenCost,
						outputTokenCost: modelData.outputTokenCost,
						isActive: true,
					},
				});

				console.log(
					`    ➕ Created model: ${model.displayName} (${model.name})`,
				);
				console.log(
					`      💰 Input: $${model.inputTokenCost}/1M tokens, Output: $${model.outputTokenCost}/1M tokens`,
				);
			}
		}

		// Print summary
		const providerCount = await prisma.provider.count();
		const modelCount = await prisma.providerModel.count();

		console.log("\n🎉 Seeding completed successfully!");
		console.log("📊 Summary:");
		console.log(`   • ${providerCount} providers created`);
		console.log(`   • ${modelCount} models created`);

		// Print provider breakdown
		const providers = await prisma.provider.findMany({
			include: {
				_count: {
					select: { models: true },
				},
			},
		});

		console.log("\n📋 Provider breakdown:");
		providers.forEach((provider) => {
			console.log(
				`   • ${provider.displayName}: ${provider._count.models} models`,
			);
		});
	} catch (error) {
		console.error("❌ Error seeding providers:", error);
		throw error;
	}
}

async function main() {
	try {
		await seedProviders();
	} catch (error) {
		console.error("❌ Seeding failed:", error);
		process.exit(1);
	} finally {
		await prisma.$disconnect();
	}
}

// Run the seed function
if (require.main === module) {
	main();
}

export { seedProviders };
