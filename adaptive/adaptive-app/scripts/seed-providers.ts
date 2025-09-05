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

				inputTokenCost: 0.5,
				outputTokenCost: 1.5,
			},
			"gpt-4o": {
				displayName: "GPT-4o",

				inputTokenCost: 3.0,
				outputTokenCost: 10.0,
			},
			"gpt-4o-mini": {
				displayName: "GPT-4o Mini",

				inputTokenCost: 0.15,
				outputTokenCost: 0.6,
			},
			"gpt-4": {
				displayName: "GPT-4",

				inputTokenCost: 30.0,
				outputTokenCost: 60.0,
			},
			"gpt-4-turbo": {
				displayName: "GPT-4 Turbo",

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

				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"claude-3.5-haiku": {
				displayName: "Claude 3.5 Haiku",

				inputTokenCost: 0.8,
				outputTokenCost: 4.0,
			},
			"claude-3-opus": {
				displayName: "Claude 3 Opus",

				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"claude-4-sonnet": {
				displayName: "Claude 4 Sonnet",

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

				inputTokenCost: 1.25,
				outputTokenCost: 10.0,
			},
			"gemini-2.5-pro-large": {
				displayName: "Gemini 2.5 Pro Large",

				inputTokenCost: 2.5,
				outputTokenCost: 15.0,
			},
			"gemini-1.5-flash": {
				displayName: "Gemini 1.5 Flash",

				inputTokenCost: 1.25,
				outputTokenCost: 5.0,
			},
			"gemini-2.0-flash": {
				displayName: "Gemini 2.0 Flash",

				inputTokenCost: 0.1,
				outputTokenCost: 0.4,
			},
			"gemini-pro": {
				displayName: "Gemini Pro",

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

				inputTokenCost: 0.27,
				outputTokenCost: 1.1,
			},
			"deepseek-reasoner": {
				displayName: "DeepSeek Reasoner",

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

				inputTokenCost: 0.11,
				outputTokenCost: 0.34,
			},
			"llama-4-maverick-17b-128e-instruct": {
				displayName: "Llama 4 Maverick 17B",

				inputTokenCost: 0.2,
				outputTokenCost: 0.6,
			},
			"llama-guard-4-12b": {
				displayName: "Llama Guard 4 12B",

				inputTokenCost: 0.2,
				outputTokenCost: 0.2,
			},
			"deepseek-r1-distill-llama-70b": {
				displayName: "DeepSeek R1 Distill Llama 70B",

				inputTokenCost: 0.75,
				outputTokenCost: 0.99,
			},
			"qwen-qwq-32b": {
				displayName: "Qwen QwQ 32B",

				inputTokenCost: 0.29,
				outputTokenCost: 0.39,
			},
			"mistral-saba-24b": {
				displayName: "Mistral Saba 24B",

				inputTokenCost: 0.79,
				outputTokenCost: 0.79,
			},
			"llama-3.3-70b-versatile": {
				displayName: "Llama 3.3 70B Versatile",

				inputTokenCost: 0.59,
				outputTokenCost: 0.79,
			},
			"llama-3.1-8b-instant": {
				displayName: "Llama 3.1 8B Instant",

				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"llama3-70b-8192": {
				displayName: "Llama 3 70B",

				inputTokenCost: 0.59,
				outputTokenCost: 0.79,
			},
			"llama3-8b-8192": {
				displayName: "Llama 3 8B",

				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"mixtral-8x7b-32768": {
				displayName: "Mixtral 8x7B",

				inputTokenCost: 0.27,
				outputTokenCost: 0.27,
			},
			"gemma-7b-it": {
				displayName: "Gemma 7B IT",

				inputTokenCost: 0.1,
				outputTokenCost: 0.1,
			},
			"gemma2-9b-it": {
				displayName: "Gemma 2 9B IT",

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

				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"grok-3-mini": {
				displayName: "Grok 3 Mini",

				inputTokenCost: 0.3,
				outputTokenCost: 0.5,
			},
			"grok-3-fast": {
				displayName: "Grok 3 Fast",

				inputTokenCost: 5.0,
				outputTokenCost: 25.0,
			},
			"grok-beta": {
				displayName: "Grok Beta",

				inputTokenCost: 38.15,
				outputTokenCost: 114.44,
			},
		},
	},
	huggingface: {
		displayName: "Hugging Face",
		description: "Open-source models hosted on Hugging Face",
		models: {
			"meta-llama/Llama-3.1-8B-Instruct": {
				displayName: "Llama 3.1 8B Instruct",

				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
			},
			"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
				displayName: "DeepSeek R1 Distill Qwen 14B",

				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
			},
			"mistralai/Mistral-7B-Instruct-v0.3": {
				displayName: "Mistral 7B Instruct v0.3",

				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
			},
			"deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
				displayName: "DeepSeek R1 Distill Llama 8B",

				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
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
					visibility: "system", // System providers are globally visible
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
						inputTokenCost: modelData.inputTokenCost,
						outputTokenCost: modelData.outputTokenCost,
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
