import { PrismaClient } from "prisma/generated";

const prisma = new PrismaClient();

// Provider pricing data
const globalPricing = {
	openai: {
		displayName: "OpenAI",
		description: "Advanced AI models from OpenAI",
		models: {
			"gpt-5": {
				displayName: "GPT-5",
				inputTokenCost: 1.25,
				outputTokenCost: 10.0,
			},
			"gpt-5-pro": {
				displayName: "GPT-5 Pro",
				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"gpt-5-mini": {
				displayName: "GPT-5 Mini",
				inputTokenCost: 0.25,
				outputTokenCost: 2.0,
			},
			"gpt-5-nano": {
				displayName: "GPT-5 Nano",
				inputTokenCost: 0.05,
				outputTokenCost: 0.4,
			},
			"gpt-4.1": {
				displayName: "GPT-4.1",
				inputTokenCost: 30.0,
				outputTokenCost: 60.0,
			},
			"gpt-4.1-mini": {
				displayName: "GPT-4.1 Mini",
				inputTokenCost: 5.0,
				outputTokenCost: 10.0,
			},
			"gpt-4.1-nano": {
				displayName: "GPT-4.1 Nano",
				inputTokenCost: 0.5,
				outputTokenCost: 1.0,
			},
			"gpt-4o": {
				displayName: "GPT-4o",
				inputTokenCost: 2.5,
				outputTokenCost: 10.0,
			},
			"gpt-4o-mini": {
				displayName: "GPT-4o Mini",
				inputTokenCost: 0.15,
				outputTokenCost: 0.6,
			},
			o3: {
				displayName: "O3 Reasoning",
				inputTokenCost: 60.0,
				outputTokenCost: 240.0,
			},
			"o3-pro": {
				displayName: "O3 Pro Reasoning",
				inputTokenCost: 120.0,
				outputTokenCost: 480.0,
			},
			"o4-mini": {
				displayName: "O4 Mini Reasoning",
				inputTokenCost: 10.0,
				outputTokenCost: 40.0,
			},
		},
	},
	anthropic: {
		displayName: "Anthropic",
		description: "Constitutional AI models by Anthropic",
		models: {
			"claude-opus-4.1": {
				displayName: "Claude Opus 4.1",
				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"claude-opus-4": {
				displayName: "Claude Opus 4",
				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"claude-sonnet-4": {
				displayName: "Claude Sonnet 4",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"claude-sonnet-3.7": {
				displayName: "Claude Sonnet 3.7",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"claude-3-5-sonnet-20241022": {
				displayName: "Claude 3.5 Sonnet",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
			"claude-3-5-haiku-20241022": {
				displayName: "Claude 3.5 Haiku",
				inputTokenCost: 0.8,
				outputTokenCost: 4.0,
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
			"gemini-2.5-flash": {
				displayName: "Gemini 2.5 Flash",
				inputTokenCost: 0.3,
				outputTokenCost: 1.2,
			},
			"gemini-2.5-flash-lite": {
				displayName: "Gemini 2.5 Flash Lite",
				inputTokenCost: 0.1,
				outputTokenCost: 0.4,
			},
			"gemini-2.0-flash": {
				displayName: "Gemini 2.0 Flash",
				inputTokenCost: 0.1,
				outputTokenCost: 0.4,
			},
			"gemini-2.0-flash-live": {
				displayName: "Gemini 2.0 Flash Live",
				inputTokenCost: 0.15,
				outputTokenCost: 0.6,
			},
		},
	},
	deepseek: {
		displayName: "DeepSeek",
		description: "Advanced reasoning AI models by DeepSeek",
		models: {
			"deepseek-chat": {
				displayName: "DeepSeek Chat V3.1",
				inputTokenCost: 0.27,
				outputTokenCost: 1.1,
			},
			"deepseek-reasoner": {
				displayName: "DeepSeek Reasoner V3.1",
				inputTokenCost: 0.55,
				outputTokenCost: 2.19,
			},
			"deepseek-v3-0324": {
				displayName: "DeepSeek V3 Enhanced",
				inputTokenCost: 0.27,
				outputTokenCost: 1.1,
			},
			"deepseek-r1": {
				displayName: "DeepSeek R1 Reasoning",
				inputTokenCost: 0.55,
				outputTokenCost: 2.19,
			},
			"deepseek-r1-0528": {
				displayName: "DeepSeek R1 Advanced",
				inputTokenCost: 0.75,
				outputTokenCost: 2.99,
			},
			"deepseek-coder-v2": {
				displayName: "DeepSeek Coder V2",
				inputTokenCost: 0.27,
				outputTokenCost: 1.1,
			},
		},
	},
	groq: {
		displayName: "Groq",
		description: "High-performance inference for open-source models",
		models: {
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
			"deepseek-r1-distill-llama-70b": {
				displayName: "DeepSeek R1 Distill Llama 70B",
				inputTokenCost: 0.75,
				outputTokenCost: 0.99,
			},
			"llama-3-groq-70b-tool-use": {
				displayName: "Llama 3 Groq 70B Tool Use",
				inputTokenCost: 0.59,
				outputTokenCost: 0.79,
			},
			"llama-3-groq-8b-tool-use": {
				displayName: "Llama 3 Groq 8B Tool Use",
				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"llama-guard-4-12b": {
				displayName: "Llama Guard 4 12B",
				inputTokenCost: 0.2,
				outputTokenCost: 0.2,
			},
		},
	},
	grok: {
		displayName: "Grok",
		description: "Advanced AI models by xAI",
		models: {
			"grok-4": {
				displayName: "Grok 4",
				inputTokenCost: 15.0,
				outputTokenCost: 75.0,
			},
			"grok-4-heavy": {
				displayName: "Grok 4 Heavy",
				inputTokenCost: 25.0,
				outputTokenCost: 125.0,
			},
			"grok-4-fast": {
				displayName: "Grok 4 Fast",
				inputTokenCost: 5.0,
				outputTokenCost: 25.0,
			},
			"grok-code-fast-1": {
				displayName: "Grok Code Fast 1",
				inputTokenCost: 3.0,
				outputTokenCost: 15.0,
			},
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
		},
	},
	huggingface: {
		displayName: "Hugging Face",
		description: "Open-source models hosted on Hugging Face",
		models: {
			"meta-llama/Llama-3.3-70B-Instruct": {
				displayName: "Llama 3.3 70B Instruct",
				inputTokenCost: 0.05,
				outputTokenCost: 0.08,
			},
			"meta-llama/Llama-3.1-8B-Instruct": {
				displayName: "Llama 3.1 8B Instruct",
				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
			},
			"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
				displayName: "DeepSeek R1 Distill Qwen 14B",
				inputTokenCost: 0.02,
				outputTokenCost: 0.04,
			},
			"deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
				displayName: "DeepSeek R1 Distill Llama 8B",
				inputTokenCost: 0.01,
				outputTokenCost: 0.02,
			},
			"Qwen/Qwen3-235B-A22B": {
				displayName: "Qwen3 235B A22B",
				inputTokenCost: 0.1,
				outputTokenCost: 0.2,
			},
			"Qwen/Qwen3-30B-A3B": {
				displayName: "Qwen3 30B A3B",
				inputTokenCost: 0.02,
				outputTokenCost: 0.04,
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
if (import.meta.url === `file://${process.argv[1]}`) {
	main();
}

export { seedProviders };
