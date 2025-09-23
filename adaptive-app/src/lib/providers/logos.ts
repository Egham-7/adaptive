/**
 * Shared provider utilities for logos and display names
 */

export const getProviderLogo = (provider: string): string | undefined => {
	const logoMap: { [key: string]: string } = {
		anthropic: "/logos/anthropic.jpeg",
		openai: "/logos/openai.webp",
		meta: "/logos/meta.png",
		huggingface: "/logos/huggingface.png",
		groq: "/logos/groq.png",
		grok: "/logos/grok.svg",
		gemini: "/logos/google.svg",
		google: "/logos/google.svg",
		deepseek: "/logos/deepseek.svg",
		cohere: "/logos/cohere.png",
		mistral: "/logos/mistral.png",
	};
	return logoMap[provider.toLowerCase()];
};

export const getProviderDisplayName = (provider: string): string => {
	const nameMap: { [key: string]: string } = {
		anthropic: "Anthropic",
		openai: "OpenAI",
		meta: "Meta",
		huggingface: "Hugging Face",
		groq: "Groq",
		grok: "Grok",
		gemini: "Gemini",
		google: "Google",
		deepseek: "DeepSeek",
		cohere: "Cohere",
		mistral: "Mistral",
	};
	return nameMap[provider.toLowerCase()] || provider;
};
