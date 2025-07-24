import type {
	ChatCompletionCreateParamsBase,
	ChatCompletion as OpenAIChatCompletion,
	ChatCompletionChunk as OpenAIChatCompletionChunk,
} from "openai/resources/chat/completions";

// Define supported providers as a union type
export type Provider =
	| "openai"
	| "anthropic"
	| "gemini"
	| "groq"
	| "deepseek"
	| "huggingface"
	| "grok"
	| "adaptive"
	| null;

// Extend OpenAI types with our custom fields
export interface ChatCompletionRequest extends ChatCompletionCreateParamsBase {
	provider_constraint?: string[];
	cost_bias?: number;
}

// Extend OpenAI types with provider field
export interface ChatCompletion extends OpenAIChatCompletion {
	provider: Provider;
}

export interface ChatCompletionChunk extends OpenAIChatCompletionChunk {
	provider: string;
}
