import type {
	ChatCompletionCreateParamsBase,
	ChatCompletion as OpenAIChatCompletion,
	ChatCompletionChunk as OpenAIChatCompletionChunk,
} from "openai/resources/chat/completions";

// Extend OpenAI types with our custom fields
export interface ChatCompletionRequest extends ChatCompletionCreateParamsBase {
	provider_constraint?: string[];
	cost_bias?: number;
}

// Extend OpenAI types with provider field
export interface ChatCompletion extends OpenAIChatCompletion {
	provider: string;
}

export interface ChatCompletionChunk extends OpenAIChatCompletionChunk {
	provider: string;
}
