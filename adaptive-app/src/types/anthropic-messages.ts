import type {
	Usage as AnthropicSDKUsage,
	ContentBlock,
	Message,
	MessageCreateParams,
	MessageParam,
	MessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages";
import { z } from "zod";
import type { ModelCapability } from "./models";
import type { Provider, ProviderConfig } from "./providers";

// Use Anthropic SDK types as base
export type AnthropicMessage = Message;
export type AnthropicUsage = AnthropicSDKUsage;
export type AnthropicStreamEvent = MessageStreamEvent;
export type AnthropicMessageParam = MessageParam;
export type AnthropicContentBlock = ContentBlock;
export type AnthropicMessageCreateParams = MessageCreateParams;

// Create schema that matches Anthropic's MessageCreateParams structure
const anthropicMessageParamSchema = z.object({
	role: z.enum(["user", "assistant"]),
	content: z.union([
		z.string(),
		z.array(z.any()), // Content blocks (text, image, tool_result, tool_use, etc.)
	]),
});

const anthropicMetadataSchema = z.object({
	user_id: z.string().optional(),
});

const anthropicToolChoiceSchema = z.union([
	z.literal("auto"),
	z.literal("any"),
	z.literal("none"),
	z.object({
		type: z.literal("tool"),
		name: z.string(),
	}),
]);

// Base Anthropic MessageCreateParams schema (matches the SDK exactly)
const baseAnthropicMessageSchema = z.object({
	// Required fields
	model: z.string(),
	max_tokens: z.number().int().min(1),
	messages: z.array(anthropicMessageParamSchema),

	// Optional fields that match Anthropic SDK
	system: z.union([z.string(), z.array(z.any())]).optional(),
	temperature: z.number().min(0).max(2).optional(),
	top_k: z.number().int().min(0).optional(),
	top_p: z.number().min(0).max(1).optional(),
	stop_sequences: z.array(z.string()).max(4).optional(),
	stream: z.boolean().optional(),
	metadata: anthropicMetadataSchema.optional(),
	tools: z.array(z.any()).optional(),
	tool_choice: anthropicToolChoiceSchema.optional(),
});

// Create reusable provider config schema
const providerConfigSchema = z.object({
	api_key: z.string().optional(),
	base_url: z.string().optional(),
	auth_type: z.enum(["bearer", "api_key", "basic", "custom"]).optional(),
	auth_header_name: z.string().optional(),
	health_endpoint: z.string().optional(),
	rate_limit_rpm: z.number().optional(),
	timeout_ms: z.number().optional(),
	retry_config: z.record(z.string(), z.unknown()).optional(),
	headers: z.record(z.string(), z.string()).optional(),
});

// Extended schema that adds our Adaptive-specific extensions
export const anthropicMessagesRequestSchema = baseAnthropicMessageSchema.extend(
	{
		// Adaptive extensions
		provider_configs: z.record(z.string(), providerConfigSchema).optional(),
		model_router: z
			.object({
				models: z.array(z.any()).optional(),
				cost_bias: z.number().min(0).max(1).optional(),
				complexity_threshold: z.number().optional(),
				token_threshold: z.number().int().optional(),
			})
			.optional(),
		semantic_cache: z
			.object({
				enabled: z.boolean(),
				semantic_threshold: z.number().optional(),
			})
			.optional(),
		prompt_cache: z
			.object({
				enabled: z.boolean().optional(),
				ttl: z.number().int().optional(),
			})
			.optional(),
		fallback: z
			.object({
				enabled: z.boolean().optional(),
				mode: z.enum(["sequential", "race"]).optional(),
			})
			.optional(),
	},
);

// Enhanced response types that extend Anthropic's Message and Usage with provider info
export interface AdaptiveAnthropicMessage extends Message {
	provider?: Provider;
}

export interface AdaptiveAnthropicUsage extends AnthropicSDKUsage {
	cache_tier?: "semantic_exact" | "semantic_similar" | "prompt_response";
}

export interface AdaptiveAnthropicResponse extends Omit<Message, "usage"> {
	provider?: Provider;
	usage?: AdaptiveAnthropicUsage;
}

// Base request type that matches Anthropic SDK exactly
export type BaseAnthropicMessageRequest = z.infer<
	typeof baseAnthropicMessageSchema
>;

// Extended request type with our Adaptive extensions
export type AnthropicMessagesRequest = z.infer<
	typeof anthropicMessagesRequestSchema
>;

// Ensure type compatibility with Anthropic SDK
export type AdaptiveMessageCreateParams = Omit<
	AnthropicMessageCreateParams,
	"stream"
> & {
	// Adaptive extensions
	provider_configs?: Record<string, ProviderConfig>;
	model_router?: {
		models?: ModelCapability[];
		cost_bias?: number;
		complexity_threshold?: number;
		token_threshold?: number;
	};
	semantic_cache?: {
		enabled: boolean;
		semantic_threshold?: number;
	};
	prompt_cache?: {
		enabled?: boolean;
		ttl?: number;
	};
	fallback?: {
		enabled?: boolean;
		mode?: "sequential" | "race";
	};
	stream?: boolean; // Make stream optional
};
