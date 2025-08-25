import type {
	Usage as AnthropicSDKUsage,
	ContentBlock,
	Message,
	MessageParam,
	MessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages";
import { z } from "zod";
import type { Provider } from "./chat-completion";
import { providerConfigSchema } from "./chat-completion";

// Use Anthropic SDK types as base
export type AnthropicMessage = Message;
export type AnthropicUsage = AnthropicSDKUsage;
export type AnthropicStreamEvent = MessageStreamEvent;
export type AnthropicMessageParam = MessageParam;
export type AnthropicContentBlock = ContentBlock;

// Enhanced request schema that extends Anthropic's MessageCreateParams
export const anthropicMessagesRequestSchema = z.object({
	// Required Anthropic parameters
	model: z.string(),
	max_tokens: z.number().int().min(1),
	messages: z.array(
		z.object({
			role: z.enum(["user", "assistant"]),
			content: z.union([
				z.string(),
				z.array(z.any()), // Content blocks (text, image, etc.)
			]),
		}),
	),

	// Optional Anthropic parameters
	system: z.union([z.string(), z.array(z.any())]).optional(),
	temperature: z.number().min(0).max(2).optional(),
	top_p: z.number().min(0).max(1).optional(),
	top_k: z.number().int().min(0).optional(),
	stop_sequences: z.array(z.string()).max(4).optional(),
	stream: z.boolean().optional(),
	metadata: z
		.object({
			user_id: z.string().optional(),
		})
		.optional(),
	tools: z.array(z.any()).optional(),
	tool_choice: z
		.union([
			z.literal("auto"),
			z.literal("none"),
			z.object({
				type: z.literal("tool"),
				name: z.string(),
			}),
		])
		.optional(),

	// Adaptive extensions - reuse the shared provider config schema
	provider_configs: z.record(z.string(), providerConfigSchema).optional(),

	// Adaptive system extensions
	protocol_manager: z
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
});

// Enhanced response type that extends Anthropic's Message with provider info
export interface AnthropicResponse extends Message {
	provider: Provider;
}

// Request type - use the zod-inferred type which will have all the properties
export type AnthropicMessagesRequest = z.infer<
	typeof anthropicMessagesRequestSchema
>;
