import { z } from "zod";

// Tool definition schema
export const toolSchema = z.object({
	type: z.literal("function"),
	function: z.object({
		name: z.string(),
		description: z.string().optional(),
		parameters: z.record(z.string(), z.any()).optional(),
	}),
});

// Tool call schema
export const toolCallSchema = z.object({
	id: z.string(),
	type: z.literal("function"),
	function: z.object({
		name: z.string(),
		arguments: z.string(),
	}),
});

// Tool choice schema
export const toolChoiceSchema = z.union([
	z.literal("none"),
	z.literal("auto"),
	z.object({
		type: z.literal("function"),
		function: z.object({
			name: z.string(),
		}),
	}),
]);

// TypeScript types derived from Zod schemas
export type Tool = z.infer<typeof toolSchema>;
export type ToolCall = z.infer<typeof toolCallSchema>;
export type ToolChoice = z.infer<typeof toolChoiceSchema>;
