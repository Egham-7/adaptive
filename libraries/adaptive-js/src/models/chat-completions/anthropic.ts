import { z } from "zod";

export const AnthropicContentItemSchema = z.object({
  type: z.string(),
  text: z.string(),
});

export const AnthropicResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  type: z.string(),
  role: z.string(),
  content: z.array(AnthropicContentItemSchema),
  stop_reason: z.string(),
  stop_sequence: z.string().nullable().optional(),
  usage: z.object({
    input_tokens: z.number(),
    output_tokens: z.number(),
  }),
});

export type AnthropicResponse = z.infer<typeof AnthropicResponseSchema>;
