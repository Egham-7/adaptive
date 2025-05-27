import { z } from "zod";

export const GroqResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  created: z.number(),
  object: z.string().optional(),
  choices: z.array(
    z.object({
      index: z.number(),
      message: z.object({
        role: z.string(),
        content: z.string(),
      }),
      finish_reason: z.string(),
    }),
  ),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
  }),
});

export type GroqResponse = z.infer<typeof GroqResponseSchema>;
