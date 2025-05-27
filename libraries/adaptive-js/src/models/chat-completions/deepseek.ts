import { z } from "zod";

export const DeepSeekLogProbsSchema = z.object({
  token_logprobs: z.array(z.number()),
  tokens: z.array(z.string()),
  top_logprobs: z.array(z.record(z.number())),
});

export const DeepSeekResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  object: z.string(),
  created: z.number(),
  choices: z.array(
    z.object({
      index: z.number(),
      message: z.object({
        role: z.string(),
        content: z.string(),
      }),
      finish_reason: z.string(),
      logprobs: DeepSeekLogProbsSchema.optional(),
    }),
  ),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
  }),
});

export type DeepSeekResponse = z.infer<typeof DeepSeekResponseSchema>;
