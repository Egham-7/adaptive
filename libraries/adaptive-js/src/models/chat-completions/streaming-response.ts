import { z } from "zod";
import { ProviderNameEnum } from "./schema";

export const ChatCompletionStreamingResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  provider: ProviderNameEnum,
  choices: z.array(
    z.object({
      delta: z.any().optional(),
      text: z.string().optional(),
      content: z.string().optional(),
      message: z.any().optional(),
    }),
  ),
  object: z.string().optional(),
  created: z.number().optional(),
});

export type ChatCompletionStreamingResponse = z.infer<
  typeof ChatCompletionStreamingResponseSchema
>;
