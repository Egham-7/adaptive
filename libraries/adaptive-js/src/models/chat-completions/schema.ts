import { z } from "zod";
import { OpenAIResponseSchema } from "./openai";
import { AnthropicResponseSchema } from "./anthropic";
import { GroqResponseSchema } from "./groq";
import { DeepSeekResponseSchema } from "./deepseek";

// 🎭 Shared Enum
export const ProviderNameEnum = z.enum([
  "openai",
  "anthropic",
  "groq",
  "deepseek",
]);
export type ProviderName = z.infer<typeof ProviderNameEnum>;

// 💬 Message Role
export const MessageRoleSchema = z.union([
  z.literal("user"),
  z.literal("assistant"),
  z.literal("system"),
  z.literal("tool"),
]);
export type MessageRole = z.infer<typeof MessageRoleSchema>;

// 💬 Chat Message
export const MessageSchema = z.object({
  role: MessageRoleSchema,
  content: z.string(),
  tool_call_id: z.string().optional(),
  function_call: z.record(z.string()).optional(),
  tool_calls: z.array(z.record(z.unknown())).optional(),
});
export type Message = z.infer<typeof MessageSchema>;

// 🤖 Chat Completion Response (non-streaming)
export const ChatAPIResponseSchema = z.discriminatedUnion("provider", [
  z.object({
    provider: z.literal("openai"),
    response: OpenAIResponseSchema,
  }),
  z.object({
    provider: z.literal("anthropic"),
    response: AnthropicResponseSchema,
  }),
  z.object({
    provider: z.literal("groq"),
    response: GroqResponseSchema,
  }),
  z.object({
    provider: z.literal("deepseek"),
    response: DeepSeekResponseSchema,
  }),
]);

export type ChatAPIResponse = z.infer<typeof ChatAPIResponseSchema>;
