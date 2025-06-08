import { z } from "zod";

// Define JSONValue type to match AI SDK exactly
type JSONValue =
  | null
  | string
  | number
  | boolean
  | {
      [key: string]: JSONValue;
    }
  | Array<JSONValue>;

// JSONValue schema (recursive)
export const jsonValueSchema: z.ZodType<JSONValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.string(),
    z.number(),
    z.boolean(),
    z.array(jsonValueSchema),
    z.record(z.string(), jsonValueSchema),
  ]),
);

// Attachment schema
export const attachmentSchema = z.object({
  name: z.string().optional(),
  contentType: z.string().optional(),
  url: z.string(),
});

// LanguageModelV1Source schema (matching AI SDK)
const languageModelV1SourceSchema = z.object({
  sourceType: z.literal("url"),
  id: z.string(),
  url: z.string(),
  name: z.string().optional(),
  description: z.string().optional(),
});

// Tool Call schema (matching AI SDK ToolCall)
const toolCallSchema = z.object({
  toolCallId: z.string(),
  toolName: z.string(),
  args: z.record(z.string(), jsonValueSchema),
});

// Tool Result schema (matching AI SDK ToolResult)
const toolResultSchema = z.object({
  toolCallId: z.string(),
  toolName: z.string(),
  args: z.record(z.string(), jsonValueSchema),
  result: jsonValueSchema,
});

// Tool Invocation schema (matching AI SDK ToolInvocation)
const toolInvocationSchema = z.union([
  // partial-call state
  toolCallSchema.extend({
    state: z.literal("partial-call"),
    step: z.number().optional(),
  }),
  // call state
  toolCallSchema.extend({
    state: z.literal("call"),
    step: z.number().optional(),
  }),
  // result state
  toolResultSchema.extend({
    state: z.literal("result"),
    step: z.number().optional(),
  }),
]);

// Reasoning detail schemas
const reasoningTextDetailSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
  signature: z.string().optional(),
});

const reasoningRedactedDetailSchema = z.object({
  type: z.literal("redacted"),
  data: z.string(),
});

// UI Part schemas (matching exact AI SDK types)
const textUIPartSchema = z.object({
  type: z.literal("text"),
  text: z.string(),
});

const reasoningUIPartSchema = z.object({
  type: z.literal("reasoning"),
  reasoning: z.string(),
  details: z.array(
    z.union([reasoningTextDetailSchema, reasoningRedactedDetailSchema]),
  ),
});

const toolInvocationUIPartSchema = z.object({
  type: z.literal("tool-invocation"),
  toolInvocation: toolInvocationSchema,
});

const sourceUIPartSchema = z.object({
  type: z.literal("source"),
  source: languageModelV1SourceSchema,
});

const fileUIPartSchema = z.object({
  type: z.literal("file"),
  mimeType: z.string(),
  data: z.string(),
});

const stepStartUIPartSchema = z.object({
  type: z.literal("step-start"),
});

// Combined UI Part schema
export const uiPartSchema = z.union([
  textUIPartSchema,
  reasoningUIPartSchema,
  toolInvocationUIPartSchema,
  sourceUIPartSchema,
  fileUIPartSchema,
  stepStartUIPartSchema,
]);

// Main SDK Message schema
export const sdkMessageSchema = z.object({
  id: z.string(),
  createdAt: z
    .union([z.date(), z.string().datetime()])
    .optional()
    .transform((val) => {
      if (typeof val === "string") {
        return new Date(val);
      }
      return val;
    }),
  content: z.string(),
  reasoning: z.string().optional(),
  experimental_attachments: z.array(attachmentSchema).optional(),
  role: z.enum(["system", "user", "assistant", "data"]),
  data: jsonValueSchema.optional(),
  annotations: z.array(jsonValueSchema).optional(),
  toolInvocations: z.array(toolInvocationSchema).optional(),
  parts: z.array(uiPartSchema).optional(),
});

// Type inference
export type SDKMessage = z.infer<typeof sdkMessageSchema>;

// Chat request schema
export const chatRequestSchema = z.object({
  message: sdkMessageSchema,
  id: z.string(),
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;

// Create message schema (for messages without required id)
export const createMessageSchema = sdkMessageSchema.extend({
  id: z.string().optional(),
});

export type CreateMessage = z.infer<typeof createMessageSchema>;

// UI Message schema (with required parts)
export const uiMessageSchema = sdkMessageSchema.extend({
  parts: z.array(uiPartSchema),
});

export type UIMessage = z.infer<typeof uiMessageSchema>;

// Message array schema
export const messagesArraySchema = z.array(sdkMessageSchema);

// Chat request options schema
export const chatRequestOptionsSchema = z.object({
  headers: z.union([z.record(z.string()), z.instanceof(Headers)]).optional(),
  body: z.record(z.string(), jsonValueSchema).optional(),
  data: jsonValueSchema.optional(),
  experimental_attachments: z.array(attachmentSchema).optional(),
  allowEmptySubmit: z.boolean().optional(),
});

export type ChatRequestOptions = z.infer<typeof chatRequestOptionsSchema>;

// Export the JSONValue type for use elsewhere
export type { JSONValue };
