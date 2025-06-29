import { z } from "zod";

export const messageRoleSchema = z.enum(["system", "user", "assistant"]);

// JSON value schema for Prisma compatibility
type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

const jsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.string(),
    z.number(),
    z.boolean(),
    z.array(jsonValueSchema),
    z.record(jsonValueSchema),
  ]),
);

const uiPartSchema = z.union([
  z.object({
    type: z.literal("text"),
    text: z.string(),
    state: z.enum(["streaming", "done"]).optional(),
  }),
  z.object({
    type: z.literal("file"),
    mediaType: z.string(),
    filename: z.string().optional(),
    url: z.string(),
  }),
  z.object({
    type: z.string().regex(/^tool-.+/),
    toolCallId: z.string(),
    state: z.enum([
      "input-streaming",
      "input-available",
      "output-available",
      "output-error",
    ]),
    input: jsonValueSchema.optional(),
    output: jsonValueSchema.optional(),
    errorText: z.string().optional(),
    providerExecuted: z.boolean().optional(),
  }),
  z
    .object({
      type: z.string(),
    })
    .passthrough(),
]);

// --- Message Schemas ---
export const createMessageSchema = z.object({
  conversationId: z.number(),
  role: messageRoleSchema,
  id: z.string().optional(),
  createdAt: z.union([z.date(), z.string().datetime()]).optional(),
  metadata: jsonValueSchema.nullable().optional(),
  annotations: z.array(jsonValueSchema).nullable().optional(),
  parts: z.array(uiPartSchema),
});

export const updateMessageSchema = z.object({
  id: z.string(),
  metadata: jsonValueSchema.nullable().optional(),
  annotations: z.array(jsonValueSchema).nullable().optional(),
  parts: z.array(uiPartSchema).nullable().optional(),
});

// --- Conversation Schemas ---
export const createConversationSchema = z.object({
  title: z.string().min(1),
  pinned: z.boolean().optional(),
});

export const updateConversationSchema = z.object({
  id: z.number(), // Conversation ID to update
  title: z.string().min(1).optional(),
  pinned: z.boolean().optional(),
});

export const getConversationsOptionsSchema = z.object({
  pinned: z.boolean().optional(),
  includeMessages: z.boolean().optional(),
});
