import type { z } from "zod";
import type { RouterInputs, RouterOutputs } from "@/trpc/react";
import type { createMessageSchema, updateMessageSchema } from "@/types/chat";

export type Message = RouterOutputs["messages"]["getById"];
export type MessageCreateInput = RouterInputs["messages"]["create"];
export type MessageUpdateInput = RouterInputs["messages"]["update"];
export type MessageListInput = RouterInputs["messages"]["listByConversation"];
export type MessageBatchUpsertInput = RouterInputs["messages"]["batchUpsert"];

// Schema-derived types for internal router use
export type CreateMessageInput = z.infer<typeof createMessageSchema>;
export type UpdateMessageInput = z.infer<typeof updateMessageSchema>;
