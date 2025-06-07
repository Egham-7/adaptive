import type { RouterInputs, RouterOutputs } from "@/trpc/react";

export type Message = RouterOutputs["messages"]["getById"];
export type MessageCreateInput = RouterInputs["messages"]["create"];
export type MessageUpdateInput = RouterInputs["messages"]["update"];
export type MessageListInput = RouterInputs["messages"]["listByConversation"];
export type MessageBatchUpsertInput = RouterInputs["messages"]["batchUpsert"];
