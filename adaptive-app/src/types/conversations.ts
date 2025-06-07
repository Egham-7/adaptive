import type { RouterInputs, RouterOutputs } from "./index";

// ---- Output Types ----

/**
 * The type for a single item in the conversation list.
 * Includes the conversation details and its most recent message.
 */
export type ConversationListItem =
	RouterOutputs["conversations"]["list"][number];

/**
 * The type for a single, fully-detailed conversation.
 * Includes all of its messages.
 */
export type ConversationDetails = RouterOutputs["conversations"]["getById"];

// ---- Input Types ----

/**
 * The type for the input when creating a new conversation.
 */
export type ConversationCreateInput = RouterInputs["conversations"]["create"];

/**
 * The type for the input when updating a conversation's title.
 */
export type ConversationUpdateInput = RouterInputs["conversations"]["update"];
