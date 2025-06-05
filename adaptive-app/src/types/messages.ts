import type { RouterOutputs, RouterInputs } from "./index";

// ---- Output Types ----

/**
 * The type for a single message.
 */
export type Message = RouterOutputs["messages"]["getById"];

// ---- Input Types ----

/**
 * The type for the input when creating a new message.
 */
export type MessageCreateInput = RouterInputs["messages"]["create"];

/**
 * The type for the input when updating an existing message.
 */
export type MessageUpdateInput = RouterInputs["messages"]["update"];
