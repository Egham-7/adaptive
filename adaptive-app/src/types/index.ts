import type { inferRouterInputs, inferRouterOutputs } from "@trpc/server";
import type { AppRouter } from "@/server/api/root";

/**
 * A set of helper types to infer the inputs and outputs of your tRPC API.
 * These are the building blocks for all other tRPC types.
 */
export type RouterOutputs = inferRouterOutputs<AppRouter>;
export type RouterInputs = inferRouterInputs<AppRouter>;

export * from "./api_keys";
// Re-export all domain-specific types for easy access
export * from "./conversations";
export * from "./credits";
export * from "./messages";
export * from "./organizations";
export * from "./projects";
