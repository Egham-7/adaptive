import { type RouterOutputs } from "@/trpc/react";

export type Provider = RouterOutputs["user"]["getPreferences"]["providers"][0];
export type UserPreferences = RouterOutputs["user"]["getPreferences"];
