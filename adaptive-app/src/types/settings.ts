import { type RouterOutputs } from "@/trpc/react";

export type Provider = RouterOutputs["user"]["getPreferences"]["providers"][0];
export type UserPreferences = RouterOutputs["user"]["getPreferences"];
export type UserMetadata = Partial<Pick<UserPreferences, "displayName" | "responseStyle" | "language" | "jobRole" | "personalPreferences" | "fullName" | "preferredName">>;
