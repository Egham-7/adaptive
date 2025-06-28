import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";
import { clerkClient } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import { z } from "zod";

const providerSchema = z.object({
	id: z.string(),
	name: z.string(),
	enabled: z.boolean(),
	costPerToken: z.number(),
});

const userPreferencesSchema = z.object({
	providers: z.array(providerSchema),
});

const userMetadataSchema = z.object({
	displayName: z.string().optional(),
	responseStyle: z
		.enum(["concise", "balanced", "detailed", "creative"])
		.optional(),
	language: z
		.enum(["english", "spanish", "french", "german", "chinese"])
		.optional(),
	fullName: z.string().optional(),
	preferredName: z.string().optional(),
	jobRole: z.string().optional(),
	personalPreferences: z.string().optional(),
});

const defaultProviders = [
	{
		id: "openai",
		name: "OpenAI",
		enabled: true,
		costPerToken: 0.002,
		logoPath: "/logos/openai.webp",
	},
	{
		id: "anthropic",
		name: "Anthropic",
		enabled: true,
		costPerToken: 0.003,
		logoPath: "/logos/anthropic.jpeg",
	},
	{
		id: "google",
		name: "Gemini",
		enabled: true,
		costPerToken: 0.001,
		logoPath: "/logos/google.svg",
	},
	{
		id: "groq",
		name: "Groq",
		enabled: true,
		costPerToken: 0.0025,
		logoPath: "/logos/groq.png",
	},
	{
		id: "deepseek",
		name: "Deepseek",
		enabled: true,
		costPerToken: 0.0018,
		logoPath: "/logos/deepseek.svg",
	},
	{
		id: "huggingface",
		name: "HuggingFace",
		enabled: true,
		costPerToken: 0.0022,
		logoPath: "/logos/huggingface.png",
	},
];

export const userRouter = createTRPCRouter({
	getPreferences: protectedProcedure.query(async ({ ctx }) => {
		const userId = ctx.clerkAuth.userId;
		if (!userId) {
			throw new TRPCError({ code: "UNAUTHORIZED" });
		}

		try {
			const user = await (await clerkClient()).users.getUser(userId);
			const preferences = user.privateMetadata.userPreferences as
				| z.infer<typeof userPreferencesSchema>
				| undefined;
			const publicMetadata = user.publicMetadata as
				| z.infer<typeof userMetadataSchema>
				| undefined;

			return {
				providers: preferences?.providers || defaultProviders,
				displayName: publicMetadata?.displayName,
				responseStyle: publicMetadata?.responseStyle || "balanced",
				language: publicMetadata?.language || "english",
				fullName: publicMetadata?.fullName,
				preferredName: publicMetadata?.preferredName,
				jobRole: publicMetadata?.jobRole,
				personalPreferences: publicMetadata?.personalPreferences,
			};
		} catch (error) {
			console.error("Error fetching user preferences:", error);
			throw new TRPCError({
				code: "INTERNAL_SERVER_ERROR",
				message: "Failed to fetch user preferences",
			});
		}
	}),

	updatePreferences: protectedProcedure
		.input(userPreferencesSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				await (await clerkClient()).users.updateUserMetadata(userId, {
					privateMetadata: {
						userPreferences: input,
					},
				});

				return { success: true };
			} catch (error) {
				console.error("Error updating user preferences:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update user preferences",
				});
			}
		}),

	updateMetadata: protectedProcedure
		.input(userMetadataSchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			try {
				await (await clerkClient()).users.updateUserMetadata(userId, {
					publicMetadata: input,
				});

				return { success: true };
			} catch (error) {
				console.error("Error updating user metadata:", error);
				throw new TRPCError({
					code: "INTERNAL_SERVER_ERROR",
					message: "Failed to update user metadata",
				});
			}
		}),
});
