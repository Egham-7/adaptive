import { clerkClient } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "@/server/api/trpc";

const providerSchema = z.object({
  id: z.string(),
  name: z.string(),
  enabled: z.boolean(),
  costPerToken: z.number(),
});

const userPreferencesSchema = z.object({
  providers: z.array(providerSchema),
});

const userMetadataSchema = z.record(z.any());

const defaultProviders = [
  { id: "1", name: "OpenAI", enabled: true, costPerToken: 0.002 },
  { id: "2", name: "Anthropic", enabled: true, costPerToken: 0.003 },
  { id: "3", name: "Gemini", enabled: true, costPerToken: 0.001 },
  { id: "4", name: "Groq", enabled: true, costPerToken: 0.0025 },
  { id: "5", name: "Deepseek", enabled: true, costPerToken: 0.0018 },
  { id: "6", name: "HuggingFace", enabled: true, costPerToken: 0.0022 },
];

export const userRouter = createTRPCRouter({
  getPreferences: protectedProcedure.query(async ({ ctx }) => {
    const userId = ctx.clerkAuth.userId;
    if (!userId) {
      throw new TRPCError({ code: "UNAUTHORIZED" });
    }

    try {
      const user = await (await clerkClient()).users.getUser(userId);
      const preferences = user.privateMetadata.userPreferences as any;
      
      return {
        providers: preferences?.providers || defaultProviders,
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