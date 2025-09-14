import { betterFetch } from "@better-fetch/fetch";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { env } from "@/env";
import { createTRPCRouter, publicProcedure } from "@/server/api/trpc";
import {
  selectModelRequestSchema,
  selectModelResponseSchema,
} from "@/types/select-model";

export const selectModelRouter = createTRPCRouter({
  selectModel: publicProcedure
    .input(
      z.object({
        request: selectModelRequestSchema,
      }),
    )
    .mutation(async ({ input }) => {
      try {
        const { request } = input;

        // Get backend URL from environment
        const backendUrl = env.ADAPTIVE_API_BASE_URL;

        // Construct URL properly to avoid double slashes
        const url = new URL("/v1/select-model", backendUrl);

        // Call the Go backend select-model endpoint with schema validation
        const { data, error } = await betterFetch(url.toString(), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(request),
          output: selectModelResponseSchema, // Runtime validation using Zod schema
        });

        if (error) {
          console.error("Backend select-model error:", error);
          throw new TRPCError({
            code: "INTERNAL_SERVER_ERROR",
            message: "Model selection failed",
          });
        }

        return data;
      } catch (error) {
        console.error("Select model error:", error);

        if (error instanceof TRPCError) {
          throw error;
        }

        throw new TRPCError({
          code: "INTERNAL_SERVER_ERROR",
          message: "Failed to select model",
        });
      }
    }),
});
