import {
  type Context,
  createTRPCRouter,
  protectedProcedure,
} from "@/server/api/trpc";
import { createFetch, createSchema } from "@better-fetch/fetch";
import { TRPCError } from "@trpc/server";
import { z } from "zod";

// Input/Output schemas
const createAPIKeySchema = z.object({
  name: z.string().min(1, "Name is required"),
  status: z.string().default("active"),
  expires_at: z.string().optional(),
});

const updateAPIKeySchema = z.object({
  id: z.string().uuid("Invalid API key ID"),
  name: z.string().min(1, "Name is required"),
  status: z.string(),
});

const apiKeySchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  expires_at: z.string().nullable(),
  user_id: z.string(),
  key_preview: z.string(),
});

const createAPIKeyResponseSchema = z.object({
  api_key: apiKeySchema,
  full_api_key: z.string(),
});

// Type definitions for better type safety
type APIKey = z.infer<typeof apiKeySchema>;
type CreateAPIKeyResponse = z.infer<typeof createAPIKeyResponseSchema>;
type APIKeyList = APIKey[];

// Create Better Fetch schema for API endpoints
const apiSchema = createSchema({
  "/api_keys/:userId": {
    output: z.array(apiKeySchema),
  },
  "/api_keys/:id": {
    output: apiKeySchema,
  },
  "/api_keys": {
    input: createAPIKeySchema,
    output: createAPIKeyResponseSchema,
  },
  "/api_keys/:id/update": {
    input: z.object({
      name: z.string(),
      status: z.string(),
    }),
    output: apiKeySchema,
  },
  "/api_keys/:id/delete": {
    output: z.object({ success: z.boolean() }),
  },
});

// Helper to extract error message from response
const getErrorMessage = async (response?: Response): Promise<string> => {
  if (!response) return "An error occurred";

  try {
    const errorData = await response.json();
    return errorData.error ?? errorData.message ?? "An error occurred";
  } catch {
    return "An error occurred";
  }
};

// Helper to create TRPC error based on status
const createTRPCErrorFromStatus = (status: number, message: string) => {
  const errorMap = {
    404: { code: "NOT_FOUND" as const, fallback: "Resource not found" },
    401: { code: "UNAUTHORIZED" as const, fallback: "Unauthorized" },
    400: { code: "BAD_REQUEST" as const, fallback: "Bad request" },
  } as const;

  const errorConfig = errorMap[status as keyof typeof errorMap];

  return new TRPCError({
    code: errorConfig?.code ?? "INTERNAL_SERVER_ERROR",
    message: message || errorConfig?.fallback || "Internal server error",
  });
};

// Create Better Fetch instance
const createBackendFetch = (authToken?: string) => {
  return createFetch({
    baseURL: process.env.ADAPTIVE_API_BASE_URL || "http://localhost:8080",
    schema: apiSchema,
    throw: true,
    headers: {
      "Content-Type": "application/json",
      ...(authToken && { Authorization: `Bearer ${authToken}` }),
    },
    timeout: 10000,
    onError: async (error) => {
      const status = error.response?.status ?? 500;
      const errorMessage = await getErrorMessage(error.response);
      throw createTRPCErrorFromStatus(status, errorMessage);
    },
  });
};

// Helper to get auth token from context
const getAuthToken = async (ctx: Context) => {
  return ctx.clerkAuth?.getToken();
};

// Helper to validate auth and get token
const validateAuthAndGetToken = async (ctx: Context) => {
  const userId = ctx.clerkAuth.userId;
  if (!userId) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "User not authenticated",
    });
  }

  const authToken = await getAuthToken(ctx);
  if (!authToken) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "No authentication token found",
    });
  }

  return { userId, authToken };
};

export const apiKeysRouter = createTRPCRouter({
  // Get all API keys for the current user
  list: protectedProcedure
    .output(z.array(apiKeySchema))
    .query(async ({ ctx }): Promise<APIKeyList> => {
      const { userId, authToken } = await validateAuthAndGetToken(ctx);
      const $fetch = createBackendFetch(authToken);
      const result = await $fetch(`/api/api_keys/${userId}`);
      return result as APIKeyList;
    }),

  // Get a specific API key by ID
  getById: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .output(apiKeySchema)
    .query(async ({ ctx, input }): Promise<APIKey> => {
      const { authToken } = await validateAuthAndGetToken(ctx);
      const $fetch = createBackendFetch(authToken);
      const result = await $fetch(`/api/api_keys/${input.id}`);
      return result as APIKey;
    }),

  // Create a new API key
  create: protectedProcedure
    .input(createAPIKeySchema)
    .output(createAPIKeyResponseSchema)
    .mutation(async ({ ctx, input }): Promise<CreateAPIKeyResponse> => {
      const { authToken } = await validateAuthAndGetToken(ctx);
      const $fetch = createBackendFetch(authToken);
      const result = await $fetch("/api/api_keys", {
        method: "POST",
        body: input,
      });
      return result as CreateAPIKeyResponse;
    }),

  // Update an existing API key
  update: protectedProcedure
    .input(updateAPIKeySchema)
    .output(apiKeySchema)
    .mutation(async ({ ctx, input }): Promise<APIKey> => {
      const { authToken } = await validateAuthAndGetToken(ctx);
      const { id, ...updateData } = input;
      const $fetch = createBackendFetch(authToken);
      const result = await $fetch(`/api/api_keys/${id}`, {
        method: "PUT",
        body: updateData,
      });
      return result as APIKey;
    }),

  // Delete an API key
  delete: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .output(z.object({ success: z.boolean() }))
    .mutation(async ({ ctx, input }): Promise<{ success: boolean }> => {
      const { authToken } = await validateAuthAndGetToken(ctx);
      const $fetch = createBackendFetch(authToken);
      await $fetch(`/api/api_keys/${input.id}`, {
        method: "DELETE",
      });
      return { success: true };
    }),

  // Verify an API key
  verify: protectedProcedure
    .input(z.object({ apiKey: z.string() }))
    .output(z.object({ valid: z.boolean() }))
    .query(async ({ input }): Promise<{ valid: boolean }> => {
      const $fetch = createFetch({
        baseURL: process.env.BACKEND_API_URL || "http://localhost:8080",
        throw: false,
        headers: {
          "X-Stainless-API-Key": input.apiKey,
        },
      });

      try {
        const { error } = await $fetch("/api/verify");
        return { valid: !error };
      } catch {
        return { valid: false };
      }
    }),
});
