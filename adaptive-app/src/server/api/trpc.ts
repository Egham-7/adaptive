// src/server/api/trpc.ts

import { initTRPC, TRPCError } from "@trpc/server";
import superjson from "superjson";
import { ZodError } from "zod";
// Import 'auth' from Clerk. We can alias it if needed, but 'auth' is fine.
import { auth as getClerkAuth } from "@clerk/nextjs/server";

import { db } from "@/server/db";

/**
 * 1. CONTEXT
 *
 * This section defines the "contexts" that are available in the backend API.
 * We will follow the Clerk documentation pattern for creating the tRPC context.
 */
export const createTRPCContext = async (opts: { headers: Headers }) => {
  // As per Clerk's tRPC documentation, await the auth() call.
  const clerkAuthResult = await getClerkAuth();

  return {
    db, // Your existing database instance
    clerkAuth: clerkAuthResult, // The entire auth object from Clerk
    userId: clerkAuthResult.userId, // Convenience access to userId from the auth object
    ...opts, // Pass along other context properties like headers
  };
};

// This type is crucial for initTRPC.
// It infers the return type of `createTRPCContext`, including the awaited auth object.
type Context = Awaited<ReturnType<typeof createTRPCContext>>;

/**
 * 2. INITIALIZATION
 *
 * This is where the tRPC API is initialized, connecting the context and transformer.
 */
const t = initTRPC.context<Context>().create({
  // Use the inferred Context type
  transformer: superjson,
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError:
          error.cause instanceof ZodError ? error.cause.flatten() : null,
      },
    };
  },
});

/**
 * Create a server-side caller.
 *
 * @see https://trpc.io/docs/server/server-side-calls
 */
export const createCallerFactory = t.createCallerFactory;

/**
 * 3. ROUTER & PROCEDURE (THE IMPORTANT BIT)
 *
 * These are the pieces you use to build your tRPC API. You should import these a lot in the
 * "/src/server/api/routers" directory.
 */

/**
 * This is how you create new routers and sub-routers in your tRPC API.
 *
 * @see https://trpc.io/docs/router
 */
export const createTRPCRouter = t.router;

/**
 * Middleware for timing procedure execution and adding an artificial delay in development.
 */
const timingMiddleware = t.middleware(async ({ next, path }) => {
  const start = Date.now();

  if (t._config.isDev) {
    const waitMs = Math.floor(Math.random() * 400) + 100;
    await new Promise((resolve) => setTimeout(resolve, waitMs));
  }

  const result = await next();

  const end = Date.now();
  console.log(`[TRPC] ${path} took ${end - start}ms to execute`);

  return result;
});

/**
 * Reusable middleware that enforces users are logged in before proceeding.
 * This aligns with the Clerk documentation's approach.
 */
const enforceUserIsAuthed = t.middleware(({ ctx, next }) => {
  // Access userId through ctx.clerkAuth, as this is where the auth object is stored.
  if (!ctx.clerkAuth.userId) {
    throw new TRPCError({ code: "UNAUTHORIZED", message: "Not authenticated" });
  }
  // If the user is authenticated, refine the context for TypeScript.
  // This ensures that in protected procedures, ctx.clerkAuth and ctx.clerkAuth.userId are known to be present.
  return next({
    ctx: {
      ...ctx,
      clerkAuth: ctx.clerkAuth, // Pass the auth object along, now known to have a userId
    },
  });
});

/**
 * Public (unauthenticated) procedure
 *
 * This is the base piece you use to build new queries and mutations on your tRPC API. It does not
 * guarantee that a user querying is authorized, but you can still access user session data if they
 * are logged in via `ctx.clerkAuth.userId`.
 */
export const publicProcedure = t.procedure.use(timingMiddleware);

/**
 * Protected (authenticated) procedure
 *
 * This is the base piece you use to build new queries and mutations on your tRPC API that require
 * a user to be authenticated. It ensures that `ctx.clerkAuth.userId` is available and non-null.
 */
export const protectedProcedure = t.procedure
  .use(timingMiddleware) // Apply timing first
  .use(enforceUserIsAuthed); // Then enforce authentication
