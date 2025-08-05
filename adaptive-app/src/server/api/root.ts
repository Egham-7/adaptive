import { createCallerFactory, createTRPCRouter } from "@/server/api/trpc";
import { adminRouter } from "./routers/admin";
import { apiKeysRouter } from "./routers/api_keys";
import { conversationRouter } from "./routers/conversations";
import { creditsRouter } from "./routers/credits";
import { messageRouter } from "./routers/message";
import { organizationsRouter } from "./routers/organizations";
import { modelPricingRouter } from "./routers/pricing";
import { projectsRouter } from "./routers/projects";
import { subscriptionRouter } from "./routers/subscription";
import { supportRouter } from "./routers/support";
import { usageRouter } from "./routers/usage";
import { userRouter } from "./routers/user";

/**
 * This is the primary router for your server.
 *
 * All routers added in /api/routers should be manually added here.
 */
export const appRouter = createTRPCRouter({
	admin: adminRouter,
	conversations: conversationRouter,
	messages: messageRouter,
	api_keys: apiKeysRouter,
	credits: creditsRouter,
	organizations: organizationsRouter,
	modelPricing: modelPricingRouter,
	projects: projectsRouter,
	subscription: subscriptionRouter,
	support: supportRouter,
	usage: usageRouter,
	user: userRouter,
});

// export type definition of API
export type AppRouter = typeof appRouter;

/**
 * Create a server-side caller for the tRPC API.
 * @example
 * const trpc = createCaller(createContext);
 * const res = await trpc.post.all();
 *       ^? Post[]
 */
export const createCaller = createCallerFactory(appRouter);
