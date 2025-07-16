import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import type { NextRequest } from "next/server";

import { env } from "@/env";
import {
	createCacheHeaders,
	createCacheInvalidationHeaders,
} from "@/server/api/cache";
import { appRouter } from "@/server/api/root";
import { createTRPCContext } from "@/server/api/trpc";

/**
 * This wraps the `createTRPCContext` helper and provides the required context for the tRPC API when
 * handling a HTTP request (e.g. when you make requests from Client Components).
 */
const createContext = async (req: NextRequest) => {
	return createTRPCContext({
		headers: req.headers,
	});
};

const handler = (req: NextRequest) =>
	fetchRequestHandler({
		endpoint: "/api/trpc",
		req,
		router: appRouter,
		createContext: () => createContext(req),
		onError:
			env.NODE_ENV === "development"
				? ({ path, error }) => {
						console.error(
							`❌ tRPC failed on ${path ?? "<no-path>"}: ${error.message}`,
						);
					}
				: undefined,
		responseMeta(opts) {
			const { ctx, errors, type } = opts;

			// Handle mutations - add cache invalidation headers
			if (type === "mutation" && errors.length === 0) {
				// For mutations, use broad invalidation
				return {
					headers: createCacheInvalidationHeaders([
						"conversations",
						"messages",
						"projects",
						"users",
					]),
				};
			}

			// Only cache successful query requests
			if (type !== "query" || errors.length > 0) {
				return {};
			}

			// Don't cache if there are any auth errors or if user is not authenticated
			if (!ctx?.clerkAuth?.userId) {
				return {};
			}

			// For queries, use a simple cache approach
			return {
				headers: createCacheHeaders(),
			};
		},
	});

export { handler as GET, handler as POST };
