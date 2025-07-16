import redis from "@/lib/redis";

export async function withCache<T>(
	key: string,
	fetchFn: () => Promise<T>,
	ttlSeconds = 300,
): Promise<T> {
	try {
		// Try to get cached result
		const cached = await redis.get(key);
		if (cached) {
			console.log(`[CACHE HIT] ${key}`);
			return JSON.parse(cached);
		}
	} catch (error) {
		console.error(`[CACHE ERROR] ${key}:`, error);
	}

	// Execute the function to get fresh data
	const result = await fetchFn();

	try {
		// Cache the result
		await redis.setEx(key, ttlSeconds, JSON.stringify(result));
		console.log(`[CACHE SET] ${key} (TTL: ${ttlSeconds}s)`);
	} catch (error) {
		console.error(`[CACHE ERROR] ${key}:`, error);
	}

	return result;
}

export async function invalidateUserCache(userId: string, patterns: string[]) {
	try {
		for (const pattern of patterns) {
			const keys = await redis.keys(`${pattern}:${userId}:*`);
			if (keys.length > 0) {
				await redis.del(keys);
				console.log(
					`[CACHE INVALIDATED] ${keys.length} keys for pattern: ${pattern}`,
				);
			}
		}
	} catch (error) {
		console.error("[CACHE INVALIDATION ERROR]:", error);
	}
}

export async function invalidateConversationCache(
	userId: string,
	conversationId?: number,
) {
	const patterns = ["conversations"];
	if (conversationId) {
		patterns.push("conversation");
	}
	await invalidateUserCache(userId, patterns);
}

export async function invalidateOrganizationCache(
	userId: string,
	organizationId?: string,
) {
	const patterns = ["organizations"];
	if (organizationId) {
		patterns.push("organization");
	}
	await invalidateUserCache(userId, patterns);
}

export async function invalidateProjectCache(
	userId: string,
	projectId?: string,
) {
	const patterns = ["projects"];
	if (projectId) {
		patterns.push("project");
	}
	await invalidateUserCache(userId, patterns);
}

export async function invalidateSubscriptionCache(userId: string) {
	const patterns = ["subscription", "subscription-status"];
	await invalidateUserCache(userId, patterns);
}

export async function invalidateAnalyticsCache(
	userId: string,
	projectId?: string,
) {
	const patterns = ["user-analytics"];
	if (projectId) {
		patterns.push("project-analytics");
	}
	await invalidateUserCache(userId, patterns);
}
