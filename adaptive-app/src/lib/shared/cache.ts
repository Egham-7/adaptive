import { redis } from "./redis";

type CacheKey = string;
type CachePattern = string;

/**
 * Cache wrapper that stores and retrieves data from Redis
 */
export async function withCache<T>(
	key: CacheKey,
	fetchFn: () => Promise<T>,
	ttlSeconds = 300,
): Promise<T> {
	try {
		const cached = await redis.get(key);
		if (cached) {
			console.log(`[CACHE HIT] ${key}`);
			return JSON.parse(cached);
		}
	} catch (error) {
		console.error(`[CACHE ERROR] Failed to get ${key}:`, error);
	}

	const result = await fetchFn();

	try {
		await redis.setEx(key, ttlSeconds, JSON.stringify(result));
		console.log(`[CACHE SET] ${key} (TTL: ${ttlSeconds}s)`);
	} catch (error) {
		console.error(`[CACHE ERROR] Failed to set ${key}:`, error);
	}

	return result;
}

/**
 * Generic cache invalidation helper
 */
async function invalidatePatterns(patterns: CachePattern[]): Promise<void> {
	try {
		const allKeys: string[] = [];

		for (const pattern of patterns) {
			const keys = await redis.keys(pattern);
			allKeys.push(...keys);
		}

		if (allKeys.length > 0) {
			await redis.del(allKeys);
			console.log(`[CACHE INVALIDATED] ${allKeys.length} keys`);
		}
	} catch (error) {
		console.error("[CACHE INVALIDATION ERROR]:", error);
	}
}

/**
 * Invalidate user-specific cache patterns
 */
export async function invalidateUserCache(
	userId: string,
	patterns: string[],
): Promise<void> {
	const cachePatterns = patterns.flatMap((pattern) => [
		`${pattern}:${userId}`, // Exact match
		`${pattern}:${userId}:*`, // Wildcard match
	]);

	await invalidatePatterns(cachePatterns);
}

// User-specific cache invalidation functions
export const invalidateConversationCache = (
	userId: string,
	conversationId?: number,
) => {
	const patterns = conversationId
		? ["conversations", "conversation"]
		: ["conversations"];
	return invalidateUserCache(userId, patterns);
};

export const invalidateOrganizationCache = (
	userId: string,
	organizationId?: string,
) => {
	const patterns = organizationId
		? ["organizations", "organization"]
		: ["organizations"];
	return invalidateUserCache(userId, patterns);
};

export const invalidateProjectCache = (userId: string, projectId?: string) => {
	const patterns = projectId ? ["projects", "project"] : ["projects"];
	return invalidateUserCache(userId, patterns);
};

export const invalidateSubscriptionCache = (userId: string) => {
	return invalidateUserCache(userId, ["subscription", "subscription-status"]);
};

export const invalidateAnalyticsCache = (
	userId: string,
	projectId?: string,
) => {
	const patterns = projectId
		? ["user-analytics", "project-analytics"]
		: ["user-analytics"];
	return invalidateUserCache(userId, patterns);
};

// Project-specific cache invalidation functions
export const invalidateProviderCache = (
	projectId: string,
	providerName?: string,
) => {
	const patterns = [
		`providers:${projectId}*`,
		`provider-configs:${projectId}*`,
	];

	if (providerName) {
		patterns.push(`provider:${providerName}:${projectId}*`);
	}

	return invalidatePatterns(patterns);
};

export const invalidateProviderConfigCache = (
	projectId: string,
	providerId?: string,
) => {
	const patterns = [`provider-configs:${projectId}*`];

	if (providerId) {
		patterns.push(`provider-config:${projectId}:${providerId}*`);
	}

	return invalidatePatterns(patterns);
};

export const invalidateClusterCache = async (
	projectId: string,
	clusterName?: string,
) => {
	const patterns: string[] = [];

	if (clusterName) {
		patterns.push(`cluster:${projectId}:${clusterName}*`);

		// Invalidate related model details
		try {
			const clusterKeys = await redis.keys(
				`cluster:${projectId}:${clusterName}`,
			);
			for (const key of clusterKeys) {
				const cluster = await redis.get(key);
				if (cluster) {
					const clusterData = JSON.parse(cluster);
					if (clusterData.id) {
						patterns.push(`model-details:${clusterData.id}*`);
					}
				}
			}
		} catch (error) {
			console.error("[CACHE ERROR] Failed to get cluster data:", error);
		}
	} else {
		patterns.push(`cluster:${projectId}:*`, "model-details:*");
	}

	return invalidatePatterns(patterns);
};
