import { createClient } from "redis";

import { env } from "@/env";

const globalForRedis = globalThis as unknown as {
	redis: ReturnType<typeof createClient> | undefined;
};

const client =
	globalForRedis.redis ??
	createClient({
		url: env.REDIS_URL,
		socket: {
			connectTimeout: 60000,
		},
	});

if (env.NODE_ENV !== "production") globalForRedis.redis = client;

// Retry connection with exponential backoff
async function connectWithRetry(
	redisClient: ReturnType<typeof createClient>,
	maxRetries = 5,
): Promise<void> {
	for (let i = 0; i < maxRetries; i++) {
		try {
			await redisClient.connect();
			console.log("Redis connected successfully");
			return;
		} catch (error) {
			// Check for fatal Redis error codes that should fail fast
			const errorMessage = (error as Error).message?.toLowerCase() || "";
			if (
				errorMessage.includes("enotfound") ||
				errorMessage.includes("econnrefused") ||
				errorMessage.includes("invalid credentials")
			) {
				console.error(
					"Fatal Redis error: Invalid credentials or connection parameters",
					error,
				);
				throw error;
			}

			const isLastAttempt = i === maxRetries - 1;

			if (isLastAttempt) {
				console.error(
					`Redis connection attempt ${i + 1} failed (final attempt), throwing error:`,
					error,
				);
				throw error;
			}
			// Calculate exponential backoff with jitter
			const baseDelay = Math.min(1000 * 2 ** i, 10000); // Cap at 10s
			const jitter = Math.random() * 250; // 0-250ms jitter
			const delay = baseDelay + jitter;

			console.error(
				`Redis connection attempt ${i + 1} failed, retrying in ${Math.round(delay)}ms...`,
				error,
			);

			await new Promise((resolve) => setTimeout(resolve, delay));
		}
	}
}

// Lazy connection function with retry
async function ensureConnected() {
	if (!client.isOpen) {
		await connectWithRetry(client);
	}
	return client;
}

// Export the lazy connection function
export const redis = {
	async get(key: string) {
		const redisClient = await ensureConnected();
		return redisClient.get(key);
	},
	async setEx(key: string, seconds: number, value: string) {
		const redisClient = await ensureConnected();
		return redisClient.setEx(key, seconds, value);
	},
	async keys(pattern: string) {
		const redisClient = await ensureConnected();
		return redisClient.keys(pattern);
	},
	async del(keys: string[]) {
		const redisClient = await ensureConnected();
		return redisClient.del(keys);
	},
	async exists(key: string) {
		const redisClient = await ensureConnected();
		return redisClient.exists(key);
	},
};

export default redis;

// Initialize connection with retry logic in production
if (env.NODE_ENV === "production") {
	connectWithRetry(client).catch(console.error);
}
