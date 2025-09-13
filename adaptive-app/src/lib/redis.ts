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

// Lazy connection function
async function ensureConnected() {
	if (!client.isOpen) {
		await client.connect();
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
