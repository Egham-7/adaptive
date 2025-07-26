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
			reconnectStrategy: (retries) => Math.min(retries * 50, 1000),
		},
	});

// Error handling
client.on('error', (err) => {
	console.error('Redis client error:', err);
});

client.on('connect', () => {
	console.log('Redis client connected');
});

client.on('disconnect', () => {
	console.log('Redis client disconnected');
});

if (env.NODE_ENV !== "production") globalForRedis.redis = client;

// Lazy connection function with error handling
async function ensureConnected() {
	try {
		if (!client.isOpen) {
			await client.connect();
		}
		return client;
	} catch (error) {
		console.error('Failed to connect to Redis:', error);
		throw error;
	}
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
};

export default redis;
