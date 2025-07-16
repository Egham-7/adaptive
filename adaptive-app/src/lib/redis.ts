import { createClient } from "redis";

import { env } from "@/env";

const globalForRedis = globalThis as unknown as {
	redis: ReturnType<typeof createClient> | undefined;
};

export const redis =
	globalForRedis.redis ??
	createClient({
		url: env.REDIS_URL,
	});

if (env.NODE_ENV !== "production") globalForRedis.redis = redis;

// Connect to Redis
if (!redis.isOpen) {
	await redis.connect();
}

export default redis;
