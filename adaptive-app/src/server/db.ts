import { env } from "@/env";
import { PrismaClient } from "../../prisma/generated";

const createPrismaClient = () =>
	new PrismaClient({
		log:
			env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"],
	});

// Retry connection with exponential backoff
async function connectWithRetry(
	prisma: PrismaClient,
	maxRetries = 5,
): Promise<void> {
	for (let i = 0; i < maxRetries; i++) {
		try {
			await prisma.$connect();
			console.log("Database connected successfully");
			return;
		} catch (error) {
			// Check for fatal Prisma error codes that should fail fast
			const errorCode = (error as { code?: string }).code;
			if (errorCode === "P1000" || errorCode === "P1003") {
				console.error(
					`Fatal database error (${errorCode}): Invalid credentials or connection parameters`,
					error,
				);
				throw error;
			}

			const isLastAttempt = i === maxRetries - 1;

			if (isLastAttempt) {
				console.error(
					`Database connection attempt ${i + 1} failed (final attempt), throwing error:`,
					error,
				);
				throw error;
			}
			// Calculate exponential backoff with jitter
			const baseDelay = Math.min(1000 * 2 ** i, 10000); // Cap at 10s
			const jitter = Math.random() * 250; // 0-250ms jitter
			const delay = baseDelay + jitter;

			console.error(
				`Database connection attempt ${i + 1} failed, retrying in ${Math.round(delay)}ms...`,
				error,
			);

			await new Promise((resolve) => setTimeout(resolve, delay));
		}
	}
}

const globalForPrisma = globalThis as unknown as {
	prisma: ReturnType<typeof createPrismaClient> | undefined;
};

export const db = globalForPrisma.prisma ?? createPrismaClient();

if (env.NODE_ENV !== "production") globalForPrisma.prisma = db;

// Initialize connection with retry logic in production
if (env.NODE_ENV === "production") {
	connectWithRetry(db).catch(console.error);
}
