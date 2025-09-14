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
			const delay = Math.min(1000 * 2 ** i, 10000); // Cap at 10s
			console.log(
				`Database connection attempt ${i + 1} failed, retrying in ${delay}ms...`,
			);
			if (i === maxRetries - 1) throw error;
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
