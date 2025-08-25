import crypto from "node:crypto";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
	createTRPCRouter,
	protectedProcedure,
	publicProcedure,
} from "@/server/api/trpc";

const API_KEY_BYTE_LENGTH = 36;
const API_KEY_PREFIX_LENGTH = 11;

// Helper to generate API key, prefix, and hash
function generateApiKey() {
	const randomBytes = crypto.randomBytes(API_KEY_BYTE_LENGTH);
	const fullKey = `sk-${randomBytes.toString("base64url")}`;
	const prefix = fullKey.slice(0, API_KEY_PREFIX_LENGTH);
	const hash = crypto.createHash("sha256").update(fullKey).digest("hex");
	return { fullKey, prefix, hash };
}

// Simple encryption helper for storing full keys temporarily
function encryptKey(key: string, secret: string): string {
	const algorithm = "aes-256-cbc";
	const keyHash = crypto.createHash("sha256").update(secret).digest();
	const iv = crypto.randomBytes(16);
	const cipher = crypto.createCipheriv(algorithm, keyHash, iv);
	let encrypted = cipher.update(key, "utf8", "hex");
	encrypted += cipher.final("hex");
	return `${iv.toString("hex")}:${encrypted}`;
}

function decryptKey(encryptedKey: string, secret: string): string {
	const algorithm = "aes-256-cbc";
	const keyHash = crypto.createHash("sha256").update(secret).digest();
	const [ivHex, encrypted] = encryptedKey.split(":");
	if (!ivHex || !encrypted) {
		throw new Error("Invalid encrypted key format");
	}
	const iv = Buffer.from(ivHex, "hex");
	const decipher = crypto.createDecipheriv(algorithm, keyHash, iv);
	let decrypted = decipher.update(encrypted, "hex", "utf8");
	decrypted += decipher.final("utf8");
	return decrypted;
}

const createAPIKeySchema = z.object({
	name: z.string().min(1),
	status: z.enum(["active", "revoked", "inactive", "deleted"]),
	expires_at: z.string().optional(),
	projectId: z.string().optional(),
});

const updateAPIKeySchema = z.object({
	id: z.string().uuid(),
	name: z.string().min(1),
	status: z.enum(["active", "revoked", "inactive", "deleted"]),
});

const apiKeySchema = z.object({
	id: z.string(),
	name: z.string(),
	status: z.enum(["active", "revoked", "inactive", "deleted"]),
	created_at: z.string(),
	updated_at: z.string(),
	expires_at: z.string().nullable(),
	user_id: z.string(),
	key_preview: z.string(),
});

type APIKey = z.infer<typeof apiKeySchema>;
type CreateAPIKeyResponse = {
	api_key: APIKey;
	reveal_token: string;
};

export const apiKeysRouter = createTRPCRouter({
	list: protectedProcedure.query(async ({ ctx }): Promise<APIKey[]> => {
		const userId = ctx.clerkAuth.userId;
		if (!userId) {
			throw new TRPCError({ code: "UNAUTHORIZED" });
		}
		const keys = await ctx.db.apiKey.findMany({
			where: {
				userId,
				status: { not: "deleted" }, // Exclude soft-deleted keys
			},
			orderBy: { createdAt: "desc" },
		});
		return keys.map((k) => ({
			id: k.id,
			name: k.name,
			status: k.status,
			created_at: k.createdAt.toISOString(),
			updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
			expires_at: k.expiresAt?.toISOString() ?? null,
			user_id: k.userId,
			key_preview: k.keyPrefix,
		}));
	}),

	getById: protectedProcedure
		.input(z.object({ id: z.string().uuid() }))
		.query(async ({ ctx, input }): Promise<APIKey> => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}
			const k = await ctx.db.apiKey.findUnique({
				where: { id: input.id },
			});
			if (!k || k.userId !== userId || k.status === "deleted") {
				throw new TRPCError({ code: "NOT_FOUND" });
			}
			return {
				id: k.id,
				name: k.name,
				status: k.status,
				created_at: k.createdAt.toISOString(),
				updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
				expires_at: k.expiresAt?.toISOString() ?? null,
				user_id: k.userId,
				key_preview: k.keyPrefix,
			};
		}),

	create: protectedProcedure
		.input(createAPIKeySchema)
		.mutation(async ({ ctx, input }): Promise<CreateAPIKeyResponse> => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			const randomBytes = crypto.randomBytes(API_KEY_BYTE_LENGTH);
			const fullKey = `sk-${randomBytes.toString("base64url")}`;
			const prefix = fullKey.slice(0, API_KEY_PREFIX_LENGTH);
			const hash = crypto.createHash("sha256").update(fullKey).digest("hex");

			const expiresAt = input.expires_at
				? new Date(input.expires_at)
				: undefined;

			// If projectId is provided, verify user has access to the project
			if (input.projectId) {
				const project = await ctx.db.project.findFirst({
					where: {
						id: input.projectId,
						OR: [
							{ members: { some: { userId } } },
							{ organization: { ownerId: userId } },
							{ organization: { members: { some: { userId } } } },
						],
					},
				});

				if (!project) {
					throw new TRPCError({
						code: "FORBIDDEN",
						message: "You don't have access to this project",
					});
				}
			}

			const k = await ctx.db.apiKey.create({
				data: {
					userId,
					name: input.name,
					status: input.status,
					keyPrefix: prefix,
					keyHash: hash,
					expiresAt,
					projectId: input.projectId,
				},
			});

			// Create a one-time reveal token
			const revealToken = crypto.randomBytes(32).toString("hex");
			if (!process.env.API_KEY_ENCRYPTION_SECRET) {
				throw new Error(
					"Environment variable API_KEY_ENCRYPTION_SECRET is required but not set.",
				);
			}
			const encryptionSecret = process.env.API_KEY_ENCRYPTION_SECRET;
			const encryptedKey = encryptKey(fullKey, encryptionSecret);
			const revealExpiresAt = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes

			await ctx.db.apiKeyRevealToken.create({
				data: {
					apiKeyId: k.id,
					userId,
					token: revealToken,
					fullKey: encryptedKey,
					expiresAt: revealExpiresAt,
				},
			});

			const api_key: APIKey = {
				id: k.id,
				name: k.name,
				status: k.status,
				created_at: k.createdAt.toISOString(),
				updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
				expires_at: k.expiresAt?.toISOString() ?? null,
				user_id: k.userId,
				key_preview: k.keyPrefix,
			};

			return { api_key, reveal_token: revealToken };
		}),

	update: protectedProcedure
		.input(updateAPIKeySchema)
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}
			const existing = await ctx.db.apiKey.findUnique({
				where: { id: input.id },
			});
			if (
				!existing ||
				existing.userId !== userId ||
				existing.status === "deleted"
			) {
				throw new TRPCError({ code: "NOT_FOUND" });
			}
			const k = await ctx.db.apiKey.update({
				where: { id: input.id },
				data: {
					name: input.name,
					status: input.status,
				},
			});
			return {
				id: k.id,
				name: k.name,
				status: k.status,
				created_at: k.createdAt.toISOString(),
				updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
				expires_at: k.expiresAt?.toISOString() ?? null,
				user_id: k.userId,
				key_preview: k.keyPrefix,
			};
		}),

	delete: protectedProcedure
		.input(z.object({ id: z.string().uuid() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;

			const existing = await ctx.db.apiKey.findUnique({
				where: { id: input.id },
			});
			if (
				!existing ||
				existing.userId !== userId ||
				existing.status === "deleted"
			) {
				throw new TRPCError({ code: "NOT_FOUND" });
			}

			// Soft delete: set status to 'deleted' and clear sensitive data
			await ctx.db.apiKey.update({
				where: { id: input.id },
				data: {
					status: "deleted",
					keyHash: "", // Clear sensitive key hash
					keyPrefix: "", // Clear key prefix
					name: "[DELETED]", // Clear original name
				},
			});

			return { success: true };
		}),

	verify: publicProcedure
		.input(z.object({ apiKey: z.string() }))
		.query(async ({ ctx, input }) => {
			const apiKey = input.apiKey;
			const apiKeyRegex = /^sk-[A-Za-z0-9_-]+$/;
			if (!apiKeyRegex.test(apiKey)) {
				return { valid: false };
			}
			const prefix = apiKey.slice(0, 11);
			const record = await ctx.db.apiKey.findFirst({
				where: { keyPrefix: prefix, status: "active" },
			});
			if (!record) {
				return { valid: false };
			}

			// Check if key is expired
			if (record.expiresAt && record.expiresAt < new Date()) {
				return { valid: false };
			}

			const hash = crypto.createHash("sha256").update(apiKey).digest("hex");
			const isValid = hash === record.keyHash;

			if (!isValid) {
				return { valid: false };
			}

			return {
				valid: true,
				projectId: record.projectId,
				userId: record.userId,
			};
		}),

	// Get API keys for a specific project
	getByProject: protectedProcedure
		.input(z.object({ projectId: z.string() }))
		.query(async ({ ctx, input }): Promise<APIKey[]> => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			// Verify user has access to the project
			const project = await ctx.db.project.findFirst({
				where: {
					id: input.projectId,
					OR: [
						{ members: { some: { userId } } },
						{ organization: { ownerId: userId } },
						{ organization: { members: { some: { userId } } } },
					],
				},
			});

			if (!project) {
				throw new TRPCError({
					code: "FORBIDDEN",
					message: "You don't have access to this project",
				});
			}

			const keys = await ctx.db.apiKey.findMany({
				where: {
					projectId: input.projectId,
					status: { not: "deleted" }, // Exclude soft-deleted keys
				},
				orderBy: { createdAt: "desc" },
			});

			return keys.map((k) => ({
				id: k.id,
				name: k.name,
				status: k.status,
				created_at: k.createdAt.toISOString(),
				updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
				expires_at: k.expiresAt?.toISOString() ?? null,
				user_id: k.userId,
				key_preview: k.keyPrefix,
			}));
		}),

	// Create API key for a specific project
	createForProject: protectedProcedure
		.input(
			z.object({
				name: z.string().min(1),
				projectId: z.string(),
				status: z
					.enum(["active", "revoked", "inactive", "deleted"])
					.default("active"),
				expires_at: z.string().optional(),
			}),
		)
		.mutation(async ({ ctx, input }): Promise<CreateAPIKeyResponse> => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			// Verify user has permission to create API keys for this project
			const project = await ctx.db.project.findFirst({
				where: {
					id: input.projectId,
					OR: [
						{ members: { some: { userId, role: { in: ["owner", "admin"] } } } },
						{ organization: { ownerId: userId } },
						{
							organization: {
								members: { some: { userId, role: { in: ["owner", "admin"] } } },
							},
						},
					],
				},
			});

			if (!project) {
				throw new TRPCError({
					code: "FORBIDDEN",
					message:
						"You don't have permission to create API keys for this project",
				});
			}

			const { fullKey, prefix, hash } = generateApiKey();

			const expiresAt = input.expires_at
				? new Date(input.expires_at)
				: undefined;

			const k = await ctx.db.apiKey.create({
				data: {
					userId,
					name: input.name,
					status: input.status,
					keyPrefix: prefix,
					keyHash: hash,
					expiresAt,
					projectId: input.projectId,
				},
			});

			// Create a one-time reveal token
			const revealToken = crypto.randomBytes(32).toString("hex");
			const encryptionSecret =
				process.env.API_KEY_ENCRYPTION_SECRET ||
				"default-secret-change-in-production";
			const encryptedKey = encryptKey(fullKey, encryptionSecret);
			const revealExpiresAt = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes

			await ctx.db.apiKeyRevealToken.create({
				data: {
					apiKeyId: k.id,
					userId,
					token: revealToken,
					fullKey: encryptedKey,
					expiresAt: revealExpiresAt,
				},
			});

			const api_key: APIKey = {
				id: k.id,
				name: k.name,
				status: k.status,
				created_at: k.createdAt.toISOString(),
				updated_at: (k.updatedAt ?? k.createdAt).toISOString(),
				expires_at: k.expiresAt?.toISOString() ?? null,
				user_id: k.userId,
				key_preview: k.keyPrefix,
			};

			return { api_key, reveal_token: revealToken };
		}),

	// Reveal API key using one-time token
	revealApiKey: protectedProcedure
		.input(z.object({ token: z.string() }))
		.mutation(async ({ ctx, input }) => {
			const userId = ctx.clerkAuth.userId;
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}

			// Find the reveal token
			const revealToken = await ctx.db.apiKeyRevealToken.findFirst({
				where: {
					token: input.token,
					userId,
					revealed: false,
					expiresAt: { gt: new Date() },
				},
			});

			if (!revealToken) {
				throw new TRPCError({
					code: "NOT_FOUND",
					message: "Invalid or expired reveal token",
				});
			}

			// Mark as revealed and decrypt the key
			await ctx.db.apiKeyRevealToken.update({
				where: { id: revealToken.id },
				data: { revealed: true },
			});

			const encryptionSecret =
				process.env.API_KEY_ENCRYPTION_SECRET ||
				"default-secret-change-in-production";
			const fullKey = decryptKey(revealToken.fullKey, encryptionSecret);

			// Clean up expired tokens periodically
			await ctx.db.apiKeyRevealToken.deleteMany({
				where: {
					expiresAt: { lt: new Date() },
				},
			});

			return { full_api_key: fullKey };
		}),
});
