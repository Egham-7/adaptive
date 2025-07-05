import crypto from "node:crypto";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
	createTRPCRouter,
	protectedProcedure,
	publicProcedure,
} from "@/server/api/trpc";

const createAPIKeySchema = z.object({
	name: z.string().min(1),
	status: z.string(),
	expires_at: z.string().optional(),
	projectId: z.string().optional(),
});

const updateAPIKeySchema = z.object({
	id: z.string().uuid(),
	name: z.string().min(1),
	status: z.string(),
});

const apiKeySchema = z.object({
	id: z.string(),
	name: z.string(),
	status: z.string(),
	created_at: z.string(),
	updated_at: z.string(),
	expires_at: z.string().nullable(),
	user_id: z.string(),
	key_preview: z.string(),
});

type APIKey = z.infer<typeof apiKeySchema>;
type CreateAPIKeyResponse = {
	api_key: APIKey;
	full_api_key: string;
};

export const apiKeysRouter = createTRPCRouter({
	list: protectedProcedure.query(async ({ ctx }): Promise<APIKey[]> => {
		const userId = ctx.clerkAuth.userId;
		if (!userId) {
			throw new TRPCError({ code: "UNAUTHORIZED" });
		}
		const keys = await ctx.db.apiKey.findMany({
			where: { userId },
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
			if (!k || k.userId !== userId) {
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

			const fullKey = crypto.randomBytes(32).toString("hex");
			const prefix = fullKey.slice(0, 8);
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

			return { api_key, full_api_key: fullKey };
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
			if (!existing || existing.userId !== userId) {
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
			if (!userId) {
				throw new TRPCError({ code: "UNAUTHORIZED" });
			}
			const existing = await ctx.db.apiKey.findUnique({
				where: { id: input.id },
			});
			if (!existing || existing.userId !== userId) {
				throw new TRPCError({ code: "NOT_FOUND" });
			}
			await ctx.db.apiKey.delete({ where: { id: input.id } });
			return { success: true };
		}),

	verify: publicProcedure
		.input(z.object({ apiKey: z.string() }))
		.query(async ({ ctx, input }) => {
			const apiKey = input.apiKey;
			if (apiKey.length < 8) {
				return { valid: false };
			}
			const prefix = apiKey.slice(0, 8);
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
			return { valid: hash === record.keyHash };
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
				where: { projectId: input.projectId },
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
				status: z.string().default("active"),
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

			const fullKey = crypto.randomBytes(32).toString("hex");
			const prefix = fullKey.slice(0, 8);
			const hash = crypto.createHash("sha256").update(fullKey).digest("hex");

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

			return { api_key, full_api_key: fullKey };
		}),
});
