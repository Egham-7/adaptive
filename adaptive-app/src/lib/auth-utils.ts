import crypto from "node:crypto";
import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Context } from "@/server/api/trpc";

// API key validation constants
const API_KEY_REGEX = /^sk-[A-Za-z0-9_-]+$/;
const API_KEY_PREFIX_LENGTH = 11;

// Utility to normalize and validate API key
export const normalizeAndValidateApiKey = (apiKey: string) => {
	const normalizedKey = apiKey.trim();

	if (!API_KEY_REGEX.test(normalizedKey)) {
		throw new TRPCError({
			code: "UNAUTHORIZED",
			message: "Invalid API key format",
		});
	}

	const prefix = normalizedKey.slice(0, API_KEY_PREFIX_LENGTH);
	const hash = crypto.createHash("sha256").update(normalizedKey).digest("hex");

	return { normalizedKey, prefix, hash };
};

// Utility to validate API key and check if it exists in database
export const validateAndAuthenticateApiKey = async (
	apiKey: string,
	db: Context["db"],
) => {
	const { prefix, hash } = normalizeAndValidateApiKey(apiKey);

	const record = await db.apiKey.findFirst({
		where: { keyPrefix: prefix, keyHash: hash, status: "active" },
	});

	if (!record || (record.expiresAt && record.expiresAt < new Date())) {
		throw new TRPCError({
			code: "UNAUTHORIZED",
			message: "Invalid or expired API key",
		});
	}

	return record;
};

export type AuthResult =
	| {
			authType: "api_key";
			apiKey: {
				id: string;
				projectId: string;
				keyPrefix: string;
				project: { id: string; name: string };
			};
			project: { id: string; name: string };
	  }
	| {
			authType: "user";
			userId: string;
			project: { id: string; name: string };
	  };

// Helper function to authenticate and get project access
export const authenticateAndGetProject = async (
	ctx: Context,
	input: { projectId: string; apiKey?: string },
): Promise<AuthResult> => {
	// Try API key authentication first
	if (input.apiKey) {
		const { prefix, hash } = normalizeAndValidateApiKey(input.apiKey);

		const record = await ctx.db.apiKey.findFirst({
			where: {
				keyPrefix: prefix,
				keyHash: hash,
				status: "active",
			},
			include: { project: true },
		});

		if (!record || (record.expiresAt && record.expiresAt < new Date())) {
			throw new TRPCError({
				code: "UNAUTHORIZED",
				message: "Invalid or expired API key",
			});
		}

		// Guard: Check project access
		if (!record.projectId) {
			throw new TRPCError({
				code: "FORBIDDEN",
				message: "API key is not associated with any project",
			});
		}

		if (!record.project) {
			throw new TRPCError({
				code: "FORBIDDEN",
				message: "Project not found for API key",
			});
		}

		if (record.projectId !== input.projectId) {
			throw new TRPCError({
				code: "FORBIDDEN",
				message: "API key does not have access to this project",
			});
		}

		return {
			authType: "api_key" as const,
			apiKey: {
				id: record.id,
				projectId: record.projectId,
				keyPrefix: record.keyPrefix,
				project: { id: record.project.id, name: record.project.name },
			},
			project: { id: record.project.id, name: record.project.name },
		};
	}

	// Fall back to user authentication
	const clerkAuthResult = await getClerkAuth();
	if (!clerkAuthResult.userId) {
		throw new TRPCError({
			code: "UNAUTHORIZED",
			message:
				"Authentication required - provide either API key or user authentication",
		});
	}

	const userId = clerkAuthResult.userId;
	const whereClause = {
		id: input.projectId,
		organization: {
			OR: [{ ownerId: userId }, { members: { some: { userId } } }],
		},
	};

	const project = await ctx.db.project.findFirst({ where: whereClause });

	if (!project) {
		throw new TRPCError({
			code: "FORBIDDEN",
			message: "You don't have access to this project",
		});
	}

	return { authType: "user" as const, userId, project };
};

// Helper for API key only authentication (for REST API routes)
export const authenticateApiKey = async (
	apiKey: string,
	db: Context["db"],
): Promise<{
	apiKey: { id: string; projectId: string };
	project: { id: string; name: string };
}> => {
	const { prefix, hash } = (() => {
		try {
			return normalizeAndValidateApiKey(apiKey);
		} catch (error) {
			if (error instanceof TRPCError) {
				throw new Error(error.message);
			}
			throw new Error("Invalid API key format");
		}
	})();

	const record = await db.apiKey.findFirst({
		where: {
			keyPrefix: prefix,
			keyHash: hash,
			status: "active",
		},
		include: { project: true },
	});

	if (!record || (record.expiresAt && record.expiresAt < new Date())) {
		throw new Error("Invalid or expired API key");
	}

	if (!record.projectId) {
		throw new Error("API key is not associated with any project");
	}

	if (!record.project) {
		throw new Error("Project not found for API key");
	}

	return {
		apiKey: { id: record.id, projectId: record.projectId },
		project: { id: record.project.id, name: record.project.name },
	};
};

// Helper to get consistent cache keys
export const getCacheKey = (auth: AuthResult, suffix: string): string => {
	const prefix = auth.authType === "api_key" ? auth.apiKey.id : auth.userId;
	return `llm-clusters:${auth.authType}:${prefix}:${suffix}`;
};
