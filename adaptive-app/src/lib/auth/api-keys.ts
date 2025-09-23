import crypto from "node:crypto";
import { auth as getClerkAuth } from "@clerk/nextjs/server";
import { TRPCError } from "@trpc/server";
import type { Context } from "@/server/api/trpc";
import type { AuthResult } from "@/types/auth";

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

// =============================================================================
// PROVIDER API KEY ENCRYPTION
// =============================================================================

// Encryption configuration for provider API keys
const ALGORITHM = "aes-256-gcm";
const KEY_LENGTH = 32; // 32 bytes for AES-256
const IV_LENGTH = 16; // 16 bytes for AES-GCM
const TAG_LENGTH = 16; // 16 bytes for authentication tag

/**
 * Derives an encryption key from the environment variable
 * Uses PBKDF2 for key derivation with a fixed salt for consistency
 */
function getEncryptionKey(): Buffer {
	const encryptionSecret = process.env.ENCRYPTION_SECRET;
	if (!encryptionSecret) {
		throw new Error(
			"ENCRYPTION_SECRET environment variable is required for provider API key encryption",
		);
	}

	// Use a fixed salt derived from the secret itself for deterministic key derivation
	const salt = crypto
		.createHash("sha256")
		.update(`${encryptionSecret}:provider-api-keys`)
		.digest()
		.subarray(0, 16);

	return crypto.pbkdf2Sync(
		encryptionSecret,
		salt,
		100000,
		KEY_LENGTH,
		"sha512",
	);
}

/**
 * Encrypts a provider API key using AES-256-GCM
 * Returns base64-encoded encrypted data with IV and auth tag
 */
export function encryptProviderApiKey(apiKey: string): string {
	if (!apiKey || apiKey.trim().length === 0) {
		throw new Error("Provider API key cannot be empty");
	}

	try {
		const key = getEncryptionKey();
		const iv = crypto.randomBytes(IV_LENGTH);
		const cipher = crypto.createCipheriv(ALGORITHM, key, iv);
		cipher.setAAD(Buffer.from("provider-api-key", "utf8"));

		let encrypted = cipher.update(apiKey.trim(), "utf8");
		encrypted = Buffer.concat([encrypted, cipher.final()]);
		const tag = cipher.getAuthTag();

		// Combine IV + tag + encrypted data and encode as base64
		const combined = Buffer.concat([iv, tag, encrypted]);
		return combined.toString("base64");
	} catch (error) {
		throw new Error(
			`Failed to encrypt provider API key: ${error instanceof Error ? error.message : "Unknown error"}`,
		);
	}
}

/**
 * Decrypts a provider API key using AES-256-GCM
 * Expects base64-encoded encrypted data with IV and auth tag
 */
export function decryptProviderApiKey(encryptedData: string): string {
	if (!encryptedData || encryptedData.trim().length === 0) {
		throw new Error("Encrypted provider API key data cannot be empty");
	}

	try {
		const key = getEncryptionKey();
		const combined = Buffer.from(encryptedData.trim(), "base64");

		// Extract IV, tag, and encrypted data
		const iv = combined.subarray(0, IV_LENGTH);
		const tag = combined.subarray(IV_LENGTH, IV_LENGTH + TAG_LENGTH);
		const encrypted = combined.subarray(IV_LENGTH + TAG_LENGTH);

		const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
		decipher.setAuthTag(tag);
		decipher.setAAD(Buffer.from("provider-api-key", "utf8"));

		let decrypted = decipher.update(encrypted, undefined, "utf8");
		decrypted += decipher.final("utf8");

		return decrypted;
	} catch (error) {
		throw new Error(
			`Failed to decrypt provider API key: ${error instanceof Error ? error.message : "Invalid or corrupted encrypted data"}`,
		);
	}
}

/**
 * Validates that an encrypted provider API key can be successfully decrypted
 * Used for testing encryption/decryption roundtrip
 */
export function validateEncryptedProviderApiKey(
	encryptedData: string,
): boolean {
	try {
		const decrypted = decryptProviderApiKey(encryptedData);
		return decrypted.length > 0;
	} catch {
		return false;
	}
}

/**
 * Securely compares a plain provider API key with an encrypted one
 * This prevents timing attacks on API key comparison
 */
export function secureCompareProviderApiKeys(
	plainKey: string,
	encryptedKey: string,
): boolean {
	try {
		const decryptedKey = decryptProviderApiKey(encryptedKey);
		return crypto.timingSafeEqual(
			Buffer.from(plainKey.trim(), "utf8"),
			Buffer.from(decryptedKey, "utf8"),
		);
	} catch {
		return false;
	}
}
