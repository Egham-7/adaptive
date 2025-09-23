/**
 * Authentication and authorization utilities
 */

// API Key management and encryption
export {
	authenticateAndGetProject,
	authenticateApiKey,
	decryptProviderApiKey,
	encryptProviderApiKey,
	getCacheKey,
	normalizeAndValidateApiKey,
	secureCompareProviderApiKeys,
	validateAndAuthenticateApiKey,
	validateEncryptedProviderApiKey,
} from "./api-keys";

// Clerk authentication helpers
export {
	createAuthError,
	createErrorResponse,
	createSuccessResponse,
	extractApiKey,
} from "./clerk";
