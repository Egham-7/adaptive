import { SignJWT } from 'jose';

// JWT configuration
export const JWT_CONFIG = {
  algorithm: 'HS256' as const,
  expiresIn: '1h',
  issuer: 'adaptive-app',
  audience: 'adaptive-backend',
};

/**
 * Signs a JWT token for backend authentication using jose library
 * @param payload - The payload to include in the JWT
 * @returns Promised signed JWT token
 */
export async function signJWT(payload: Record<string, unknown>): Promise<string> {
  const secret = process.env.JWT_SECRET;
  
  if (!secret) {
    throw new Error('JWT_SECRET environment variable is required');
  }

  const secretBytes = new TextEncoder().encode(secret);

  return await new SignJWT(payload)
    .setProtectedHeader({ alg: JWT_CONFIG.algorithm })
    .setIssuedAt()
    .setIssuer(JWT_CONFIG.issuer)
    .setAudience(JWT_CONFIG.audience)
    .setExpirationTime(JWT_CONFIG.expiresIn)
    .sign(secretBytes);
}

/**
 * Creates a JWT token for backend API requests
 * @param apiKey - The API key being used
 * @param userId - Optional user ID
 * @param organizationId - Optional organization ID
 * @returns JWT token for backend authentication
 */
export async function createBackendJWT(
  apiKey: string,
  userId?: string,
  organizationId?: string
): Promise<string> {
  const payload = {
    apiKey,
    userId,
    organizationId,
  };

  return await signJWT(payload);
}