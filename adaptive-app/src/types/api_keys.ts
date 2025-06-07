import type { RouterInputs, RouterOutputs } from "./index";

// ---- Output Types ----

/**
 * The type for a single API key item in the list.
 * Includes all API key details like name, status, creation date, etc.
 */
export type APIKeyListItem = RouterOutputs["api_keys"]["list"][number];

/**
 * The type for a single, fully-detailed API key.
 * Same as list item but explicitly typed for single item operations.
 */
export type APIKeyDetails = RouterOutputs["api_keys"]["getById"];

/**
 * The type for the response when creating a new API key.
 * Includes both the API key details and the full key value.
 */
export type APIKeyCreateResponse = RouterOutputs["api_keys"]["create"];

/**
 * The type for the response when updating an API key.
 * Returns the updated API key details.
 */
export type APIKeyUpdateResponse = RouterOutputs["api_keys"]["update"];

/**
 * The type for the response when deleting an API key.
 * Simple success confirmation.
 */
export type APIKeyDeleteResponse = RouterOutputs["api_keys"]["delete"];

/**
 * The type for the response when verifying an API key.
 * Contains validation status.
 */
export type APIKeyVerifyResponse = RouterOutputs["api_keys"]["verify"];

/**
 * The type for the complete list of API keys.
 */
export type APIKeysList = RouterOutputs["api_keys"]["list"];

// ---- Input Types ----

/**
 * The type for the input when creating a new API key.
 */
export type APIKeyCreateInput = RouterInputs["api_keys"]["create"];

/**
 * The type for the input when updating an API key.
 */
export type APIKeyUpdateInput = RouterInputs["api_keys"]["update"];

/**
 * The type for the input when getting an API key by ID.
 */
export type APIKeyGetByIdInput = RouterInputs["api_keys"]["getById"];

/**
 * The type for the input when deleting an API key.
 */
export type APIKeyDeleteInput = RouterInputs["api_keys"]["delete"];

/**
 * The type for the input when verifying an API key.
 */
export type APIKeyVerifyInput = RouterInputs["api_keys"]["verify"];

// ---- Utility Types ----

/**
 * The status values that an API key can have.
 */
export type APIKeyStatus = APIKeyListItem["status"];

/**
 * The core API key data without metadata like created_at, updated_at.
 */
export type APIKeyCore = Pick<
  APIKeyListItem,
  "id" | "name" | "status" | "key_preview"
>;

/**
 * API key data for forms (without server-generated fields).
 */
export type APIKeyFormData = Pick<
  APIKeyCreateInput,
  "name" | "status" | "expires_at"
>;

/**
 * API key data for updates (without immutable fields).
 */
export type APIKeyUpdateData = Omit<APIKeyUpdateInput, "id">;
