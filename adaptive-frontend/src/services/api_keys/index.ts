import axios from "axios";

import { API_BASE_URL } from "../common";
import {
  APIKey,
  CreateAPIKeyResponse,
  CreateAPIKeyRequest,
  UpdateAPIKeyRequest,
} from "./types";

/**
 * Gets all API keys for a specific user
 *
 * @param userId - The user ID
 * @returns Promise with an array of API keys
 */
export const getAPIKeysByUserId = async (userId: string): Promise<APIKey[]> => {
  const response = await axios.get<APIKey[]>(
    `${API_BASE_URL}/api/api_keys/${userId}`,
  );
  return response.data;
};

/**
 * Gets a specific API key by ID
 *
 * @param id - The API key ID
 * @returns Promise with the API key data
 */
export const getAPIKeyById = async (id: string): Promise<APIKey> => {
  const response = await axios.get<APIKey>(
    `${API_BASE_URL}/api/api_keys/${id}`,
  );
  return response.data;
};

/**
 * Creates a new API key
 *
 * @param apiKeyData - The API key data to create
 * @returns Promise with the created API key
 */
export const createAPIKey = async (
  apiKeyData: CreateAPIKeyRequest,
): Promise<CreateAPIKeyResponse> => {
  const response = await axios.post<CreateAPIKeyResponse>(
    `${API_BASE_URL}/api/api_keys`,
    apiKeyData,
  );
  return response.data;
};

/**
 * Updates an API key
 *
 * @param id - The API key ID
 * @param apiKeyData - The updated API key data
 * @returns Promise with the updated API key
 */
export const updateAPIKey = async (
  id: string,
  apiKeyData: UpdateAPIKeyRequest,
): Promise<APIKey> => {
  const response = await axios.put<APIKey>(
    `${API_BASE_URL}/api/api_keys/${id}`,
    apiKeyData,
  );
  return response.data;
};

/**
 * Deletes an API key
 *
 * @param id - The API key ID
 * @returns Promise that resolves when the API key is deleted
 */
export const deleteAPIKey = async (id: string): Promise<void> => {
  await axios.delete(`${API_BASE_URL}/api/api_keys/${id}`);
};
