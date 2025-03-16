import axios from "axios";
import { API_BASE_URL } from "../common";
import { Conversation } from "./types";

/**
 * Creates a new conversation
 *
 * @param title - The title of the conversation
 * @param authToken - The authentication token
 * @returns Promise with the created conversation
 */
export const createConversation = async (
  authToken: string,
  title?: string,
): Promise<Conversation> => {
  const response = await axios.post<Conversation>(
    `${API_BASE_URL}/api/conversations`,
    { title },
    {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    },
  );
  return response.data;
};

/**
 * Gets all conversations
 *
 * @param authToken - The authentication token
 * @returns Promise with an array of conversations
 */
export const getConversations = async (
  authToken: string,
): Promise<Conversation[]> => {
  const response = await axios.get<Conversation[]>(
    `${API_BASE_URL}/api/conversations`,
    {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    },
  );
  return response.data;
};

/**
 * Gets a specific conversation by ID
 *
 * @param id - The conversation ID
 * @param authToken - The authentication token
 * @returns Promise with the conversation data
 */
export const getConversation = async (
  id: number,
  authToken: string,
): Promise<Conversation> => {
  const response = await axios.get<Conversation>(
    `${API_BASE_URL}/api/conversations/${id}`,
    {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    },
  );
  return response.data;
};

/**
 * Updates a conversation's title
 *
 * @param id - The conversation ID
 * @param title - The new title
 * @param authToken - The authentication token
 * @returns Promise with the updated conversation
 */
export const updateConversation = async (
  id: number,
  title: string,
  authToken: string,
): Promise<Conversation> => {
  const response = await axios.put<Conversation>(
    `${API_BASE_URL}/api/conversations/${id}`,
    { title },
    {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    },
  );
  return response.data;
};

/**
 * Deletes a conversation
 *
 * @param id - The conversation ID
 * @param authToken - The authentication token
 * @returns Promise with the deletion result
 */
export const deleteConversation = async (
  id: number,
  authToken: string,
): Promise<void> => {
  await axios.delete(`${API_BASE_URL}/api/conversations/${id}`, {
    headers: {
      Authorization: `Bearer ${authToken}`,
    },
  });
};

/**
 * Deletes all messages for a conversation
 *
 * @param conversationId - The conversation ID
 * @param authToken - The authentication token
 * @returns Promise that resolves when all messages are deleted
 */
export const deleteAllMessages = async (
  conversationId: number,
  authToken: string,
): Promise<void> => {
  await axios.delete(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/batch`,
    {
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
    },
  );
};
