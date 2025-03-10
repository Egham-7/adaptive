import axios from "axios";
import { API_BASE_URL } from "../common";
import { Conversation } from "./types";

/**
 * Creates a new conversation
 *
 * @param title - The title of the conversation
 * @returns Promise with the created conversation
 */
export const createConversation = async (
  title?: string,
): Promise<Conversation> => {
  const response = await axios.post<Conversation>(
    `${API_BASE_URL}/api/conversations`,
    { title },
  );
  return response.data;
};

/**
 * Gets all conversations
 *
 * @returns Promise with an array of conversations
 */
export const getConversations = async (): Promise<Conversation[]> => {
  const response = await axios.get<Conversation[]>(
    `${API_BASE_URL}/api/conversations`,
  );
  return response.data;
};

/**
 * Gets a specific conversation by ID
 *
 * @param id - The conversation ID
 * @returns Promise with the conversation data
 */
export const getConversation = async (id: number): Promise<Conversation> => {
  const response = await axios.get<Conversation>(
    `${API_BASE_URL}/api/conversations/${id}`,
  );
  return response.data;
};

/**
 * Updates a conversation's title
 *
 * @param id - The conversation ID
 * @param title - The new title
 * @returns Promise with the updated conversation
 */
export const updateConversation = async (
  id: number,
  title: string,
): Promise<Conversation> => {
  const response = await axios.put<Conversation>(
    `${API_BASE_URL}/api/conversations/${id}`,
    { title },
  );
  return response.data;
};

/**
 * Deletes a conversation
 *
 * @param id - The conversation ID
 * @returns Promise with the deletion result
 */
export const deleteConversation = async (id: number): Promise<void> => {
  await axios.delete(`${API_BASE_URL}/api/conversations/${id}`);
};

/**
 * Deletes all messages for a conversation
 *
 * @param conversationId - The conversation ID
 * @returns Promise that resolves when all messages are deleted
 */
export const deleteAllMessages = async (
  conversationId: number,
): Promise<void> => {
  await axios.delete(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/batch`,
  );
};
