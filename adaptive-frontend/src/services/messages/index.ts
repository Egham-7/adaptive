import axios from "axios";
import { API_BASE_URL } from "../common";
import { DBMessage } from "./types";
import { Message } from "../llms/types";

/**
 * Gets all messages for a conversation
 *
 * @param conversationId - The conversation ID
 * @returns Promise with an array of messages
 */
export const getMessages = async (
  conversationId: number,
): Promise<DBMessage[]> => {
  const response = await axios.get<DBMessage[]>(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages`,
  );
  return response.data;
};

/**
 * Creates a new message in a conversation
 *
 * @param conversationId - The conversation ID
 * @param message - The message data (role and content)
 * @returns Promise with the created message
 */
export const createMessage = async (
  conversationId: number,
  message: { role: string; content: string },
): Promise<DBMessage> => {
  const response = await axios.post<DBMessage>(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages`,
    message,
  );
  return response.data;
};

/**
 * Gets a specific message by ID
 *
 * @param conversationId - The conversation ID
 * @param messageId - The message ID
 * @returns Promise with the message data
 */
export const getMessage = async (
  conversationId: number,
  messageId: number,
): Promise<DBMessage> => {
  const response = await axios.get<DBMessage>(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/${messageId}`,
  );
  return response.data;
};

/**
 * Updates a message
 *
 * @param conversationId - The conversation ID
 * @param messageId - The message ID
 * @param updates - The message updates (role and/or content)
 * @returns Promise with the updated message
 */
export const updateMessage = async (
  conversationId: number,
  messageId: number,
  updates: { role?: string; content?: string },
): Promise<DBMessage> => {
  const response = await axios.put<DBMessage>(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/${messageId}`,
    updates,
  );
  return response.data;
};

/**
 * Deletes a message
 *
 * @param conversationId - The conversation ID
 * @param messageId - The message ID
 * @returns Promise with the deletion result
 */
export const deleteMessage = async (
  conversationId: number,
  messageId: number,
): Promise<void> => {
  await axios.delete(
    `${API_BASE_URL}/api/conversations/${conversationId}/messages/${messageId}`,
  );
};

/**
 * Deletes batch of messages
 * 
 * @param messageIds - Array of message IDs to delete
 * @returns Promise with the deletion result
 */

export const deleteMessages = async (
  messageIds: number[]
) : Promise<void> => {

  await axios.delete(
    `${API_BASE_URL}/api/messages/batch`,
    {
      data: { messageIds }
    }
  )

}

/**
 * Converts database messages to the API message format used for chat completion
 *
 * @param dbMessages - Array of database messages
 * @returns Array of messages in the format expected by chat completion API
 */
export const convertToApiMessages = (dbMessages: DBMessage[]): Message[] => {
  return dbMessages.map((msg) => ({
    id: msg.id,
    role: msg.role,
    content: msg.content,
  }));
};
