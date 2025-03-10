import { Message } from "../llms/types";

/**
 * Conversation model
 */
export interface Conversation {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages?: Message[];
}
