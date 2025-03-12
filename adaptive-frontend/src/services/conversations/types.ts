import { DBMessage } from "../messages/types";

/**
 * Conversation model
 */
export interface Conversation {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages?: DBMessage[];
}
