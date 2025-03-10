import { Message } from "../llms/types";

/**
 * Database Message model (extends the Message interface from types.ts)
 */
export interface DBMessage extends Message {
  id: number;
  conversation_id: number;
  created_at: string;
  updated_at: string;
}
