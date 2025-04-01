import { MessageRole } from "../llms/types";

/**
 * Base Message interface containing common properties
 * shared between DBMessage and CreateDBMessage
 */
export interface BaseMessage {
  role: MessageRole;
  content: string;
  provider?: string;
  model?: string;
}

/**
 * Database Message model (extends the Message interface from types.ts)
 */

export interface DBMessage extends BaseMessage {
  id: number;
  conversation_id: number;
  created_at: string;
  updated_at: string;
}

export type CreateDBMessage = BaseMessage;
