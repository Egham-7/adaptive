import type { UIMessage } from "@ai-sdk/react";
import type { ReactElement } from "react";

// Message part types
export type MessageTextPart = Extract<
  UIMessage["parts"][number],
  { type: "text" }
>;
export type MessageToolPart = Extract<
  UIMessage["parts"][number],
  { type: `tool-${string}` }
>;

// State management types
export type MessageAction =
  | { type: "CANCEL_TOOL_INVOCATIONS"; messageId: string }
  | { type: "DELETE_MESSAGE_AND_AFTER"; messageId: string }
  | { type: "SET_MESSAGES"; messages: UIMessage[] }
  | { type: "EDIT_MESSAGE"; messageId: string; content: string }
  | { type: "RETRY_MESSAGE"; messageId: string }
  | { type: "CLEAR_EDITING" };

export interface MessageState {
  messages: UIMessage[];
  editingMessageId: string | null;
  editingContent: string;
}

// Core chat configuration
export interface ChatConfig {
  showWelcomeInterface?: boolean;
  suggestions: string[];
  userId?: string;
}

// Input handling types
export interface ChatInputProps {
  input: string;
  handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { files?: FileList },
  ) => void;
  transcribeAudio?: (blob: Blob) => Promise<string>;
}

// Message handling types
export interface ChatMessageProps {
  messages: UIMessage[];
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  sendMessage: (message: { text: string }) => void;
  isGenerating: boolean;
  stop?: () => void;
}

// Status and limits types
export interface ChatLimitsProps {
  hasReachedLimit?: boolean;
  remainingMessages?: number;
  isUnlimited?: boolean;
  limitsLoading?: boolean;
}

// Error handling types
export interface ChatErrorProps {
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
}

// Rating functionality types
export interface ChatRatingProps {
  onRateResponse?: (
    messageId: string,
    rating: "thumbs-up" | "thumbs-down",
  ) => void;
}

// Main chat component props
export interface ChatProps 
  extends ChatConfig, 
          ChatInputProps, 
          ChatMessageProps, 
          ChatLimitsProps, 
          ChatErrorProps, 
          ChatRatingProps {
  className?: string;
}

export interface ChatFormProps {
  className?: string;
  isPending: boolean;
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { files?: FileList },
  ) => void;
  children: (props: {
    files: File[] | null;
    setFiles: React.Dispatch<React.SetStateAction<File[] | null>>;
  }) => ReactElement;
  hasReachedLimit?: boolean;
}

