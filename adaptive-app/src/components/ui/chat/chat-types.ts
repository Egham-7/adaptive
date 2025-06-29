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

// Component prop types
export interface ChatPropsBase {
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { files?: FileList },
  ) => void;
  messages: UIMessage[];
  input: string;
  className?: string;
  handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  isGenerating: boolean;
  stop?: () => void;
  onRateResponse?: (
    messageId: string,
    rating: "thumbs-up" | "thumbs-down",
  ) => void;
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  transcribeAudio?: (blob: Blob) => Promise<string>;
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
  hasReachedLimit?: boolean;
  remainingMessages?: number;
  isUnlimited?: boolean;
  limitsLoading?: boolean;
  userId?: string;
  showWelcomeInterface?: boolean;
}

export interface ChatPropsWithSuggestions extends ChatPropsBase {
  sendMessage: (message: { text: string }) => void;
  suggestions: string[];
}

export type ChatProps = ChatPropsWithSuggestions;

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

