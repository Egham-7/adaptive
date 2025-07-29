import type { UIMessage } from "@ai-sdk/react";
import type { ReactElement } from "react";
import type { ChatStatus, RatingType } from "./constants/chat-constants";

// Message part types
export type MessageTextPart = Extract<
  UIMessage["parts"][number],
  { type: "text" }
>;
export type MessageToolPart = Extract<
  UIMessage["parts"][number],
  { type: `tool-${string}` }
>;
export type MessageReasoningPart = Extract<
  UIMessage["parts"][number],
  { type: "reasoning" }
>;
export type MessageFilePart = Extract<
  UIMessage["parts"][number],
  { type: "file" }
>;

// Enhanced message capabilities
export interface MessageCapabilities {
  canEdit: boolean;
  canRetry: boolean;
  canDelete: boolean;
  canRate: boolean;
  isEditing: boolean;
}

// State management types
export type MessageAction =
  | { type: "CANCEL_TOOL_INVOCATIONS"; messageId: string }
  | { type: "DELETE_MESSAGE_AND_AFTER"; messageId: string }
  | { type: "SET_MESSAGES"; messages: UIMessage[] }
  | { type: "EDIT_MESSAGE"; messageId: string; content: string }
  | { type: "RETRY_MESSAGE"; messageId: string }
  | { type: "CLEAR_EDITING" }
  | { type: "UPDATE_EDITING_CONTENT"; messageId: string; content: string };

export interface MessageState {
  messages: UIMessage[];
  editingMessageId: string | null;
  editingContent: string;
}

// Computed state for better performance
export interface MessageComputedState {
  lastMessage: UIMessage | undefined;
  lastAssistantMessage: UIMessage | undefined;
  isEmpty: boolean;
  userMessageCount: number;
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
  handleSuggestionSubmit?: (text: string) => Promise<void>;
  sendMessage?: (options: { text: string }) => Promise<void>;
  transcribeAudio?: (blob: Blob) => Promise<string>;
}

// Message handling types
export interface ChatMessageProps {
  messages: UIMessage[];
  setMessages: React.Dispatch<React.SetStateAction<UIMessage[]>>;
  isGenerating: boolean;
  stop?: () => void;
}

// Status and limits types
export interface ChatLimitsState {
  hasReachedLimit: boolean;
  remainingMessages?: number;
  isUnlimited: boolean;
  limitsLoading: boolean;
  usedMessages: number;
  displayRemainingMessages?: number;
  shouldShowCounter: boolean;
  shouldShowWarning: boolean;
  limitStatus: ChatStatus;
}

export interface ChatLimitsProps {
  hasReachedLimit?: boolean;
  remainingMessages?: number;
  isUnlimited?: boolean;
  limitsLoading?: boolean;
}

// Error handling types
export interface ChatErrorState {
  isError: boolean;
  error?: Error;
  canRetry: boolean;
}

export interface ChatErrorProps {
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
}

// Rating functionality types
export interface ChatRatingState {
  canRate: boolean;
}

export interface ChatRatingProps {
  onRateResponse?: (messageId: string, rating: RatingType) => void;
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
