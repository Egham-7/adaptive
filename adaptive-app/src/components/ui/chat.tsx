import { useDeleteMessage } from "@/hooks/messages/use-delete-message";
import {
  AlertTriangle,
  ArrowDown,
  Edit3,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
  Trash2,
} from "lucide-react";
import {
  type ReactElement,
  forwardRef,
  useCallback,
  useReducer,
  useRef,
  useState,
  useEffect,
} from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { MessageInput } from "@/components/ui/message-input";
import { MessageList } from "@/components/ui/message-list";
import { PromptSuggestions } from "@/components/ui/prompt-suggestions";
import { useAutoScroll } from "@/hooks/use-auto-scroll";
import { cn } from "@/lib/utils";
import type { Message, UIMessage } from "ai";
import { DAILY_MESSAGE_LIMIT } from "@/lib/chat/message-limits";
import SubscribeButton from "@/app/_components/stripe/subscribe-button";

// Infer specific part types from UIMessage for safety and clarity
type MessageTextPart = Extract<UIMessage["parts"][number], { type: "text" }>;
type MessageToolInvocationPart = Extract<
  UIMessage["parts"][number],
  { type: "tool-invocation" }
>;

type MessageAction =
  | { type: "CANCEL_TOOL_INVOCATIONS"; messageId: string }
  | { type: "DELETE_MESSAGE_AND_AFTER"; messageId: string }
  | { type: "DELETE_MESSAGES_AFTER"; messageIndex: number }
  | { type: "SET_MESSAGES"; messages: Message[] }
  | { type: "EDIT_MESSAGE"; messageId: string; content: string }
  | {
      type: "RATE_MESSAGE";
      messageId: string;
      rating: "thumbs-up" | "thumbs-down";
    }
  | { type: "RETRY_MESSAGE"; messageId: string; content: string }
  | { type: "ADD_MESSAGE"; message: Message };

interface MessageState {
  messages: Message[];
  editingMessageId: string | null;
  editingContent: string;
}

function messageReducer(
  state: MessageState,
  action: MessageAction
): MessageState {
  switch (action.type) {
    case "CANCEL_TOOL_INVOCATIONS": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId
      );
      if (messageIndex === -1) return state;

      const message = state.messages[messageIndex] as UIMessage;
      if (!message.parts?.length) return state;

      let needsUpdate = false;
      const updatedParts = message.parts.map((part) => {
        if (
          part.type === "tool-invocation" &&
          (part as MessageToolInvocationPart).toolInvocation?.state === "call"
        ) {
          needsUpdate = true;
          return {
            ...part,
            toolInvocation: {
              ...(part as MessageToolInvocationPart).toolInvocation,
              state: "result",
              result: {
                content: "Tool execution was cancelled",
                __cancelled: true,
              },
            },
          } as MessageToolInvocationPart;
        }
        return part;
      });

      if (!needsUpdate) return state;

      const newMessages = [...state.messages];
      newMessages[messageIndex] = {
        ...message,
        parts: updatedParts,
      };

      return {
        ...state,
        messages: newMessages,
      };
    }

    case "DELETE_MESSAGE_AND_AFTER": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId
      );
      const newMessages =
        messageIndex === -1
          ? state.messages
          : state.messages.slice(0, messageIndex);

      return {
        ...state,
        messages: newMessages,
        editingMessageId: null,
        editingContent: "",
      };
    }

    case "DELETE_MESSAGES_AFTER": {
      return {
        ...state,
        messages: state.messages.slice(0, action.messageIndex),
        editingMessageId: null,
        editingContent: "",
      };
    }

    case "SET_MESSAGES": {
      return {
        ...state,
        messages: action.messages,
      };
    }

    case "EDIT_MESSAGE": {
      return {
        ...state,
        editingMessageId: action.messageId,
        editingContent: action.content,
      };
    }

    case "RATE_MESSAGE": {
      // Note: This would typically update message metadata or trigger external API call
      // For now, we'll just return the current state as rating is handled externally
      return state;
    }

    case "RETRY_MESSAGE": {
      const messageIndex = state.messages.findIndex(
        (m) => m.id === action.messageId
      );
      const newMessages =
        messageIndex === -1
          ? state.messages
          : state.messages.slice(0, messageIndex);

      return {
        ...state,
        messages: newMessages,
        editingMessageId: null,
        editingContent: "",
      };
    }

    case "ADD_MESSAGE": {
      return {
        ...state,
        messages: [...state.messages, action.message],
      };
    }

    default:
      return state;
  }
}

interface ChatPropsBase {
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { experimental_attachments?: FileList }
  ) => void;
  messages: Array<UIMessage>;
  input: string;
  className?: string;
  handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  isGenerating: boolean;
  stop?: () => void;
  onRateResponse?: (
    messageId: string,
    rating: "thumbs-up" | "thumbs-down"
  ) => void;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  transcribeAudio?: (blob: Blob) => Promise<string>;
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
  hasReachedLimit?: boolean;
  remainingMessages?: number;
  isUnlimited?: boolean;
  limitsLoading?: boolean;
  userId?: string; // Add this
}

interface ChatPropsWithoutSuggestions extends ChatPropsBase {
  append?: never;
  suggestions?: never;
}

interface ChatPropsWithSuggestions extends ChatPropsBase {
  append: (message: { role: "user"; content: string }) => void;
  suggestions: string[];
}

type ChatProps = ChatPropsWithoutSuggestions | ChatPropsWithSuggestions;

export function Chat({
  messages,
  handleSubmit,
  input,
  handleInputChange,
  stop,
  isGenerating,
  append,
  suggestions,
  className,
  onRateResponse,
  setMessages,
  transcribeAudio,
  isError,
  error,
  onRetry,
  hasReachedLimit = false,
  remainingMessages,
  isUnlimited = true,
  limitsLoading = false,
  userId, // Add this parameter
}: ChatProps & { userId?: string }) {
  const deleteMessageMutation = useDeleteMessage();
  const [state, dispatch] = useReducer(messageReducer, {
    messages,
    editingMessageId: null,
    editingContent: "",
  });

  // Add local state for optimistic updates
  const [optimisticRemainingMessages, setOptimisticRemainingMessages] =
    useState(remainingMessages);

  // Sync with prop changes
  useEffect(() => {
    setOptimisticRemainingMessages(remainingMessages);
  }, [remainingMessages]);

  // Optimistically update counter when messages change
  const messagesRef = useRef(messages);
  useEffect(() => {
    if (
      messagesRef.current.length < messages.length &&
      remainingMessages !== undefined
    ) {
      // New message added, optimistically decrease counter
      const newUserMessages = messages.filter((m) => m.role === "user").length;
      const oldUserMessages = messagesRef.current.filter(
        (m) => m.role === "user"
      ).length;

      if (newUserMessages > oldUserMessages) {
        setOptimisticRemainingMessages((prev) => Math.max(0, (prev || 0) - 1));
      }
    }
    messagesRef.current = messages;
  }, [messages, remainingMessages]);

  // Use optimistic value for display
  const displayRemainingMessages =
    optimisticRemainingMessages ?? remainingMessages;
  const usedMessages =
    displayRemainingMessages !== undefined
      ? DAILY_MESSAGE_LIMIT - displayRemainingMessages
      : 0;

  // Sync external messages with internal state
  const externalMessagesRef = useRef(messages);
  if (externalMessagesRef.current !== messages) {
    externalMessagesRef.current = messages;
    dispatch({ type: "SET_MESSAGES", messages });
  }

  // Sync internal state changes back to external setMessages
  const internalMessagesRef = useRef(state.messages);
  if (internalMessagesRef.current !== state.messages) {
    internalMessagesRef.current = state.messages;
    setMessages(state.messages);
  }

  const lastMessage = state.messages.at(-1);
  const isEmpty = state.messages.length === 0;
  const isTyping = lastMessage?.role === "user";

  const handleStop = useCallback(() => {
    stop?.();

    const lastAssistantMessage = state.messages.findLast(
      (m: Message) => m.role === "assistant"
    );

    if (lastAssistantMessage) {
      dispatch({
        type: "CANCEL_TOOL_INVOCATIONS",
        messageId: lastAssistantMessage.id,
      });
    }
  }, [stop, state.messages]);

  const handleEditMessage = useCallback(
    (messageId: string, content: string) => {
      dispatch({
        type: "EDIT_MESSAGE",
        messageId,
        content,
      });
    },
    []
  );

  const handleSaveEdit = useCallback(
    (messageId: string) => {
      if (!state.editingContent.trim()) return;

      const messageIndex = state.messages.findIndex((m) => m.id === messageId);
      if (messageIndex !== -1) {
        // Delete subsequent messages from database
        const messagesToDelete = state.messages.slice(messageIndex);
        for (const msg of messagesToDelete) {
          deleteMessageMutation.mutate({ id: msg.id });
        }

        setMessages(messages.slice(0, messageIndex));
        append?.({
          role: "user",
          content: state.editingContent.trim(),
        });

        // Clear editing state
        dispatch({ type: "EDIT_MESSAGE", messageId: "", content: "" });
      }
    },
    [
      state.messages,
      state.editingContent,
      deleteMessageMutation,
      setMessages,
      messages,
      append,
    ]
  );

  const handleCancelEdit = useCallback(() => {
    dispatch({
      type: "EDIT_MESSAGE",
      messageId: "",
      content: "",
    });
  }, []);

  const handleRetryMessage = useCallback(
    (message: UIMessage) => {
      if (!append) return;

      const messageIndex = state.messages.findIndex((m) => m.id === message.id);
      if (messageIndex !== -1) {
        // Delete subsequent messages from database
        const messagesToDelete = state.messages.slice(messageIndex);
        for (const msg of messagesToDelete) {
          deleteMessageMutation.mutate({ id: msg.id });
        }
      }

      dispatch({
        type: "RETRY_MESSAGE",
        messageId: message.id,
        content: message.content,
      });

      // Re-send the message
      append({ role: "user", content: message.content });
    },
    [append, state.messages, deleteMessageMutation]
  );

  const handleDeleteMessage = useCallback(
    (messageId: string) => {
      const messageIndex = state.messages.findIndex((m) => m.id === messageId);
      if (messageIndex !== -1) {
        // Delete this message and all subsequent messages from database
        const messagesToDelete = state.messages.slice(messageIndex);
        for (const msg of messagesToDelete) {
          deleteMessageMutation.mutate({ id: msg.id });
        }
      }

      dispatch({
        type: "DELETE_MESSAGE_AND_AFTER",
        messageId,
      });
    },
    [state.messages, deleteMessageMutation]
  );

  const handleRateMessage = useCallback(
    (messageId: string, rating: "thumbs-up" | "thumbs-down") => {
      dispatch({
        type: "RATE_MESSAGE",
        messageId,
        rating,
      });

      // Call external rating handler
      onRateResponse?.(messageId, rating);
    },
    [onRateResponse]
  );

  const messageOptions = useCallback(
    (message: UIMessage) => {
      const isUserMessage = message.role === "user";
      const canEdit = isUserMessage && !isGenerating;
      const canRetry = isUserMessage && !isGenerating;
      const canDelete = !isGenerating;

      if (isUserMessage) {
        return {
          actions: (
            <>
              {canEdit && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => handleEditMessage(message.id, message.content)}
                  disabled={state.editingMessageId === message.id}
                >
                  <Edit3 className="h-4 w-4" />
                </Button>
              )}
              {canRetry && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6"
                  onClick={() => handleRetryMessage(message)}
                >
                  <RotateCcw className="h-4 w-4" />
                </Button>
              )}
              {canDelete && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 text-destructive hover:text-destructive"
                  onClick={() => handleDeleteMessage(message.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </>
          ),
          isEditing: state.editingMessageId === message.id,
          editingContent: state.editingContent,
          onEditingContentChange: (content: string) =>
            dispatch({
              type: "EDIT_MESSAGE",
              messageId: state.editingMessageId || "",
              content,
            }),
          onSaveEdit: () => handleSaveEdit(message.id),
          onCancelEdit: handleCancelEdit,
        };
      }

      // Assistant message actions
      return {
        actions: onRateResponse ? (
          <>
            <div className="border-r pr-1">
              <CopyButton
                content={
                  (
                    message.parts?.find(
                      (p) => p.type === "text"
                    ) as MessageTextPart
                  )?.text || message.content
                }
                copyMessage="Copied response to clipboard!"
              />
            </div>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => handleRateMessage(message.id, "thumbs-up")}
            >
              <ThumbsUp className="h-4 w-4" />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={() => handleRateMessage(message.id, "thumbs-down")}
            >
              <ThumbsDown className="h-4 w-4" />
            </Button>
          </>
        ) : (
          <CopyButton
            content={
              (message.parts?.find((p) => p.type === "text") as MessageTextPart)
                ?.text || message.content
            }
            copyMessage="Copied response to clipboard!"
          />
        ),
      };
    },
    [
      onRateResponse,
      isGenerating,
      state.editingMessageId,
      state.editingContent,
      handleEditMessage,
      handleSaveEdit,
      handleCancelEdit,
      handleRetryMessage,
      handleDeleteMessage,
      handleRateMessage,
    ]
  );

  return (
    <ChatContainer className={className}>
      {isEmpty && append && suggestions ? (
        <PromptSuggestions
          label="Try these prompts ✨"
          append={append}
          suggestions={suggestions}
        />
      ) : null}

      {state.messages.length > 0 ? (
        <ChatMessages messages={state.messages as UIMessage[]}>
          <MessageList
            messages={state.messages as UIMessage[]}
            isTyping={isTyping}
            messageOptions={messageOptions}
          />
        </ChatMessages>
      ) : null}

      {isError && <ChatErrorDisplay error={error} onRetry={onRetry} />}

      {/* Message Counter - use optimistic value */}
      {!limitsLoading &&
        !isUnlimited &&
        displayRemainingMessages !== undefined && (
          <div className="mx-4 mb-2 text-center">
            <span
              className={`text-xs px-2 py-1 rounded-full ${
                hasReachedLimit
                  ? "bg-red-100 text-red-700"
                  : displayRemainingMessages <= 2
                  ? "bg-orange-100 text-orange-700"
                  : "bg-gray-100 text-gray-600"
              }`}
            >
              {usedMessages}/{DAILY_MESSAGE_LIMIT} messages used today
            </span>
          </div>
        )}
      {/* Message limit warning - use optimistic value */}
      {!limitsLoading &&
        !isUnlimited &&
        displayRemainingMessages !== undefined &&
        displayRemainingMessages <= 5 && (
          <div className="mx-4 mt-4 text-center">
            <div className="inline-block rounded-lg border border-orange-200 bg-orange-50 p-4 text-left">
              <p className="text-sm text-orange-800">
                {displayRemainingMessages > 0
                  ? `You have ${displayRemainingMessages} messages remaining today.`
                  : "You've reached your daily message limit."}
                {userId && (
                  <SubscribeButton
                    userId={userId}
                    variant="link"
                    className="ml-1 text-orange-800 hover:text-orange-900"
                  >
                    Upgrade to Pro
                  </SubscribeButton>
                )}{" "}
                for unlimited messages.
              </p>
            </div>
          </div>
        )}

      <ChatForm
        className="mt-auto"
        isPending={isGenerating || isTyping}
        handleSubmit={handleSubmit}
        hasReachedLimit={hasReachedLimit}
      >
        {({ files, setFiles }) => (
          <MessageInput
            value={input}
            onChange={handleInputChange}
            allowAttachments
            files={files}
            setFiles={setFiles}
            stop={handleStop}
            isGenerating={isGenerating}
            transcribeAudio={transcribeAudio}
            disabled={hasReachedLimit}
            placeholder={
              hasReachedLimit
                ? "Daily message limit reached - upgrade to continue"
                : undefined
            }
          />
        )}
      </ChatForm>
    </ChatContainer>
  );
}

function ChatErrorDisplay({
  error,
  onRetry,
}: {
  error?: Error;
  onRetry?: () => void;
}) {
  if (!error) return null;

  // Extract error message from JSON if needed
  const getErrorMessage = (error: Error): string => {
    try {
      const parsed = JSON.parse(error.message);
      if (parsed.error) {
        return parsed.error;
      }
      if (parsed.message) {
        return parsed.message;
      }
    } catch {
      // Not JSON, continue with original message
    }

    return error.message || "An unexpected error occurred.";
  };

  const errorMessage = getErrorMessage(error);
  const isLimitError = errorMessage
    .toLowerCase()
    .includes("daily message limit");

  return (
    <div className="w-full max-w-2xl px-4 pb-4">
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          <p>{errorMessage}</p>
          {onRetry && !isLimitError && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onRetry}
              className="mt-2"
            >
              Retry
            </Button>
          )}
        </AlertDescription>
      </Alert>
    </div>
  );
}

export function ChatMessages({
  messages,
  children,
}: React.PropsWithChildren<{
  messages: UIMessage[];
}>) {
  const {
    containerRef,
    scrollToBottom,
    handleScroll,
    shouldAutoScroll,
    handleTouchStart,
  } = useAutoScroll([messages]);

  return (
    <div
      className="grid grid-cols-1 overflow-y-auto pb-4"
      ref={containerRef}
      onScroll={handleScroll}
      onTouchStart={handleTouchStart}
    >
      <div className="max-w-full [grid-column:1/1] [grid-row:1/1]">
        {children}
      </div>

      {!shouldAutoScroll && (
        <div className="pointer-events-none flex flex-1 items-end justify-end [grid-column:1/1] [grid-row:1/1]">
          <div className="sticky bottom-0 left-0 flex w-full justify-end">
            <Button
              onClick={scrollToBottom}
              className="fade-in-0 slide-in-from-bottom-1 pointer-events-auto h-8 w-8 animate-in rounded-full ease-in-out"
              size="icon"
              variant="ghost"
            >
              <ArrowDown className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

export const ChatContainer = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  return (
    <div
      ref={ref}
      className={cn("grid max-h-full w-full grid-rows-[1fr_auto]", className)}
      {...props}
    />
  );
});
ChatContainer.displayName = "ChatContainer";

interface ChatFormProps {
  className?: string;
  isPending: boolean;
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { experimental_attachments?: FileList }
  ) => void;
  children: (props: {
    files: File[] | null;
    setFiles: React.Dispatch<React.SetStateAction<File[] | null>>;
  }) => ReactElement;
  hasReachedLimit?: boolean;
}

export const ChatForm = forwardRef<HTMLFormElement, ChatFormProps>(
  ({ children, handleSubmit, className, hasReachedLimit = false }, ref) => {
    const [files, setFiles] = useState<File[] | null>(null);

    const onSubmit = (event: React.FormEvent) => {
      // Prevent submission if limit is reached
      if (hasReachedLimit) {
        event.preventDefault();
        return;
      }

      if (!files) {
        handleSubmit(event);
        return;
      }

      const fileList = createFileList(files);
      handleSubmit(event, { experimental_attachments: fileList });
      setFiles(null);
    };

    return (
      <form ref={ref} onSubmit={onSubmit} className={className}>
        {children({ files, setFiles })}
      </form>
    );
  }
);
ChatForm.displayName = "ChatForm";

function createFileList(files: File[] | FileList): FileList {
  const dataTransfer = new DataTransfer();
  for (const file of Array.from(files)) {
    dataTransfer.items.add(file);
  }
  return dataTransfer.files;
}
