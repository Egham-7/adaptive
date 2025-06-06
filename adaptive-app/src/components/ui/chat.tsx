import {
  forwardRef,
  useCallback,
  useRef,
  useState,
  type ReactElement,
} from "react";
import { ArrowDown, ThumbsDown, ThumbsUp } from "lucide-react";

import { cn } from "@/lib/utils";
import { useAutoScroll } from "@/hooks/use-auto-scroll";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { MessageInput } from "@/components/ui/message-input";
import { MessageList } from "@/components/ui/message-list";
import { PromptSuggestions } from "@/components/ui/prompt-suggestions";
// Import UIMessage and other relevant types from 'ai' package
import type { UIMessage } from "ai";

// Infer specific part types from UIMessage for safety and clarity
type MessageTextPart = Extract<UIMessage["parts"][number], { type: "text" }>;
type MessageToolInvocationPart = Extract<
  UIMessage["parts"][number],
  { type: "tool-invocation" }
>;

interface ChatPropsBase {
  handleSubmit: (
    event?: { preventDefault?: () => void },
    options?: { experimental_attachments?: FileList },
  ) => void;
  messages: Array<UIMessage>;
  input: string;
  className?: string;
  handleInputChange: React.ChangeEventHandler<HTMLTextAreaElement>;
  isGenerating: boolean;
  stop?: () => void;
  onRateResponse?: (
    messageId: string,
    rating: "thumbs-up" | "thumbs-down",
  ) => void;
  setMessages?: (messages: UIMessage[]) => void;
  transcribeAudio?: (blob: Blob) => Promise<string>;
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
}: ChatProps) {
  const lastMessage = messages.at(-1);
  const isEmpty = messages.length === 0;
  const isTyping = lastMessage?.role === "user";

  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const handleStop = useCallback(() => {
    stop?.();

    if (!setMessages) return;

    const latestMessages = [...messagesRef.current];
    const lastAssistantMessage = latestMessages.findLast(
      (m: UIMessage) => m.role === "assistant",
    );

    if (!lastAssistantMessage) return;

    let needsUpdate = false;
    let updatedMessage: UIMessage = { ...lastAssistantMessage };

    // Handle deprecated `toolInvocations` field directly,
    // though preferring `parts` is recommended by AI SDK.
    if (lastAssistantMessage.toolInvocations) {
      const updatedToolInvocations = lastAssistantMessage.toolInvocations.map(
        (toolInvocation) => {
          if (toolInvocation.state === "call") {
            needsUpdate = true;
            return {
              ...toolInvocation,
              state: "result",
              result: {
                content: "Tool execution was cancelled",
                __cancelled: true,
              },
            } as const; // This cast keeps the specific literal type
          }
          return toolInvocation;
        },
      );

      if (needsUpdate) {
        updatedMessage = {
          ...updatedMessage,
          toolInvocations: updatedToolInvocations,
        };
      }
    }

    if (lastAssistantMessage.parts && lastAssistantMessage.parts.length > 0) {
      const updatedParts = lastAssistantMessage.parts.map((part) => {
        // Use type guard to safely access properties of ToolInvocationUIPart
        if (
          part.type === "tool-invocation" &&
          (part as MessageToolInvocationPart).toolInvocation &&
          (part as MessageToolInvocationPart).toolInvocation.state === "call"
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
          } as MessageToolInvocationPart; // Cast back to the specific part type
        }
        return part;
      });

      if (needsUpdate) {
        updatedMessage = {
          ...updatedMessage,
          parts: updatedParts,
        };
      }
    }

    if (needsUpdate) {
      const messageIndex = latestMessages.findIndex(
        (m) => m.id === lastAssistantMessage.id,
      );
      if (messageIndex !== -1) {
        latestMessages[messageIndex] = updatedMessage;
        setMessages(latestMessages);
      }
    }
  }, [stop, setMessages, messagesRef]);

  const messageOptions = useCallback(
    (message: UIMessage) => ({
      actions: onRateResponse ? (
        <>
          <div className="border-r pr-1">
            <CopyButton
              // Safely extract text content from the parts array
              content={
                (
                  message.parts?.find(
                    (p) => p.type === "text",
                  ) as MessageTextPart
                )?.text || ""
              }
              copyMessage="Copied response to clipboard!"
            />
          </div>
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={() => onRateResponse(message.id, "thumbs-up")}
          >
            <ThumbsUp className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={() => onRateResponse(message.id, "thumbs-down")}
          >
            <ThumbsDown className="h-4 w-4" />
          </Button>
        </>
      ) : (
        <CopyButton
          content={
            (message.parts?.find((p) => p.type === "text") as MessageTextPart)
              ?.text || ""
          }
          copyMessage="Copied response to clipboard!"
        />
      ),
    }),
    [onRateResponse],
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

      {messages.length > 0 ? (
        <ChatMessages messages={messages}>
          <MessageList
            messages={messages} // Now consistently UIMessage[]
            isTyping={isTyping}
            messageOptions={messageOptions}
          />
        </ChatMessages>
      ) : null}

      <ChatForm
        className="mt-auto"
        isPending={isGenerating || isTyping}
        handleSubmit={handleSubmit}
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
          />
        )}
      </ChatForm>
    </ChatContainer>
  );
}

export function ChatMessages({
  messages,
  children,
}: React.PropsWithChildren<{
  messages: UIMessage[]; // Consistently use UIMessage[]
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
              className="pointer-events-auto h-8 w-8 rounded-full ease-in-out animate-in fade-in-0 slide-in-from-bottom-1"
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
    options?: { experimental_attachments?: FileList },
  ) => void;
  children: (props: {
    files: File[] | null;
    setFiles: React.Dispatch<React.SetStateAction<File[] | null>>;
  }) => ReactElement;
}

export const ChatForm = forwardRef<HTMLFormElement, ChatFormProps>(
  ({ children, handleSubmit, isPending, className }, ref) => {
    const [files, setFiles] = useState<File[] | null>(null);

    const onSubmit = (event: React.FormEvent) => {
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
  },
);
ChatForm.displayName = "ChatForm";

function createFileList(files: File[] | FileList): FileList {
  const dataTransfer = new DataTransfer();
  for (const file of Array.from(files)) {
    dataTransfer.items.add(file);
  }
  return dataTransfer.files;
}
