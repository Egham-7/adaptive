"use client";

import { type VariantProps, cva } from "class-variance-authority";
import { motion } from "framer-motion";
import { Check, ChevronRight, Code2, RotateCcw, Terminal, X } from "lucide-react";
import React, { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { FilePreview } from "./file-preview";
import { CircularLoader, DotsLoader } from "./loader";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { ResponseStream } from "./response-stream";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

import type { UIMessage } from "@ai-sdk/react";

// Define the specific part types we need
type TextUIPart = Extract<UIMessage["parts"][number], { type: "text" }>;
type ReasoningUIPart = Extract<
  UIMessage["parts"][number],
  { type: "reasoning" }
>;
type FileUIPart = Extract<UIMessage["parts"][number], { type: "file" }>;
type ToolInvocationUIPart = Extract<
  UIMessage["parts"][number],
  { type: "tool-invocation" }
>;

const chatBubbleVariants = cva(
  "group/message relative w-full max-w-[50%] break-words rounded-lg p-3 text-sm shadow-sm transition-colors",
  {
    variants: {
      isUser: {
        true: "bg-primary text-primary-foreground",
        false: "bg-muted text-foreground",
      },
      animation: {
        none: "",
        slide: "fade-in-0 animate-in duration-300",
        scale: "fade-in-0 zoom-in-75 animate-in duration-300",
        fade: "fade-in-0 animate-in duration-500",
      },
    },
    compoundVariants: [
      {
        isUser: true,
        animation: "slide",
        class: "slide-in-from-right",
      },
      {
        isUser: false,
        animation: "slide",
        class: "slide-in-from-left",
      },
      {
        isUser: true,
        animation: "scale",
        class: "origin-bottom-right",
      },
      {
        isUser: false,
        animation: "scale",
        class: "origin-bottom-left",
      },
    ],
  },
);

type Animation = VariantProps<typeof chatBubbleVariants>["animation"];

export interface ChatMessageProps extends UIMessage {
  showTimeStamp?: boolean;
  animation?: Animation;
  actions?: React.ReactNode;
  isEditing?: boolean;
  editingContent?: string;
  onEditingContentChange?: (content: string) => void;
  onSaveEdit?: () => void;
  onCancelEdit?: () => void;
  enableStreaming?: boolean;
  streamingMode?: "typewriter" | "fade";
  streamingSpeed?: number;
  isError?: boolean;
  error?: Error;
  onRetryError?: () => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  parts,
  showTimeStamp = false,
  animation = "scale",
  actions,
  isEditing = false,
  editingContent = "",
  onEditingContentChange,
  onSaveEdit,
  onCancelEdit,
  enableStreaming = false,
  streamingMode = "typewriter",
  streamingSpeed = 30,
  isError = false,
  error,
  onRetryError,
  ...message
}) => {
  const content =
    (parts?.find((p) => p.type === "text") as TextUIPart)?.text || "";
  const createdAt =
    message.metadata &&
    typeof message.metadata === "object" &&
    "timestamp" in message.metadata &&
    typeof message.metadata.timestamp === "number"
      ? new Date(message.metadata.timestamp)
      : undefined;
  const userFiles = useMemo(() => {
    if (role === "user" && parts) {
      return parts
        .filter((part): part is FileUIPart => part.type === "file")
        .map((filePart, index) => {
          if (filePart.url.startsWith("data:")) {
            const base64Content = filePart.url.split(",")[1];
            if (!base64Content) return null;
            return base64ToNewFile(
              base64Content,
              filePart.mediaType || "application/octet-stream",
              filePart.filename ?? `attachment-${index}`,
            );
          }
          return null;
        })
        .filter(Boolean) as File[];
    }
    return null;
  }, [role, parts]);

  const isUser = role === "user";
  const formattedTime = createdAt?.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (isUser) {
    return (
      <div
        className={cn("flex flex-col", isUser ? "items-end" : "items-start")}
      >
        {userFiles && userFiles.length > 0 ? (
          <div className="mb-1 flex flex-wrap gap-2">
            {userFiles.map((file) => {
              return (
                <FilePreview key={`${file.name}-${file.size}`} file={file} />
              );
            })}
          </div>
        ) : null}

        <div className={cn(chatBubbleVariants({ isUser, animation }))}>
          {isEditing ? (
            <div className="space-y-2">
              <Textarea
                value={editingContent}
                onChange={(e) => onEditingContentChange?.(e.target.value)}
                className="min-h-[150px] resize-none border-0 bg-transparent text-primary-foreground placeholder:text-primary-foreground/70 focus-visible:ring-0"
                placeholder="Edit your message..."
                autoFocus
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={onSaveEdit}
                  className="h-6 px-2"
                >
                  <Check className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={onCancelEdit}
                  className="h-6 px-2"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ) : (
            <MarkdownRenderer>{content}</MarkdownRenderer>
          )}

          {actions && !isEditing ? (
            <div className="-bottom-4 absolute right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
              {actions}
            </div>
          ) : null}
        </div>

        {showTimeStamp && createdAt ? (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50",
              animation !== "none" && "fade-in-0 animate-in duration-500",
            )}
          >
            {formattedTime}
          </time>
        ) : null}
      </div>
    );
  }

  if (role === "assistant" && parts && parts.length > 0) {
    return (
      <div className={cn("flex flex-col", "items-start")}>
        {parts.map((part, index) => {
          if (part.type === "text") {
            return (
              // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
              <React.Fragment key={`text-${index}`}>
                <div className={cn(chatBubbleVariants({ isUser, animation }))}>
                  {enableStreaming && index === parts.length - 1 ? (
                    <ResponseStream
                      textStream={part.text}
                      mode={streamingMode}
                      speed={streamingSpeed}
                      as="div"
                      className="[&>*]:text-inherit [&>*]:leading-inherit"
                    />
                  ) : (
                    <MarkdownRenderer>{part.text}</MarkdownRenderer>
                  )}
                  {actions && index === parts.length - 1 ? (
                    <div className="-bottom-4 absolute right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
                      {actions}
                    </div>
                  ) : null}
                </div>
                {showTimeStamp && createdAt && index === parts.length - 1 ? (
                  <time
                    dateTime={createdAt.toISOString()}
                    className={cn(
                      "mt-1 block px-1 text-xs opacity-50",
                      animation !== "none" &&
                        "fade-in-0 animate-in duration-500",
                    )}
                  >
                    {formattedTime}
                  </time>
                ) : null}
              </React.Fragment>
            );
          }
          if (part.type === "reasoning") {
            // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
            return <ReasoningBlock key={`reasoning-${index}`} part={part} />;
          }
          if (part.type === "tool-invocation") {
            return (
              <ToolCallBlock
                // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
                key={`tool-${index}`}
                toolPart={part as ToolInvocationUIPart}
              />
            );
          }
          if (part.type === "file") {
            const filePart = part as FileUIPart;
            const file = base64ToNewFile(
              filePart.url.split(",")[1] || filePart.url,
              filePart.mediaType,
              filePart.filename || `ai-file-${index}`,
            );
            return file ? (
              // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
              <div key={`file-${index}`} className="mb-2">
                <FilePreview file={file} />
              </div>
            ) : null;
          }
          return null;
        })}
      </div>
    );
  }

  if (role === "assistant" && content.length > 0) {
    return (
      <div className={cn("flex flex-col", "items-start")}>
        <div className={cn(chatBubbleVariants({ isUser, animation }))}>
          <MarkdownRenderer>{content}</MarkdownRenderer>
          {actions ? (
            <div className="-bottom-2 absolute right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
              {actions}
            </div>
          ) : null}
        </div>

        {showTimeStamp && createdAt ? (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50",
              animation !== "none" && "fade-in-0 animate-in duration-500",
            )}
          >
            {formattedTime}
          </time>
        ) : null}
      </div>
    );
  }

  if (isError) {
    return (
      <div className={cn("flex flex-col", "items-start")}>
        <div 
          className={cn(chatBubbleVariants({ isUser: false, animation }), "border-destructive/20 bg-destructive/10 text-destructive-foreground")}
          role="alert"
          aria-live="polite"
        >
          <div className="flex items-start gap-3">
            <X className="h-4 w-4 mt-1 flex-shrink-0" aria-hidden="true" />
            <div className="flex-1">
              <h4 className="font-medium text-sm mb-1">Error generating response</h4>
              <p className="text-xs opacity-90 mb-2">
                {error?.message || "Something went wrong. Please try again."}
              </p>
              {onRetryError && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onRetryError}
                  className="h-7 px-2 text-xs border-destructive/20 hover:bg-destructive/20"
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  Try again
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn(chatBubbleVariants({ isUser, animation }))}>
      <CircularLoader size="sm" className="text-muted-foreground" />
    </div>
  );
};

function base64ToUint8Array(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function base64ToNewFile(
  base64Data: string,
  mimeType: string,
  name: string,
): File | null {
  try {
    const uint8Array = base64ToUint8Array(base64Data);
    return new File([uint8Array], name, { type: mimeType });
  } catch (error) {
    console.error("Failed to convert base64 to File:", error);
    return null;
  }
}

const ReasoningBlock = ({ part }: { part: ReasoningUIPart }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mb-2 flex flex-col items-start sm:max-w-[70%]">
      <Collapsible
        open={isOpen}
        onOpenChange={setIsOpen}
        className="group w-full overflow-hidden rounded-lg border bg-muted/50"
      >
        <div className="flex items-center p-2">
          <CollapsibleTrigger asChild>
            <button
              type="button"
              className="flex items-center gap-2 text-muted-foreground text-sm hover:text-foreground"
            >
              <ChevronRight className="h-4 w-4 transition-transform group-data-[state=open]:rotate-90" />
              <span>Thinking</span>
            </button>
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent forceMount>
          <motion.div
            initial={false}
            animate={isOpen ? "open" : "closed"}
            variants={{
              open: { height: "auto", opacity: 1 },
              closed: { height: 0, opacity: 0 },
            }}
            transition={{ duration: 0.3, ease: [0.04, 0.62, 0.23, 0.98] }}
            className="border-t"
          >
            <div className="p-2">
              <div className="whitespace-pre-wrap text-xs">{part.text}</div>
            </div>
          </motion.div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

interface ToolCallBlockProps {
  toolPart: ToolInvocationUIPart;
}

function ToolCallBlock({ toolPart }: ToolCallBlockProps) {
  const { toolName, state, result } = toolPart;

  switch (state) {
    case "partial-call":
    case "call":
      return (
        <div className="flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-2 text-muted-foreground text-sm">
          <Terminal className="h-4 w-4" />
          <span>
            Calling{" "}
            <span className="font-mono">
              {"`"}
              {toolName}
              {"`"}
            </span>
            ...
          </span>
          <DotsLoader size="sm" />
        </div>
      );
    case "result":
      return (
        <div className="flex flex-col gap-1.5 rounded-lg border bg-muted/50 px-3 py-2 text-sm">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Code2 className="h-4 w-4" />
            <span>
              Result from{" "}
              <span className="font-mono">
                {"`"}
                {toolName}
                {"`"}
              </span>
            </span>
          </div>
          <pre className="overflow-x-auto whitespace-pre-wrap text-foreground">
            {JSON.stringify(result || {}, null, 2)}
          </pre>
        </div>
      );
    default:
      return null;
  }
}
