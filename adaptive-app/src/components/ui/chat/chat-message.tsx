"use client";

import { type VariantProps, cva } from "class-variance-authority";
import { motion } from "framer-motion";
import {
  Check,
  ChevronRight,
  RotateCcw,
  Terminal,
  X,
  Search,
  ExternalLink,
  Globe,
} from "lucide-react";
import React, { useMemo, useState } from "react";
import Image from "next/image";

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { FilePreview } from "./file-preview";
import { DotsLoader, TypingLoader } from "./loader";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";
import { Textarea } from "@/components/ui/textarea";
import { useAnimatedText } from "@/components/ui/animated-text";
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers";
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
  "relative break-words text-sm transition-colors",
  {
    variants: {
      isUser: {
        true: "bg-primary text-primary-foreground  max-w-max rounded-lg p-4",
        false: "text-foreground w-full max-w-none p-0 bg-transparent",
      },
      animation: {
        none: "",
        slide: "fade-in-0 animate-in duration-300",
        scale: "fade-in-0 zoom-in-75 animate-in duration-300",
        fade: "fade-in-0 animate-in duration-500",
      },
    },
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
  isError?: boolean;
  error?: Error;
  onRetryError?: () => void;
  isStreaming?: boolean;
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
  isError = false,
  error,
  onRetryError,
  isStreaming = false,
  ...message
}) => {
  const content =
    (parts?.find((p) => p.type === "text") as TextUIPart)?.text || "";
  const animatedContent = useAnimatedText(content, " ");

  const provider = (message.metadata as any)?.providerMetadata?.adaptive
    ?.provider;
  const modelId = (message.metadata as any)?.response?.modelId;

  const createdAt =
    message.metadata &&
    typeof message.metadata === "object" &&
    "timestamp" in message.metadata &&
    typeof message.metadata.timestamp === "number"
      ? new Date(message.metadata.timestamp)
      : undefined;

  console.log("Message metadata:", message.metadata);

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
      <div className="w-full">
        {userFiles && userFiles.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2 justify-start">
            {userFiles.map((file) => (
              <FilePreview key={`${file.name}-${file.size}`} file={file} />
            ))}
          </div>
        )}

        <div className="relative group/message">
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
          </div>

          {actions && !isEditing && (
            <div className="mt-2 flex justify-start">
              <div className="flex space-x-1 p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100 z-20 shadow-sm">
                {actions}
              </div>
            </div>
          )}
        </div>

        {showTimeStamp && createdAt && (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50 text-right",
              animation !== "none" && "fade-in-0 animate-in duration-500",
            )}
          >
            {formattedTime}
          </time>
        )}
      </div>
    );
  }

  if (role === "assistant" && parts && parts.length > 0) {
    return (
      <div className="w-full">
        {parts.map((part, index) => {
          if (part.type === "text") {
            return (
              // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
              <React.Fragment key={`text-${index}`}>
                <div className="relative group/message w-full">
                  <div
                    className={cn(chatBubbleVariants({ isUser, animation }))}
                  >
                    <MarkdownRenderer>
                      {isStreaming ? animatedContent : part.text}
                    </MarkdownRenderer>
                  </div>
                  {index === parts.length - 1 && (
                    <div className="mt-2 opacity-0 transition-opacity group-hover/message:opacity-100 z-20">
                      {actions}
                    </div>
                  )}
                </div>
                {showTimeStamp && createdAt && index === parts.length - 1 && (
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
                )}
              </React.Fragment>
            );
          }
          if (part.type === "reasoning") {
            // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
            return <ReasoningBlock key={`reasoning-${index}`} part={part} />;
          }
          if (part.type === "tool-invocation") {
            const toolInvocation = part as ToolInvocationUIPart;
            const { toolName, state } = toolInvocation.toolInvocation;

            // Render tool-specific UI based on tool name
            if (toolName === "webSearch") {
              return (
                // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
                <WebSearchBlock
                  key={`websearch-${index}`}
                  toolPart={toolInvocation}
                />
              );
            }

            // Default fallback for unknown tools
            return (
              // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
              <div
                key={`tool-${index}`}
                className="mb-2 rounded-lg border bg-muted/50 px-3 py-2 text-sm"
              >
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Terminal className="h-4 w-4" />
                  <span>Tool: {toolName}</span>
                  {state === "call" && <DotsLoader size="sm" />}
                </div>
              </div>
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
      <div className="w-full">
        <div className="relative group/message w-full">
          <MarkdownRenderer>
            {isStreaming ? animatedContent : content}
          </MarkdownRenderer>

          <>
            {(provider || modelId) && (
              <div className="flex justify-between items-center mt-4 text-xs text-muted-foreground gap-3">
                <div className="flex items-center gap-3">
                  {provider && (
                    <div className="flex items-center gap-1">
                      {getProviderLogo(provider) && (
                        <Image
                          src={getProviderLogo(provider)!}
                          alt={provider}
                          width={16}
                          height={16}
                          className="rounded-sm"
                        />
                      )}
                      <span>{getProviderDisplayName(provider)}</span>
                    </div>
                  )}
                  {modelId && <span>Model: {modelId}</span>}
                </div>
                <div className="flex justify-end items-center  opacity-0 transition-opacity group-hover/message:opacity-100 z-20 ">
                  {actions}
                </div>
              </div>
            )}
          </>
        </div>

        {showTimeStamp && createdAt && (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50",
              animation !== "none" && "fade-in-0 animate-in duration-500",
            )}
          >
            {formattedTime}
          </time>
        )}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="w-full">
        <div
          className={cn(
            chatBubbleVariants({ isUser: false, animation }),
            "border-destructive/20 bg-destructive/10 text-destructive-foreground",
          )}
          role="alert"
          aria-live="polite"
        >
          <div className="flex items-start gap-3">
            <X className="h-4 w-4 mt-1 flex-shrink-0" aria-hidden="true" />
            <div className="flex-1">
              <h4 className="font-medium text-sm mb-1">
                Error generating response
              </h4>
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

  return <TypingLoader size="sm" className="text-muted-foreground" />;
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
    <div className="mb-2 max-w-3xl">
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

interface WebSearchBlockProps {
  toolPart: ToolInvocationUIPart;
}

function WebSearchBlock({ toolPart }: WebSearchBlockProps) {
  const { state } = toolPart.toolInvocation;
  const result =
    toolPart.toolInvocation.state === "result"
      ? toolPart.toolInvocation.result
      : undefined;
  const args =
    toolPart.toolInvocation.state === "call" ||
    toolPart.toolInvocation.state === "result"
      ? toolPart.toolInvocation.args
      : undefined;

  switch (state) {
    case "partial-call":
    case "call":
      return (
        <div className="mb-3 rounded-lg border bg-muted/50 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
              <Search className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-foreground">
                  Searching the web
                </span>
                <DotsLoader size="sm" />
              </div>
              {args?.query && (
                <p className="text-xs text-muted-foreground mt-1">
                  "{args.query}"
                </p>
              )}
            </div>
          </div>
        </div>
      );

    case "result": {
      if (!result) return null;

      const searchData = result as {
        query: string;
        results: Array<{ title: string; url: string; snippet: string }>;
      };

      return (
        <div className="mb-3 rounded-lg border bg-muted/50 px-4 py-3">
          <div className="flex items-center gap-3 mb-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-success/10">
              <Globe className="h-4 w-4 text-success" />
            </div>
            <div className="flex-1">
              <span className="text-sm font-medium text-foreground">
                Web Search Results
              </span>
              <p className="text-xs text-muted-foreground">
                "{searchData.query}"
              </p>
            </div>
          </div>

          <div className="space-y-3">
            {searchData.results.map((item, idx) => (
              <div
                key={idx}
                className="rounded-lg border bg-card p-3 transition-colors hover:bg-accent"
              >
                <div className="flex items-start gap-2">
                  <div className="flex-1">
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group flex items-center gap-2 text-sm font-medium text-foreground hover:text-primary transition-colors"
                    >
                      <span className="line-clamp-1">{item.title}</span>
                      <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>
                    <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                      {item.snippet}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground font-mono">
                      {new URL(item.url).hostname}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    default:
      return null;
  }
}
