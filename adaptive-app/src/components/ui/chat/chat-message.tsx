"use client";

import { type VariantProps, cva } from "class-variance-authority";
import { motion } from "framer-motion";
import {
  Check,
  ChevronRight,
  ChevronDown,
  RotateCcw,
  Terminal,
  X,
  Search,
  Globe,
} from "lucide-react";
import React, { useMemo, useState } from "react";
import Image from "next/image";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
import { getProviderLogo, getProviderDisplayName } from "@/lib/providers/logos";
import { cn } from "@/lib/shared/utils";

import type { UIMessage } from "@ai-sdk/react";

// Extract part types from UIMessage
type MessagePart = UIMessage["parts"][number];
type TextUIPart = Extract<MessagePart, { type: "text" }>;
type ReasoningUIPart = Extract<MessagePart, { type: "reasoning" }>;
type FileUIPart = Extract<MessagePart, { type: "file" }>;

// Extract tool part types - any part that starts with "tool-"
type ToolPart = Extract<MessagePart, { type: `tool-${string}` }>;

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
    // Separate reasoning and other parts, ensuring reasoning comes first
    const reasoningParts = parts.filter((part) => part.type === "reasoning");
    const otherParts = parts.filter((part) => part.type !== "reasoning");
    const orderedParts = [...reasoningParts, ...otherParts];

    return (
      <div className="w-full">
        {orderedParts.map((part, index) => {
          if (part.type === "reasoning") {
            // biome-ignore lint/suspicious/noArrayIndexKey: Message parts don't have stable IDs, index is appropriate here
            return <ReasoningBlock key={`reasoning-${index}`} part={part} />;
          }
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

          // Handle tool parts following AI SDK 5.0 pattern
          if (part.type === "tool-webSearch") {
            const callId = part.toolCallId;

            switch (part.state) {
              case "input-streaming":
                return (
                  <Card key={callId} className="mb-3">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                          <Search className="h-4 w-4 text-primary" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-foreground">
                              Preparing search...
                            </span>
                            <DotsLoader size="sm" />
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              case "input-available":
                return (
                  <Card key={callId} className="mb-3">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10">
                          <Search className="h-4 w-4 text-primary" />
                        </div>
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center gap-2">
                            <Globe className="w-4 h-4 text-muted-foreground" />
                            <span className="text-sm font-medium text-foreground">
                              Searching the web
                            </span>
                            <DotsLoader size="sm" />
                          </div>
                          {(part as any).input?.query && (
                            <Badge variant="outline" className="text-xs">
                              "{(part as any).input.query}"
                            </Badge>
                          )}
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-primary animate-pulse rounded-full"></div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              case "output-available": {
                const result = (part as any).output;
                if (!result) return null;

                const searchData = result as {
                  query: string;
                  results: Array<{
                    title: string;
                    url: string;
                    snippet: string;
                  }>;
                };

                return (
                  <SearchResultsComponent
                    key={callId}
                    searchData={searchData}
                  />
                );
              }
              case "output-error":
                return (
                  <div
                    key={callId}
                    className="mb-3 rounded-lg border bg-destructive/10 px-4 py-3"
                  >
                    <div className="flex items-center gap-3">
                      <X className="h-4 w-4 text-destructive" />
                      <span className="text-sm text-destructive">
                        Search failed: {(part as any).errorText}
                      </span>
                    </div>
                  </div>
                );
              default:
                return null;
            }
          }

          // Handle other tool types with generic fallback
          if (part.type.startsWith("tool-")) {
            const toolPart = part as ToolPart;
            const toolName = part.type.replace("tool-", "");
            const callId = (toolPart as any).toolCallId;

            switch ((toolPart as any).state) {
              case "input-streaming":
              case "input-available":
                return (
                  <div
                    key={callId}
                    className="mb-2 rounded-lg border bg-muted/50 px-3 py-2 text-sm"
                  >
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Terminal className="h-4 w-4" />
                      <span>Tool: {toolName}</span>
                      <DotsLoader size="sm" />
                    </div>
                  </div>
                );
              case "output-available":
                return (
                  <div
                    key={callId}
                    className="mb-2 rounded-lg border bg-muted/50 px-3 py-2 text-sm"
                  >
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Terminal className="h-4 w-4" />
                      <span>Tool: {toolName} completed</span>
                    </div>
                  </div>
                );
              case "output-error":
                return (
                  <div
                    key={callId}
                    className="mb-2 rounded-lg border bg-destructive/10 px-3 py-2 text-sm"
                  >
                    <div className="flex items-center gap-2 text-destructive">
                      <X className="h-4 w-4" />
                      <span>
                        Tool {toolName} failed: {(toolPart as any).errorText}
                      </span>
                    </div>
                  </div>
                );
              default:
                return null;
            }
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
    return new File([uint8Array as BlobPart], name, { type: mimeType });
  } catch (error) {
    console.error("Failed to convert base64 to File:", error);
    return null;
  }
}

const SearchResultsComponent = ({
  searchData,
}: {
  searchData: {
    query: string;
    results: Array<{
      title: string;
      url: string;
      snippet: string;
    }>;
  };
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <Card className="mb-3 w-full shadow-2xl">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Globe className="w-5 h-5 text-muted-foreground" />
            <span className="text-foreground text-sm font-medium">
              {searchData.query}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground h-auto p-1"
          >
            <Badge variant="secondary" className="text-xs">
              {searchData.results.length} results
            </Badge>
            <ChevronDown
              className={`w-4 h-4 transition-transform ${isExpanded ? "" : "rotate-180"}`}
            />
          </Button>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="pt-0">
          <div className="max-h-80 overflow-y-auto space-y-1">
            {searchData.results.map((item, idx) => (
              <div
                key={idx}
                className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-accent cursor-pointer transition-colors"
              >
                <div className="flex-shrink-0">
                  {new URL(item.url).hostname === "pinterest.com" ? (
                    <div className="w-4 h-4 bg-adaptive-scarlet rounded-full flex items-center justify-center text-white text-xs font-bold">
                      P
                    </div>
                  ) : (
                    <Globe className="w-4 h-4 text-muted-foreground" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-foreground text-sm truncate hover:text-primary transition-colors block"
                  >
                    {item.title}
                  </a>
                </div>
                <Badge variant="outline" className="text-xs">
                  {new URL(item.url).hostname}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  );
};

const ReasoningBlock = ({ part }: { part: ReasoningUIPart }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mb-3 max-w-3xl">
      <Collapsible
        open={isOpen}
        onOpenChange={setIsOpen}
        className="group w-full overflow-hidden rounded-lg border bg-muted/50"
      >
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="w-full flex items-center gap-3 p-4 text-left hover:bg-muted/70 transition-colors"
          >
            {isOpen ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            )}
            <span className="text-muted-foreground text-sm font-medium">
              Thinking
            </span>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent forceMount>
          <motion.div
            initial={false}
            animate={isOpen ? "open" : "closed"}
            variants={{
              open: { height: "auto", opacity: 1 },
              closed: { height: 0, opacity: 0 },
            }}
            transition={{ duration: 0.3, ease: [0.04, 0.62, 0.23, 0.98] }}
            className="border-t border-border"
          >
            <div className="p-4 max-h-80 overflow-y-auto">
              <div className="text-muted-foreground text-sm leading-relaxed whitespace-pre-wrap">
                {part.text.split("\n\n").map((paragraph, index) => (
                  <div key={index} className="mb-4 last:mb-0">
                    {paragraph.split("\n").map((line, lineIndex) => (
                      <div
                        key={lineIndex}
                        className={cn(
                          line.startsWith("•") || line.startsWith("-")
                            ? "ml-4"
                            : "",
                          "mb-1 last:mb-0",
                        )}
                      >
                        {line}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};
