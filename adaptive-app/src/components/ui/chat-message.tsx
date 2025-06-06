"use client";

import React, { useMemo, useState } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { motion } from "framer-motion";
import {
  Ban,
  ChevronRight,
  Code2,
  Loader2,
  Terminal,
  Check,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { FilePreview } from "@/components/ui/file-preview";
import { MarkdownRenderer } from "@/components/ui/markdown-renderer";

import type { UIMessage, ToolInvocation as AIToolInvocation } from "ai";

type MessageReasoningPart = Extract<
  UIMessage["parts"][number],
  { type: "reasoning" }
>;

const chatBubbleVariants = cva(
  "group/message relative break-words rounded-lg p-3 text-sm w-full max-w-[50%] shadow-sm transition-colors",
  {
    variants: {
      isUser: {
        true: "bg-primary text-primary-foreground",
        false: "bg-muted text-foreground",
      },
      animation: {
        none: "",
        slide: "duration-300 animate-in fade-in-0",
        scale: "duration-300 animate-in fade-in-0 zoom-in-75",
        fade: "duration-500 animate-in fade-in-0",
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
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  role,
  content,
  createdAt,
  showTimeStamp = false,
  animation = "scale",
  actions,
  experimental_attachments,
  parts,
  isEditing = false,
  editingContent = "",
  onEditingContentChange,
  onSaveEdit,
  onCancelEdit,
}) => {
  const userFiles = useMemo(() => {
    if (role === "user" && experimental_attachments) {
      return experimental_attachments
        .map((attachment, index) => {
          if (attachment.url.startsWith("data:")) {
            const base64Content = attachment.url.split(",")[1];
            if (!base64Content) return null;
            return base64ToNewFile(
              base64Content,
              attachment.contentType || "application/octet-stream",
              attachment.name ?? `attachment-${index}`,
            );
          }
          return null;
        })
        .filter(Boolean) as File[];
    }
    return null;
  }, [role, experimental_attachments]);

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
            {userFiles.map((file, index) => {
              return <FilePreview key={index} file={file} />;
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
            <div className="absolute -bottom-4 right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
              {actions}
            </div>
          ) : null}
        </div>

        {showTimeStamp && createdAt ? (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50",
              animation !== "none" && "duration-500 animate-in fade-in-0",
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
              <React.Fragment key={`text-${index}`}>
                <div className={cn(chatBubbleVariants({ isUser, animation }))}>
                  <MarkdownRenderer>{part.text}</MarkdownRenderer>
                  {actions && index === parts.length - 1 ? (
                    <div className="absolute -bottom-4 right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
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
                        "duration-500 animate-in fade-in-0",
                    )}
                  >
                    {formattedTime}
                  </time>
                ) : null}
              </React.Fragment>
            );
          } else if (part.type === "reasoning") {
            return <ReasoningBlock key={`reasoning-${index}`} part={part} />;
          } else if (part.type === "tool-invocation") {
            return (
              <ToolCallBlock
                key={`tool-invocation-${index}`}
                toolInvocation={part.toolInvocation}
              />
            );
          } else if (part.type === "file") {
            const file = base64ToNewFile(
              part.data,
              part.mimeType,
              `ai-file-${index}`,
            );
            return file ? (
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
            <div className="absolute -bottom-2 right-2 flex space-x-1 rounded-lg border bg-background p-1 text-foreground opacity-0 transition-opacity group-hover/message:opacity-100">
              {actions}
            </div>
          ) : null}
        </div>

        {showTimeStamp && createdAt ? (
          <time
            dateTime={createdAt.toISOString()}
            className={cn(
              "mt-1 block px-1 text-xs opacity-50",
              animation !== "none" && "duration-500 animate-in fade-in-0",
            )}
          >
            {formattedTime}
          </time>
        ) : null}
      </div>
    );
  }

  return (
    <div className={cn(chatBubbleVariants({ isUser, animation }))}>
      <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent text-muted-foreground" />
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

const ReasoningBlock = ({ part }: { part: MessageReasoningPart }) => {
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
            <button className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
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
              <div className="whitespace-pre-wrap text-xs">
                {part.reasoning}
              </div>
            </div>
          </motion.div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

interface ToolCallBlockProps {
  toolInvocation?: AIToolInvocation;
  toolInvocations?: AIToolInvocation[];
}

function ToolCallBlock({
  toolInvocation,
  toolInvocations,
}: ToolCallBlockProps) {
  const invocationsToRender = toolInvocation
    ? [toolInvocation]
    : toolInvocations;

  if (!invocationsToRender?.length) return null;

  return (
    <div className="flex flex-col items-start gap-2">
      {invocationsToRender.map((invocation, index) => {
        const isCancelled =
          invocation.state === "result" &&
          (invocation.result as any)?.__cancelled === true;

        if (isCancelled) {
          return (
            <div
              key={index}
              className="flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-2 text-sm text-muted-foreground"
            >
              <Ban className="h-4 w-4" />
              <span>
                Cancelled{" "}
                <span className="font-mono">
                  {"`"}
                  {invocation.toolName}
                  {"`"}
                </span>
              </span>
            </div>
          );
        }

        switch (invocation.state) {
          case "partial-call":
          case "call":
            return (
              <div
                key={index}
                className="flex items-center gap-2 rounded-lg border bg-muted/50 px-3 py-2 text-sm text-muted-foreground"
              >
                <Terminal className="h-4 w-4" />
                <span>
                  Calling{" "}
                  <span className="font-mono">
                    {"`"}
                    {invocation.toolName}
                    {"`"}
                  </span>
                  ...
                </span>
                <Loader2 className="h-3 w-3 animate-spin" />
              </div>
            );
          case "result":
            return (
              <div
                key={index}
                className="flex flex-col gap-1.5 rounded-lg border bg-muted/50 px-3 py-2 text-sm"
              >
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Code2 className="h-4 w-4" />
                  <span>
                    Result from{" "}
                    <span className="font-mono">
                      {"`"}
                      {invocation.toolName}
                      {"`"}
                    </span>
                  </span>
                </div>
                <pre className="overflow-x-auto whitespace-pre-wrap text-foreground">
                  {JSON.stringify(invocation.result || {}, null, 2)}
                </pre>
              </div>
            );
          default:
            return null;
        }
      })}
    </div>
  );
}
