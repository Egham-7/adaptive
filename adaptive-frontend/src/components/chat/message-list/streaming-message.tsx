import { Bot } from "lucide-react";
import { memo } from "react";
import { Markdown } from "@/components/markdown";

interface StreamingMessageProps {
  content: string;
  isStreaming?: boolean;
}

const StreamingMessage = memo(
  ({ content, isStreaming = false }: StreamingMessageProps) => {
    return (
      <div className="flex w-full py-4 px-2 group border-b border-border">
        <Bot className="w-5 h-5 mt-1 shrink-0 text-muted-foreground" />
        <div className="ml-3 w-full">
          <div className="markdown-container relative">
            <Markdown>{content || "AI is thinking..."}</Markdown>
            {isStreaming && (
              <span className="typing-cursor inline-block h-4 w-[2px] ml-[1px] align-middle bg-primary animate-cursor-blink" />
            )}
          </div>
          {isStreaming && (
            <div className="mt-2 text-xs text-muted-foreground flex items-center gap-1">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
              </span>
              <span>AI is typing...</span>
            </div>
          )}
        </div>
      </div>
    );
  },
);

StreamingMessage.displayName = "StreamingMessage";
export default StreamingMessage;
