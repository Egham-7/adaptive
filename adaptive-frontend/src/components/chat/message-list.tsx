import { Loader2, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/services/llms/types";
import { useEffect, useRef } from "react";

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export function MessageList({ messages, isLoading, error }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messagesEndRef]);

  return (
    <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        messages.map((msg: Message, index: number) => (
          <div
            key={index}
            className={cn(
              "flex w-full max-w-[90%] rounded-2xl p-4",
              msg.role === "user"
                ? "ml-auto bg-primary text-primary-foreground"
                : "bg-muted",
            )}
          >
            <div className="prose-sm prose dark:prose-invert">
              {msg.content}
            </div>
          </div>
        ))
      )}

      {isLoading && (
        <div className="flex items-center gap-3 rounded-2xl bg-muted p-4 max-w-[90%]">
          <Loader2 className="w-5 h-5 animate-spin text-primary" />
          <p>Processing your request...</p>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-3 rounded-2xl bg-destructive/10 p-4 max-w-[90%] text-destructive border border-destructive/20">
          <p>Error: {error}</p>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}

function EmptyState() {
  return (
    <>
      <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
        <MessageSquare className="w-8 h-8 text-primary" />
      </div>
      <h2 className="mb-2 text-xl font-medium">Welcome to Adaptive</h2>
      <div className="max-w-md text-muted-foreground">
        Send a message to start a conversation. Adaptive will select the best AI
        model for your specific query.
      </div>
    </>
  );
}
