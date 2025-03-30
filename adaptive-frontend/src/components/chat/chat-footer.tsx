import { MessageInput } from "./message-input";

interface ChatFooterProps {
  isLoading: boolean;
  sendMessage: (message: string) => void;
  isStreaming: boolean;
  abortStreaming?: () => void;
  isError: boolean;
}

export function ChatFooter({
  isLoading,
  sendMessage,
  isStreaming = false,
  abortStreaming,
  isError,
}: ChatFooterProps) {
  return (
    <footer className="sticky bottom-0 z-40 pt-3 pb-5 border-t bg-background/95 backdrop-blur-sm">
      <div className="w-full max-w-5xl px-4 mx-auto space-y-3">
        <MessageInput
          isLoading={isLoading}
          sendMessage={sendMessage}
          disabled={(isStreaming && !abortStreaming) || isError}
          isStreaming={isStreaming}
          abortStreaming={abortStreaming}
        />
        <p className="pt-1 text-xs text-center text-muted-foreground">
          Results may vary. Verify important information.
        </p>
      </div>
    </footer>
  );
}
