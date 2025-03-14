import { ActionButtons } from "./action-buttons";
import { MessageInput } from "./message-input";
import { Button } from "@/components/ui/button";
import { StopCircle } from "lucide-react";

interface ChatFooterProps {
  isLoading: boolean;
  sendMessage: (message: string) => void;
  showActions: boolean;
  toggleActions: () => void;
  isStreaming?: boolean;
  abortStreaming?: () => void;
}

export function ChatFooter({
  isLoading,
  sendMessage,
  showActions,
  toggleActions,
  isStreaming = false,
  abortStreaming,
}: ChatFooterProps) {
  return (
    <footer className="sticky bottom-0 z-40 pt-3 pb-5 border-t bg-background/95 backdrop-blur-sm">
      <div className="w-full max-w-5xl px-4 mx-auto space-y-3">
        <ActionButtons visible={showActions} />
        {isStreaming && abortStreaming ? (
          <div className="flex justify-center">
            <Button
              variant="destructive"
              size="sm"
              onClick={abortStreaming}
              className="flex items-center gap-2"
            >
              <StopCircle className="w-4 h-4" />
              Stop generating
            </Button>
          </div>
        ) : (
          <MessageInput
            isLoading={isLoading}
            sendMessage={sendMessage}
            showActions={showActions}
            toggleActions={toggleActions}
            disabled={isStreaming}
          />
        )}
        <p className="pt-1 text-xs text-center text-muted-foreground">
          Results may vary. Verify important information.
        </p>
      </div>
    </footer>
  );
}
