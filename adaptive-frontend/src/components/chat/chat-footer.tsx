import { ActionButtons } from "./action-buttons";
import { MessageInput } from "./message-input";

interface ChatFooterProps {
  isLoading: boolean;
  sendMessage: (message: string) => void;
  showActions: boolean;
  toggleActions: () => void;
}

export function ChatFooter({ isLoading, sendMessage, showActions, toggleActions }: ChatFooterProps) {
  return (
    <footer className="fixed bottom-0 left-0 right-0 z-40 pt-3 pb-5 border-t bg-background/95 backdrop-blur-xs">
      <div className="w-full max-w-5xl px-4 mx-auto space-y-3">
        <ActionButtons visible={showActions} />
        <MessageInput 
          isLoading={isLoading} 
          sendMessage={sendMessage} 
          showActions={showActions} 
          toggleActions={toggleActions} 
        />
        <p className="pt-1 text-xs text-center text-muted-foreground">
          Results may vary. Verify important information.
        </p>
      </div>
    </footer>
  );
}