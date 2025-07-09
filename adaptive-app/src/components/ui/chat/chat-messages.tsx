import { ArrowDown } from "lucide-react";
import type { UIMessage } from "@ai-sdk/react";

import { Button } from "@/components/ui/button";
import { useAutoScroll } from "@/hooks/use-auto-scroll";

interface ChatMessagesProps {
  messages: UIMessage[];
  children: React.ReactNode;
  isStreaming?: boolean;
}

export function ChatMessages({
  messages,
  children,
  isStreaming,
}: ChatMessagesProps) {
  const {
    containerRef,
    scrollToBottom,
    handleScroll,
    shouldAutoScroll,
    handleTouchStart,
  } = useAutoScroll([messages], isStreaming);

  return (
    <div className="relative h-full">
      <div
        className="h-full overflow-y-auto"
        ref={containerRef}
        onScroll={handleScroll}
        onTouchStart={handleTouchStart}
      >
        {children}
      </div>

      {!shouldAutoScroll && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20">
          <Button
            onClick={() => {
              scrollToBottom();
            }}
            className="fade-in-0 slide-in-from-bottom-1 h-8 w-8 animate-in rounded-full ease-in-out shadow-lg bg-background border"
            size="icon"
            variant="ghost"
          >
            <ArrowDown className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
