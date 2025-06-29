import { ArrowDown } from "lucide-react";
import type { UIMessage } from "@ai-sdk/react";

import { Button } from "@/components/ui/button";
import { useAutoScroll } from "@/hooks/use-auto-scroll";

interface ChatMessagesProps {
  messages: UIMessage[];
  children: React.ReactNode;
}

export function ChatMessages({ messages, children }: ChatMessagesProps) {
  const {
    containerRef,
    scrollToBottom,
    handleScroll,
    shouldAutoScroll,
    handleTouchStart,
  } = useAutoScroll([messages]);

  return (
    <div
      className="grid grid-cols-1 overflow-y-auto pb-4"
      ref={containerRef}
      onScroll={handleScroll}
      onTouchStart={handleTouchStart}
    >
      <div className="max-w-full [grid-column:1/1] [grid-row:1/1]">
        {children}
      </div>

      {!shouldAutoScroll && (
        <div className="pointer-events-none flex flex-1 items-end justify-end [grid-column:1/1] [grid-row:1/1]">
          <div className="sticky bottom-0 left-0 flex w-full justify-end">
            <Button
              onClick={scrollToBottom}
              className="fade-in-0 slide-in-from-bottom-1 pointer-events-auto h-8 w-8 animate-in rounded-full ease-in-out"
              size="icon"
              variant="ghost"
            >
              <ArrowDown className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}