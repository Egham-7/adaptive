import { Bot } from "lucide-react";
import { memo, useEffect, useState, useRef } from "react";
import Markdown from "@/components/markdown";

interface StreamingMessageProps {
  content: string;
}

const StreamingMessage = memo(({ content }: StreamingMessageProps) => {
  const [displayedContent, setDisplayedContent] = useState("");
  const [isTyping, setIsTyping] = useState(true);
  const contentRef = useRef("");
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Typing effect logic
  useEffect(() => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // If content hasn't changed, don't restart typing
    if (contentRef.current === content) {
      return;
    }

    contentRef.current = content;

    // Calculate how much new content we have
    const newContentLength = content.length - displayedContent.length;

    // If we have new content, animate it
    if (newContentLength > 0) {
      setIsTyping(true);

      // Determine typing speed based on content length
      // Faster for longer content to keep it responsive
      const typingSpeed = Math.max(10, Math.min(50, 100 - content.length / 20));

      // Schedule the next character to appear
      timeoutRef.current = setTimeout(() => {
        setDisplayedContent(content.substring(0, displayedContent.length + 1));
      }, typingSpeed);
    } else {
      setIsTyping(false);
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [content, displayedContent]);

  return (
    <div className="flex flex-col w-full max-w-[90%] rounded-2xl p-4 bg-muted">
      <div className="flex items-start gap-3 w-full">
        <Bot className="w-5 h-5 mt-1 shrink-0 text-muted-foreground" />
        <div className="w-full">
          <div className="markdown-container relative">
            <Markdown content={displayedContent} />
            {isTyping && (
              <span className="typing-cursor inline-block h-4 w-[2px] ml-[1px] align-middle bg-primary animate-cursor-blink" />
            )}
          </div>
          {isTyping && (
            <div className="flex items-center mt-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                </span>
                <span>AI is typing...</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

StreamingMessage.displayName = "StreamingMessage";

export default StreamingMessage;
