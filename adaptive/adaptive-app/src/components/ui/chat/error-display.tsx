import { ChatErrorDisplay } from "./chat-error-display";

interface ErrorDisplayProps {
  isError?: boolean;
  error?: Error;
  onRetry?: () => void;
  className?: string;
}

export function ErrorDisplay({ isError, error, onRetry, className = "mx-4 mb-4" }: ErrorDisplayProps) {
  if (!isError) return null;

  return (
    <div className={className}>
      <ChatErrorDisplay error={error} onRetry={onRetry} />
    </div>
  );
}