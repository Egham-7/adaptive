import { AlertTriangle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { getErrorMessage } from "./chat-utils";

interface ChatErrorDisplayProps {
  error?: Error;
  onRetry?: () => void;
}

export function ChatErrorDisplay({ error, onRetry }: ChatErrorDisplayProps) {
  if (!error) return null;

  const errorMessage = getErrorMessage(error);
  const isLimitError = errorMessage
    .toLowerCase()
    .includes("daily message limit");

  return (
    <div className="w-full max-w-2xl px-4 pb-4">
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          <p>{errorMessage}</p>
          {onRetry && !isLimitError && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onRetry}
              className="mt-2"
            >
              Retry
            </Button>
          )}
        </AlertDescription>
      </Alert>
    </div>
  );
}