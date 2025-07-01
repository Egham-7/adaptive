import { Edit3, RotateCcw, ThumbsDown, ThumbsUp, Trash2 } from "lucide-react";
import type { UIMessage } from "@ai-sdk/react";

import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { getMessageContent } from "./chat-utils";
import type { MessageTextPart } from "./chat-types";

interface MessageActionsProps {
  message: UIMessage;
  canEdit: boolean;
  canRetry: boolean;
  canDelete: boolean;
  canRate: boolean;
  isEditing: boolean;
  onEdit: (messageId: string, content: string) => void;
  onRetry: (message: UIMessage) => void;
  onDelete: (messageId: string) => void;
  onRate?: (messageId: string, rating: "thumbs-up" | "thumbs-down") => void;
}

export function MessageActions({
  message,
  canEdit,
  canRetry,
  canDelete,
  canRate,
  isEditing,
  onEdit,
  onRetry,
  onDelete,
  onRate,
}: MessageActionsProps) {
  const isUserMessage = message.role === "user";

  if (isUserMessage) {
    return (
      <>
        {canEdit && (
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={() =>
              onEdit(message.id, getMessageContent(message))
            }
            disabled={isEditing}
          >
            <Edit3 className="h-4 w-4" />
          </Button>
        )}
        {canRetry && (
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6"
            onClick={() => onRetry(message)}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        )}
        {canDelete && (
          <Button
            size="icon"
            variant="ghost"
            className="h-6 w-6 text-destructive hover:text-destructive"
            onClick={() => onDelete(message.id)}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </>
    );
  }

  // Assistant message actions
  const messageContent =
    (message.parts?.find((p) => p.type === "text") as MessageTextPart)
      ?.text || getMessageContent(message);

  return canRate ? (
    <>
      <div className="border-r pr-1">
        <CopyButton
          content={messageContent}
          copyMessage="Copied response to clipboard!"
        />
      </div>
      <Button
        size="icon"
        variant="ghost"
        className="h-6 w-6"
        onClick={() => onRate?.(message.id, "thumbs-up")}
      >
        <ThumbsUp className="h-4 w-4" />
      </Button>
      <Button
        size="icon"
        variant="ghost"
        className="h-6 w-6"
        onClick={() => onRate?.(message.id, "thumbs-down")}
      >
        <ThumbsDown className="h-4 w-4" />
      </Button>
    </>
  ) : (
    <CopyButton
      content={messageContent}
      copyMessage="Copied response to clipboard!"
    />
  );
}