import { memo } from "react";
import { Pencil, Bot, User, RefreshCw } from "lucide-react";
import { Markdown } from "@/components/markdown";
import { DBMessage } from "@/services/messages/types";
import { Button } from "@/components/ui/button";
import DeleteMessageDialog from "./delete-message-dialog";
import ProviderBadge from "./provider-badge";
import { cn } from "@/lib/utils";

const MessageContent = memo(
  ({
    message,
    onEdit,
    onRetry,
    onDelete,
    hasSubsequentMessages,
    isProcessing,
  }: {
    message: DBMessage;
    onEdit: () => void;
    onRetry: () => void;
    onDelete: () => void;
    hasSubsequentMessages: boolean;
    isProcessing: boolean;
  }) => (
    <div className="flex flex-col w-full">
      <div className="flex items-start justify-between w-full">
        <div className="flex items-start gap-3 w-full overflow-hidden">
          {message.role === "user" ? (
            <User className="w-5 h-5 mt-5 shrink-0 text-primary-foreground" />
          ) : (
            <Bot className="w-5 h-5 mt-5 shrink-0 text-muted-foreground" />
          )}
          <div className="overflow-hidden w-full">
            <Markdown>{message.content}</Markdown>
          </div>
        </div>
        {message.role === "user" && (
          <div className="flex gap-1 ml-2 transition-opacity opacity-0 group-hover:opacity-100 shrink-0">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-8 w-8",
                message.role === "user"
                  ? "hover:bg-primary-foreground/10"
                  : "hover:bg-background/80",
              )}
              onClick={onRetry}
              disabled={isProcessing}
            >
              <RefreshCw
                className={cn("w-4 h-4", isProcessing && "animate-spin")}
              />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-8 w-8",
                message.role === "user"
                  ? "hover:bg-primary-foreground/10"
                  : "hover:bg-background/80",
              )}
              onClick={onEdit}
              disabled={isProcessing}
            >
              <Pencil className="w-4 h-4" />
            </Button>
            <DeleteMessageDialog
              userMessage={message.role === "user"}
              hasSubsequentMessages={hasSubsequentMessages}
              onDelete={onDelete}
              disabled={isProcessing}
            />
          </div>
        )}
      </div>

      {/* Provider and model info - only show for AI messages */}
      {message.role === "assistant" && message.provider && message.model && (
        <div className="mt-2 ml-8">
          <ProviderBadge
            provider={message.provider}
            model={message.model}
            className="opacity-70 hover:opacity-100 transition-opacity"
          />
        </div>
      )}
    </div>
  ),
);

MessageContent.displayName = "MessageContent";

export default MessageContent;
