import { memo, useState, useCallback } from "react";
import { Loader2, Pencil, Save, X, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { useUpdateMessage } from "@/lib/hooks/conversations/use-update-message";
import { useDeleteMessage } from "@/lib/hooks/conversations/use-delete-message";
import { DBMessage } from "@/services/messages/types";
import DeleteMessageDialog from "./delete-message-dialog";
import Markdown from "@/components/markdown";

interface MessageItemProps {
  message: DBMessage;
  conversationId: number;
  index: number;
  messages: DBMessage[];
}

// Message content display component
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
    <div className="flex items-start justify-between w-full">
      <div className="flex items-start gap-3 w-full">
        {message.role === "user" ? (
          <User className="w-5 h-5 mt-1 shrink-0 text-primary-foreground" />
        ) : (
          <Bot className="w-5 h-5 mt-1 shrink-0 text-muted-foreground" />
        )}
        <Markdown
          content={message.content}
          className={
            message.role === "user"
              ? "text-primary-foreground w-full"
              : "w-full"
          }
        />
      </div>
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
          <Loader2 className={cn("w-4 h-4", isProcessing && "animate-spin")} />
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
    </div>
  ),
);
MessageContent.displayName = "MessageContent";

// Message edit form component with corrected prop types
const MessageEditForm = memo(
  ({
    message,
    editedContent,
    onChangeContent, // Changed from setEditedContent
    onCancel,
    onSave,
    isUpdating,
    isProcessing,
  }: {
    message: DBMessage;
    editedContent: string;
    onChangeContent: (e: React.ChangeEvent<HTMLTextAreaElement>) => void; // Fixed type
    onCancel: () => void;
    onSave: () => void;
    isUpdating: boolean;
    isProcessing: boolean;
  }) => (
    <div className="flex flex-col w-full gap-2">
      <div className="flex items-start gap-3 w-full">
        {message.role === "user" ? (
          <User className="w-5 h-5 mt-1 shrink-0 text-primary-foreground" />
        ) : (
          <Bot className="w-5 h-5 mt-1 shrink-0 text-muted-foreground" />
        )}
        <Textarea
          value={editedContent}
          onChange={onChangeContent}
          className={cn(
            "min-h-[100px] bg-transparent border-muted-foreground/20 w-full",
            message.role === "user" ? "text-primary-foreground" : "",
          )}
        />
      </div>
      <div className="flex justify-end gap-2">
        <Button
          size="sm"
          variant="ghost"
          className={cn(
            message.role === "user"
              ? "hover:bg-primary-foreground/10"
              : "hover:bg-background/80",
          )}
          onClick={onCancel}
          disabled={isProcessing}
        >
          <X className="w-4 h-4 mr-2" />
          Cancel
        </Button>
        <Button
          size="sm"
          variant="outline"
          className={cn(
            message.role === "user"
              ? "bg-primary-foreground/10 hover:bg-primary-foreground/20"
              : "bg-background/80 hover:bg-background",
          )}
          onClick={onSave}
          disabled={isProcessing}
        >
          {isUpdating ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Save className="w-4 h-4 mr-2" />
          )}
          Save
        </Button>
      </div>
    </div>
  ),
);
MessageEditForm.displayName = "MessageEditForm";

// Main MessageItem component
export const MessageItem = memo(function MessageItem({
  message,
  conversationId,
  index,
  messages,
}: MessageItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

  const { mutateAsync: updateMessage, isPending: isUpdating } =
    useUpdateMessage();
  const { mutateAsync: deleteMessage, isPending: isDeleting } =
    useDeleteMessage();

  const hasSubsequentMessages = index < messages.length - 1;
  const isProcessing = isUpdating || isDeleting;

  // Memoized handlers
  const handleEdit = useCallback(() => {
    setIsEditing(true);
  }, []);

  const handleRetry = useCallback(async () => {
    await updateMessage({
      conversationId,
      messageId: message.id,
      updates: { content: message.content },
      index,
      messages,
    });
  }, [
    conversationId,
    message.id,
    message.content,
    index,
    messages,
    updateMessage,
  ]);

  const handleSaveEdit = useCallback(async () => {
    await updateMessage({
      conversationId,
      messageId: message.id,
      updates: { role: "user", content: editedContent },
      index,
      messages,
    });
    setIsEditing(false);
  }, [
    conversationId,
    message.id,
    editedContent,
    index,
    messages,
    updateMessage,
  ]);

  const handleDelete = useCallback(async () => {
    await deleteMessage({
      conversationId,
      messageId: message.id,
      messages,
      index,
    });
  }, [conversationId, message.id, messages, index, deleteMessage]);

  const handleCancel = useCallback(() => {
    setEditedContent(message.content);
    setIsEditing(false);
  }, [message.content]);

  const handleEditedContentChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setEditedContent(e.target.value);
    },
    [],
  );

  return (
    <div
      className={cn(
        "flex flex-col w-full max-w-[90%] rounded-2xl p-4 group",
        message.role === "user"
          ? "ml-auto bg-primary text-primary-foreground"
          : "bg-muted",
        isProcessing && "opacity-60",
      )}
    >
      {!isEditing ? (
        <MessageContent
          message={message}
          onEdit={handleEdit}
          onRetry={handleRetry}
          onDelete={handleDelete}
          hasSubsequentMessages={hasSubsequentMessages}
          isProcessing={isProcessing}
        />
      ) : (
        <MessageEditForm
          message={message}
          editedContent={editedContent}
          onChangeContent={handleEditedContentChange} // Changed prop name
          onCancel={handleCancel}
          onSave={handleSaveEdit}
          isUpdating={isUpdating}
          isProcessing={isProcessing}
        />
      )}
    </div>
  );
});
