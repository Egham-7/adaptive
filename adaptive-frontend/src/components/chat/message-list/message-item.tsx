import { useState, useCallback, memo } from "react";
import { Loader2, Save, X, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { DBMessage } from "@/services/messages/types";
import MessageContent from "./message-content";
import { useDeleteMessage } from "@/hooks/conversations/use-delete-message";

interface MessageItemProps {
  message: DBMessage;
  conversationId: number;
  index: number;
  messages: DBMessage[];
  updateMessage: (params: {
    conversationId: number;
    messageId: number;
    updates: { role?: string; content?: string };
    index: number;
    messages: DBMessage[];
  }) => Promise<DBMessage>;
  isUpdating: boolean;
}

// Message edit form component with corrected prop types
const MessageEditForm = memo(
  ({
    message,
    editedContent,
    onChangeContent,
    onCancel,
    onSave,
    isUpdating,
    isProcessing,
  }: {
    message: DBMessage;
    editedContent: string;
    onChangeContent: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
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
            "min-h-[100px] bg-transparent border-muted-foreground/20 w-full overflow-hidden break-words",
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
  updateMessage,
  isUpdating,
}: MessageItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);
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
      updates: { content: message.content, role: "user" },
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

  // Different styling for user vs AI messages
  if (message.role === "user") {
    return (
      <div
        className={cn(
          "flex flex-col w-full max-w-[90%] rounded-2xl p-4 group overflow-hidden ml-auto bg-primary text-primary-foreground",
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
            onChangeContent={handleEditedContentChange}
            onCancel={handleCancel}
            onSave={handleSaveEdit}
            isUpdating={isUpdating}
            isProcessing={isProcessing}
          />
        )}
      </div>
    );
  } else {
    // AI message - full width, no bubble styling
    return (
      <div
        className={cn(
          "flex flex-col w-full py-4 px-2 group border-b border-border",
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
            onChangeContent={handleEditedContentChange}
            onCancel={handleCancel}
            onSave={handleSaveEdit}
            isUpdating={isUpdating}
            isProcessing={isProcessing}
          />
        )}
      </div>
    );
  }
});
