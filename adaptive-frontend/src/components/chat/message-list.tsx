import { Loader2, MessageSquare, Pencil, Save, X, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/services/llms/types";
import { useEffect, useRef, useState } from "react";
import Markdown from "../markdown";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
  DrawerClose,
} from "@/components/ui/drawer";
import { useMediaQuery } from "@/lib/hooks/use-media-query";
import { useDeleteMessage } from "@/lib/hooks/conversations/use-delete-message";
import { useUpdateMessage } from "@/lib/hooks/conversations/use-update-message";

interface MessageListProps {
  conversationId: number;
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export function MessageList({
  conversationId,
  messages,
  isLoading,
  error,
}: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        messages.map((msg: Message, index) => (
          <MessageItem
            key={msg.id}
            message={msg}
            conversationId={conversationId}
            index={index}
            messages={messages}
          />
        ))
      )}
      {isLoading && <LoadingIndicator />}
      {error && <ErrorMessage error={error} />}
      <div ref={messagesEndRef} />
    </div>
  );
}

function MessageItem({
  message,
  conversationId,
  index,
  messages,
}: {
  message: Message;
  conversationId: number;
  index: number;
  messages: Message[];
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(message.content);

  const { mutate: updateMessage } = useUpdateMessage();
  const { mutate: deleteMessage } = useDeleteMessage();

  const hasSubsequentMessages = index < messages.length - 1;

  // Dialog state for confirming edits that will delete subsequent messages
  const [showEditConfirmDialog, setShowEditConfirmDialog] = useState(false);
  const isMobile = useMediaQuery("(max-width: 640px)");

  const handleEdit = () => {
    if (hasSubsequentMessages) {
      setShowEditConfirmDialog(true);
    } else {
      setIsEditing(true);
    }
  };

  const handleSaveEdit = () => {
    // First update the current message
    updateMessage(
      {
        conversationId,
        messageId: message.id,
        updates: { content: editedContent },
      },
      {
        onSuccess: () => {
          setIsEditing(false);
          // Then delete all subsequent messages if there are any
          if (hasSubsequentMessages) {
            const subsequentMessageIds = messages
              .slice(index + 1)
              .map((msg) => msg.id);

            subsequentMessageIds.forEach((id) =>
              deleteMessage({ conversationId, messageId: id })
            );
          }
        },
      }
    );
  };

  const handleDelete = () => {
    deleteMessage(
      {
        conversationId,
        messageId: message.id,
      },
      {
        onSuccess: () => {
          // Delete all subsequent messages if there are any
          if (hasSubsequentMessages) {
            const subsequentMessageIds = messages
              .slice(index + 1)
              .map((msg) => msg.id);

            subsequentMessageIds.forEach((id) =>
              deleteMessage({ conversationId, messageId: id })
            );
          }
        },
      }
    );
  };

  return (
    <div
      className={cn(
        "flex flex-col w-full max-w-[90%] rounded-2xl p-4 group",
        message.role === "user"
          ? "ml-auto bg-primary text-primary-foreground"
          : "bg-muted"
      )}
    >
      {!isEditing ? (
        <div className="flex items-start justify-between w-full">
          <Markdown
            content={message.content}
            className={message.role === "user" ? "text-primary-foreground" : ""}
          />
          <div className="flex gap-1 ml-2 transition-opacity opacity-0 group-hover:opacity-100">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-8 w-8",
                message.role === "user"
                  ? "hover:bg-primary-foreground/10"
                  : "hover:bg-background/80"
              )}
              onClick={handleEdit}
            >
              <Pencil className="w-4 h-4" />
            </Button>
            <DeleteMessageDialog
              userMessage={message.role === "user"}
              hasSubsequentMessages={hasSubsequentMessages}
              onDelete={handleDelete}
            />
          </div>
        </div>
      ) : (
        <div className="flex flex-col w-full gap-2">
          <Textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className={cn(
              "min-h-[100px] bg-transparent border-muted-foreground/20",
              message.role === "user" ? "text-primary-foreground" : ""
            )}
          />
          <div className="flex justify-end gap-2">
            <Button
              size="sm"
              variant="ghost"
              className={cn(
                message.role === "user"
                  ? "hover:bg-primary-foreground/10"
                  : "hover:bg-background/80"
              )}
              onClick={() => {
                setEditedContent(message.content);
                setIsEditing(false);
              }}
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
                  : "bg-background/80 hover:bg-background"
              )}
              onClick={handleSaveEdit}
            >
              <Save className="w-4 h-4 mr-2" />
              Save
            </Button>
          </div>
        </div>
      )}

      {/* Confirmation dialog for editing with subsequent messages */}
      {showEditConfirmDialog &&
        (isMobile ? (
          <EditConfirmDrawer
            onConfirm={() => {
              setShowEditConfirmDialog(false);
              setIsEditing(true);
            }}
            onCancel={() => setShowEditConfirmDialog(false)}
          />
        ) : (
          <EditConfirmDialog
            onConfirm={() => {
              setShowEditConfirmDialog(false);
              setIsEditing(true);
            }}
            onCancel={() => setShowEditConfirmDialog(false)}
          />
        ))}
    </div>
  );
}

function DeleteMessageDialog({
  userMessage,
  hasSubsequentMessages,
  onDelete,
}: {
  userMessage: boolean;
  hasSubsequentMessages: boolean;
  onDelete: () => void;
}) {
  const isMobile = useMediaQuery("(max-width: 640px)");
  const dialogTitle = hasSubsequentMessages
    ? "Delete Messages"
    : "Delete Message";

  const dialogDescription = hasSubsequentMessages
    ? "Editing this message will also delete all subsequent messages in the conversation. This action cannot be undone."
    : "Are you sure you want to delete this message? This action cannot be undone.";

  if (isMobile) {
    return (
      <Drawer>
        <DrawerTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "h-8 w-8",
              userMessage
                ? "hover:bg-primary-foreground/10"
                : "hover:bg-background/80"
            )}
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </DrawerTrigger>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle>{dialogTitle}</DrawerTitle>
            <DrawerDescription>{dialogDescription}</DrawerDescription>
          </DrawerHeader>
          <DrawerFooter className="pt-2">
            <DrawerClose asChild>
              <Button variant="destructive" onClick={onDelete}>
                Delete
              </Button>
            </DrawerClose>
            <DrawerClose asChild>
              <Button variant="outline">Cancel</Button>
            </DrawerClose>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    );
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "h-8 w-8",
            userMessage
              ? "hover:bg-primary-foreground/10"
              : "hover:bg-background/80"
          )}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
          <DialogDescription>{dialogDescription}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button variant="destructive" onClick={onDelete}>
              Delete
            </Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EditConfirmDialog({
  onConfirm,
  onCancel,
}: {
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Dialog defaultOpen>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Message</DialogTitle>
          <DialogDescription>
            Editing this message will also delete all subsequent messages in the
            conversation. This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="default" onClick={onConfirm}>
            Continue with Edit
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EditConfirmDrawer({
  onConfirm,
  onCancel,
}: {
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Drawer defaultOpen>
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle>Edit Message</DrawerTitle>
          <DrawerDescription>
            Editing this message will also delete all subsequent messages in the
            conversation. This action cannot be undone.
          </DrawerDescription>
        </DrawerHeader>
        <DrawerFooter>
          <Button variant="default" onClick={onConfirm}>
            Continue with Edit
          </Button>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-muted p-4 max-w-[90%]">
      <Loader2 className="w-5 h-5 animate-spin text-primary" />
      <p>Processing your request...</p>
    </div>
  );
}

function ErrorMessage({ error }: { error: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl bg-destructive/10 p-4 max-w-[90%] text-destructive border border-destructive/20">
      <p>Error: {error}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center w-full text-center">
      <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
        <MessageSquare className="w-8 h-8 text-primary" />
      </div>
      <h2 className="mb-2 text-xl font-medium">Welcome to Adaptive</h2>
      <div className="max-w-md text-muted-foreground">
        Send a message to start a conversation. Adaptive will select the best AI
        model for your specific query.
      </div>
    </div>
  );
}
