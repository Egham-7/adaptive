import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { useIsMobile } from "@/hooks/use-mobile";
import { Conversation } from "@/services/conversations/types";
import { Trash2 } from "lucide-react";
import { useDeleteConversation } from "@/hooks/conversations/use-delete-conversation";
import { useNavigate, useRouter } from "@tanstack/react-router";

interface DeleteConversationDialogProps {
  conversation: Conversation;
}

export function DeleteConversationDialog({
  conversation,
}: DeleteConversationDialogProps) {
  const isMobile = useIsMobile();
  const deleteConversationMutation = useDeleteConversation();
  const navigate = useNavigate();
  const router = useRouter();

  const handleDelete = async () => {
    await deleteConversationMutation.mutateAsync(conversation.id);
    // Navigate to home if we're deleting the current conversation

    // Check if we're currently viewing the conversation that's being deleted
    const currentRoute = router.state.location.pathname;
    const isCurrentConversation =
      currentRoute === `/conversations/${conversation.id}`;

    // Navigate to home if we're deleting the current conversation
    if (isCurrentConversation) {
      navigate({ to: "/home" });
    }
  };

  const DeleteTrigger = (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 text-destructive hover:text-destructive/90 hover:bg-destructive/10"
    >
      <Trash2 className="h-4 w-4" />
      <span className="sr-only">Delete</span>
    </Button>
  );

  if (isMobile) {
    return (
      <Drawer>
        <DrawerTrigger asChild>{DeleteTrigger}</DrawerTrigger>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle>Delete Conversation</DrawerTitle>
            <DrawerDescription>
              Are you sure you want to delete this conversation? This action
              cannot be undone.
            </DrawerDescription>
          </DrawerHeader>
          <DrawerFooter>
            <DrawerClose asChild>
              <Button variant="destructive" onClick={handleDelete}>
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
      <DialogTrigger asChild>{DeleteTrigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Conversation</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete this conversation? This action
            cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Cancel</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
