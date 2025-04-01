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
import { useMediaQuery } from "@/hooks/use-media-query";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

export default function DeleteMessageDialog({
  userMessage,
  hasSubsequentMessages,
  onDelete,
  disabled,
}: {
  userMessage: boolean;
  hasSubsequentMessages: boolean;
  onDelete: () => void;
  disabled: boolean;
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
                : "hover:bg-background/80",
            )}
            disabled={disabled}
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
              : "hover:bg-background/80",
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
