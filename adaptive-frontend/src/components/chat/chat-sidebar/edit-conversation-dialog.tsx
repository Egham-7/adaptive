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
import { Pencil } from "lucide-react";
import { useUpdateConversation } from "@/hooks/conversations/use-update-conversation";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";

const formSchema = z.object({
  title: z.string().min(1, "Title is required"),
});

interface EditConversationDialogProps {
  conversation: Conversation;
}

export function EditConversationDialog({
  conversation,
}: EditConversationDialogProps) {
  const isMobile = useIsMobile();
  const updateConversationMutation = useUpdateConversation();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      title: conversation.title,
    },
  });

  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      await updateConversationMutation.mutateAsync({
        id: conversation.id,
        title: values.title,
      });
    } catch (error) {
      console.error("Failed to update conversation:", error);
    }
  };

  const EditTrigger = (
    <Button variant="ghost" size="icon" className="h-7 w-7">
      <Pencil className="h-4 w-4" />
      <span className="sr-only">Edit</span>
    </Button>
  );

  const EditContent = (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <FormField
            control={form.control}
            name="title"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Conversation Title</FormLabel>
                <FormControl>
                  <Input placeholder="Enter a title" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <div className={isMobile ? "mt-4" : "flex justify-end gap-2 mt-4"}>
          <Button type="submit">Save changes</Button>
        </div>
      </form>
    </Form>
  );

  if (isMobile) {
    return (
      <Drawer>
        <DrawerTrigger asChild>{EditTrigger}</DrawerTrigger>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle>Edit Conversation</DrawerTitle>
            <DrawerDescription>
              Change the title of your conversation.
            </DrawerDescription>
          </DrawerHeader>
          <div className="px-4 pb-4">{EditContent}</div>
          <DrawerFooter>
            <DrawerClose asChild>
              <Button
                variant="outline"
                type="button"
                onClick={() => form.reset()}
              >
                Cancel
              </Button>
            </DrawerClose>
          </DrawerFooter>
        </DrawerContent>
      </Drawer>
    );
  }

  return (
    <Dialog>
      <DialogTrigger asChild>{EditTrigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Conversation</DialogTitle>
          <DialogDescription>
            Change the title of your conversation.
          </DialogDescription>
        </DialogHeader>
        {EditContent}
        <DialogFooter>
          <DialogClose asChild>
            <Button
              variant="outline"
              type="button"
              onClick={() => form.reset()}
            >
              Cancel
            </Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
