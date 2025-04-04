import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
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
import { useState } from "react";

const formSchema = z.object({
  title: z.string().min(1, "Title is required"),
});

interface EditConversationFormProps {
  conversation: Conversation;
  onSubmit: (values: z.infer<typeof formSchema>) => Promise<void>;
  onCancel: () => void;
}

function EditConversationForm({
  conversation,
  onSubmit,
  onCancel,
}: EditConversationFormProps) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      title: conversation.title,
    },
  });

  const handleSubmit = form.handleSubmit(onSubmit);
  const isSubmitting = form.formState.isSubmitting;

  return (
    <Form {...form}>
      <form onSubmit={handleSubmit}>
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
        <div className="mt-4 flex justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              form.reset();
              onCancel();
            }}
          >
            Cancel
          </Button>
          <Button disabled={isSubmitting} type="submit" onClick={handleSubmit}>
            {isSubmitting ? "Submitting..." : "Save changes"}
          </Button>
        </div>
      </form>
    </Form>
  );
}

interface EditConversationDialogProps {
  conversation: Conversation;
}

export function EditConversationDialog({
  conversation,
}: EditConversationDialogProps) {
  const isMobile = useIsMobile();
  const updateConversationMutation = useUpdateConversation();
  const [isOpen, setIsOpen] = useState(false);

  const handleSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      await updateConversationMutation.mutateAsync({
        id: conversation.id,
        title: values.title,
      });
      setIsOpen(false); // Close the dialog/drawer after successful submission
    } catch (error) {
      console.error("Failed to update conversation:", error);
    }
  };

  const handleCancel = () => {
    setIsOpen(false);
  };

  const EditTrigger = (
    <Button variant="ghost" size="icon" className="h-7 w-7">
      <Pencil className="h-4 w-4" />
      <span className="sr-only">Edit</span>
    </Button>
  );

  if (isMobile) {
    return (
      <Drawer open={isOpen} onOpenChange={setIsOpen}>
        <DrawerTrigger asChild>{EditTrigger}</DrawerTrigger>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle>Edit Conversation</DrawerTitle>
            <DrawerDescription>
              Change the title of your conversation.
            </DrawerDescription>
          </DrawerHeader>
          <div className="px-4 pb-4">
            <EditConversationForm
              conversation={conversation}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
            />
          </div>
        </DrawerContent>
      </Drawer>
    );
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>{EditTrigger}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Conversation</DialogTitle>
          <DialogDescription>
            Change the title of your conversation.
          </DialogDescription>
        </DialogHeader>
        <EditConversationForm
          conversation={conversation}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
        />
      </DialogContent>
    </Dialog>
  );
}
