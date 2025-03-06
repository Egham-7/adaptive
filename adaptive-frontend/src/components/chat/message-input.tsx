import { ChevronDown, Loader2, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { cn } from "@/lib/utils";

// Form schema
const formSchema = z.object({
  message: z.string().min(1, {
    message: "Please enter a message.",
  }),
});

interface MessageInputProps {
  isLoading: boolean;
  sendMessage: (message: string) => void;
  showActions: boolean;
  toggleActions: () => void;
}

export function MessageInput({ isLoading, sendMessage, showActions, toggleActions }: MessageInputProps) {
  // Setup form with react-hook-form
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      message: "",
    },
  });

  // Form submission handler
  function onSubmit(values: z.infer<typeof formSchema>) {
    sendMessage(values.message);
    form.reset();
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="relative">
        <div className="p-1 border rounded-full shadow-lg bg-card">
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="rounded-full"
              onClick={toggleActions}
              title="Toggle quick actions"
            >
              <ChevronDown
                className={cn(
                  "h-5 w-5 transition-transform",
                  showActions ? "rotate-180" : ""
                )}
              />
            </Button>
            <FormField
              control={form.control}
              name="message"
              render={({ field }) => (
                <FormItem className="flex-1 m-0">
                  <FormControl>
                    <Input
                      {...field}
                      placeholder="Message Adaptive..."
                      className="flex-1 text-base bg-transparent border-none focus:outline-none focus:ring-0 placeholder:text-muted-foreground"
                      disabled={isLoading}
                    />
                  </FormControl>
                </FormItem>
              )}
            />
            <Button
              type="submit"
              variant="default"
              size="icon"
              className="rounded-full bg-primary text-primary-foreground"
              disabled={isLoading || !form.formState.isValid}
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </Button>
          </div>
        </div>
      </form>
    </Form>
  );
}