import { Loader2, Send, StopCircle, CircleDot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";

// Form schema
const formSchema = z.object({
  message: z.string().min(1, {
    message: "Please enter a message.",
  }),
});

interface MessageInputProps {
  isLoading: boolean;
  sendMessage: (message: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  abortStreaming?: () => void;
}

export function MessageInput({
  isLoading,
  sendMessage,
  disabled,
  isStreaming = false,
  abortStreaming,
}: MessageInputProps) {
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
    <div className="p-4 border-t border-muted">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="relative">
          <div className="relative rounded-xl bg-card p-2 shadow-lg">
            {/* Input field */}
            <div className="flex items-center mb-4">
              <FormField
                control={form.control}
                name="message"
                render={({ field }) => (
                  <FormItem className="flex-1 m-0">
                    <FormControl>
                      <textarea
                        {...field}
                        placeholder="Message Adaptive..."
                        className="w-full bg-transparent text-foreground outline-none text-lg px-2 resize-none h-24 max-h-24 overflow-y-auto border-none focus:outline-none focus:ring-0"
                        disabled={isLoading || disabled}
                        rows={1}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            if (form.formState.isValid) {
                              form.handleSubmit(onSubmit)();
                            }
                          }
                        }}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              {/* Right side icons */}
              <div className="flex items-center space-x-2">
                <div className="flex items-center bg-white/10 rounded-full p-1">
                  <div className="bg-teal-500 rounded-full p-1">
                    <CircleDot className="h-4 w-4 text-white" />
                  </div>
                  <div className="bg-teal-600 rounded-full p-1 -ml-1">
                    <CircleDot className="h-4 w-4 text-white" />
                  </div>
                </div>

                {isStreaming && abortStreaming ? (
                  <Button
                    type="button"
                    variant="destructive"
                    size="icon"
                    className="rounded-lg p-3"
                    onClick={abortStreaming}
                    title="Stop generating"
                  >
                    <StopCircle className="w-5 h-5" />
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    variant="default"
                    className="rounded-lg p-3"
                    disabled={isLoading || !form.formState.isValid}
                  >
                    {isLoading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </Button>
                )}
              </div>
            </div>
          </div>
        </form>
      </Form>
    </div>
  );
}
