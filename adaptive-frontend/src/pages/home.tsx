import {
  Plus,
  Cpu,
  ImageIcon,
  Send,
  FileText,
  GraduationCap,
  Eye,
  MoreHorizontal,
  Loader2,
} from "lucide-react";
import { useEffect, useRef } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConversation } from "@/lib/hooks/use-conversation";
import { cn } from "@/lib/utils";
import { Message } from "@/services/llms/types";

// Form schema
const formSchema = z.object({
  message: z.string().min(1, {
    message: "Please enter a message.",
  }),
});

export default function Home() {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    sendMessage,
    isLoading,
    error,
    resetConversation,
    lastResponse,
  } = useConversation();

  // Setup form with react-hook-form
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      message: "",
    },
  });

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Form submission handler
  function onSubmit(values: z.infer<typeof formSchema>) {
    sendMessage(values.message);
    form.reset();
  }

  // Helper to capitalize first letter
  const capitalizeFirstLetter = (str?: string) => {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  // Get current model info from the last response
  const currentProvider = lastResponse?.provider;
  const currentModel = lastResponse?.response?.model;

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-3xl space-y-4 flex flex-col h-[calc(100vh-2rem)]">
        {/* Logo and Heading */}
        <div className="flex flex-col items-center gap-4">
          <div className="relative w-16 h-16">
            <img
              src="https://v0.dev/placeholder.svg"
              className="rounded-lg"
              alt="Adaptive Logo"
            />
          </div>
          <h1 className="text-4xl font-display font-semibold text-center">
            Adaptive
          </h1>
        </div>

        {/* Model Display (Read-only) */}
        <div className="flex items-center justify-center gap-2">
          <Cpu className="w-5 h-5 text-muted-foreground" />
          <div className="bg-secondary rounded-md px-3 py-2 w-[180px] text-sm truncate">
            {currentModel || "Auto-selected model"}
          </div>
          <div className="text-xs text-muted-foreground">
            {currentProvider
              ? `${capitalizeFirstLetter(currentProvider)}`
              : "Auto-selected based on query"}
          </div>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 rounded-lg bg-secondary/30 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <p className="mb-2">Send a message to start a conversation</p>
              <p className="text-sm">
                Adaptive will select the best model for your query
              </p>
            </div>
          ) : (
            messages.map((msg: Message, index: number) => (
              <div
                key={index}
                className={cn(
                  "flex max-w-[80%] rounded-lg p-4",
                  msg.role === "user"
                    ? "ml-auto bg-primary text-primary-foreground"
                    : "bg-muted",
                )}
              >
                {msg.content}
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex items-center gap-2 rounded-lg bg-muted p-4 max-w-[80%]">
              <Loader2 className="h-4 w-4 animate-spin" />
              <p className="text-sm">Processing your request...</p>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-destructive/20 p-4 max-w-[80%] text-destructive">
              <p className="text-sm">Error: {error}</p>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Message Input Area with shadcn Form */}
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="relative">
            <div className="rounded-lg bg-card p-2">
              <div className="flex items-center gap-2 p-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="rounded-md"
                  onClick={resetConversation}
                  title="Reset conversation"
                >
                  <Plus className="h-5 w-5" />
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
                          className="flex-1 bg-transparent border-none focus:outline-none focus:ring-0 placeholder:text-muted-foreground"
                          disabled={isLoading}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <Button
                  type="submit"
                  variant="ghost"
                  size="icon"
                  className="rounded-md"
                  disabled={isLoading || !form.formState.isValid}
                >
                  {isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </Button>
              </div>
            </div>
          </form>
        </Form>

        {/* Action Buttons */}
        <div className="flex flex-wrap justify-center gap-2">
          <Button variant="secondary" className="rounded-md gap-2">
            <ImageIcon className="h-4 w-4" />
            Create image
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <FileText className="h-4 w-4" />
            Summarize text
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <Eye className="h-4 w-4" />
            Analyze images
          </Button>
          <Button variant="secondary" className="rounded-md gap-2">
            <GraduationCap className="h-4 w-4" />
            Get advice
          </Button>
          <Button variant="secondary" className="rounded-md">
            <MoreHorizontal className="h-4 w-4" />
            More
          </Button>
        </div>

        {/* Footer */}
        <p className="text-center text-sm text-muted-foreground">
          Results may vary. Verify important information.
        </p>
      </div>
    </div>
  );
}
