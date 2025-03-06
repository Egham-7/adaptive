"use client";

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
  MessageSquare,
  ChevronDown,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useConversation } from "@/lib/hooks/use-conversation";
import { cn } from "@/lib/utils";
import type { Message } from "@/services/llms/types";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";

// Form schema
const formSchema = z.object({
  message: z.string().min(1, {
    message: "Please enter a message.",
  }),
});

export default function Home() {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showActions, setShowActions] = useState(true);
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
  }, [messagesEndRef]); // Removed 'messages' dependency

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

  // Toggle action buttons visibility
  const toggleActions = () => {
    setShowActions(!showActions);
  };

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      {/* Sticky Header */}
      <header className="fixed top-0 left-0 right-0 z-50 py-3 border-b bg-background/95 backdrop-blur-sm">
        <div className="w-full max-w-5xl px-4 mx-auto">
          <div className="flex items-center justify-between">
            {/* Logo and Heading */}
            <div className="flex items-center gap-3">
              <div className="relative w-10 h-10 overflow-hidden">
                <img
                  src="https://v0.dev/placeholder.svg"
                  className="object-cover w-full h-full rounded-lg"
                  alt="Adaptive Logo"
                />
              </div>
              <h1 className="text-2xl font-semibold font-display">Adaptive</h1>
            </div>

            {/* Model Display (Read-only) */}
            <div className="flex items-center gap-3">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2 bg-secondary/50 hover:bg-secondary rounded-full px-3 py-1.5 text-sm transition-colors">
                      <Cpu className="w-4 h-4 text-muted-foreground" />
                      <span className="hidden sm:inline truncate max-w-[120px]">
                        {currentModel || "Model not selected"}
                      </span>
                      {currentProvider && (
                        <Badge
                          variant="outline"
                          className="text-xs font-normal"
                        >
                          {capitalizeFirstLetter(currentProvider)}
                        </Badge>
                      )}
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    {currentModel || "Model not selected"}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <Button
                variant="outline"
                size="sm"
                onClick={resetConversation}
                className="items-center hidden gap-1 sm:flex"
              >
                <Plus className="w-4 h-4" />
                <span>New Chat</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full max-w-5xl mx-auto flex flex-col h-screen pt-[80px] pb-[140px]">
        {/* Chat Messages Area */}
        <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto">
          {messages.length === 0 ? (
            <>
              <div className="flex items-center justify-center w-16 h-16 mb-4 rounded-full bg-primary/10">
                <MessageSquare className="w-8 h-8 text-primary" />
              </div>
              <h2 className="mb-2 text-xl font-medium">Welcome to Adaptive</h2>
              <div className="max-w-md text-muted-foreground">
                Send a message to start a conversation. Adaptive will select the
                best AI model for your specific query.
              </div>
            </>
          ) : (
            messages.map((msg: Message, index: number) => (
              <div
                key={index}
                className={cn(
                  "flex w-full max-w-[90%] rounded-2xl p-4",
                  msg.role === "user"
                    ? "ml-auto bg-primary text-primary-foreground"
                    : "bg-muted"
                )}
              >
                <div className="prose-sm prose dark:prose-invert">
                  {msg.content}
                </div>
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex items-center gap-3 rounded-2xl bg-muted p-4 max-w-[90%]">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              <p>Processing your request...</p>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-3 rounded-2xl bg-destructive/10 p-4 max-w-[90%] text-destructive border border-destructive/20">
              <p>Error: {error}</p>
            </div>
          )}
        </div>

        <div ref={messagesEndRef} />
      </main>

      {/* Fixed Bottom Section */}
      <footer className="fixed bottom-0 left-0 right-0 z-40 pt-3 pb-5 border-t bg-background/95 backdrop-blur-sm">
        <div className="w-full max-w-5xl px-4 mx-auto space-y-3">
          {/* Action Buttons */}
          {showActions && (
            <div className="flex flex-wrap justify-center gap-2 pb-2 duration-200 animate-in fade-in slide-in-from-bottom-2">
              <Button
                variant="secondary"
                size="sm"
                className="gap-2 rounded-full"
              >
                <ImageIcon className="w-4 h-4" />
                <span>Create image</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="gap-2 rounded-full"
              >
                <FileText className="w-4 h-4" />
                <span>Summarize text</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="gap-2 rounded-full"
              >
                <Eye className="w-4 h-4" />
                <span>Analyze images</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="gap-2 rounded-full"
              >
                <GraduationCap className="w-4 h-4" />
                <span>Get advice</span>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="rounded-full"
                  >
                    <MoreHorizontal className="w-4 h-4" />
                    <span className="sr-only">More options</span>
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>Code generation</DropdownMenuItem>
                  <DropdownMenuItem>Data analysis</DropdownMenuItem>
                  <DropdownMenuItem>Translation</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )}

          {/* Message Input Area with shadcn Form */}
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

          {/* Footer */}
          <p className="pt-1 text-xs text-center text-muted-foreground">
            Results may vary. Verify important information.
          </p>
        </div>
      </footer>
    </div>
  );
}
