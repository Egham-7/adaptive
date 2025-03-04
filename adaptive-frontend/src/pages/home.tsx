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
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Sticky Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm py-3 border-b">
        <div className="max-w-5xl mx-auto w-full px-4">
          <div className="flex items-center justify-between">
            {/* Logo and Heading */}
            <div className="flex items-center gap-3">
              <div className="relative w-10 h-10 overflow-hidden">
                <img
                  src="https://v0.dev/placeholder.svg"
                  className="rounded-lg object-cover w-full h-full"
                  alt="Adaptive Logo"
                />
              </div>
              <h1 className="text-2xl font-display font-semibold">Adaptive</h1>
            </div>

            {/* Model Display (Read-only) */}
            <div className="flex items-center gap-3">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex items-center gap-2 bg-secondary/50 hover:bg-secondary rounded-full px-3 py-1.5 text-sm transition-colors">
                      <Cpu className="w-4 h-4 text-muted-foreground" />
                      <span className="hidden sm:inline truncate max-w-[120px]">
                        {currentModel || "Auto-selected model"}
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
                    <p>Model is auto-selected based on your query</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>

              <Button
                variant="outline"
                size="sm"
                onClick={resetConversation}
                className="hidden sm:flex items-center gap-1"
              >
                <Plus className="h-4 w-4" />
                <span>New Chat</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full max-w-5xl mx-auto flex flex-col h-screen pt-[80px] pb-[140px]">
        {/* Chat Messages Area */}
        {messages.length === 0 ? (
          <>
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <MessageSquare className="w-8 h-8 text-primary" />
            </div>
            <h2 className="text-xl font-medium mb-2">Welcome to Adaptive</h2>
            <div className="text-muted-foreground max-w-md">
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
                  : "bg-muted",
              )}
            >
              <div className="prose prose-sm dark:prose-invert">
                {msg.content}
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex items-center gap-3 rounded-2xl bg-muted p-4 max-w-[90%]">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p>Processing your request...</p>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-3 rounded-2xl bg-destructive/10 p-4 max-w-[90%] text-destructive border border-destructive/20">
            <p>Error: {error}</p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Fixed Bottom Section */}
      <footer className="fixed bottom-0 left-0 right-0 bg-background/95 backdrop-blur-sm pt-3 pb-5 border-t z-40">
        <div className="max-w-5xl mx-auto w-full px-4 space-y-3">
          {/* Action Buttons */}
          {showActions && (
            <div className="flex flex-wrap justify-center gap-2 pb-2 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <Button
                variant="secondary"
                size="sm"
                className="rounded-full gap-2"
              >
                <ImageIcon className="h-4 w-4" />
                <span>Create image</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="rounded-full gap-2"
              >
                <FileText className="h-4 w-4" />
                <span>Summarize text</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="rounded-full gap-2"
              >
                <Eye className="h-4 w-4" />
                <span>Analyze images</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                className="rounded-full gap-2"
              >
                <GraduationCap className="h-4 w-4" />
                <span>Get advice</span>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="sm"
                    className="rounded-full"
                  >
                    <MoreHorizontal className="h-4 w-4" />
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
              <div className="rounded-full bg-card p-1 shadow-lg border">
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
                        showActions ? "rotate-180" : "",
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
                            className="flex-1 bg-transparent border-none focus:outline-none focus:ring-0 placeholder:text-muted-foreground text-base"
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
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Send className="h-5 w-5" />
                    )}
                  </Button>
                </div>
              </div>
            </form>
          </Form>

          {/* Footer */}
          <p className="text-center text-xs text-muted-foreground pt-1">
            Results may vary. Verify important information.
          </p>
        </div>
      </footer>
    </div>
  );
}
