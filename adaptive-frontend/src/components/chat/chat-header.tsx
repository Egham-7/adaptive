import { useState, useRef } from "react";
import { Cpu, Plus, Edit2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { SidebarTrigger, useSidebar } from "../ui/sidebar";
import { SiOpenai } from "react-icons/si";

interface ChatHeaderProps {
  currentModel?: string;
  currentProvider?: string;
  resetConversation: () => void;
  title: string;
  setTitle: (title: string) => void;
}

function ProviderIcon({ provider }: { provider: string }) {
  const providerLower = provider.toLowerCase();

  // Return the appropriate icon based on the provider
  switch (providerLower) {
    case "openai":
      return <SiOpenai className="w-4 h-4" />;
    case "groq":
      return <img src="/groq.png" alt="Groq" className="w-4 h-4" />;
    case "deepseek":
      return <img src="/deepseek.svg" alt="Deepseek" className="w-4 h-4" />;
    default:
      return (
        <div className="w-4 h-4 flex items-center justify-center">
          <Cpu className="w-full h-full" />
        </div>
      );
  }
}

export default function ChatHeader({
  currentModel,
  currentProvider,
  resetConversation,
  title,
  setTitle,
}: ChatHeaderProps) {
  const [isEditing, setIsEditing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newTitle = inputRef.current?.value.trim();
    if (newTitle) {
      setTitle(newTitle);
    }
    setIsEditing(false);
  };

  const { open, openMobile } = useSidebar();

  return (
    <header className="sticky top-0 z-50 p-3 border-b bg-background/95 backdrop-blur-sm">
      <div className="w-full max-w-5xl px-4 mx-auto">
        <div className="flex items-center justify-between">
          {/* Logo and Heading */}
          <div className="flex items-center gap-3">
            {!open && !openMobile && <SidebarTrigger />}
            <div className="relative w-10 h-10 overflow-hidden">
              <img
                src="https://v0.dev/placeholder.svg"
                className="object-cover w-full h-full rounded-lg"
                alt="Adaptive Logo"
              />
            </div>
            {isEditing ? (
              <form onSubmit={handleSubmit} onBlur={handleSubmit}>
                <Input
                  ref={inputRef}
                  defaultValue={title}
                  autoFocus
                  placeholder="Conversation title"
                  className="h-8 min-w-[200px] text-lg font-semibold font-display"
                  onKeyDown={(e) => e.key === "Escape" && setIsEditing(false)}
                />
              </form>
            ) : (
              <div
                className="flex items-center gap-2 cursor-pointer"
                onClick={() => setIsEditing(true)}
              >
                <h1 className="text-2xl font-semibold font-display truncate max-w-[300px]">
                  {title}
                </h1>
                <Button
                  size="icon"
                  variant="ghost"
                  className="h-6 w-6 opacity-50 hover:opacity-100"
                >
                  <Edit2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            )}
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
                      <Badge variant="outline" className="text-xs font-normal">
                        <ProviderIcon provider={currentProvider} />
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
  );
}
