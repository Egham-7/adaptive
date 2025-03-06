import { Cpu, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ChatHeaderProps {
  currentModel?: string;
  currentProvider?: string;
  resetConversation: () => void;
}

export function ChatHeader({ currentModel, currentProvider, resetConversation }: ChatHeaderProps) {
  // Helper to capitalize first letter
  const capitalizeFirstLetter = (str?: string) => {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  return (
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
                      <Badge variant="outline" className="text-xs font-normal">
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
  );
}