import { Pin } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { EditConversationDialog } from "./edit-conversation-dialog";
import { DeleteConversationDialog } from "./delete-conversation-dialog";
import type { ConversationListItem } from "@/types";

interface ConversationItemProps {
  conversation: ConversationListItem;
  isActive: boolean;
  onPin: (id: number, isPinned: boolean) => void;
}

export function ConversationItem({
  conversation,
  isActive,
  onPin,
}: ConversationItemProps) {
  const lastMessage = conversation.messages[0];

  return (
    <SidebarMenuItem>
      <Link
        href={`/chat-platform/chats/${conversation.id}`}
        className="w-full"
        prefetch={true}
      >
        <SidebarMenuButton
          tooltip={conversation.title}
          className={cn(
            "relative group",
            isActive && "bg-accent text-accent-foreground",
          )}
        >
          <div className="flex-1 overflow-hidden">
            <div className="flex justify-between items-center">
              <span className="font-medium truncate flex items-center gap-1">
                {conversation.title}
                {conversation.pinned && (
                  <Pin className="h-3 w-3 inline fill-current text-primary" />
                )}
              </span>
            </div>
            {lastMessage && (
              <p className="text-xs text-muted-foreground truncate">
                {lastMessage.role === "user" ? "You: " : "AI: "}
                {lastMessage.content}
              </p>
            )}
          </div>

          <div
            className={cn(
              "absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 bg-background/80 backdrop-blur-xs rounded p-0.5 opacity-0 transition-opacity",
              "group-hover:opacity-100",
              isActive && "opacity-100 bg-accent/80",
            )}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
          >
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                      "h-7 w-7",
                      conversation.pinned && "text-primary",
                    )}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onPin(conversation.id, conversation.pinned);
                    }}
                  >
                    <Pin
                      className={cn(
                        "h-4 w-4",
                        conversation.pinned && "fill-current",
                      )}
                    />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {conversation.pinned
                    ? "Unpin conversation"
                    : "Pin conversation"}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <EditConversationDialog conversation={conversation} />
            <DeleteConversationDialog conversation={conversation} />
          </div>
        </SidebarMenuButton>
      </Link>
    </SidebarMenuItem>
  );
}
