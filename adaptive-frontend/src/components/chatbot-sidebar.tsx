import { PlusCircle, Search, Pin } from "lucide-react";
import { UserButton } from "@clerk/clerk-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { formatDistanceToNow, isToday, subDays } from "date-fns";
import { Conversation } from "@/services/conversations/types";
import CommonSidebarHeader from "./sidebar-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useCreateConversation } from "@/hooks/conversations/use-create-conversation";
import { ModeToggle } from "./mode-toggle";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { dark, shadesOfPurple } from "@clerk/themes";
import { usePinConversation } from "@/hooks/conversations/use-pin-conversation";
import { useConversations } from "@/hooks/conversations/use-conversations";
import { EditConversationDialog } from "./chat/chat-sidebar/edit-conversation-dialog";
import { DeleteConversationDialog } from "./chat/chat-sidebar/delete-conversation-dialog";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function ChatbotSidebar() {
  const { data: conversations, isLoading, error } = useConversations();
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();
  const createConversationMutation = useCreateConversation();
  const pinConversationMutation = usePinConversation();
  const { theme } = useTheme();
  const path = usePathname();

  // Helper function to get the most recent message from a conversation
  const getLastMessage = (conversation: Conversation) => {
    if (!conversation.messages || conversation.messages.length === 0) {
      return { content: "No messages", time: "", role: "" };
    }
    // Get the last message in the array
    const lastMessage = conversation.messages[conversation.messages.length - 1];
    return {
      content: lastMessage.content,
      time: formatRelativeTime(lastMessage.updated_at),
      role: lastMessage.role,
    };
  };

  const formatRelativeTime = (timestamp: string) => {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
  };

  const handleCreateNewChat = async () => {
    const newConversation = await createConversationMutation.mutateAsync({
      title: "New Conversation",
    });
    router.push(`/conversations/${newConversation.id}`);
  };

  const handlePinConversation = async (conversationId: number) => {
    const conversation = conversations?.find((c) => c.id === conversationId);
    if (!conversation) return;

    await pinConversationMutation.mutateAsync({
      conversationId: conversationId,
      isPinned: conversation.pinned,
    });
  };

  // Check if a conversation is active
  const isConversationActive = (conversationId: number) => {
    return path === `/conversations/${conversationId}`;
  };

  // Group conversations by time periods
  const groupConversations = (conversations: Conversation[] | undefined) => {
    if (!conversations)
      return { pinned: [], today: [], last30Days: [], older: [] };

    // First separate pinned conversations
    const pinned: Conversation[] = [];
    const unpinned: Conversation[] = [];

    conversations.forEach((conversation) => {
      if (conversation.pinned) {
        pinned.push(conversation);
      } else {
        unpinned.push(conversation);
      }
    });

    // Sort pinned by updated_at
    pinned.sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    );

    // Group unpinned by time periods
    const now = new Date();
    const thirtyDaysAgo = subDays(now, 30);

    return unpinned.reduce(
      (groups, conversation) => {
        const updatedAt = new Date(conversation.updated_at);
        if (isToday(updatedAt)) {
          groups.today.push(conversation);
        } else if (updatedAt >= thirtyDaysAgo) {
          groups.last30Days.push(conversation);
        } else {
          groups.older.push(conversation);
        }
        return groups;
      },
      {
        pinned,
        today: [] as Conversation[],
        last30Days: [] as Conversation[],
        older: [] as Conversation[],
      },
    );
  };

  // Filter conversations based on search query
  const filteredConversations = conversations?.filter(
    (conversation) =>
      conversation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      conversation.messages?.some((msg) =>
        msg.content.toLowerCase().includes(searchQuery.toLowerCase()),
      ),
  );

  const groupedConversations = groupConversations(filteredConversations);

  const renderConversationItem = (conversation: Conversation) => {
    const lastMessage = getLastMessage(conversation);
    const isActive = isConversationActive(conversation.id);

    return (
      <SidebarMenuItem key={conversation.id}>
        <Link
          to={`/conversations/$conversationId`}
          params={{ conversationId: String(conversation.id) }}
          className="w-full"
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
                <span className="text-xs text-muted-foreground">
                  {lastMessage.time}
                </span>
              </div>
              <p className="text-xs text-muted-foreground truncate">
                {lastMessage.role === "user" ? "You: " : "AI: "}
                {lastMessage.content}
              </p>
            </div>

            {/* Action buttons that appear on hover */}
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
                        handlePinConversation(conversation.id);
                      }}
                    >
                      <Pin
                        className={cn(
                          "h-4 w-4",
                          conversation.pinned && "fill-current",
                        )}
                      />
                      <span className="sr-only">
                        {conversation.pinned ? "Unpin" : "Pin"}
                      </span>
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
  };

  const renderConversationGroup = (
    groupConversations: Conversation[],
    title: string,
  ) => {
    if (groupConversations.length === 0) return null;
    return (
      <SidebarGroup>
        <SidebarGroupLabel>{title}</SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu>
            {groupConversations.map(renderConversationItem)}
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  };

  return (
    <Sidebar className="h-screen">
      <div className="space-x-2 flex items-center justify-between p-2">
        <CommonSidebarHeader />
        <SidebarTrigger />
      </div>
      <SidebarSeparator />
      <div className="p-3">
        <Button
          onClick={handleCreateNewChat}
          className="w-full flex items-center gap-2"
          variant="outline"
        >
          <PlusCircle className="h-4 w-4" />
          New Chat
        </Button>
      </div>
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search conversations..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>
      <SidebarContent className="px-1">
        {isLoading ? (
          <div className="p-4 text-center text-muted-foreground">
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-muted rounded"></div>
              <div className="h-4 bg-muted rounded w-5/6 mx-auto"></div>
              <div className="h-4 bg-muted rounded w-4/6 mx-auto"></div>
            </div>
          </div>
        ) : error ? (
          <div className="p-4 text-center text-destructive">
            <p>Error loading conversations</p>
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => window.location.reload()}
            >
              Retry
            </Button>
          </div>
        ) : filteredConversations && filteredConversations.length > 0 ? (
          <>
            {groupedConversations.pinned.length > 0 && (
              <SidebarGroup>
                <SidebarGroupLabel>
                  <span className="flex items-center gap-1">
                    <Pin className="h-3 w-3 fill-current" />
                    Pinned
                  </span>
                </SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {groupedConversations.pinned.map(renderConversationItem)}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            )}
            {renderConversationGroup(groupedConversations.today, "Today")}
            {renderConversationGroup(
              groupedConversations.last30Days,
              "Last 30 Days",
            )}
            {renderConversationGroup(groupedConversations.older, "Older")}
          </>
        ) : (
          <div className="p-4 text-center text-muted-foreground">
            {searchQuery ? (
              <>
                <p>No matching conversations</p>
                <Button
                  variant="link"
                  size="sm"
                  onClick={() => setSearchQuery("")}
                  className="mt-1"
                >
                  Clear search
                </Button>
              </>
            ) : (
              <>
                <p>No conversations yet</p>
                <p className="text-sm mt-1">Start a new chat to begin</p>
              </>
            )}
          </div>
        )}
      </SidebarContent>
      <SidebarFooter>
        <SidebarSeparator />
        <SidebarMenu className="flex-row items-center justify-between p-2">
          <SidebarMenuItem>
            <SidebarMenuButton asChild>
              <UserButton
                appearance={{
                  baseTheme: theme === "dark" ? dark : shadesOfPurple,
                  elements: {
                    userButtonAvatarBox: "w-8 h-8",
                  },
                }}
              />
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild>
              <ModeToggle />
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
