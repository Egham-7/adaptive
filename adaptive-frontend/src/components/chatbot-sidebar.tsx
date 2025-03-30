import { PlusCircle, Search } from "lucide-react";
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
import { useConversations } from "@/lib/hooks/conversations/use-conversations";
import { formatDistanceToNow, isToday, subDays } from "date-fns";
import { Conversation } from "@/services/conversations/types";
import CommonSidebarHeader from "./sidebar-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useCreateConversation } from "@/lib/hooks/conversations/use-create-conversation";
import { Link, useNavigate } from "@tanstack/react-router";

export function ChatbotSidebar() {
  const { data: conversations, isLoading, error } = useConversations();
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();
  const createConversationMutation = useCreateConversation();

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
    try {
      const newConversation =
        await createConversationMutation.mutateAsync("New Conversation");
      navigate({ to: `/conversations/${newConversation.id}` });
    } catch (error) {
      console.error("Failed to create new conversation:", error);
    }
  };

  // Group conversations by time periods
  const groupConversations = (conversations: Conversation[] | undefined) => {
    if (!conversations) return { today: [], last30Days: [], older: [] };

    const now = new Date();
    const thirtyDaysAgo = subDays(now, 30);

    return conversations.reduce(
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
            {groupConversations.map((conversation) => {
              const lastMessage = getLastMessage(conversation);
              return (
                <SidebarMenuItem key={conversation.id}>
                  <Link
                    to={`/conversations/$conversationId`}
                    params={{ conversationId: String(conversation.id) }}
                    className="w-full"
                  >
                    <SidebarMenuButton tooltip={conversation.title}>
                      <div className="flex-1 overflow-hidden">
                        <div className="flex justify-between items-center">
                          <span className="font-medium truncate">
                            {conversation.title}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {lastMessage.time}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {lastMessage.role === "user" ? "You: " : ""}
                          {lastMessage.content}
                        </p>
                      </div>
                    </SidebarMenuButton>
                  </Link>
                </SidebarMenuItem>
              );
            })}
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

      <SidebarContent>
        {isLoading ? (
          <div className="p-4 text-center text-muted-foreground">
            Loading conversations...
          </div>
        ) : error ? (
          <div className="p-4 text-center text-destructive">
            Error loading conversations
          </div>
        ) : filteredConversations && filteredConversations.length > 0 ? (
          <>
            {renderConversationGroup(groupedConversations.today, "Today")}
            {renderConversationGroup(
              groupedConversations.last30Days,
              "Last 30 Days",
            )}
            {renderConversationGroup(groupedConversations.older, "Older")}
          </>
        ) : (
          <div className="p-4 text-center text-muted-foreground">
            {searchQuery ? "No matching conversations" : "No conversations yet"}
          </div>
        )}
      </SidebarContent>

      <SidebarFooter>
        <SidebarSeparator />
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton>
              <div className="flex items-center gap-2 w-full">
                <UserButton />
                <div className="flex-1">
                  <span className="font-medium">Account</span>
                </div>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
