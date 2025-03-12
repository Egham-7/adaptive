import { MessageSquare, PlusCircle, Search } from "lucide-react";
import { UserButton } from "@clerk/clerk-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { useConversations } from "@/lib/hooks/conversations/use-conversations";
import { formatDistanceToNow } from "date-fns";
import { Conversation } from "@/services/conversations/types";

export function ChatbotSidebar() {
  const { data: conversations, isLoading, error } = useConversations();

  // Helper function to get the most recent message from a conversation
  const getLastMessage = (conversation: Conversation) => {
    if (!conversation.messages || conversation.messages.length === 0) {
      return { content: "No messages", time: "" };
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

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center justify-between px-4 py-2">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-bold">ChatBot</h1>
          </div>
          <SidebarTrigger />
        </div>
        <SidebarGroup className="py-2">
          <SidebarGroupContent className="relative">
            <SidebarInput placeholder="Search..." className="pl-8" />
            <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="flex justify-between items-center">
            Recent Conversations
            <SidebarMenuAction showOnHover>
              <PlusCircle className="h-4 w-4" />
            </SidebarMenuAction>
          </SidebarGroupLabel>
          <SidebarGroupContent>
            {isLoading ? (
              <div className="p-4 text-center text-muted-foreground">
                Loading conversations...
              </div>
            ) : error ? (
              <div className="p-4 text-center text-destructive">
                Error loading conversations
              </div>
            ) : (
              <SidebarMenu>
                {conversations && conversations.length > 0 ? (
                  conversations.map((conversation) => {
                    const lastMessage = getLastMessage(conversation);
                    return (
                      <SidebarMenuItem key={conversation.id}>
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
                      </SidebarMenuItem>
                    );
                  })
                ) : (
                  <div className="p-4 text-center text-muted-foreground">
                    No conversations yet
                  </div>
                )}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
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
