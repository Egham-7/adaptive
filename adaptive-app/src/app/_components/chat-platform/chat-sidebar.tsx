"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import {
  Sidebar,
  SidebarContent,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import CommonSidebarHeader from "@/components/sidebar-header";
import { useConversations } from "@/hooks/conversations/use-conversations";
import { useCreateConversation } from "@/hooks/conversations/use-create-conversation";
import { usePinConversation } from "@/hooks/conversations/use-pin-conversation";
import { SidebarActions } from "./chat-sidebar/sidebar-actions";
import { ConversationList } from "./chat-sidebar/conversation-list";
import { SidebarNavFooter } from "./chat-sidebar/sidebar-nav-footer";
import type { ConversationListItem } from "@/types";

interface ChatbotSidebarClientProps {
  initialConversations: ConversationListItem[];
}

export function ChatbotSidebar({
  initialConversations,
}: ChatbotSidebarClientProps) {
  const path = usePathname();
  const [searchQuery, setSearchQuery] = useState("");

  const { conversations, isLoading, error } =
    useConversations(initialConversations);
  const createConversationMutation = useCreateConversation();
  const pinConversationMutation = usePinConversation();

  const handlePinConversation = (conversationId: number, isPinned: boolean) => {
    pinConversationMutation.mutate({
      id: conversationId,
      pinned: !isPinned,
    });
  };

  const isConversationActive = (conversationId: number) => {
    return path === `/chat-platform/chats/${conversationId}`;
  };

  return (
    <Sidebar className="h-screen">
      <div className="space-x-2 flex items-center justify-between p-2">
        <CommonSidebarHeader href="/chat-platform/" />
        <SidebarTrigger />
      </div>
      <SidebarSeparator />

      <SidebarActions
        searchQuery={searchQuery}
        onSearchChange={(e) => setSearchQuery(e.target.value)}
        onCreateClick={() =>
          createConversationMutation.mutate({ title: "New Conversation" })
        }
        isCreatePending={createConversationMutation.isPending}
      />

      <SidebarContent className="px-1">
        <ConversationList
          conversations={conversations}
          isLoading={isLoading}
          error={error}
          searchQuery={searchQuery}
          onClearSearch={() => setSearchQuery("")}
          onPin={handlePinConversation}
          isConversationActive={isConversationActive}
        />
      </SidebarContent>

      <SidebarNavFooter />

      <SidebarRail />
    </Sidebar>
  );
}
