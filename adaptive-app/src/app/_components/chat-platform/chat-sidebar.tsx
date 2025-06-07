"use client";

import CommonSidebarHeader from "@/components/sidebar-header";
import {
	Sidebar,
	SidebarContent,
	SidebarRail,
	SidebarSeparator,
	SidebarTrigger,
} from "@/components/ui/sidebar";
import { useConversations } from "@/hooks/conversations/use-conversations";
import { useCreateConversation } from "@/hooks/conversations/use-create-conversation";
import { usePinConversation } from "@/hooks/conversations/use-pin-conversation";
import type { ConversationListItem } from "@/types";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { ConversationList } from "./chat-sidebar/conversation-list";
import { SidebarActions } from "./chat-sidebar/sidebar-actions";
import { SidebarNavFooter } from "./chat-sidebar/sidebar-nav-footer";

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
			<div className="flex items-center justify-between space-x-2 p-2">
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
