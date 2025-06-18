import {
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarMenu,
} from "@/components/ui/sidebar";
import type { ConversationListItem } from "@/types";
import { ConversationItem } from "./conversation-item";

interface ConversationGroupProps {
	title: React.ReactNode;
	conversations: ConversationListItem[];
	onPin: (id: number, isPinned: boolean) => void;
	isConversationActive: (id: number) => boolean;
}

export function ConversationGroup({
	title,
	conversations,
	onPin,
	isConversationActive,
}: ConversationGroupProps) {
	if (conversations.length === 0) {
		return null;
	}

	return (
		<SidebarGroup>
			<SidebarGroupLabel>{title}</SidebarGroupLabel>
			<SidebarGroupContent>
				<SidebarMenu>
					{conversations.map((convo) => (
						<ConversationItem
							key={convo.id}
							conversation={convo}
							isActive={isConversationActive(convo.id)}
							onPin={onPin}
						/>
					))}
				</SidebarMenu>
			</SidebarGroupContent>
		</SidebarGroup>
	);
}
