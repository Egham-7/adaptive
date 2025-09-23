import { Pin } from "lucide-react";
import Link from "next/link";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import {
	Tooltip,
	TooltipContent,
	TooltipProvider,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/shared/utils";
import type { ConversationListItem } from "@/types";
import { DeleteConversationDialog } from "./delete-conversation-dialog";
import { EditConversationDialog } from "./edit-conversation-dialog";

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
	return (
		<SidebarMenuItem className="group relative">
			<Link
				href={`/chat-platform/chats/${conversation.id}`}
				className="w-full"
				prefetch={true}
			>
				<SidebarMenuButton
					className={cn(
						"relative",
						isActive && "bg-accent text-accent-foreground",
					)}
				>
					<div className="flex-1 overflow-hidden">
						<div className="flex items-center justify-between">
							<span className="flex items-center gap-1 truncate font-medium">
								{conversation.title}
								{conversation.pinned && (
									<Pin className="inline h-3 w-3 fill-current text-primary" />
								)}
							</span>
						</div>
					</div>
				</SidebarMenuButton>
			</Link>

			<div
				className={
					"-translate-y-1/2 absolute top-1/2 right-2 z-10 flex items-center gap-1 rounded bg-background/80 p-0.5 opacity-0 backdrop-blur-xs transition-opacity hover:opacity-100"
				}
			>
				<TooltipProvider>
					<Tooltip>
						<TooltipTrigger asChild>
							<button
								type="button"
								className={cn(
									"inline-flex h-7 w-7 items-center justify-center rounded-md hover:bg-accent hover:text-accent-foreground",
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
							</button>
						</TooltipTrigger>
						<TooltipContent side="bottom">
							{conversation.pinned ? "Unpin conversation" : "Pin conversation"}
						</TooltipContent>
					</Tooltip>
				</TooltipProvider>

				<EditConversationDialog conversation={conversation} />
				<DeleteConversationDialog conversation={conversation} />
			</div>
		</SidebarMenuItem>
	);
}
