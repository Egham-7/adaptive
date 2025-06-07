import { Button } from "@/components/ui/button";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import {
	Tooltip,
	TooltipContent,
	TooltipProvider,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { ConversationListItem } from "@/types";
import { Pin } from "lucide-react";
import Link from "next/link";
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
						"group relative",
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
						{lastMessage && (
							<p className="truncate text-muted-foreground text-xs">
								{lastMessage.role === "user" ? "You: " : "AI: "}
								{lastMessage.content}
							</p>
						)}
					</div>
					{/* 
            biome-ignore lint/a11y/useKeyWithClickEvents: This div's onClick is solely for event propagation control (stopping bubbling to parent Link), 
            not for direct user interaction via mouse or keyboard. The actual interactive elements (buttons) are nested inside.
          */}
					<div
						className={cn(
							"-translate-y-1/2 absolute top-1/2 right-2 flex items-center gap-1 rounded bg-background/80 p-0.5 opacity-0 backdrop-blur-xs transition-opacity",
							"group-hover:opacity-100",
							isActive && "bg-accent/80 opacity-100",
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
