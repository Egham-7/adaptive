// src/components/chat/chat-sidebar/delete-conversation-dialog.tsx

"use client";

import { Trash2 } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import {
	Drawer,
	DrawerClose,
	DrawerContent,
	DrawerDescription,
	DrawerFooter,
	DrawerHeader,
	DrawerTitle,
	DrawerTrigger,
} from "@/components/ui/drawer";
import { useDeleteConversation } from "@/hooks/conversations/use-delete-conversation";
import { useIsMobile } from "@/hooks/use-mobile";
import type { ConversationListItem } from "@/types";

interface DeleteConversationDialogProps {
	conversation: ConversationListItem;
}

export function DeleteConversationDialog({
	conversation,
}: DeleteConversationDialogProps) {
	const isMobile = useIsMobile();
	const router = useRouter();
	const pathname = usePathname();
	const deleteConversationMutation = useDeleteConversation();

	const handleDelete = () => {
		// Check if we're currently viewing the conversation that's being deleted
		const isCurrentConversation =
			pathname === `/chat-platform/chats/${conversation.id}`;

		deleteConversationMutation.mutate({ id: conversation.id });

		// If we deleted the conversation we were viewing, navigate to the home page.
		if (isCurrentConversation) {
			router.push("/chat-platform");
		}
	};

	const DeleteTrigger = (
		<Button
			variant="ghost"
			size="icon"
			className="h-7 w-7 text-destructive hover:bg-destructive/10 hover:text-destructive/90"
			disabled={deleteConversationMutation.isPending}
		>
			<Trash2 className="h-4 w-4" />
			<span className="sr-only">Delete</span>
		</Button>
	);

	const DialogContentComponent = (
		<>
			<DialogHeader>
				<DialogTitle>Delete Conversation</DialogTitle>
				<DialogDescription>
					Are you sure you want to delete "{conversation.title}"? This action
					cannot be undone.
				</DialogDescription>
			</DialogHeader>
			<DialogFooter>
				<DialogClose asChild>
					<Button variant="outline">Cancel</Button>
				</DialogClose>
				<DialogClose asChild>
					<Button
						variant="destructive"
						onClick={handleDelete}
						disabled={deleteConversationMutation.isPending}
					>
						{deleteConversationMutation.isPending ? "Deleting..." : "Delete"}
					</Button>
				</DialogClose>
			</DialogFooter>
		</>
	);

	if (isMobile) {
		return (
			<Drawer>
				<DrawerTrigger asChild>{DeleteTrigger}</DrawerTrigger>
				<DrawerContent>
					<DrawerHeader className="text-left">
						<DrawerTitle>Delete Conversation</DrawerTitle>
						<DrawerDescription>
							Are you sure you want to delete "{conversation.title}"? This
							action cannot be undone.
						</DrawerDescription>
					</DrawerHeader>
					<DrawerFooter className="pt-4">
						<Button
							variant="destructive"
							onClick={handleDelete}
							disabled={deleteConversationMutation.isPending}
						>
							{deleteConversationMutation.isPending ? "Deleting..." : "Delete"}
						</Button>
						<DrawerClose asChild>
							<Button variant="outline">Cancel</Button>
						</DrawerClose>
					</DrawerFooter>
				</DrawerContent>
			</Drawer>
		);
	}

	return (
		<Dialog>
			<DialogTrigger asChild>{DeleteTrigger}</DialogTrigger>
			<DialogContent>{DialogContentComponent}</DialogContent>
		</Dialog>
	);
}
