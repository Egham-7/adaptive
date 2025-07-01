// src/components/chat/chat-sidebar/SidebarActions.tsx

import { PlusCircle, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SidebarActionsProps {
	searchQuery: string;
	onSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
	onCreateClick: () => void;
	isCreatePending: boolean;
}

export function SidebarActions({
	searchQuery,
	onSearchChange,
	onCreateClick,
	isCreatePending,
}: SidebarActionsProps) {
	return (
		<>
			<div className="p-3">
				<Button
					onClick={onCreateClick}
					className="flex w-full items-center gap-2"
					variant="outline"
					disabled={isCreatePending}
				>
					<PlusCircle className="h-4 w-4" />
					New Chat
				</Button>
			</div>
			<div className="px-3 pb-2">
				<div className="relative">
					<Search className="absolute top-2.5 left-2 h-4 w-4 text-muted-foreground" />
					<Input
						placeholder="Search conversations..."
						className="pl-8"
						value={searchQuery}
						onChange={onSearchChange}
					/>
				</div>
			</div>
		</>
	);
}
