import { Button } from "@/components/ui/button";

export function ConversationListLoading() {
	return (
		<div className="p-4 text-center text-muted-foreground">
			<div className="animate-pulse space-y-2">
				<div className="h-4 rounded bg-muted" />
				<div className="mx-auto h-4 w-5/6 rounded bg-muted" />
				<div className="mx-auto h-4 w-4/6 rounded bg-muted" />
			</div>
		</div>
	);
}

export function ConversationListError({
	error,
}: {
	error: { message: string };
}) {
	return (
		<div className="p-4 text-center text-destructive">
			<p>Error: {error.message}</p>
		</div>
	);
}

interface ConversationListEmptyProps {
	searchQuery: string;
	onClearSearch: () => void;
}

export function ConversationListEmpty({
	searchQuery,
	onClearSearch,
}: ConversationListEmptyProps) {
	return (
		<div className="p-4 text-center text-muted-foreground">
			{searchQuery ? (
				<>
					<p>No matching conversations</p>
					<Button
						variant="link"
						size="sm"
						onClick={onClearSearch}
						className="mt-1"
					>
						Clear search
					</Button>
				</>
			) : (
				<>
					<p>No conversations yet</p>
					<p className="mt-1 text-sm">Start a new chat to begin</p>
				</>
			)}
		</div>
	);
}
