import { Button } from "@/components/ui/button";

export function ConversationListLoading() {
  return (
    <div className="p-4 text-center text-muted-foreground">
      <div className="animate-pulse space-y-2">
        <div className="h-4 bg-muted rounded"></div>
        <div className="h-4 bg-muted rounded w-5/6 mx-auto"></div>
        <div className="h-4 bg-muted rounded w-4/6 mx-auto"></div>
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
          <p className="text-sm mt-1">Start a new chat to begin</p>
        </>
      )}
    </div>
  );
}
