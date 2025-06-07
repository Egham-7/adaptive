import type { ConversationListItem } from "@/types";
import { isToday, subDays } from "date-fns";
import { Pin } from "lucide-react";
import { useMemo } from "react";
import { ConversationGroup } from "./conversation-group";
import {
  ConversationListEmpty,
  ConversationListError,
  ConversationListLoading,
} from "./conversation-list-states";
import type { useConversations } from "@/hooks/conversations/use-conversations";

type UseConversationsResult = ReturnType<typeof useConversations>;
type ConversationsErrorType = UseConversationsResult["error"];

interface ConversationListProps {
  conversations: ConversationListItem[] | undefined;
  isLoading: boolean;
  error: ConversationsErrorType;
  searchQuery: string;
  onClearSearch: () => void;
  onPin: (id: number, isPinned: boolean) => void;
  isConversationActive: (id: number) => boolean;
}

export function ConversationList({
  conversations,
  isLoading,
  error,
  searchQuery,
  onClearSearch,
  onPin,
  isConversationActive,
}: ConversationListProps) {
  const groupedConversations = useMemo(() => {
    const filtered =
      conversations?.filter(
        (c) =>
          c.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          c.messages[0]?.content
            .toLowerCase()
            .includes(searchQuery.toLowerCase()),
      ) ?? [];

    const pinned: ConversationListItem[] = [];
    const unpinned: ConversationListItem[] = [];

    for (const conversation of filtered) {
      if (conversation.pinned) {
        pinned.push(conversation);
      } else {
        unpinned.push(conversation);
      }
    }

    const now = new Date();
    const thirtyDaysAgo = subDays(now, 30);

    return unpinned.reduce(
      (groups, conversation) => {
        const updatedAt = new Date(conversation.updatedAt);
        if (isToday(updatedAt)) groups.today.push(conversation);
        else if (updatedAt >= thirtyDaysAgo)
          groups.last30Days.push(conversation);
        else groups.older.push(conversation);
        return groups;
      },
      {
        pinned,
        today: [] as ConversationListItem[],
        last30Days: [] as ConversationListItem[],
        older: [] as ConversationListItem[],
      },
    );
  }, [conversations, searchQuery]);

  if (isLoading && !conversations) {
    return <ConversationListLoading />;
  }

  if (error) {
    return <ConversationListError error={error} />;
  }

  const noResults =
    !conversations ||
    (conversations.length > 0 &&
      groupedConversations.pinned.length === 0 &&
      groupedConversations.today.length === 0 &&
      groupedConversations.last30Days.length === 0 &&
      groupedConversations.older.length === 0) ||
    conversations.length === 0;

  if (noResults) {
    return (
      <ConversationListEmpty
        searchQuery={searchQuery}
        onClearSearch={onClearSearch}
      />
    );
  }

  return (
    <>
      <ConversationGroup
        title={
          <span className="flex items-center gap-1">
            <Pin className="h-3 w-3 fill-current" />
            Pinned
          </span>
        }
        conversations={groupedConversations.pinned}
        onPin={onPin}
        isConversationActive={isConversationActive}
      />
      <ConversationGroup
        title="Today"
        conversations={groupedConversations.today}
        onPin={onPin}
        isConversationActive={isConversationActive}
      />
      <ConversationGroup
        title="Last 30 Days"
        conversations={groupedConversations.last30Days}
        onPin={onPin}
        isConversationActive={isConversationActive}
      />
      <ConversationGroup
        title="Older"
        conversations={groupedConversations.older}
        onPin={onPin}
        isConversationActive={isConversationActive}
      />
    </>
  );
}
