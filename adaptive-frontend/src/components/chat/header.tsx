import { useState, useEffect } from "react";
import { Search, Plus } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useDebounce } from "@/hooks/use-debounce";
import { ModeToggle } from "../mode-toggle";
import { useNavigate } from "@tanstack/react-router";
import { SidebarTrigger } from "../ui/sidebar";
import { useConversations } from "@/hooks/conversations/use-conversations";
import { Conversation } from "@/services/conversations/types";
import { useCreateConversation } from "@/hooks/conversations/use-create-conversation";

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredConversations, setFilteredConversations] = useState<
    Conversation[]
  >([]);
  const debouncedSearchQuery = useDebounce(searchQuery, 300) as string;
  const navigate = useNavigate();

  const { data: conversations, isLoading, error } = useConversations();
  const { mutateAsync } = useCreateConversation();

  // Filter conversations based on search query
  useEffect(() => {
    if (!conversations || debouncedSearchQuery.trim() === "") {
      setFilteredConversations([]);
      return;
    }

    const filtered = conversations.filter((conversation) =>
      conversation.title
        .toLowerCase()
        .includes(debouncedSearchQuery.toLowerCase()),
    );
    setFilteredConversations(filtered);
  }, [debouncedSearchQuery, conversations]);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleConversationClick = (conversationId: number) => {
    navigate({
      to: `/conversations/$conversationId`,
      params: { conversationId: String(conversationId) },
    });
    setSearchOpen(false);
    setSearchQuery("");
  };

  const handleCreateNewChat = async () => {
    const converation = await mutateAsync("New Conversation");

    const conversationId = String(converation.id);

    navigate({
      to: "/conversations/$conversationId",
      params: { conversationId },
    });
  };

  return (
    <header className="w-full bg-background border-b border-border">
      <div className="flex items-center justify-between h-14 px-4">
        <div className="flex items-center space-x-4">
          <SidebarTrigger />

          <Popover open={searchOpen} onOpenChange={setSearchOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="w-8 h-8 text-muted-foreground hover:text-foreground"
              >
                <Search size={20} />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="p-0 border-bord ml-4">
              <div className="flex items-center p-2">
                <Search className="mr-2 h-4 w-4 shrink-0 text-muted-foreground" />
                <Input
                  placeholder="Search conversations..."
                  className="border-none bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-sm text-foreground"
                  value={searchQuery}
                  onChange={handleSearchChange}
                  autoFocus
                />
              </div>
              <div className="border-t border-border py-2 px-4">
                {isLoading ? (
                  <p className="text-sm text-muted-foreground">
                    Loading conversations...
                  </p>
                ) : error ? (
                  <p className="text-sm text-destructive">
                    Error loading conversations
                  </p>
                ) : filteredConversations.length > 0 ? (
                  <div className="mt-2 space-y-1 max-h-60 overflow-y-auto">
                    {filteredConversations.map((conversation) => (
                      <div
                        key={conversation.id}
                        className="text-sm text-foreground cursor-pointer hover:bg-accent p-2 rounded flex items-center"
                        onClick={() => handleConversationClick(conversation.id)}
                      >
                        <span className="truncate">{conversation.title}</span>
                      </div>
                    ))}
                  </div>
                ) : searchQuery.trim() !== "" ? (
                  <p className="text-sm text-muted-foreground">
                    No conversations found
                  </p>
                ) : conversations && conversations.length > 0 ? (
                  <>
                    <p className="text-xs text-muted-foreground">
                      Recent conversations
                    </p>
                    <div className="mt-2 space-y-1 max-h-60 overflow-y-auto">
                      {conversations.slice(0, 5).map((conversation) => (
                        <div
                          key={conversation.id}
                          className="text-sm text-foreground cursor-pointer hover:bg-accent p-2 rounded flex items-center"
                          onClick={() =>
                            handleConversationClick(conversation.id)
                          }
                        >
                          <span className="truncate">{conversation.title}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No recent conversations
                  </p>
                )}
              </div>
            </PopoverContent>
          </Popover>

          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 text-muted-foreground hover:text-foreground"
            onClick={handleCreateNewChat}
          >
            <Plus size={20} />
          </Button>
        </div>

        <div className="flex items-center space-x-4">
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}
