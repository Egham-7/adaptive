import { toast } from "sonner";
import { useConversationMessages } from "@/lib/hooks/conversations/use-conversation-message";
// Components
import { Skeleton } from "@/components/ui/skeleton";
import { Cpu, User } from "lucide-react";
import Chat from "@/components/chat/chat";
import { useParams } from "@tanstack/react-router";

export default function ConversationPage() {
  const { conversationId } = useParams({
    from: "/_home/conversations/$conversationId",
  });
  const numericConversationId = Number(conversationId);

  // State for model info

  // Fetch conversation messages
  const {
    data: messages = [],
    isLoading: isLoadingMessages,
    error: messagesError,
  } = useConversationMessages(numericConversationId);

  // Combined loading state for UI elements
  const initialLoading = isLoadingMessages;

  // Simplified Message Skeleton
  const MessageSkeleton = () => (
    <div className="flex-1 w-full py-6 space-y-6 overflow-y-auto px-4">
      {/* Welcome message skeleton */}
      <div className="flex flex-col items-center justify-center text-center opacity-70 mb-8">
        <Skeleton className="w-12 h-12 mb-3 rounded-full" />
        <Skeleton className="w-48 mb-2 h-6" />
        <Skeleton className="w-64 h-4 mb-1" />
        <Skeleton className="w-56 h-4" />
      </div>
      {/* Message skeletons - simplified to just 3 messages */}
      {/* User message */}
      <div className="flex items-start gap-3 justify-end mb-6">
        <div className="flex flex-col gap-1 p-3 rounded-2xl max-w-[90%] bg-primary/20 w-[220px]">
          <Skeleton className="w-full h-4" />
          <Skeleton className="h-4 w-[90%]" />
          <Skeleton className="self-end w-12 h-3 mt-1 opacity-50" />
        </div>
        <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/20">
          <User className="w-4 h-4 text-primary/40" />
        </div>
      </div>
      {/* Assistant message */}
      <div className="flex items-start gap-3 justify-start mb-6">
        <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/10">
          <Cpu className="w-4 h-4 text-primary/40" />
        </div>
        <div className="flex flex-col gap-1 p-3 rounded-2xl max-w-[90%] bg-muted/80 w-[320px]">
          <Skeleton className="w-full h-4" />
          <Skeleton className="h-4 w-[90%]" />
          <Skeleton className="h-4 w-[80%]" />
          <Skeleton className="h-4 w-[85%]" />
          <Skeleton className="self-end w-12 h-3 mt-1 opacity-50" />
        </div>
      </div>
      {/* User message */}
      <div className="flex items-start gap-3 justify-end">
        <div className="flex flex-col gap-1 p-3 rounded-2xl max-w-[90%] bg-primary/20 w-[180px]">
          <Skeleton className="w-full h-4" />
          <Skeleton className="h-4 w-[90%]" />
          <Skeleton className="self-end w-12 h-3 mt-1 opacity-50" />
        </div>
        <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/20">
          <User className="w-4 h-4 text-primary/40" />
        </div>
      </div>
      {/* Typing indicator */}
      <div className="flex items-start gap-3 mt-4">
        <div className="flex items-center justify-center flex-shrink-0 w-8 h-8 rounded-full bg-primary/10">
          <Cpu className="w-4 h-4 text-primary/40" />
        </div>
        <div className="flex items-center gap-1 p-2 rounded-2xl bg-muted/80 w-[60px]">
          <Skeleton className="w-2 h-2 rounded-full" />
          <Skeleton className="w-2 h-2 rounded-full" />
          <Skeleton className="w-2 h-2 rounded-full" />
        </div>
      </div>
    </div>
  );

  // Show toast for non-fatal message loading error
  if (messagesError) {
    toast.error("Failed to load messages", {
      description:
        messagesError instanceof Error
          ? messagesError.message
          : "Please try again",
    });
  }

  // Render different content based on loading and error states
  const renderContent = () => {
    // Fatal error - conversation not found or cannot be loaded

    // Initial loading state
    if (initialLoading) {
      return <MessageSkeleton />;
    }

    return <Chat conversationId={numericConversationId} messages={messages} />;
  };

  return (
    <div className="flex flex-col w-full h-full min-h-full bg-background text-foreground">
      {renderContent()}
    </div>
  );
}
