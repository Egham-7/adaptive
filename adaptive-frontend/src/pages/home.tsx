import { useState } from "react";
import { useConversation } from "@/lib/hooks/use-conversation";
import { ChatFooter } from "@/components/chat/chat-footer";
import { ChatHeader } from "@/components/chat/chat-header";
import { MessageList } from "@/components/chat/message-list";
import { useParams } from "@tanstack/react-router";

export default function ConversationPage() {
  const [showActions, setShowActions] = useState(true);

  const { conversationId } = useParams({
    from: "/_home/conversations/$conversationId",
  });

  const {
    messages,
    sendMessage,
    isLoading,
    error,
    resetConversation,
    lastResponse,
    title,
    setTitle,
  } = useConversation({ conversationId: Number(conversationId) });

  // Toggle action buttons visibility
  const toggleActions = () => {
    setShowActions(!showActions);
  };

  // Get current model info from the last response
  const currentProvider = lastResponse?.provider;
  const currentModel = lastResponse?.response?.model;

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      <ChatHeader
        currentModel={currentModel}
        currentProvider={currentProvider}
        resetConversation={resetConversation}
        title={title}
        setTitle={setTitle}
      />
      <main className="w-full max-w-5xl mx-auto flex flex-col h-screen pt-[80px] pb-[140px]">
        <MessageList messages={messages} isLoading={isLoading} error={error} />
      </main>
      <ChatFooter
        isLoading={isLoading}
        sendMessage={sendMessage}
        showActions={showActions}
        toggleActions={toggleActions}
      />
    </div>
  );
}
