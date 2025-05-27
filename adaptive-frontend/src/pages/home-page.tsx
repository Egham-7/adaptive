import { useState } from "react";
import { useRouter } from "@tanstack/react-router";
import { useCreateConversation } from "@/hooks/conversations/use-create-conversation";
import { useCreateMessage } from "@/hooks/conversations/use-create-message";
import { EmptyState } from "@/components/chat/empty-state";
import { ChatFooter } from "@/components/chat/chat-footer";

export default function Home() {
  const router = useRouter();
  const { mutateAsync: createConversation, isPending: isCreatingConversation } =
    useCreateConversation();
  const { mutateAsync: createMessage, isPending: isCreatingMessage } =
    useCreateMessage();
  const [submitting, setSubmitting] = useState(false);

  const handleSend = async (message: string) => {
    if (!message.trim()) return;
    setSubmitting(true);
    try {
      // Create a title based on message length
      const words = message.trim().split(" ");
      const title =
        words.length <= 2
          ? message.trim()
          : words.slice(0, 2).join(" ") + "...";

      // First create the conversation
      const conversation = await createConversation({
        title,
      });

      // Then create the initial message
      await createMessage({
        convId: conversation.id,
        message: {
          content: message.trim(),
          role: "user",
        },
      });

      router.navigate({
        to: "/conversations/$conversationId",
        params: { conversationId: String(conversation.id) },
      });
    } finally {
      setSubmitting(false);
    }
  };

  const isLoading = isCreatingConversation || isCreatingMessage || submitting;

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex-1 relative">
        <main className="absolute inset-0 overflow-y-auto">
          <EmptyState onSendMessage={handleSend} isLoading={isLoading} />
        </main>
      </div>

      <ChatFooter
        isLoading={isLoading}
        sendMessage={handleSend}
        isStreaming={false}
        isError={false}
      />
    </div>
  );
}
