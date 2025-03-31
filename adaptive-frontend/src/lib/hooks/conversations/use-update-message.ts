import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateMessage } from "@/services/messages";
import { DBMessage } from "@/services/messages/types";
import { useDeleteMessages } from "./use-delete-messages";
import { useCreateMessage } from "./use-create-message";
import { Message, MessageRole } from "@/services/llms/types";
import { useAuth } from "@clerk/clerk-react";
import { toast } from "sonner";
import { useStreamingChatCompletion } from "./use-streaming-chat-completion";

export function useUpdateMessage() {
  const queryClient = useQueryClient();
  const { mutateAsync: deleteMessages } = useDeleteMessages();
  const { mutateAsync: createMessage } = useCreateMessage();
  const {
    streamChatCompletion,
    streamingContent,
    isStreaming,
    abortStreaming,
  } = useStreamingChatCompletion();
  const { getToken, isLoaded, isSignedIn } = useAuth();

  return {
    mutation: useMutation({
      mutationFn: async ({
        conversationId,
        messageId,
        updates,
        index,
        messages,
      }: {
        conversationId: number;
        messageId: number;
        updates: { role?: string; content?: string };
        index: number;
        messages: DBMessage[];
      }) => {
        if (!isLoaded || !isSignedIn) {
          throw new Error("User is not signed in");
        }

        const token = await getToken();
        if (!token) {
          throw new Error("User is not signed in");
        }

        // Update the message
        const updatedMessage = await updateMessage(messageId, updates, token);

        // Handle subsequent messages deletion if any exist
        const hasSubsequentMessages = index < messages.length - 1;
        if (hasSubsequentMessages) {
          const subsequentMessageIds = messages
            .slice(index + 1)
            .map((msg) => msg.id);
          await deleteMessages({
            messageIds: subsequentMessageIds,
            conversationId,
          });
        }

        // Prepare updated message list
        const updatedMessages = [
          ...messages.slice(0, index),
          { ...messages[index], ...updates },
        ];

        const formattedMessages: Message[] = updatedMessages.map((dbMsg) => ({
          role: dbMsg.role as MessageRole,
          content: dbMsg.content,
        }));

        // Stream the AI response
        await streamChatCompletion({
          request: {
            messages: formattedMessages,
          },
          onComplete: async (content) => {
            // Create the message only after we have the complete content
            await createMessage({
              convId: conversationId,
              message: { role: "assistant", content },
            });
          },
          onError: (error) => {
            toast.error("Failed to generate response");
            console.error("Streaming error:", error);
          },
        });

        return updatedMessage;
      },
      onSuccess: (_updatedMessage, variables) => {
        queryClient.invalidateQueries({
          queryKey: ["conversations", variables.conversationId],
        });
        queryClient.invalidateQueries({
          queryKey: ["messages", variables.conversationId],
        });
      },
      onError: () => {
        toast.error("Failed to update message");
      },
    }),
    streamingContent,
    isStreaming,
    abortStreaming,
  };
}
