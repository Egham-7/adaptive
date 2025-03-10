import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateMessage } from "@/services/messages";
import { DBMessage } from "@/services/messages/types";
import { useDeleteMessages } from "./use-delete-messages";
import { useCreateMessage } from "./use-create-message";
import { typeProviderResponse } from "@/services/llms";
import { useChatCompletion } from "./use-chat-completion";
import { Message, MessageRole } from "@/services/llms/types";

export function useUpdateMessage() {
  const queryClient = useQueryClient();

  const { mutateAsync: deleteMessages } = useDeleteMessages();
  const { mutateAsync: createMessage } = useCreateMessage();
  const { mutateAsync: createChatCompletion } = useChatCompletion();

  return useMutation({
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
      // First update the message
      const updatedMessage = await updateMessage(messageId, updates);

      // Handle additional operations if needed
      // Delete subsequent messages if there are any
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

      // Create updated message list (current message updated + no subsequent messages)
      const updatedMessages = [
        ...messages.slice(0, index),
        { ...messages[index], ...updates },
      ];

      const formattedMessages: Message[] = updatedMessages.map((dbMsg) => ({
        role: dbMsg.role as MessageRole,
        content: dbMsg.content,
      }));

      // Generate new AI response
      const response = await createChatCompletion({
        messages: formattedMessages,
      });
      const providerResponse = typeProviderResponse(
        response.provider,
        response.response,
      );

      // Extract content and create new assistant message
      const assistantContent = providerResponse.choices[0].message.content;
      await createMessage({
        convId: conversationId,
        message: { role: "assistant", content: assistantContent },
      });

      return updatedMessage;
    },

    onSuccess: (_updatedMessage, variables) => {
      // Invalidate and refetch relevant queries
      queryClient.invalidateQueries({
        queryKey: ["conversations", variables.conversationId],
      });

      queryClient.invalidateQueries({
        queryKey: ["messages", variables.conversationId],
      });
    },
  });
}

