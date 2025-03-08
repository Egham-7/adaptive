import { useCallback } from "react";
import { Message } from "@/services/llms/types";
import { typeProviderResponse } from "@/services/llms";
import { useChatCompletion } from "./use-chat-completion";
import { useCreateConversation } from "./use-create-conversation";
import { useCreateMessage } from "./use-create-message";

export const useSendMessage = (conversationId: number, messages: Message[]) => {
  const createConversationMutation = useCreateConversation();
  const createMessageMutation = useCreateMessage();
  const chatCompletionMutation = useChatCompletion();

  const sendMessage = useCallback(
    async (content: string) => {
      // Create message object
      const userMessage: Message = { role: "user", content };

      try {
        // Save user message to database
        await createMessageMutation.mutateAsync({
          convId: conversationId,
          message: userMessage,
        });

        // Get response from AI
        const allMessages = [...messages, userMessage];
        const response = await chatCompletionMutation.mutateAsync({
          messages: allMessages,
        });

        const providerResponse = typeProviderResponse(
          response.provider,
          response.response,
        );

        // Extract content from response and save AI's reply
        const assistantContent = providerResponse.choices[0].message.content;
        await createMessageMutation.mutateAsync({
          convId: conversationId,
          message: { role: "assistant", content: assistantContent },
        });

        return { response, conversationId: conversationId };
      } catch (error) {
        console.error("Error sending message:", error);
        throw error;
      }
    },
    [conversationId, messages, createMessageMutation, chatCompletionMutation],
  );

  const isLoading =
    createConversationMutation.isPending ||
    createMessageMutation.isPending ||
    chatCompletionMutation.isPending;

  const error =
    createConversationMutation.error ||
    createMessageMutation.error ||
    chatCompletionMutation.error;

  return {
    sendMessage,
    isLoading,
    error,
  };
};
