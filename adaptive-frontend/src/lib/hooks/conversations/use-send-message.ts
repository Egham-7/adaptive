import { useMutation } from "@tanstack/react-query";
import { Message } from "@/services/llms/types";
import { useCreateMessage } from "./use-create-message";
import {
  ModelInfo,
  useStreamingChatCompletion,
} from "./use-streaming-chat-completion";
import { CreateDBMessage } from "@/services/messages/types";
import { convertToApiMessage } from "@/services/messages";

export function useSendMessage(conversationId: number, messages: Message[]) {
  const createMessage = useCreateMessage();
  const {
    streamChatCompletion,
    abortStreaming,
    isStreaming,
    streamingContent,
    isPending: isStreamingPending,
    error: streamingError,
    modelInfo,
  } = useStreamingChatCompletion();

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      const userMessage: CreateDBMessage = {
        role: "user",
        content,
      };
      await createMessage.mutateAsync({
        convId: conversationId,
        message: userMessage,
      });

      const chatCompletionsUserMessage = convertToApiMessage(userMessage);

      // Start streaming completion
      const result = await streamChatCompletion({
        request: { messages: [...messages, chatCompletionsUserMessage] },
        onComplete: async (finalContent, modelInfo?: ModelInfo) => {
          const { provider, model } = modelInfo || {};
          if (finalContent) {
            await createMessage.mutateAsync({
              convId: conversationId,
              message: {
                role: "assistant",
                content: finalContent,
                provider,
                model,
              },
            });
          }
        },
      });

      return {
        abortFunction: result.abortFn,
        conversationId,
      };
    },
  });

  const isLoading =
    createMessage.isPending ||
    isStreamingPending ||
    isStreaming ||
    sendMessageMutation.isPending;

  const error =
    sendMessageMutation.error || createMessage.error || streamingError;

  return {
    sendMessage: sendMessageMutation.mutateAsync,
    abortStreaming,
    isLoading,
    isStreaming,
    streamingContent,
    modelInfo,
    error,
  };
}
