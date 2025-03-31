import { useMutation } from "@tanstack/react-query";
import { Message } from "@/services/llms/types";
import { useCreateMessage } from "./use-create-message";
import { useStreamingChatCompletion } from "./use-streaming-chat-completion";

export function useSendMessage(conversationId: number, messages: Message[]) {
  const createMessage = useCreateMessage();
  const {
    streamChatCompletion,
    abortStreaming,
    isStreaming,
    streamingContent,
    modelInfo,
    isPending: isStreamingPending,
    error: streamingError,
  } = useStreamingChatCompletion();

  const sendMessageMutation = useMutation({
    mutationFn: async (content: string) => {
      // Store user message
      const userMessage: Message = { role: "user", content };
      await createMessage.mutateAsync({
        convId: conversationId,
        message: userMessage,
      });

      // Start streaming completion
      const result = await streamChatCompletion({
        request: { messages: [...messages, userMessage] },
        onComplete: async (finalContent) => {
          if (finalContent) {
            await createMessage.mutateAsync({
              convId: conversationId,
              message: { role: "assistant", content: finalContent },
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
