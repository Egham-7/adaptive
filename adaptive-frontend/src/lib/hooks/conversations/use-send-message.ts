import { useCallback, useState } from "react";
import { Message } from "@/services/llms/types";
import { useCreateMessage } from "./use-create-message";
import { useStreamingChatCompletion } from "./use-streaming-chat-completion";

export function useSendMessage(conversationId: number, messages: Message[]) {
  // Mutations
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

  // State
  const [sendError, setSendError] = useState<Error | null>(null);

  const sendMessage = useCallback(
    async (content: string) => {
      try {
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
            // Store assistant message when streaming completes
            if (finalContent) {
              await createMessage.mutateAsync({
                convId: conversationId,
                message: { role: "assistant", content: finalContent },
              });
            }
          },
          onError: (error) => {
            setSendError(new Error("Failed to send message: " + error.message));
          },
        });

        return {
          abortFunction: result.abortFn,
          conversationId,
        };
      } catch (error) {
        const formattedError =
          error instanceof Error ? error : new Error(String(error));
        setSendError(
          new Error("Failed to send message: " + formattedError.message),
        );
        console.error("Error sending message:", formattedError);
        throw formattedError;
      }
    },
    [conversationId, messages, createMessage, streamChatCompletion],
  );

  const isLoading =
    createMessage.isPending || isStreamingPending || isStreaming;

  // Combine all possible errors
  const error = sendError || createMessage.error || streamingError;

  return {
    sendMessage,
    abortStreaming,
    isLoading,
    isStreaming,
    streamingContent,
    modelInfo,
    error,
  };
}
