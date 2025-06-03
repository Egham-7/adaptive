import { useMutation } from "@tanstack/react-query";
import { useCreateMessage } from "./use-create-message";
import {
  useStreamingChatCompletion,
  ModelInfo,
} from "../llms/chat-completions/use-streaming-chat-completion";
import { CreateDBMessage } from "@/services/messages/types";
import { convertToApiMessage } from "@/services/messages";

import OpenAI from "openai";

export function useSendMessage(
  conversationId: number,
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
) {
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
        messages: [...messages, chatCompletionsUserMessage],
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

  const isError = sendMessageMutation.isError || createMessage.isError;

  return {
    sendMessage: sendMessageMutation.mutateAsync,
    abortStreaming,
    isLoading,
    isStreaming,
    streamingContent,
    modelInfo,
    error,
    isError,
  };
}
