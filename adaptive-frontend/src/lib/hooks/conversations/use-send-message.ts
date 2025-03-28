import { useCallback, useState } from "react";
import { Message, StreamingResponse } from "@/services/llms/types";
import { extractContentFromStreamingResponse } from "@/services/llms/types";
import { useCreateMessage } from "./use-create-message";
import { useStreamingChatCompletion } from "./use-streaming-chat-completion";

interface ModelInfo {
  provider?: string;
  model?: string;
}

export function useSendMessage(conversationId: number, messages: Message[]) {
  // Mutations
  const createMessage = useCreateMessage();
  const streamingCompletion = useStreamingChatCompletion();

  // State
  const [streamingContent, setStreamingContent] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [sendError, setSendError] = useState<Error | null>(null);

  // Memoize helper functions to avoid dependency issues
  const resetStreamingState = useCallback(() => {
    setStreamingContent("");
    setIsStreaming(true);
    setSendError(null);
    setModelInfo(null);
  }, []);

  const handleStreamingError = useCallback((error: unknown) => {
    setIsStreaming(false);
    const formattedError =
      error instanceof Error ? error : new Error(String(error));
    setSendError(
      new Error("Failed to send message: " + formattedError.message),
    );
    console.error("Error sending message:", formattedError);
    return formattedError;
  }, []);

  const storeAssistantMessage = useCallback(
    async (content: string) => {
      if (!content) return;

      await createMessage.mutateAsync({
        convId: conversationId,
        message: { role: "assistant", content },
      });
    },
    [createMessage, conversationId],
  );

  const handleModelInfo = useCallback(
    (chunk: StreamingResponse) => {
      if ((chunk.model && chunk.provider) && !modelInfo) {
        setModelInfo({
          provider: chunk.provider,
          model: chunk.model,
        });
      }
    },
    [modelInfo],
  );

  const sendMessage = useCallback(
    async (content: string) => {
      let accumulatedContent = "";

      try {
        // Move storeUserMessage inside to avoid dependency issues
        const userMessage: Message = { role: "user", content };
        await createMessage.mutateAsync({
          convId: conversationId,
          message: userMessage,
        });

        resetStreamingState();

        // Start streaming completion
        const abortFunction = await streamingCompletion.mutateAsync({
          request: { messages: [...messages, userMessage] },

          onChunk: (chunk: StreamingResponse) => {
            handleModelInfo(chunk);

            // Update content
            const newContent = extractContentFromStreamingResponse(chunk);
            if (newContent) {
              accumulatedContent += newContent;
              requestAnimationFrame(() => {
                setStreamingContent(accumulatedContent);
              });
            }
          },

          onComplete: async () => {
            setIsStreaming(false);
            await storeAssistantMessage(accumulatedContent);
          },

          onError: (error) => {
            handleStreamingError(error);
          },
        });

        return { abortFunction, conversationId };
      } catch (error) {
        const formattedError = handleStreamingError(error);
        throw formattedError;
      }
    },
    [
      conversationId,
      messages,
      createMessage,
      streamingCompletion,
      resetStreamingState,
      handleModelInfo,
      storeAssistantMessage,
      handleStreamingError,
    ],
  );

  const abortStreaming = useCallback(() => {
    if (streamingCompletion.data) {
      streamingCompletion.data();
      setIsStreaming(false);
    }
  }, [streamingCompletion]);

  const isLoading =
    createMessage.isPending || streamingCompletion.isPending || isStreaming;

  // Combine all possible errors
  const error = sendError || createMessage.error || streamingCompletion.error;

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
