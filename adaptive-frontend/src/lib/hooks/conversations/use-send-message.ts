import { useCallback, useState, useEffect } from "react";
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

  // Monitor streaming mutation errors
  useEffect(() => {
    if (streamingCompletion.error && isStreaming) {
      setIsStreaming(false);
    }
  }, [streamingCompletion.error, isStreaming]);

  const sendMessage = useCallback(
    async (content: string) => {
      const userMessage: Message = { role: "user", content };
      let accumulatedContent = "";

      try {
        // Store user message
        await createMessage.mutateAsync({
          convId: conversationId,
          message: userMessage,
        });

        // Reset streaming state
        setStreamingContent("");
        setIsStreaming(true);
        setModelInfo(null);

        // Start streaming completion
        const abortFunction = await streamingCompletion.mutateAsync({
          request: { messages: [...messages, userMessage] },
          onChunk: (chunk: StreamingResponse) => {
            // Set model info if available
            if (chunk.model && !modelInfo) {
              setModelInfo({
                provider: chunk.provider,
                model: chunk.model,
              });
            }

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

            // Store assistant message if there's content
            if (accumulatedContent) {
              await createMessage.mutateAsync({
                convId: conversationId,
                message: { role: "assistant", content: accumulatedContent },
              });
            }
          },
        });

        return { abortFunction, conversationId };
      } catch (error) {
        setIsStreaming(false);
        console.error("Error sending message:", error);
        throw error;
      }
    },
    [conversationId, messages, createMessage, streamingCompletion, modelInfo],
  );

  const abortStreaming = useCallback(() => {
    if (streamingCompletion.data) {
      streamingCompletion.data();
      setIsStreaming(false);
    }
  }, [streamingCompletion]);

  const isLoading =
    createMessage.isPending || streamingCompletion.isPending || isStreaming;

  const error = createMessage.error || streamingCompletion.error;

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
