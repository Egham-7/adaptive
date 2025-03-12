import { useCallback, useState } from "react";
import { Message, StreamingResponse } from "@/services/llms/types";
import { extractContentFromStreamingResponse } from "@/services/llms/types";
import { useCreateMessage } from "./use-create-message";
import { useStreamingChatCompletion } from "./use-streaming-chat-completion";

interface ModelInfo {
  provider?: string;
  model?: string;
}

export const useSendMessage = (conversationId: number, messages: Message[]) => {
  const createMessageMutation = useCreateMessage();
  const streamingMutation = useStreamingChatCompletion();

  // State to track streaming and model information
  const [streamingContent, setStreamingContent] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  const updateStreamingContent = useCallback((newContent: string) => {
    // Using requestAnimationFrame for smoother UI updates
    requestAnimationFrame(() => {
      setStreamingContent((prev) => prev + newContent);
    });
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      // Create message object
      const userMessage: Message = {
        role: "user",
        content,
      };

      try {
        // Save user message to database
        await createMessageMutation.mutateAsync({
          convId: conversationId,
          message: userMessage,
        });

        // Reset streaming state
        setStreamingContent("");
        setIsStreaming(true);
        setModelInfo(null);

        // Get all messages including the new user message
        const allMessages = [...messages, userMessage];

        // Start streaming the AI response
        const abortFunction = await streamingMutation.mutateAsync({
          request: {
            messages: allMessages,
          },
          onChunk: (chunk: StreamingResponse) => {
            console.log("Chunk:", chunk);
            // Extract content
            const newContent = extractContentFromStreamingResponse(chunk);

            // Update model info if available in the chunk
            if (chunk.model && !modelInfo) {
              setModelInfo({
                provider: chunk.provider,
                model: chunk.model,
              });
            }

            // Update streaming content
            if (newContent) {
              updateStreamingContent(newContent);
            }
          },
          onComplete: () => {
            setIsStreaming(false);

            // Save the complete AI response to the database
            if (streamingContent) {
              createMessageMutation.mutate({
                convId: conversationId,
                message: { role: "assistant", content: streamingContent },
              });
            }
          },
        });

        return {
          abortFunction,
          conversationId,
        };
      } catch (error) {
        setIsStreaming(false);
        console.error("Error sending message:", error);
        throw error;
      }
    },
    [
      conversationId,
      messages,
      createMessageMutation,
      streamingMutation,
      streamingContent,
      modelInfo,
      updateStreamingContent,
    ],
  );

  const abortStreaming = useCallback(() => {
    if (streamingMutation.data) {
      streamingMutation.data();
      setIsStreaming(false);
    }
  }, [streamingMutation]);

  const isLoading =
    createMessageMutation.isPending ||
    streamingMutation.isPending ||
    isStreaming;

  const error = createMessageMutation.error || streamingMutation.error;

  return {
    sendMessage,
    abortStreaming,
    isLoading,
    isStreaming,
    streamingContent,
    modelInfo,
    error,
  };
};
