import { useMutation } from "@tanstack/react-query";
import { useState, useCallback, useRef } from "react";
import {
  Adaptive,
  ChatCompletionStreamingResponse,
} from "@adaptive-llm/adaptive-js";
import { Message } from "@adaptive-llm/adaptive-js";

export interface ModelInfo {
  provider: string;
  model: string;
}

/** * Parameters for streaming chat completion */
export type StreamingChatCompletionParams = {
  /** The request to send to the LLM */
  messages: Message[];
  /** Callback for each chunk of the streaming response */
  onChunk?: (chunk: ChatCompletionStreamingResponse) => void;
  /** Callback when streaming is complete */
  onComplete?: (content: string, modelInfo?: ModelInfo) => void;
  /** Callback when an error occurs during streaming */
  onError?: (error: Error) => void;
};

// === Adaptive Client ===
const client = new Adaptive({
  apiKey: import.meta.env.VITE_ADAPTIVE_API_KEY,
  baseUrl: import.meta.env.VITE_API_BASE_URL,
});

// === Hook ===
export const useStreamingChatCompletion = () => {
  const [streamingContent, setStreamingContent] = useState<string | undefined>(
    undefined,
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | undefined>(undefined);
  const modelInfoRef = useRef<ModelInfo | undefined>(undefined);

  const resetStreamingState = useCallback(() => {
    setStreamingContent(undefined);
    setIsStreaming(true);
    setModelInfo(undefined);
    modelInfoRef.current = undefined;
  }, []);

  const mutation = useMutation({
    mutationFn: async ({
      messages,
      onChunk,
      onComplete,
      onError,
    }: StreamingChatCompletionParams) => {
      let accumulatedContent = "";
      const abortController = new AbortController();

      try {
        resetStreamingState();

        const stream = (await client.chat.completions.create({
          messages,
          stream: true,
        })) as AsyncIterable<ChatCompletionStreamingResponse>;

        for await (const chunk of stream) {
          console.log("Chunk received:", chunk);

          // Extract content
          const delta = chunk?.choices?.[0]?.delta?.content;
          if (delta) {
            accumulatedContent += delta;
            setStreamingContent((prev) => (prev ?? "") + delta);
          }

          // Extract model info if present
          if (chunk.provider && chunk.model) {
            const newModelInfo = {
              provider: chunk.provider,
              model: chunk.model,
            };
            setModelInfo(newModelInfo);
            modelInfoRef.current = newModelInfo;
          }

          if (onChunk) onChunk(chunk);
        }

        setIsStreaming(false);
        if (onComplete) {
          onComplete(accumulatedContent, modelInfoRef.current);
        }

        return {
          abortFn: () => abortController.abort(),
          getContent: () => accumulatedContent,
        };
      } catch (error) {
        setIsStreaming(false);
        const err = error instanceof Error ? error : new Error(String(error));
        console.error("Streaming error:", err);
        if (onError) onError(err);
        throw err;
      }
    },
    onError: (error) => {
      setIsStreaming(false);
      console.error("Mutation error:", error);
    },
  });

  const abortStreaming = useCallback(() => {
    if (mutation.data?.abortFn) {
      mutation.data.abortFn();
      setIsStreaming(false);
    }
  }, [mutation.data]);

  return {
    streamChatCompletion: mutation.mutateAsync,
    abortStreaming,
    isStreaming,
    streamingContent,
    modelInfo,
    isPending: mutation.isPending,
    error: mutation.error,
    reset: resetStreamingState,
  };
};
