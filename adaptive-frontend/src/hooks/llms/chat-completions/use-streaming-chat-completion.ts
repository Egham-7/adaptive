import { useMutation } from "@tanstack/react-query";
import { useState, useCallback, useRef } from "react";
import OpenAI from "openai";

export interface ModelInfo {
  provider: string;
  model: string;
}

/** * Parameters for streaming chat completion */
export type StreamingChatCompletionParams = {
  /** The request to send to the LLM */
  messages: OpenAI.Chat.ChatCompletionMessageParam[];
  /** The model to use for the completion */
  model?: string;
  /** Callback for each chunk of the streaming response */
  onChunk?: (chunk: OpenAI.Chat.Completions.ChatCompletionChunk) => void;
  /** Callback when streaming is complete */
  onComplete?: (content: string, modelInfo?: ModelInfo) => void;
  /** Callback when an error occurs during streaming */
  onError?: (error: Error) => void;
};

// === OpenAI Client ===
const client = new OpenAI({
  baseURL: `${import.meta.env.VITE_BASE_API_URL}/api`,
  apiKey: import.meta.env.VITE_OPENAI_API_KEY,
  dangerouslyAllowBrowser: true,
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
      model = "gpt-3.5-turbo",
      onChunk,
      onComplete,
      onError,
    }: StreamingChatCompletionParams) => {
      let accumulatedContent = "";
      const abortController = new AbortController();

      try {
        resetStreamingState();

        const stream = await client.chat.completions.create(
          {
            messages,
            stream: true,
            model,
          },
          {
            signal: abortController.signal,
          },
        );

        for await (const chunk of stream) {
          // Extract content
          const delta = chunk?.choices?.[0]?.delta?.content;
          if (delta) {
            accumulatedContent += delta;
            setStreamingContent((prev) => (prev ?? "") + delta);
          }

          // Extract model info if present
          if (chunk.model) {
            const newModelInfo = {
              provider: "openai", // Since we're using OpenAI SDK
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
