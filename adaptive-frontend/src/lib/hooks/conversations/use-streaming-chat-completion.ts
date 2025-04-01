import { useMutation } from "@tanstack/react-query";
import {
  ChatCompletionRequest,
  StreamingResponse,
} from "@/services/llms/types";
import { createStreamingChatCompletion } from "@/services/llms";
import { useState, useCallback, useRef } from "react";
import { extractContentFromStreamingResponse } from "@/services/llms/types";
import { isErrorResponse } from "@/services/llms/types";

/**
 * Parameters for streaming chat completion
 */
export type StreamingChatCompletionParams = {
  /** The request to send to the LLM */
  request: ChatCompletionRequest;
  /** Callback for each chunk of the streaming response */
  onChunk?: (chunk: StreamingResponse) => void;
  /** Callback when streaming is complete */
  onComplete?: (content: string, modelInfo?: ModelInfo) => void;
  /** Callback when an error occurs during streaming */
  onError?: (error: Error) => void;
};

export interface ModelInfo {
  provider: string;
  model: string;
}

/**
 * Enhanced hook for streaming chat completions with built-in state management
 */
export const useStreamingChatCompletion = () => {
  // State
  const [streamingContent, setStreamingContent] = useState<string | undefined>(
    undefined,
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | undefined>(undefined);

  // Use a ref to track the latest model info for callbacks
  const modelInfoRef = useRef<ModelInfo | undefined>(undefined);

  // Reset state function
  const resetStreamingState = useCallback(() => {
    setStreamingContent(undefined);
    setIsStreaming(true);
    setModelInfo(undefined);
    modelInfoRef.current = undefined;
  }, []);

  const mutation = useMutation({
    mutationFn: async ({
      request,
      onChunk,
      onComplete,
      onError,
    }: StreamingChatCompletionParams) => {
      let accumulatedContent = "";
      try {
        resetStreamingState();
        // Start the streaming process and get the abort function
        const abortFn = createStreamingChatCompletion(
          request,
          (chunk: StreamingResponse) => {
            // Handle model info
            if (!isErrorResponse(chunk) && chunk.model && chunk.provider) {
              const newModelInfo = {
                provider: chunk.provider,
                model: chunk.model,
              };
              // Update both state and ref
              setModelInfo(newModelInfo);
              modelInfoRef.current = newModelInfo;
            }
            // Extract and accumulate content
            const newContent = extractContentFromStreamingResponse(chunk);
            if (newContent) {
              accumulatedContent += newContent;
              requestAnimationFrame(() => {
                setStreamingContent(accumulatedContent);
              });
            }
            // Call external onChunk if provided
            if (onChunk) {
              onChunk(chunk);
            }
          },
          () => {
            setIsStreaming(false);
            if (onComplete) {
              // Use the ref to get the latest model info
              onComplete(accumulatedContent, modelInfoRef.current);
            }
          },
          (error) => {
            setIsStreaming(false);
            console.error("Streaming error:", error);
            if (onError) onError(error);
          },
        );

        // Return the abort function and accumulated content
        return {
          abortFn,
          getContent: () => accumulatedContent,
        };
      } catch (error) {
        setIsStreaming(false);
        const formattedError =
          error instanceof Error ? error : new Error(String(error));
        console.error("Chat completion setup error:", formattedError);
        if (onError) {
          onError(formattedError);
        }
        throw formattedError;
      }
    },
    onError: (error) => {
      setIsStreaming(false);
      console.error("Chat completion mutation error:", error);
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
