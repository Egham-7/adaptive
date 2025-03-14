import { useMutation, UseMutationResult } from "@tanstack/react-query";
import {
  ChatCompletionRequest,
  StreamingResponse,
} from "@/services/llms/types";
import { createStreamingChatCompletion } from "@/services/llms";

// Create a structured input type for the mutation
export type StreamingChatCompletionParams = {
  request: ChatCompletionRequest;
  onChunk: (chunk: StreamingResponse) => void;
  onComplete?: () => void;
};

/**
 * Custom hook for streaming chat completions with TanStack Query
 *
 * @returns A mutation object with an abort function in the data field
 */
export const useStreamingChatCompletion = (): UseMutationResult<
  () => void, // Return type is the abort function
  Error,
  StreamingChatCompletionParams
> => {
  return useMutation({
    mutationFn: ({
      request,
      onChunk,
      onComplete,
    }: StreamingChatCompletionParams) => {
      // Create a Promise that resolves to the abort function
      return new Promise<() => void>((resolve) => {
        // Start the streaming process
        const abortFn = createStreamingChatCompletion(
          request,
          onChunk,
          onComplete,
          (error) => {
            console.error("Chat completion error:", error);
          },
        );

        // Immediately resolve with the abort function
        resolve(abortFn);
      });
    },
    onError: (error) => {
      console.error("Chat completion error:", error);
    },
  });
};
